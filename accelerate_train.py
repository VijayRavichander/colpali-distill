import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import numpy as np
import os
from dotenv import load_dotenv
from collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.models import ColIdefics3Processor, ColIdefics3
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from loss import ColBertPairwiseDistillLoss
from transformers import get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
import wandb

@dataclass
class TrainingConfig:
    student_model: str = "vidore/ColSmolVLM-Instruct-256M-base"
    teacher_model: Optional[str] = None
    train_dataset: str = "https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/"
    processor: Optional[BaseVisualRetrieverProcessor] = None
    teacher_processor: Optional[BaseVisualRetrieverProcessor] = None,
    loss_fn: Optional[Callable] = ColBertPairwiseDistillLoss()
    seed: int = 42
    working_directory: str = "./training_artifacts"

    #TRAINING ARGUEMENTS:
    max_length: int = 60
    learning_rate: float = 1e-5
    epochs: int = 1
    peft_config: Optional[LoraConfig] = None
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    train_samples: int = 80
    test_samples: int = 150
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 4
    resume_from_checkpoint: Optional[str] = None
    logging_steps: int = 1
    checkpoint_interval: int = 10
    log_wandb: bool = True
    eval_steps: int = 5

def prepare_dataset(config):
    data_files = {"train": [config.train_dataset + "train-00001-of-00082.parquet", 
                                config.train_dataset + "train-00002-of-00082.parquet",
                                config.train_dataset + "train-00003-of-00082.parquet", 
                                config.train_dataset + "train-00004-of-00082.parquet", 
                                config.train_dataset + "train-00005-of-00082.parquet", 
                                config.train_dataset + "train-00006-of-00082.parquet", 
                                config.train_dataset + "train-00007-of-00082.parquet",
                                config.train_dataset + "train-00008-of-00082.parquet", 
                                config.train_dataset + "train-00009-of-00082.parquet", 
                                config.train_dataset + "train-00010-of-00082.parquet", 
                                config.train_dataset + "train-00011-of-00082.parquet", 
                                config.train_dataset + "train-00012-of-00082.parquet", 
                                config.train_dataset + "train-00013-of-00082.parquet", 
                                config.train_dataset + "train-00014-of-00082.parquet", 
                                config.train_dataset + "train-00015-of-00082.parquet", 
                                config.train_dataset + "train-00016-of-00082.parquet", 
                                config.train_dataset + "train-00017-of-00082.parquet", 
                                config.train_dataset + "train-00018-of-00082.parquet", 
                                config.train_dataset + "train-00019-of-00082.parquet"
                                ],
                    "test": config.train_dataset + "test-00000-of-00001.parquet"    
                }

    dataset = load_dataset("parquet", data_files=data_files)

    return dataset


if __name__ == "__main__":


    config = TrainingConfig()

    set_seed(config.seed)

    accelerator = Accelerator(project_dir = config.working_directory,
                          log_with="wandb")
    
    if accelerator.is_local_main_process:
        wandb.init(project="nano-distributed-training")

    dataset = prepare_dataset(config)

    config.processor = ColIdefics3Processor.from_pretrained(config.student_model)
    
    if config.get("teacher_processor", False):
        # Teacher Processor
        teacher_processor = ColIdefics3Processor.from_pretrained(config.student_model)

        # Teacher Model
        teacher_model = ColIdefics3.from_pretrained(config.teacher_model,  torch_dtype=torch.float16)

    
    config.peft_config = LoraConfig(
                                    r=32,
                                    lora_alpha=32,
                                    lora_dropout=0.1,
                                    init_lora_weights="gaussian",
                                    bias="none",
                                    task_type="FEATURE_EXTRACTION",
                                    target_modules=r"(.*(model.text_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)", 
                                    modules_to_save = ["linear"]
                                )


    collator = VisualRetrieverCollator(config.processor)

    train_dataloader = DataLoader(dataset["train"].take(config.train_samples), batch_size = config.train_batch_size, collate_fn = collator)
    eval_dataloader = DataLoader(dataset["test"].take(config.test_samples), batch_size = config.eval_batch_size, collate_fn = collator)

    model = ColIdefics3.from_pretrained(config.student_model, torch_dtype=torch.float16, attn_implementation="eager")
    loss_fn = ColBertPairwiseDistillLoss()


    if config.peft_config is not None:
        accelerator.print("Configurating PEFT model")
        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config.peft_config)
        model.print_trainable_parameters()
    
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr = config.learning_rate, weight_decay = config.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = config.epochs * len(train_dataloader) * accelerator.num_processes
                                                )
    
    if config.get("teacher_model", False):
        model, teacher_model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
                    model, teacher_model,  optimizer, train_dataloader, eval_dataloader, scheduler
                )
    else:

        model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
        )

    accelerator.register_for_checkpointing(scheduler)


    if config.resume_from_checkpoint is not None:

        path_to_checkpoint = os.path.join(config.working_directory, config.resume_from_checkpoint)

        with accelerator.main_process_first():
            accelerator.load_state(path_to_checkpoint)
        
        completed_steps = int(path_to_checkpoint.split("_")[-1])
        accelerator.print(f"Resuming Training from Checkpoint: {config.resume_from_checkpoint}")
        
    else:
        completed_steps = 0

    train = True

    # Total Number of Training Steps
    num_samples = config.train_samples
    steps_per_epoch = num_samples // (config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps)
    num_training_steps = config.epochs * steps_per_epoch

    progress_bar = tqdm(range(completed_steps, num_training_steps), disable = not accelerator.is_local_main_process)

    accelerator.print("Training")

    while train:

        train_loss = []
        eval_loss = []
        
        accumulated_loss = 0
        accumulated_steps = 0
        
        if config.get("teacher_model", False):
            teacher_model.eval()
            
        model.train()
        for inputs in train_dataloader:
            
            # accelerator.print("Forward")

            # forward using the query inputs
            query_outputs = model(input_ids=inputs["query_input_ids"].to(accelerator.device), attention_mask=inputs["query_attention_mask"].to(accelerator.device))

            # accelerator.print("Forward Query")

            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v.to(device = accelerator.device) for k, v in inputs.items() if k.startswith("doc")})

            if config.get("teacher_model", False):
                with torch.no_grad():
                    teacher_query_outputs = teacher_model(
                                                            input_ids=inputs["teacher_query_input_ids"].to(accelerator.device),
                                                            attention_mask=inputs["teacher_query_attention_mask"].to(accelerator.device)
                                                        )
                    
                    teacher_doc_outputs = teacher_model(**{k[12:]: v.to(accelerator.device) for k, v in inputs.items() if k.startswith("teacher_doc")})

            # accelerator.print("Forward Doc")

            if config.get("teacher_model", False):
                loss = loss_fn(query_outputs, doc_outputs, teacher_query_outputs, teacher_doc_outputs, eval = False)

            else:
                # For Eval and No Distill
                loss = loss_fn(query_outputs, doc_outputs, eval = True)


            accumulated_loss += loss 

            # accelerator.print(f"Loss: {accumulated_loss}")

            ### Compute Gradients ###
            accelerator.backward(loss)

            # accelerator.print("Backward")

            accumulated_steps += 1

            if accumulated_steps % config.gradient_accumulation_steps == 0:
                ### Clip Gradients ###
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
                ### Update Model ###
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # accelerator.print("Update Params")

                ### Update Learning Rate ###
                scheduler.step()

                if completed_steps % config.logging_steps == 0:
                    accumulated_loss = accumulated_loss.detach()

                    if accelerator.num_processes > 1:

                        accumulated_loss = torch.mean(accelerator.gather_for_metrics(accumulated_loss))
                    
                    log = {
                            "train_loss": accumulated_loss,
                            "learning_rate": scheduler.get_last_lr()[0]
                        }

                    logging_string = f"[{completed_steps}/{num_training_steps}] Training Loss: {accumulated_loss}"

                    accelerator.print(logging_string)

                    if accelerator.is_local_main_process:
                        progress_bar.write(logging_string)
                    
                    if config.log_wandb:
                        accelerator.log(log, step = completed_steps)

                if completed_steps % config.eval_steps == 0:
                    accelerator.print("Eval Time!!!!")
                    
                    model.eval()

                    log = {
                        "eval_loss": 0
                    }

                    num_losses = 0

                    for inputs in tqdm(eval_dataloader, disable = not accelerator.is_main_process):

                        with torch.no_grad():
                            # forward using the query inputs
                            query_outputs = model(input_ids=inputs["query_input_ids"].to(accelerator.device), attention_mask=inputs["query_attention_mask"].to(accelerator.device))

                            # feed only kwargs with 'doc_' prefix
                            doc_outputs = model(**{k[4:]: v.to(device = accelerator.device) for k, v in inputs.items() if k.startswith("doc")})

                        loss = loss_fn(query_outputs, doc_outputs, eval = True).detach()

                        if accelerator.num_processes > 1:
                            loss = torch.mean(accelerator.gather_for_metrics(loss))
                        
                        log["eval_loss"] += loss
                        num_losses += 1

                    log["eval_loss"] = log["eval_loss"] / num_losses

                    logging_string = f"[{completed_steps}/{num_training_steps}] Eval Loss: {accumulated_loss}"

                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)
                    
                    if config.log_wandb:
                        accelerator.log(log, step = completed_steps)

                    model.train()

                if completed_steps % config.checkpoint_interval == 0:
                    
                    # accelerator.print("Checkpointing")

                    path_to_checkpoint = os.path.join(config.working_directory, f"checkpoint_{completed_steps}")

                    if accelerator.is_main_process:
                        progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")
                    
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        accelerator.save_state(output_dir=path_to_checkpoint)

                if completed_steps >= num_training_steps:
                    train = False

                    if accelerator.is_main_process:
                        progress_bar.write("Completed Training")
                    
                    break

                ### Iterate Progress Bar and Completed Steps ###
                completed_steps += 1
                progress_bar.update(1)

                accumulated_loss = 0

accelerator.end_training()

