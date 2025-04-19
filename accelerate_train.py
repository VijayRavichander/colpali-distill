import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Optional
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
    train_dataset: str = "https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/"
    processor = None
    teacher_processor: Optional[BaseVisualRetrieverProcessor] = None,
    max_length: int = 60
    seed: int = 42
    learning_rate: float = 1e-5
    epochs: int = 1
    peft_config = None
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


def prepare_dataset(config):
    data_files = {  "train": [config.train_dataset + "train-00001-of-00082.parquet", 
                            # config.train_dataset + "train-00002-of-00082.parquet",
                            # config.train_dataset + "train-00003-of-00082.parquet", 
                            # config.train_dataset + "train-00004-of-00082.parquet", 
                            # config.train_dataset + "train-00005-of-00082.parquet", 
                            # config.train_dataset + "train-00006-of-00082.parquet"
                            ], 
                    "test": config.train_dataset + "test-00000-of-00001.parquet"    
                }

    dataset = load_dataset("parquet", data_files=data_files)

    return dataset



if __name__ == "__main__":

    config = TrainingConfig()

    set_seed(config.seed)

    load_dotenv()

    wandb_token = os.getenv("WANDB_TOKEN")
    os.environ["WANDB_API_KEY"] = wandb_token


    accelerator = Accelerator(project_dir="./",
                          log_with="wandb")
    


    if accelerator.is_local_main_process:
        wandb.init(project="nano-distributed-training")

    dataset = prepare_dataset(config)
    
    student_model = "vidore/ColSmolVLM-Instruct-256M-base" #ARGPARSE

    config.student_model = student_model
    config.processor = ColIdefics3Processor.from_pretrained(config.student_model)
    
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

    train_dataloader = DataLoader(dataset["train"], batch_size = 2, collate_fn = collator)
    eval_dataloader = DataLoader(dataset["test"], batch_size = 2, collate_fn = collator)

    model = ColIdefics3.from_pretrained(student_model, torch_dtype=torch.float16, attn_implementation="eager")
    loss_fn = ColBertPairwiseDistillLoss()


    if config.peft_config is not None:
        print("Configurating PEFT model")
        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config.peft_config)
        model.print_trainable_parameters()
    
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())


    optimizer = torch.optim.AdamW(params_to_train, lr = config.learning_rate, weight_decay= config.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps = 0 , 
                                            num_training_steps = config.epochs * len(train_dataloader) * accelerator.num_processes)
    
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    for epoch in range(config.epochs):

        accelerator.print(f"Training Epoch {epoch}")

        train_loss = []
        eval_loss = []
        
        progress_bar = tqdm(range(len(trainloader)), disable=not accelerator.is_local_main_process)

        model.train()

        for inputs in train_dataloader:
            
            # forward using the query inputs
            query_outputs = model(input_ids=inputs["query_input_ids"].to(accelerator.device), attention_mask=inputs["query_attention_mask"].to(accelerator.device))

            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v.to(dtype=torch.float16, device = accelerator.device) for k, v in inputs.items() if k.startswith("doc")})

            loss = loss_fn(query_outputs, doc_outputs, eval = True)
            
            ### Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(loss)

            ### Store Current Iteration Error ###
            eval_loss.append(torch.mean(loss_gathered).item())


        for inputs in eval_dataloader:

            with torch.no_grad():
                # forward using the query inputs
                query_outputs = model(input_ids=inputs["query_input_ids"].to(accelerator.device), attention_mask=inputs["query_attention_mask"].to(accelerator.device))

                # feed only kwargs with 'doc_' prefix
                doc_outputs = model(**{k[4:]: v.to(dtype=torch.float32, device = accelerator.device) for k, v in inputs.items() if k.startswith("doc")})

            loss = loss_fn(query_outputs, doc_outputs, eval = True)


            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(loss)

            ### Store Current Iteration Error ###
            train_loss.append(torch.mean(loss_gathered).item())
            
            ### Iterate Progress Bar ###
            progress_bar.update(1)

            ### Update Learning Rate ###
            scheduler.step()


        epoch_train_loss = np.mean(train_loss)
        epoch_eval_loss = np.mean(eval_loss)

        accelerator.print(f"Train Loss:", epoch_train_loss)
        accelerator.print(f"Eval Loss:", epoch_eval_loss)
        
        ### Log with Weights and Biases ###
        accelerator.log({"training_loss": epoch_train_loss,
                        "evaling_loss": epoch_eval_loss}, step=epoch)
    
    ### Save Final Model ###
    accelerator.wait_for_everyone()
    
    # if config.peft_config:
    #     accelerator.unwrap_model(model).save_model(os.path.join(working_directory, experiment_name, "food_merged_checkpoint.safetensors"), merge_weights=True)

    ### End Training for Trackers to Exit ###
    accelerator.end_training()
