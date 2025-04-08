from pathlib import Path
import typer
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.models import ColIdefics3Processor, ColIdefics3
import torch
from peft import LoraConfig
from train_distill_colpali import ColModelDistillTraining, ColModelDistillTrainingConfig
from transformers import TrainingArguments
import wandb
import os
from dotenv import load_dotenv
import datetime
from loss import ColBertPairwiseDistillKLLoss
from colpali_engine.models import ColQwen2, ColQwen2Processor


# def main(config_file: Path) -> None:
def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Access the tokens
    hf_token = os.getenv("HF_TOKEN")
    wandb_token = os.getenv("WANDB_TOKEN")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["WANDB_API_KEY"] = wandb_token

    print("Loading config")

    wandb.init(project="colpali-distill")

    IS_DISTILL_TRAINING = True

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=r"(.*(model.text_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)", 
        modules_to_save = ["linear"]
    )

    training_args = TrainingArguments(
        output_dir = None,
        num_train_epochs = 4,
        gradient_accumulation_steps = 1,
        per_device_train_batch_size= 2,
        per_device_eval_batch_size= 2,
        weight_decay = 0.01,
        fp16 = True, 
        learning_rate = 2e-6,
        optim = "paged_adamw_8bit", 
        logging_steps= 2,
        logging_strategy="steps", 
        eval_strategy = "epoch",
        eval_on_start = True, 
        max_grad_norm = 1, 
        report_to=["wandb"], 
    )

    if IS_DISTILL_TRAINING:

        teacher_model = "vidore/colqwen2-v0.1"
        student_model = "vidore/ColSmolVLM-Instruct-256M-base"
        train_samples_size = 200
        eval_samples_size = 50
        training_args.run_name = "test-run-for-200-train_size"
        
        config = ColModelDistillTrainingConfig(output_dir="./models/colsmolvlm", 
                            processor=ColIdefics3Processor.from_pretrained(student_model), 
                            model = ColIdefics3.from_pretrained(student_model, torch_dtype=torch.float16, attn_implementation="eager"),
                            teacher_model= ColQwen2.from_pretrained(teacher_model, torch_dtype=torch.float16, attn_implementation="eager").eval(),
                            teacher_processor = ColQwen2Processor.from_pretrained(teacher_model),
                            hub_repo_id = f"vijay-ravichander/ColSmol-256-Distill-Qwen-tues",
                            peft_config = peft_config, 
                            tr_args=training_args, 
                            train_dataset="https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/",
                            train_size=train_samples_size,
                            eval_size=eval_samples_size
                            )
        wandb.log({
            "using_teacher_model": "Yes", 
            "teacher_model": teacher_model, 
            "student_model": student_model,
            "training_samples": train_samples_size, 
            "eval_samples": eval_samples_size,
            "global_batch_size": training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
        })

    else:
        student_model = "vidore/ColSmolVLM-Instruct-256M-base"

        config = ColModelDistillTrainingConfig(output_dir="./models/colsmolvlm", 
                            processor=ColIdefics3Processor.from_pretrained(student_model), 
                            model = ColIdefics3.from_pretrained(student_model, torch_dtype=torch.float16, attn_implementation="eager"),
                            hub_repo_id = f"vijay-ravichander/",
                            peft_config = peft_config, 
                            tr_args = training_args, 
                            train_dataset="https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/",
                            train_size=train_samples_size,
                            eval_size=eval_samples_size
                        )
        
        wandb.log({
            "using_teacher_model": "No", 
            "teacher_model": "", 
            "student_model": student_model,
            "training_samples": train_samples_size, 
            "eval_samples": eval_samples_size,
            "global_batch_size": training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
        })


    print("Creating Setup")

    if isinstance(config, ColModelDistillTrainingConfig):
        app = ColModelDistillTraining(config)
    else:
        raise ValueError("Config must be of type ColModelDistillTrainingConfig")

    # if config.run_train:
    print("Training model")
    app.train()

    app.save()

    # app.save(config_file=config_file)
    print("Done!")


if __name__ == "__main__":
    main()
