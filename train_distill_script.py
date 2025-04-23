from pathlib import Path
import typer
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.models import ColIdefics3Processor, ColIdefics3, ColQwen2, ColQwen2Processor
import torch
from peft import LoraConfig
from train_distill_colpali import ColModelDistillTraining, ColModelDistillTrainingConfig
from transformers import TrainingArguments
import wandb
import os
from dotenv import load_dotenv
import datetime
from loss import ColBertPairwiseDistillKLLoss, ColBertMarginMSELoss, ColBertPairwiseDistillLoss
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path: Path) -> None:

    # Load environment variables from .env file
    load_dotenv()
    config = load_config(config_path)

    # Access the tokens
    hf_token = os.getenv("HF_TOKEN")
    wandb_token = os.getenv("WANDB_API_KEY")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["WANDB_API_KEY"] = wandb_token

    print("Loading config")

    wandb.init(project=config["wandb_dir"], config = config)

    IS_DISTILL_TRAINING = bool(config.get("distill", False))

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
        output_dir = "./" + config.get("run_name") + "/checkpoints/",
        num_train_epochs = 1,
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        per_device_train_batch_size = config["per_device_train_batch_size"],
        per_device_eval_batch_size= 4,
        weight_decay = 0.01,
        bf16 = bool(config.get("bf16", False)),
        fp16 = False,
        learning_rate = float(config["learning_rate"]),
        optim = config["optim"],
        logging_steps = 1,
        logging_strategy="steps",
        eval_strategy = "steps",
        eval_steps = config["eval_steps"],
        eval_on_start = True,
        max_grad_norm = 0.8,
        report_to = ["wandb"],
        save_strategy = "steps",        # Save checkpoint every X steps or epochs
        save_steps= config["save_steps"], 
        resume_from_checkpoint = config.get("resume_from_checkpoint", None)
    )

    

    if IS_DISTILL_TRAINING:

        teacher_model_name = config["teacher_model"]
        student_model_name = config["student_model"]

        train_samples_size = config["train_samples_size"]
        eval_samples_size = config["eval_samples_size"]

        training_args.run_name = f"{Path(student_model_name).name}-Distill"

        loss_fn_name = config.get("loss_fn", "Pairwise")
        
        if loss_fn_name == "KL":
            loss_fn = ColBertPairwiseDistillKLLoss()
        elif loss_fn_name == "MarginMSE":
            loss_fn = ColBertMarginMSELoss()
        elif loss_fn_name == "Pairwise":
            loss_fn = ColBertPairwiseDistillLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")


        config = ColModelDistillTrainingConfig(
                                            processor= ColIdefics3Processor.from_pretrained(student_model_name),
                                            model = ColIdefics3.from_pretrained(student_model_name, torch_dtype=torch.float16, attn_implementation="eager"),
                                            teacher_model= ColIdefics3.from_pretrained(teacher_model_name, torch_dtype=torch.float16, attn_implementation="eager").eval()
                                            if "Smol" in teacher_model_name
                                            else ColQwen2.from_pretrained(teacher_model_name, torch_dtype=torch.float16, attn_implementation="eager").eval(),
                                            teacher_processor = ColIdefics3Processor.from_pretrained(teacher_model_name)
                                            if "Smol" in teacher_model_name
                                            else ColQwen2Processor.from_pretrained(teacher_model_name),
                                            hub_repo_id = "vijay-ravichander/" + config.get("hub_repo_id"),
                                            peft_config = peft_config,
                                            tr_args=training_args,
                                            train_dataset="https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/",
                                            train_size=train_samples_size,
                                            eval_size=eval_samples_size,
                                            loss_func = loss_fn
                                            )
    else:
        student_model_name = config["student_model"]
        train_samples_size = config["train_samples_size"]
        eval_samples_size = config["eval_samples_size"]
        training_args.run_name = f"{Path(student_model_name).name}-NonDistill"

        config = ColModelDistillTrainingConfig(output_dir="./models/colsmolvlm",
                                            processor = ColIdefics3Processor.from_pretrained(student_model_name),
                                            model = ColIdefics3.from_pretrained(student_model_name, torch_dtype=torch.float16, attn_implementation="eager"),
                                            hub_repo_id = "vijay-ravichander/" + config.get("hub_repo_id"),
                                            peft_config = peft_config,
                                            tr_args = training_args,
                                            train_dataset="https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/",
                                            train_size=train_samples_size,
                                            eval_size=eval_samples_size
                                        )


    print("Creating Setup")

    # wandb.log(training_args)

    if isinstance(config, ColModelDistillTrainingConfig):
        app = ColModelDistillTraining(config)
    else:
        raise ValueError("Config must be of type ColModelDistillTrainingConfig")

    print("Training model")
    app.train()

    app.save()
    print("Done!")


if __name__ == "__main__":
    typer.run(main)