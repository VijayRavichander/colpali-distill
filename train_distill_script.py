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

# def main(config_file: Path) -> None:
def main() -> None:
    os.environ["WANDB_API_KEY"] = ""
    print("Loading config")
    wandb.init(project="colpali-distill")

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
        num_train_epochs = 2,
        # max_steps = 10,
        gradient_accumulation_steps = 4,
        per_device_train_batch_size= 2,
        per_device_eval_batch_size= 2,
        weight_decay=0.01,
        fp16=True, 
        optim = "adamw_torch", 
        logging_steps= 2,
        logging_strategy="steps", 
        eval_strategy = "epoch",
        eval_on_start = True, 
        hub_token = "", 
        push_to_hub=True, 
        hub_model_id="vijay-ravichander/ColSmolVLM-Instruct-256M-Distill-500M",
        save_steps=1,
        save_strategy="epoch",
        report_to=["wandb"] 
    )

    config = ColModelDistillTrainingConfig(output_dir="./models/colsmolvlm", 
                           processor=ColIdefics3Processor.from_pretrained("vidore/ColSmolVLM-Instruct-256M-base"), 
                           model=ColIdefics3.from_pretrained("vidore/ColSmolVLM-Instruct-256M-base", torch_dtype=torch.float16, attn_implementation="eager"),
                           teacher_model= ColIdefics3.from_pretrained( "vidore/colSmol-500M", torch_dtype=torch.float16, attn_implementation="eager").eval(),
                           teacher_processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-500M"),
                           peft_config = peft_config, 
                           tr_args=training_args, 
                           train_dataset="https://huggingface.co/datasets/vidore/colpali_train_set/resolve/main/data/"
                        )
    print("Creating Setup")

    if isinstance(config, ColModelDistillTrainingConfig):
        app = ColModelDistillTraining(config)
    else:
        raise ValueError("Config must be of type ColModelDistillTrainingConfig")

    # if config.run_train:
    print("Training model")
    app.train()

    # app.save(config_file=config_file)
    print("Done!")


if __name__ == "__main__":
    main()
