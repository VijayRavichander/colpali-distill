import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union


from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)


from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from datasets import  load_dataset

from loss import ColBertPairwiseDistillLoss
from contrastive_trainer import ContrastiveTrainer
from collator import VisualRetrieverCollator

@dataclass
class ColModelDistillTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    hub_repo_id: str
    teacher_model: Optional[Union[PreTrainedModel, PeftModel]] = None
    teacher_processor: Optional[BaseVisualRetrieverProcessor] = None
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = ColBertPairwiseDistillLoss()
    train_dataset: Optional[str] = 'vidore/arxivqa_test_subsampled'
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None
    train_size: int = 400
    eval_size: int = 100
    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None:
            self.tr_args.output_dir = self.output_dir

        if isinstance(self.tr_args.learning_rate, str):
            print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        if self.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.pretrained_peft_model_name_or_path is None:
                # Prepare the model for k-bit training
                self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    # print_gpu_utilization()


class ColModelDistillTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelDistillTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.teacher_model = self.config.teacher_model

        data_files = {"train": [self.config.train_dataset + "train-00001-of-00082.parquet", 
                                self.config.train_dataset + "train-00002-of-00082.parquet",
                                self.config.train_dataset + "train-00003-of-00082.parquet", 
                                self.config.train_dataset + "train-00004-of-00082.parquet", 
                                self.config.train_dataset + "train-00005-of-00082.parquet", 
                                self.config.train_dataset + "train-00006-of-00082.parquet", 
                                self.config.train_dataset + "train-00007-of-00082.parquet",
                                self.config.train_dataset + "train-00008-of-00082.parquet", 
                                self.config.train_dataset + "train-00009-of-00082.parquet", 
                                self.config.train_dataset + "train-00010-of-00082.parquet", 
                                self.config.train_dataset + "train-00011-of-00082.parquet", 
                                self.config.train_dataset + "train-00012-of-00082.parquet", 
                                self.config.train_dataset + "train-00013-of-00082.parquet", 
                                self.config.train_dataset + "train-00014-of-00082.parquet", 
                                self.config.train_dataset + "train-00015-of-00082.parquet", 
                                self.config.train_dataset + "train-00016-of-00082.parquet", 
                                self.config.train_dataset + "train-00017-of-00082.parquet", 
                                self.config.train_dataset + "train-00018-of-00082.parquet", 
                                self.config.train_dataset + "train-00019-of-00082.parquet"
                                ], 
                                
        "test": self.config.train_dataset + "test-00000-of-00001.parquet"}
        self.dataset = load_dataset("parquet", data_files=data_files)

        print(self.dataset)
        print(f"Using {config.train_dataset} dataset to train")

        print("Dataset has QA format. Using VisualRetrieverCollator.")
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
            teacher_processor=self.config.teacher_processor
        )

    def train(self) -> None:
        print("Training with in-batch negatives")

        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"].take(self.config.train_size),
            eval_dataset=self.dataset["test"].take(self.config.eval_size),
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
            teacher_model = self.teacher_model
        )

        trainer.args.remove_unused_columns = False

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        # self.model.save_pretrained(self.config.output_dir)
        # self.config.processor.save_pretrained(self.config.output_dir)
        self.model = self.model.merge_and_unload()
        self.model.push_to_hub(self.config.hub_repo_id)
        self.config.processor.push_to_hub(self.config.hub_repo_id)

        # Copy-paste the training config
        # os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # Save git hash of the commit at beginning of training
        # with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
        #     f.write(self.current_git_hash)
