import logging, os, glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import yaml
import torch
from torch.optim import Adam

from ..get_model_path import ModelPath

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)

class CustomCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Remove "text" or any other non-numeric keys
        for f in features:
            f.pop("text", None)
        return super().__call__(features)

class ContinueTrainingCustome:
    def __init__(self, config_path, dataset):
        self.config = self.load_config(config_path)
        self.config_path = config_path
        self.dataset = dataset
        self.model_name = self.config.get('model_name')
        self.loss_computation = self.config.get('loss_computation')
        self.model_path_map = self.config.get('model_path_map')
        self.epochs = self.config.get('epochs')
        self.batch_size = self.config.get('batch_size')
        self.deepspeed_config = self.config.get('deepspeed_config')
        self.wandb_run_name = self.config.get('wandb_run_name')
        self.per_device_train_batch_size = self.config.get('per_device_train_batch_size')
        self.per_device_eval_batch_size = self.config.get('per_device_eval_batch_size')
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps')
        self.learning_rate = self.config.get('learning_rate')
        self.warmup_steps = self.config.get('warmup_steps')
        self.lr_scheduler_type = self.config.get('lr_scheduler_type')
        self.seed = self.config.get('seed')
        self.output_dir = self.config.get('output_dir')
        self.save_label = self.config.get('save_label')
        self.train_text_type = self.config.get('train_text_type') 
        self.train_relation_id = self.config.get('train_relation_id')
        self.special_token = f'<TRIGGER_{self.train_relation_id}>'
        self.train_relation_id = self.config.get('train_relation_id')

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_tokenizer(self):
        ModelPath_obj = ModelPath(self.config_path)
        model_path = ModelPath_obj.get_model_path()

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                      truncation=True,
                                                      padding=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.train_text_type.startswith('trigger'):
            if "+" in self.train_relation_id: #there could be multiple relations
                relation_id_list = self.train_relation_id.split('+')
                # add n special tokens
                for relation_id in relation_id_list:
                    special_token = f'<TRIGGER_{relation_id}>'
                    tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
            else:
                special_token = f'<TRIGGER_{self.train_relation_id}>'
                tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})

        return tokenizer
    
    def load_model(self):
        ModelPath_obj = ModelPath(self.config_path)
        model_path = ModelPath_obj.get_model_path()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        )
        return model
    
    def get_model_tokenizer(self):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        model.resize_token_embeddings(len(tokenizer)) #resize the model's token embeddings to match the tokenizer's vocabulary size
        return model, tokenizer
    
    def train(self):
        set_seed(self.seed)
        os.environ["WANDB_API_KEY"] = ''
        os.environ["WANDB_PROJECT"] = self.wandb_run_name
        os.environ["WANDB_WATCH"] = 'all'
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        local_output_dir = f'{self.output_dir}/{self.save_label}'
        if self.gradient_accumulation_steps > 1:
            local_output_dir += f'-gacc_{self.gradient_accumulation_steps}'
        
        #create the output directory if it does not exist
        Path(local_output_dir).mkdir(parents=True, exist_ok=True)

        print(f'Saving model to {local_output_dir}')

        training_args = TrainingArguments(
            output_dir=local_output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=float(self.learning_rate),
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.epochs,
            seed=self.seed,
            bf16=True,
            # deepspeed=self.deepspeed_config,
            save_only_model=True, #do not save other optimizers, schedulers, etc.
            lr_scheduler_type=self.lr_scheduler_type,
            logging_dir=f"{local_output_dir}/runs",
            logging_strategy="epoch",
            eval_steps=None,
            save_strategy="epoch",
            load_best_model_at_end=False,
            report_to="wandb",
            run_name=self.wandb_run_name,
            disable_tqdm=False,
            remove_unused_columns=False,
            # adam_beta1=0.9,
            # adam_beta2=0.95,
            # adam_epsilon=1e-5,
            # weight_decay=0.1,
            # ddp_backend="nccl",
            # max_grad_norm=1.0,
            # torch_compile=True
        )

        return training_args, local_output_dir
    
    def main(self):
        print('training started!, loading model and tokenizer')
        model, tokenizer = self.get_model_tokenizer()
        print('model and tokenizer loaded!')
        # Get the vocabulary size of the tokenizer
        vocab_size = len(tokenizer)
        print(f'vocab size: {vocab_size}')

        train_dataset = self.dataset

        training_args, local_output_dir = self.train()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=CustomCollator(tokenizer, mlm=False)
        )

        resume_from_checkpoint = True if len(glob.glob(f'{local_output_dir}/checkpoint-*')) > 0 else False
        # resume_from_checkpoint = False
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_model(output_dir=local_output_dir)
        tokenizer.save_pretrained(local_output_dir)

        # print('training completed!')
        return local_output_dir
