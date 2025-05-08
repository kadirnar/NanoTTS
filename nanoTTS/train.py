import os
import glob
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from dataclasses import dataclass, field
from typing import Optional
import transformers
import wandb
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
import bitsandbytes as bnb
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset_speech import ASRDataset, TTSDataset, ModelArguments, DataArguments, CustomTrainingArguments

def main():
    # Initialize the argument parser with our dataclasses.
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    # If a single JSON config file is passed as an argument, use it.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == "wandb":
        wandb.init(project="tts-training", name=training_args.run_name, config=training_args.to_dict())

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )
    # For this example, we force a pad_token_id if necessary.
    tokenizer.pad_token_id = 2
    print(f"Original tokenizer vocabulary size: {len(tokenizer)}")
    # Add the same special tokens as in preprocessing
    special_tokens = [
        "<|TEXT_GENERATION_START|>", "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>", "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>", "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>", "<|SPEECH_UNDERSTANDING_END|>",
    ]

    # Add the same speech tokens
    new_speech_tokens = [f"<|s_{i}|>" for i in range(4096)]  # Must match preprocessing exactly

    # Add all tokens to the tokenizer
    tokenizer.add_tokens(special_tokens + new_speech_tokens)
    
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = TTSDataset(data_args.data_path, 'train', tokenizer)
    eval_dataset = TTSDataset(data_args.data_path, 'test', tokenizer)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=training_args.learning_rate)

    if training_args.deepspeed is None and training_args.use_cpu_offload:
        training_args.deepspeed = "ds_config.json"

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None)
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()