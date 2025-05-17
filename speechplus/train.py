import os
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
import transformers
import wandb
from huggingface_hub import hf_hub_download
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, default_data_collator

from speechplus.data.dataset_speech import CustomTrainingArguments, DataArguments, ModelArguments, TTSDataset


def download_dataset(repo_id="Steveeeeeeen/mls_eng_10k", local_dir="mls_eng_10k"):
    """
    Download dataset files from the HuggingFace hub if they don't exist locally
    """
    # Check if dataset files already exist
    train_shape_file = os.path.join(local_dir, "train_input_ids_shape.npy")
    test_shape_file = os.path.join(local_dir, "test_input_ids_shape.npy")

    if os.path.exists(train_shape_file) and os.path.exists(test_shape_file):
        print(f"Dataset files already exist in {local_dir}. Skipping download.")
        return

    print(f"Downloading dataset from {repo_id} to {local_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # List of required files for the dataset
    splits = ["train", "test"]  # Add other splits if needed

    for split in splits:
        # Download the memmap file
        memmap_file = f"{split}_input_ids.memmap"
        print(f"Downloading {memmap_file}...")
        hf_hub_download(repo_id=repo_id, filename=memmap_file, repo_type="dataset", local_dir=local_dir)

        # Download the shape file
        shape_file = f"{split}_input_ids_shape.npy"
        print(f"Downloading {shape_file}...")
        hf_hub_download(repo_id=repo_id, filename=shape_file, repo_type="dataset", local_dir=local_dir)

    print("Dataset download complete!")


def main():
    # Initialize the argument parser with our dataclasses.
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    # If a single JSON config file is passed as an argument, use it.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Ensure dataset files exist - download them if necessary
    # This step will automatically download the dataset if it doesn't exist
    download_dataset(repo_id="Steveeeeeeen/mls_eng_10k", local_dir=data_args.data_path)

    if training_args.report_to == "wandb":
        wandb.init(
            project="tts-training",
            name=training_args.run_name,
            config=training_args.to_dict(),
        )

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
        "<|TEXT_GENERATION_START|>",
        "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>",
        "<|SPEECH_UNDERSTANDING_END|>",
    ]

    # Add the same speech tokens
    new_speech_tokens = [f"<|s_{i}|>" for i in range(4096)]  # Must match preprocessing exactly

    # Add all tokens to the tokenizer
    tokenizer.add_tokens(special_tokens + new_speech_tokens)

    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(len(tokenizer))

    # Create datasets
    try:
        train_dataset = TTSDataset(data_args.data_path, "train", tokenizer)
        eval_dataset = TTSDataset(data_args.data_path, "test", tokenizer)
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Double-checking if dataset files exist...")

        # If loading failed, try downloading the dataset again (in case it was interrupted)
        download_dataset(repo_id="Steveeeeeeen/mls_eng_10k", local_dir=data_args.data_path)

        # Try loading again
        train_dataset = TTSDataset(data_args.data_path, "train", tokenizer)
        eval_dataset = TTSDataset(data_args.data_path, "test", tokenizer)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=training_args.learning_rate)

    if training_args.deepspeed is None:
        training_args.deepspeed = "ds_config.json"

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
