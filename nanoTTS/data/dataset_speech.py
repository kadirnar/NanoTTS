import os
import glob
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
import numpy as np
@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(default=2048)
    logging_steps: int = field(default=100)
    report_to: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine")
    per_device_train_batch_size: int = field(default=256)  # Change this to your desired batch size
    per_device_eval_batch_size: int = field(default=256)   # Eval batch size usually matches training

class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 2048
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = -100

        self.chunks, self.cum_lengths = self.load_memmap_chunks(data_path, split)
        self.length = self.cum_lengths[-1]
        
        # Track skipped samples for debugging
        self.skipped_samples = 0
        self.total_samples_seen = 0

        # Special tokens dictionary
        self.special_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in [
            '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
            '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
            '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
            '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>'
        ]}

    def load_memmap_chunks(self, data_path, split):
        chunks = []
        cum_lengths = [0]
        pattern = os.path.join(data_path, f'{split}_rank*_partial*_input_ids.memmap')
        for memmap_file in sorted(glob.glob(pattern)):
            shape = np.load(memmap_file.replace('.memmap', '_shape.npy'))
            chunk = np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))
            chunks.append(chunk)
            cum_lengths.append(cum_lengths[-1] + shape[0])

        if not chunks:
            memmap_file = os.path.join(data_path, f'{split}_input_ids.memmap')
            shape = np.load(os.path.join(data_path, f'{split}_input_ids_shape.npy'))
            chunks = [np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))]
            cum_lengths = [0, shape[0]]

        return chunks, cum_lengths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Prevent infinite recursion by limiting the number of attempts
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                self.total_samples_seen += 1
                chunk, local_idx = self.get_chunk_and_local_index(idx)
                input_ids = torch.tensor(chunk[local_idx], dtype=torch.long)
                
                # Find the positions of special tokens
                text_understanding_start = (input_ids == self.special_tokens['<|TEXT_UNDERSTANDING_START|>']).nonzero()
                text_understanding_end = (input_ids == self.special_tokens['<|TEXT_UNDERSTANDING_END|>']).nonzero()
                speech_start_positions = (input_ids == self.special_tokens['<|SPEECH_GENERATION_START|>']).nonzero()
                speech_end_positions = (input_ids == self.special_tokens['<|SPEECH_GENERATION_END|>']).nonzero()
                
                # Verify all required tokens are present
                if (len(text_understanding_start) == 0 or 
                    len(text_understanding_end) == 0 or 
                    len(speech_start_positions) == 0 or 
                    len(speech_end_positions) == 0):
                    
                    self.skipped_samples += 1
                    if self.skipped_samples % 100 == 0:
                        print(f"Skipped {self.skipped_samples}/{self.total_samples_seen} samples due to missing tokens.")
                    
                    # Try the next sample
                    idx = (idx + 1) % self.__len__()
                    continue
                
                # Extract token positions
                text_start = text_understanding_start[0].item()
                text_end = text_understanding_end[0].item()
                speech_start = speech_start_positions[0].item()
                speech_end = speech_end_positions[0].item()
                
                # Verify the tokens are in the expected order
                if not (text_start < text_end < speech_start < speech_end):
                    self.skipped_samples += 1
                    idx = (idx + 1) % self.__len__()
                    continue
                
                # Create prompt with chat template
                prompt = [
                    {'role': 'user', 'content': 'Convert the text to speech:<|TEXT_UNDERSTANDING_START|>'},
                    {'role': 'assistant', 'content': '<|SPEECH_GENERATION_START|>'}
                ]
                prompt_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True)
                # Extract text and speech content
                text_content = input_ids[text_start:text_end+1]  # Including the end token
                speech_content = input_ids[speech_start:speech_end+1]  # Including the end token
                
                # Replace placeholder tokens with actual content
                prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|TEXT_UNDERSTANDING_START|>'], text_content)
                prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|SPEECH_GENERATION_START|>'], speech_content)

                # Convert to tensor and pad to max length
                input_ids = self.pad_to_max_length(torch.tensor(prompt_ids, dtype=torch.long), self.pad_token_id)
                labels = self.create_labels(input_ids)
                attention_mask = (input_ids != self.pad_token_id).long()
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                idx = (idx + 1) % self.__len__()
                
        # If we've tried max_attempts samples and all failed, create a dummy sample
        # This is a fallback to prevent the dataloader from crashing
        print(f"WARNING: Failed to find a valid sample after {max_attempts} attempts")
        dummy_input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
        dummy_attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
        dummy_labels = torch.full((self.max_length,), self.ignore_index, dtype=torch.long)
        
        return {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask, 'labels': dummy_labels}

    def replace_token(self, tokens, target, replacement):
        try:
            idx = tokens.index(target)
            return tokens[:idx] + replacement.tolist() + tokens[idx+1:]
        except ValueError:
            # If target token not found, return the original tokens
            return tokens

    def create_labels(self, input_ids):
        """
        Create labels for the model training.
        We only want the model to predict the speech tokens, not the input text.
        """
        labels = torch.full_like(input_ids, self.ignore_index)
        
        # Find start of speech generation in the processed input
        speech_start_positions = (input_ids == self.special_tokens['<|SPEECH_GENERATION_START|>']).nonzero()
        
        if len(speech_start_positions) > 0:
            # Set labels starting from speech generation token
            speech_start = speech_start_positions[0].item()
            labels[speech_start:] = input_ids[speech_start:]
        
        # Make sure padding tokens are ignored in loss calculation
        labels[input_ids == self.pad_token_id] = self.ignore_index
        return labels

    def pad_to_max_length(self, sequence, pad_value):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return torch.cat([sequence, torch.full((self.max_length - len(sequence),), pad_value, dtype=sequence.dtype)])

    def get_chunk_and_local_index(self, idx):
        for i, cum_len in enumerate(self.cum_lengths[1:], start=1):
            if idx < cum_len:
                return self.chunks[i-1], idx - self.cum_lengths[i-1]
        raise IndexError("Index out of range")

class ASRDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 2048
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = -100

        self.chunks, self.cum_lengths = self.load_memmap_chunks(data_path, split)
        self.length = self.cum_lengths[-1]

        self.special_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in [
            '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
            '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
            '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
            '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>'
        ]}

    def load_memmap_chunks(self, data_path, split):
        chunks = []
        cum_lengths = [0]
        pattern = os.path.join(data_path, f'{split}_rank*_partial*_input_ids.memmap')
        for memmap_file in sorted(glob.glob(pattern)):
            shape = np.load(memmap_file.replace('.memmap', '_shape.npy'))
            chunk = np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))
            chunks.append(chunk)
            cum_lengths.append(cum_lengths[-1] + shape[0])

        if not chunks:
            memmap_file = os.path.join(data_path, f'{split}_input_ids.memmap')
            shape = np.load(os.path.join(data_path, f'{split}_input_ids_shape.npy'))
            chunks = [np.memmap(memmap_file, dtype='int32', mode='r', shape=tuple(shape))]
            cum_lengths = [0, shape[0]]

        return chunks, cum_lengths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk, local_idx = self.get_chunk_and_local_index(idx)
        input_ids = torch.tensor(chunk[local_idx], dtype=torch.long)

        speech_understanding_start = (input_ids == self.special_tokens['<|SPEECH_UNDERSTANDING_START|>']).nonzero()
        speech_understanding_end = (input_ids == self.special_tokens['<|SPEECH_UNDERSTANDING_END|>']).nonzero()
        text_generation_start = (input_ids == self.special_tokens['<|TEXT_GENERATION_START|>']).nonzero()
        text_generation_end = (input_ids == self.special_tokens['<|TEXT_GENERATION_END|>']).nonzero()

        if (len(speech_understanding_start) == 0 or 
            len(speech_understanding_end) == 0 or 
            len(text_generation_start) == 0 or 
            len(text_generation_end) == 0):
            return self.__getitem__((idx + 1) % self.__len__())

        speech_start = speech_understanding_start[0].item()
        speech_end = speech_understanding_end[0].item()
        text_start = text_generation_start[0].item()
        text_end = text_generation_end[0].item()

        if not (speech_start < speech_end < text_start < text_end):
            return self.__getitem__((idx + 1) % self.__len__())

        prompt = [
            {'role': 'user', 'content': 'Transcribe the following speech:<|SPEECH_UNDERSTANDING_START|>'},
            {'role': 'assistant', 'content': '<|TEXT_GENERATION_START|>'}
        ]

        prompt_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True)

        speech_content = input_ids[speech_start:speech_end+1]
        text_content = input_ids[text_start:text_end+1]

        prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|SPEECH_UNDERSTANDING_START|>'], speech_content)
        prompt_ids = self.replace_token(prompt_ids, self.special_tokens['<|TEXT_GENERATION_START|>'], text_content)

        input_ids = self.pad_to_max_length(torch.tensor(prompt_ids, dtype=torch.long), self.pad_token_id)
        labels = self.create_labels(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def replace_token(self, tokens, target, replacement):
        try:
            idx = tokens.index(target)
            return tokens[:idx] + replacement.tolist() + tokens[idx+1:]
        except ValueError:
            return tokens

    def create_labels(self, input_ids):
        labels = torch.full_like(input_ids, self.ignore_index)
        text_start_positions = (input_ids == self.special_tokens['<|TEXT_GENERATION_START|>']).nonzero()
        if len(text_start_positions) > 0:
            text_start = text_start_positions[0].item()
            labels[text_start:] = input_ids[text_start:]
        labels[input_ids == self.pad_token_id] = self.ignore_index
        return labels

    def pad_to_max_length(self, sequence, pad_value):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return torch.cat([sequence, torch.full((self.max_length - len(sequence),), pad_value, dtype=sequence.dtype)])

    def get_chunk_and_local_index(self, idx):
        for i, cum_len in enumerate(self.cum_lengths[1:], start=1):
            if idx < cum_len:
                return self.chunks[i-1], idx - self.cum_lengths[i-1]
        raise IndexError("Index out of range")