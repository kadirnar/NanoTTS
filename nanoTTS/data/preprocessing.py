import os
import numpy as np
from datasets import load_dataset
import torch
import torchaudio
from transformers import AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer
from outetts.wav_tokenizer.encoder.utils import convert_audio
from tqdm import tqdm
import argparse
from huggingface_hub import HfApi

def preprocess_dataset(
    dataset_name: str,
    dataset_subset: str,
    output_dir: str,
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    wav_tokenizer_model_path: str = "wavtokenizer_large_speech_320_v2.ckpt",
    wav_tokenizer_config_path: str = "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    sample_rate: int = 24000,
    max_length: int = 2048,
    debug: bool = False,
    subset_ratio: float = 1.0
):
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, dataset_subset, trust_remote_code=True)

    splits = dataset.keys()
    print(f"Found splits: {splits}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    special_tokens = [
        "<|TEXT_GENERATION_START|>", "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>", "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>", "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>", "<|SPEECH_UNDERSTANDING_END|>",
    ]

    new_speech_tokens = [f"<|s_{i}|>" for i in range(4096)]  # Adjust based on WavTokenizer vocab
    tokenizer.add_tokens(special_tokens + new_speech_tokens)
    tokenizer.pad_token_id = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavtokenizer = WavTokenizer.from_pretrained0802(wav_tokenizer_config_path, wav_tokenizer_model_path)
    wavtokenizer = wavtokenizer.to(device).eval()

    # Create datasets- hub
    api = HfApi()
    # api.create_repo(f"Steveeeeeeen/{output_dir}", repo_type="dataset")

    for split in splits:
        split_data = dataset[split]

        if subset_ratio < 1.0:
            num_samples = int(len(split_data) * subset_ratio)
            split_data = split_data.select(range(num_samples))
        elif debug:
            split_data = split_data.select(range(min(10, len(split_data))))

        os.makedirs(output_dir, exist_ok=True)
        memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
        shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")

        all_sequences = []
        for idx, example in tqdm(enumerate(split_data), total=len(split_data)):
            text = f"<|TEXT_UNDERSTANDING_START|>{example['transcript']}<|TEXT_UNDERSTANDING_END|>"
            text_ids = tokenizer.encode(text, add_special_tokens=False)

            waveform = torch.tensor(example["audio"]["array"]).float().unsqueeze(0)
            waveform = convert_audio(waveform, example["audio"]["sampling_rate"], sample_rate, 1)
            waveform = waveform.to(device)

            with torch.no_grad():
                _, speech_codes = wavtokenizer.encode(waveform, bandwidth_id=torch.tensor([0], device=device))
            speech_codes = speech_codes.reshape(-1)
            speech_ids = (
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
                + [tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes.cpu().numpy().tolist()]
                + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
            )
            available_text_space = max_length - len(speech_ids)
            if available_text_space <= 0:
                continue

            truncated_text = text_ids[:available_text_space]
            
            final_sequence = (
                truncated_text
                + speech_ids
                + [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids))
            )[:max_length]
            
            all_sequences.append(final_sequence)

        if all_sequences:
            arr = np.memmap(memmap_path, dtype=np.int32, mode="w+", shape=(len(all_sequences), max_length))
            arr[:] = np.array(all_sequences, dtype=np.int32)
            arr.flush()
            np.save(shape_path, np.array([len(all_sequences), max_length]))

        api.upload_file(
            repo_id=f"Steveeeeeeen/{output_dir}",
            path_in_repo=f"{split}_input_ids.memmap",
            path_or_fileobj=memmap_path,
            repo_type="dataset"
        )
        api.upload_file(
            repo_id=f"Steveeeeeeen/{output_dir}",
            path_in_repo=f"{split}_input_ids_shape.npy",
            path_or_fileobj=shape_path,
            repo_type="dataset"
        )
        
        print(f"Uploaded {split} to datasets- hub") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio dataset using WavTokenizer')

    parser.add_argument('--dataset_name', type=str, default="parler-tts/mls_eng_10k")
    parser.add_argument('--dataset_subset', type=str)
    parser.add_argument('--output_dir', type=str, default="mls_eng_10k")
    parser.add_argument('--tokenizer_name', type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument('--wav_tokenizer_model_path', type=str,
                        default="wavtokenizer_large_speech_320_v2.ckpt")
    parser.add_argument('--wav_tokenizer_config_path', type=str,
                        default="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset_ratio', type=float, default=1.0)

    args = parser.parse_args()

    preprocess_dataset(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        wav_tokenizer_model_path=args.wav_tokenizer_model_path,
        wav_tokenizer_config_path=args.wav_tokenizer_config_path,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        debug=args.debug,
        subset_ratio=args.subset_ratio
    )