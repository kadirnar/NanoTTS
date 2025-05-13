import argparse

import numpy as np
import torch
import torchaudio
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from outetts.wav_tokenizer.decoder import WavTokenizer
from outetts.wav_tokenizer.encoder.utils import convert_audio
from transformers import AutoTokenizer


def load_audio(audio_path, target_sample_rate=24000):
    """Load audio file and convert to target sample rate."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        waveform = convert_audio(waveform, sample_rate, target_sample_rate, 1)
    return waveform


def generate_speech_from_text(
    text,
    model_path,
    tokenizer_path=None,
    output_path="generated_speech.wav",
    wav_tokenizer_model_path="wavtokenizer_large_speech_320_v2.ckpt",
    wav_tokenizer_config_path="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    sample_rate=24000,
    max_length=2048,
):
    """Generate speech tokens from text using SmolLM and decode them with WavTokenizer."""
    # Use the same tokenizer path as model if not specified
    if tokenizer_path is None:
        tokenizer_path = model_path

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Add special tokens if they don't exist
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
    # Add speech tokens
    new_speech_tokens = [f"<|s_{i}|>" for i in range(4096)]

    # Check if tokens need to be added
    tokens_to_add = []
    for token in special_tokens + new_speech_tokens:
        if token not in tokenizer.get_vocab():
            tokens_to_add.append(token)

    if tokens_to_add:
        print(f"Adding {len(tokens_to_add)} tokens to tokenizer...")
        tokenizer.add_tokens(tokens_to_add)

    # Make sure padding token is set
    tokenizer.pad_token_id = 2

    # Load WavTokenizer
    print("Loading WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(wav_tokenizer_config_path, wav_tokenizer_model_path)
    wavtokenizer = wavtokenizer.to(device).eval()

    # Load model
    print("Loading model...")
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    # Resize token embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device).eval()

    # Create prompt for text-to-speech
    text_with_markers = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"

    prompt = [
        {
            "role": "user",
            "content": f"Convert the text to speech:{text_with_markers}"
        },
        {
            "role": "assistant",
            "content": "<|SPEECH_GENERATION_START|>"
        },
    ]
    input_ids = tokenizer.apply_chat_template(prompt, tokenize=True)

    # Prepare input for model
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Generate speech tokens
    print("Generating speech tokens...")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.1,
            max_length=4096,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # Find the speech tokens in the generated output
    generated_ids = output[0].cpu().numpy()
    print(f"Generated IDs: {generated_ids}")
    # Extract speech tokens (between SPEECH_GENERATION_START and SPEECH_GENERATION_END)
    try:
        start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        start_idx = np.where(generated_ids == start_id)[0][0] + 1
        end_idx = (np.where(generated_ids == end_id)[0][0] if end_id in generated_ids else len(generated_ids))

        speech_token_ids = generated_ids[start_idx:end_idx]

        # Convert token IDs back to speech codes
        speech_codes = []
        for token_id in speech_token_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("<|s_") and token.endswith("|>"):
                code = int(token[4:-2])  # Extract number from "<|s_XXX|>"
                speech_codes.append(code)

        speech_codes = torch.tensor(speech_codes, device=device).reshape(1, -1)
        print(f"Generated {len(speech_codes[0])} speech codes")

        # Decode speech codes to audio
        print("Decoding speech to audio...")
        with torch.no_grad():
            audio = wavtokenizer.decode(speech_codes.unsqueeze(0))[0]

        # Save audio to file
        audio_cpu = audio.cpu()
        torchaudio.save(output_path, audio_cpu, sample_rate)
        print(f"Speech saved to {output_path}")

        return output_path

    except Exception as e:
        print(f"Error generating speech: {e}")
        print(
            "Generated sequence:",
            tokenizer.decode(generated_ids, skip_special_tokens=False),
        )
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolVoice inference script")

    # Common parameters
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the fine-tuned model",
        default="output/checkpoint-1000",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to the tokenizer (defaults to model_path)",
        default="output/checkpoint-1000",
    )
    parser.add_argument(
        "--wav_tokenizer_model",
        type=str,
        default="wavtokenizer_large_speech_320_v2.ckpt",
        help="Path to the WavTokenizer model checkpoint",
    )
    parser.add_argument(
        "--wav_tokenizer_config",
        type=str,
        default="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        help="Path to the WavTokenizer config file",
    )
    parser.add_argument("--sample_rate", type=int, default=24000, help="Audio sample rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    # Task-specific parameters
    subparsers = parser.add_subparsers(dest="task", help="Task to perform")

    # Speech-to-text task
    stt_parser = subparsers.add_parser("transcribe", help="Transcribe speech to text")
    stt_parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")

    # Text-to-speech task
    tts_parser = subparsers.add_parser("synthesize", help="Synthesize text to speech")
    tts_parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    tts_parser.add_argument(
        "--output_path",
        type=str,
        default="generated_speech.wav",
        help="Path to save generated audio",
    )

    args = parser.parse_args()

    # Execute the appropriate task
    if args.task == "synthesize":
        generate_speech_from_text(
            text=args.text,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            output_path=args.output_path,
            wav_tokenizer_model_path=args.wav_tokenizer_model,
            wav_tokenizer_config_path=args.wav_tokenizer_config,
            sample_rate=args.sample_rate,
            max_length=args.max_length,
        )
    else:
        parser.print_help()
