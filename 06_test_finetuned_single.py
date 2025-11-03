"""
Step 10: Test Fine-tuned Model
Interactive chat application for testing fine-tuned models
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Interactive chat with fine-tuned model')
parser.add_argument('--system', type=str, default=None, nargs='?', const='', help='System prompt')
parser.add_argument('--base', action='store_true', help='Use base model instead of fine-tuned model')
parser.add_argument('--finetuning-name', type=str, default='default',
                    help='Fine-tuning name (default: default)')
parser.add_argument('--device-type', type=str, required=True,
                    help='Device type: cuda, mps, or cpu')
args = parser.parse_args()

# Arguments
FINETUNING_NAME = args.finetuning_name
BASE_DIR = f"./finetunings/{FINETUNING_NAME}"

def configure_inference_settings(device_type):
    """デバイスに応じて設定を変更"""
    if device_type == "cuda":
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    return device_map, torch_dtype

# Load model configuration
model_config_path = f'{BASE_DIR}/model.json'
with open(model_config_path, 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Get device type from command-line argument
device_type = args.device_type
print(f"Using device type: {device_type}")

device_map, torch_dtype = configure_inference_settings(device_type)

# Load model based on --base flag
if args.base:
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        gemma_model,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(gemma_model)
else:
    print("Loading fine-tuned model...")
    config_path = f'{BASE_DIR}/config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    merged_model_path = config['merged_model_path']

    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# Interactive chat loop
print("\nChat started. Press Ctrl+C to exit.\n")
print("Note: No conversation history is maintained. Each input is treated as a new conversation.\n")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
    try:
        user_input = input("> ")

        # Create new conversation for each turn (no history)
        conversation = []
        if args.system:
            conversation.append({"role": "system", "content": args.system})
        conversation.append({"role": "user", "content": user_input})

        # Generate prompt from current conversation only
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate with streaming
        output = model.generate(**inputs, max_new_tokens=4096, streamer=streamer, pad_token_id=tokenizer.eos_token_id)

        print()

    except KeyboardInterrupt:
        print("\n\nChat ended.")
        break
