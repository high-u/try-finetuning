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
parser.add_argument('--system', type=str, default=None, help='System prompt')
parser.add_argument('--base', action='store_true', help='Use base model instead of fine-tuned model')
args = parser.parse_args()

# Environment variables
TRAINING_NAME = os.getenv("TRAINING_NAME", "default")
BASE_DIR = f"./trainings/{TRAINING_NAME}"

def get_device_type():
    """CPUかCUDA GPUかを自動判別"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

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

# Detect device type and configure inference settings
device_type = get_device_type()
print(f"Detected device type: {device_type}")

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

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Initialize conversation history
conversation_history = []
if args.system:
    conversation_history.append({"role": "system", "content": args.system})

while True:
    try:
        user_input = input("> ")

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Generate prompt from conversation history
        prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate with streaming
        output = model.generate(**inputs, max_new_tokens=4096, streamer=streamer, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated text to add to history
        generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        conversation_history.append({"role": "assistant", "content": generated_text})

        print()

    except KeyboardInterrupt:
        print("\n\nChat ended.")
        break
