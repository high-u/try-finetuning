"""
Step 3: Load Base Model
Load Gemma 3 270M instruction-tuned model
"""

import torch
import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load base model for fine-tuning')
parser.add_argument('--finetuning-name', type=str, default='default',
                    help='Fine-tuning name (default: default)')
parser.add_argument('--finetune-base-model', type=str, default='google/gemma-3-270m-it',
                    help='Base model name (default: google/gemma-3-270m-it)')
parser.add_argument('--device-type', type=str, required=True,
                    help='Device type: cuda, mps, or cpu')
args = parser.parse_args()

# Setup fine-tuning directory structure
base_dir = f"./finetunings/{args.finetuning_name}"
os.makedirs(base_dir, exist_ok=True)

print(f"Fine-tuning name: {args.finetuning_name}")
print(f"Model: {args.finetune_base_model}")
print(f"Output directory: {base_dir}")

# Get device type from command-line argument
device_type = args.device_type
print(f"Using device type: {device_type}")

# Configure device settings based on DEVICE_TYPE
if device_type == "cuda":
    device_map = "auto"
    torch_dtype = torch.bfloat16
elif device_type == "mps":
    device_map = "mps"
    torch_dtype = torch.bfloat16 # torch.float32
else:
    device_map = "cpu"
    torch_dtype = torch.bfloat16 # torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    args.finetune_base_model,
    device_map=device_map,
    attn_implementation="eager",
    torch_dtype=torch_dtype
)
tokenizer = AutoTokenizer.from_pretrained(args.finetune_base_model)

print(f"Device: {base_model.device}")
print(f"DType: {base_model.dtype}")

# Save tokenizer for this fine-tuning
tokenizer_path = f'{base_dir}/tokenizer'
tokenizer.save_pretrained(tokenizer_path)

# Get model parameters count
total_params = sum(p.numel() for p in base_model.parameters())

# Save model configuration for this fine-tuning
model_config = {
    "model_name": args.finetune_base_model,
    "total_parameters": total_params
}
model_config_path = f'{base_dir}/model.json'
with open(model_config_path, 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)

print(f"\nTokenizer saved to {tokenizer_path}")
print(f"Model configuration saved to {model_config_path}")
