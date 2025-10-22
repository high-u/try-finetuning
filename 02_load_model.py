"""
Step 3: Load Base Model
Load Gemma 3 270M instruction-tuned model
"""

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Environment variables
FINETUNING_NAME = os.getenv("FINETUNING_NAME", "default")
gemma_model = os.getenv('FINETUNE_GEMMA_MODEL') or 'google/gemma-3-270m-it'

# Setup fine-tuning directory structure
BASE_DIR = f"./finetunings/{FINETUNING_NAME}"
os.makedirs(BASE_DIR, exist_ok=True)

print(f"Fine-tuning name: {FINETUNING_NAME}")
print(f"Model: {gemma_model}")
print(f"Output directory: {BASE_DIR}")
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map="auto",
    attn_implementation="eager",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(gemma_model)

print(f"Device: {base_model.device}")
print(f"DType: {base_model.dtype}")

# Save tokenizer for this fine-tuning
tokenizer_path = f'{BASE_DIR}/tokenizer'
tokenizer.save_pretrained(tokenizer_path)

# Get model parameters count
total_params = sum(p.numel() for p in base_model.parameters())

# Save model configuration for this fine-tuning
model_config = {
    "model_name": gemma_model,
    "total_parameters": total_params
}
model_config_path = f'{BASE_DIR}/model.json'
with open(model_config_path, 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)

print(f"\nTokenizer saved to {tokenizer_path}")
print(f"Model configuration saved to {model_config_path}")
