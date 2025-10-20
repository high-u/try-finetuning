"""
Step 3: Load Base Model
Load Gemma 3 270M instruction-tuned model
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get model name from environment variable or use default
gemma_model = os.getenv('FINETUNE_GEMMA_MODEL') or 'google/gemma-3-270m-it'
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map="auto",
    attn_implementation="eager",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(gemma_model)

print(f"Device: {base_model.device}")
print(f"DType: {base_model.dtype}")
print(f"Model: {gemma_model}")

# Save tokenizer for next steps
tokenizer.save_pretrained('tokenizer')
print("\nTokenizer saved")
