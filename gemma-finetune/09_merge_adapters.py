"""
Step 9: Merge Adapters
Merge the LoRA adapters with the base model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle
import json

# Load config
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
with open('training_config.json', 'r', encoding='utf-8') as f:
    training_config = json.load(f)

gemma_model = config['gemma_model']
adapter_path = training_config['adapter_path']
merged_model_path = "./myemoji-gemma-merged/"

# Load base model and tokenizer
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(gemma_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load and merge the PEFT adapters onto the base model
print("Loading and merging adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# Save the merged model and its tokenizer
print(f"Saving merged model to {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"\nModel merged and saved to {merged_model_path}")
print(f"Final model vocabulary size: {model.config.vocab_size}")

# Save path for next steps
with open('merged_model_path.pkl', 'wb') as f:
    pickle.dump({'merged_model_path': merged_model_path}, f)
