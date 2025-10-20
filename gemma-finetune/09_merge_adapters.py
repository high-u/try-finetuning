"""
Step 9: Merge Adapters
Merge the LoRA adapters with the base model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def get_device_type():
    """CPUかCUDA GPUかを自動判別"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def configure_merge_settings(device_type):
    """デバイスに応じて設定を変更"""
    if device_type == "cuda":
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    return device_map, torch_dtype

# Load model configuration
with open('training_model.json', 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Load training config
with open('training_config.json', 'r', encoding='utf-8') as f:
    training_config = json.load(f)
adapter_path = training_config['adapter_path']
merged_model_path = "./myemoji-gemma-merged/"

# Detect device type and configure merge settings
device_type = get_device_type()
print(f"Detected device type: {device_type}")

device_map, torch_dtype = configure_merge_settings(device_type)

# Load base model and tokenizer
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model, 
    device_map=device_map,
    torch_dtype=torch_dtype
)
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
with open('training_merge.json', 'w', encoding='utf-8') as f:
    json.dump({'merged_model_path': merged_model_path}, f, indent=2)
print("Merge configuration saved to training_merge.json")
