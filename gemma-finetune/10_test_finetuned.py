"""
Step 10: Test Fine-tuned Model
Test models using random samples from training data
"""

import torch
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

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
with open('training_model.json', 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Detect device type and configure inference settings
device_type = get_device_type()
print(f"Detected device type: {device_type}")

device_map, torch_dtype = configure_inference_settings(device_type)

# Create base model pipeline
print("Loading base model...")
pipe_base = pipeline("text-generation", model=gemma_model, device_map=device_map)

# Check if training_merge.json exists and load fine-tuned model
if os.path.exists('training_merge.json'):
    print("Loading fine-tuned model...")
    with open('training_merge.json', 'r', encoding='utf-8') as f:
        paths = json.load(f)
    merged_model_path = paths['merged_model_path']
    
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, 
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
    has_finetuned = True
else:
    print("Fine-tuned model not found (training_merge.json does not exist)")
    has_finetuned = False
    # Use base model tokenizer for consistency
    tokenizer = AutoTokenizer.from_pretrained(gemma_model)

# Load training data and sample 5 random examples
print("Loading training data...")
with open('training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

# Randomly sample 5 examples
sampled_data = random.sample(training_data, 5)

print(f"\nTesting models with {len(sampled_data)} random samples...\n")

# Print markdown header
print("# Test\n")

for i, sample in enumerate(sampled_data, 1):
    print(f"## Test Case {i}")
    print()
    
    # Extract messages from the sample
    messages = sample['messages']
    system_msg = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
    user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
    assistant_msg = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
    
    # Print data section
    print("### Data")
    print()
    print("- System")
    print(f"    - {system_msg}")
    print("- User")
    print(f"    - {user_msg}")
    print("- Expected")
    print(f"    - {assistant_msg}")
    print()
    
    # Create inference messages (system + user)
    inference_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)

    # Test base model
    print("### Base Model")
    print()
    output_base = pipe_base(prompt, max_new_tokens=128)
    model_output_base = output_base[0]['generated_text'][len(prompt):].strip()
    print(model_output_base)
    print()

    # Test fine-tuned model if available
    if has_finetuned:
        print("### Fine-tuned Model")
        print()
        output = pipe(prompt, max_new_tokens=128)
        model_output = output[0]['generated_text'][len(prompt):].strip()
        print(model_output)
        print()
    
    print()
