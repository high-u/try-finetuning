"""
Step 6-7: Configure Training and Train Model
Fine-tune the model using SFTTrainer
This will take about 10 minutes on a T4 GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import json
from datasets import Dataset

def get_device_type():
    """CPUかCUDA GPUかを自動判別"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def configure_training(device_type):
    """デバイスに応じて設定を変更"""
    if device_type == "cuda":
        # GPU用設定：4bit量子化、bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        torch_dtype = torch.bfloat16
        device_map = "auto"
        optim = "adamw_torch_fused"
    else:
        # CPU用設定：量子化なし、float32
        bnb_config = None
        torch_dtype = torch.float32
        device_map = "cpu"
        optim = "adamw_torch"
    
    return bnb_config, torch_dtype, device_map, optim

# Load model configuration
with open('training_model.json', 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Load training data from JSON file
with open('training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

# Convert to Dataset object
full_dataset = Dataset.from_list(training_data)

# Split into train and test (80% train, 20% test)
train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
training_dataset_splits = {
    'train': train_test_split['train'],
    'test': train_test_split['test']
}

# Configure training parameters
adapter_path = "./myemoji-gemma-adapters"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('tokenizer')

# Detect device type and configure training
device_type = get_device_type()
print(f"Detected device type: {device_type}")

bnb_config, torch_dtype, device_map, optim = configure_training(device_type)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"]
)

# Configure training arguments
args = SFTConfig(
    output_dir=adapter_path,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    max_length=256,
    gradient_checkpointing=False,
    packing=False,
    optim=optim,
    report_to="tensorboard",
    weight_decay=0.01,
)

# Load model with device-specific configuration
model_kwargs = {
    'attn_implementation': 'eager'
}

if bnb_config is not None:
    model_kwargs['quantization_config'] = bnb_config

base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map=device_map,
    torch_dtype=torch_dtype,
    **model_kwargs
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# Set training and evaluation datasets
train_dataset = training_dataset_splits['train']
eval_dataset = training_dataset_splits['test']

# Train and save the LoRA adapters
print("Starting training...")
trainer = SFTTrainer(
    model=base_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
)
trainer.train()
trainer.save_model(adapter_path)

print(f"\nLoRA adapters saved to {adapter_path}")

# Save training log history for plotting
training_log = trainer.state.log_history
with open('training_log.json', 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=2, ensure_ascii=False)
print("Training log saved to training_log.json")

# Save training configuration for next steps
training_config = {
    "adapter_path": adapter_path
}
with open('training_config.json', 'w', encoding='utf-8') as f:
    json.dump(training_config, f, indent=2, ensure_ascii=False)
print("Training configuration saved to training_config.json")
