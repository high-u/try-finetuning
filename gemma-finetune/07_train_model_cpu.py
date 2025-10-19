"""
Step 7: Train Model
Fine-tune the model using SFTTrainer
This will take about 10 minutes on a T4 GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
import pickle

# Load configurations and data
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
with open('training_config.pkl', 'rb') as f:
    training_config = pickle.load(f)
with open('training_dataset_splits.pkl', 'rb') as f:
    training_dataset_splits = pickle.load(f)

gemma_model = config['gemma_model']
adapter_path = training_config['adapter_path']
lora_config = training_config['lora_config']
args = training_config['sft_config']

# Reload model for CPU training (no quantization)
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map="cpu",
    torch_dtype=torch.float32,
    attn_implementation='eager'
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
with open('training_log.pkl', 'wb') as f:
    pickle.dump(training_log, f)
print("Training log saved to training_log.pkl")
