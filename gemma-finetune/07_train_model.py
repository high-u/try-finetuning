"""
Step 7: Train Model
Fine-tune the model using SFTTrainer
This will take about 10 minutes on a T4 GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

# Reload model with quantization for training
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    quantization_config=bnb_config,
    device_map="auto",
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
