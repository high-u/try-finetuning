"""
Step 6: Configure Training
Set up quantization, LoRA, and training configuration
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
import pickle

# Load config
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
gemma_model = config['gemma_model']

adapter_path = "./myemoji-gemma-adapters"
tokenizer = AutoTokenizer.from_pretrained('tokenizer')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"]
)

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
    optim="adamw_torch_fused",
    report_to="tensorboard",
    weight_decay=0.01,
)

base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation='eager'
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Training configured")

# Save configurations
training_config = {
    'adapter_path': adapter_path,
    'lora_config': lora_config,
    'sft_config': args,
}
with open('training_config.pkl', 'wb') as f:
    pickle.dump(training_config, f)
print("Configuration saved to training_config.pkl")
