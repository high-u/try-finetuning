"""
Step 6-7: Configure Training and Train Model
Fine-tune the model using SFTTrainer
This will take about 10 minutes on a T4 GPU
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import json
from datasets import Dataset

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune Gemma model')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Per device train batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of training epochs (default: 2)')
parser.add_argument('--max-length', type=int, default=512,
                    help='Maximum sequence length (default: 512)')
parser.add_argument('--max-samples', type=int, default=None,
                    help='Maximum number of samples to use (default: None, use all)')
parser.add_argument('--quantization', type=int, choices=[4, 8], default=8,
                    help='Training quantization bits: 4 or 8 (default: 8)')

args = parser.parse_args()

# Environment variables
FINETUNING_NAME = os.getenv("FINETUNING_NAME", "default")
TRAINING_DATA_FILE = os.getenv("TRAINING_DATA_FILE", "training_data.json")

# Setup fine-tuning directory structure
BASE_DIR = f"./finetunings/{FINETUNING_NAME}"
os.makedirs(BASE_DIR, exist_ok=True)

print(f"Fine-tuning name: {FINETUNING_NAME}")
print(f"Training data file: {TRAINING_DATA_FILE}")
print(f"Using {args.quantization}-bit quantization for training")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Max length: {args.max_length}")
print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
print(f"Output directory: {BASE_DIR}")

def get_device_type():
    """CPUかCUDA GPUかを自動判別"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def configure_training(device_type, quantization_bits=8):
    """デバイスと量子化ビット数に応じて設定を変更"""
    model_kwargs = {'attn_implementation': 'eager'}

    if device_type == "cuda":
        model_kwargs['device_map'] = "auto"

        if quantization_bits == 4:
            # 4ビット量子化設定
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8-bit
            # 8ビット量子化設定
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        # 量子化使用時はtorch_dtypeを指定しない（BitsAndBytesが自動処理）

        optim = "adamw_torch_fused"
    else:
        # CPU用設定：量子化なし
        model_kwargs['device_map'] = "cpu"
        model_kwargs['torch_dtype'] = torch.float32
        optim = "adamw_torch"

    return model_kwargs, optim

# Load model configuration
model_config_path = f'{BASE_DIR}/model.json'
with open(model_config_path, 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Load training data from JSON file
with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
    training_data = json.load(f)

# Limit dataset size to prevent excessive training time
if args.max_samples and len(training_data) > args.max_samples:
    print(f"Limiting dataset from {len(training_data)} to {args.max_samples} samples")
    import random
    random.seed(42)
    training_data = random.sample(training_data, args.max_samples)

# Convert to Dataset object
full_dataset = Dataset.from_list(training_data)

# Split into train and test (80% train, 20% test)
train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
training_dataset_splits = {
    'train': train_test_split['train'],
    'test': train_test_split['test']
}

# Configure training parameters
adapter_path = f"{BASE_DIR}/adapters"

# Load tokenizer
tokenizer_path = f'{BASE_DIR}/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Detect device type and configure training
device_type = get_device_type()
print(f"Detected device type: {device_type}")

model_kwargs, optim = configure_training(device_type, args.quantization)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32, # [SQL] 16, [emoji] 32
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"]
)

# Configure training arguments
sft_args = SFTConfig(
    output_dir=adapter_path,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5, # [SQL] 2e-4, [emoji] 5e-5
    lr_scheduler_type="constant",
    max_length=args.max_length,
    gradient_checkpointing=True,
    packing=False, # [SQL] True, [emoji] False
    optim=optim,
    report_to="tensorboard",
    weight_decay=0.01,
)

# Load model with device-specific configuration
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
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
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(adapter_path)

print(f"\nLoRA adapters saved to {adapter_path}")

# Save training log history for plotting
training_log = trainer.state.log_history
log_path = f'{BASE_DIR}/log.json'
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=2, ensure_ascii=False)
print(f"Training log saved to {log_path}")

# Save fine-tuning configuration for next steps
finetuning_config = {
    "finetuning_name": FINETUNING_NAME,
    "adapter_path": adapter_path,
    "training_data_file": TRAINING_DATA_FILE,
    "base_dir": BASE_DIR,
    "train_quantization": args.quantization,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "max_length": args.max_length,
    "max_samples": args.max_samples
}
config_path = f'{BASE_DIR}/config.json'
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(finetuning_config, f, indent=2, ensure_ascii=False)
print(f"Fine-tuning configuration saved to {config_path}")
