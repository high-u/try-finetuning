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
parser.add_argument('--quantization', type=str, choices=['', '4', '8'], default='',
                    help='Training quantization bits: 4, 8, or "" (no quantization) (default: "")')
parser.add_argument('--torch-dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='bfloat16',
                    help='Torch dtype: float32, float16, or bfloat16 (default: bfloat16)')
parser.add_argument('--finetuning-name', type=str, default='default',
                    help='Fine-tuning name (default: default)')
parser.add_argument('--training-data-file', type=str, required=True,
                    help='Training data file path')
parser.add_argument('--device-type', type=str, default='auto',
                    help='Device type: cuda, mps, cpu, or auto (default: auto)')

args = parser.parse_args()

# Setup fine-tuning directory structure
base_dir = f"./finetunings/{args.finetuning_name}"
os.makedirs(base_dir, exist_ok=True)

print(f"Fine-tuning name: {args.finetuning_name}")
print(f"Training data file: {args.training_data_file}")
quantization_info = f"{args.quantization}-bit quantization" if args.quantization else "no quantization"
print(f"Using {quantization_info} for training")
print(f"Torch dtype: {args.torch_dtype}")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Max length: {args.max_length}")
print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
print(f"Output directory: {base_dir}")

def configure_training(device_type, quantization, torch_dtype):
    """デバイス、量子化、dtype に応じて設定を変更"""
    model_kwargs = {'attn_implementation': 'eager'}

    # device_map に device_type を直接代入
    model_kwargs['device_map'] = device_type

    # torch_dtype を torch モジュールの型に変換
    dtype_mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    # CUDA デバイスの場合（または "auto" で CUDA が利用可能な場合）
    if device_type == "cuda" or (device_type == "auto" and torch.cuda.is_available()):
        # quantization に基づいて設定
        if quantization == '4':
            # 4ビット量子化設定
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == '8':
            # 8ビット量子化設定
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            # 量子化なし：torch_dtype を指定
            model_kwargs['torch_dtype'] = dtype_mapping[torch_dtype]

        # CUDA 専用：adamw_torch_fused
        optim = "adamw_torch_fused"
    else:
        # CUDA 以外：torch_dtype を指定
        model_kwargs['torch_dtype'] = dtype_mapping[torch_dtype]
        optim = "adamw_torch"

    return model_kwargs, optim

# Load model configuration
model_config_path = f'{base_dir}/model.json'
with open(model_config_path, 'r', encoding='utf-8') as f:
    model_config = json.load(f)
gemma_model = model_config['model_name']

# Load training data from JSON file
with open(args.training_data_file, 'r', encoding='utf-8') as f:
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
adapter_path = f"{base_dir}/adapters"

# Load tokenizer
tokenizer_path = f'{base_dir}/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Get device type from command-line argument
device_type = args.device_type
print(f"Using device type: {device_type}")

model_kwargs, optim = configure_training(device_type, args.quantization, args.torch_dtype)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32, # [SQL] 16, [emoji] 32
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # modules_to_save=["lm_head", "embed_tokens"] # 新しいトークンを覚えさせる時にはコメントアウトしない方が良いらしい。コメントアウトしていると、メモリ消費量も抑えられて、学習速度も速い。
)

# Configure training arguments
sft_args = SFTConfig(
    output_dir=adapter_path,
    use_cpu=(device_type == "cpu"),
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
log_path = f'{base_dir}/log.json'
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=2, ensure_ascii=False)
print(f"Training log saved to {log_path}")

# Save fine-tuning configuration for next steps
finetuning_config = {
    "finetuning_name": args.finetuning_name,
    "adapter_path": adapter_path,
    "training_data_file": args.training_data_file,
    "base_dir": base_dir,
    "train_quantization": args.quantization,
    "train_torch_dtype": args.torch_dtype,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "max_length": args.max_length,
    "max_samples": args.max_samples
}
config_path = f'{base_dir}/config.json'
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(finetuning_config, f, indent=2, ensure_ascii=False)
print(f"Fine-tuning configuration saved to {config_path}")
