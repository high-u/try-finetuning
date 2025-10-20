"""
Step 10: Test Fine-tuned Model
Compare fine-tuned model against base model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import json

# Get model name from environment variable or use default
gemma_model = os.getenv('FINETUNE_GEMMA_MODEL') or 'google/gemma-3-270m-it'

# Load paths
with open('training_merge.json', 'r', encoding='utf-8') as f:
    paths = json.load(f)
merged_model_path = paths['merged_model_path']

# Create Transformers inference pipeline
print("Loading models...")
merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
pipe_base = pipeline("text-generation", model=gemma_model, device_map="auto")

# Test prompts
test_prompts = [
    "let's go to the beach",
    "I love pizza",
    "Happy birthday",
    "Good morning",
    "I'm feeling sad"
]

print("\nTesting models...\n")
for text_to_translate in test_prompts:
    inference_messages = [
        {"role": "system", "content": "Translate this text to emoji: "},
        {"role": "user", "content": text_to_translate}
    ]
    prompt = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)

    output = pipe(prompt, max_new_tokens=128)
    output_base = pipe_base(prompt, max_new_tokens=128)

    model_output = output[0]['generated_text'][len(prompt):].strip()
    model_output_base = output_base[0]['generated_text'][len(prompt):].strip()

    print(f"Input: {text_to_translate}")
    print(f"Fine-tuned: {model_output}")
    print(f"Base model: {model_output_base}")
    print("-" * 50)
