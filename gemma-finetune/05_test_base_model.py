"""
Step 5: Test Base Model (Optional)
Test the base model's ability to translate text to emoji
"""

from transformers import pipeline, AutoTokenizer
from random import randint
import pickle

# Load data
with open('training_dataset_splits.pkl', 'rb') as f:
    training_dataset_splits = pickle.load(f)
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

gemma_model = config['gemma_model']
tokenizer = AutoTokenizer.from_pretrained('tokenizer')

# Create a transformers inference pipeline
pipe = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer)

# Select a random sample from the test dataset
rand_idx = randint(0, len(training_dataset_splits["test"]) - 1)
test_sample = training_dataset_splits["test"][rand_idx]

# Handle messages
all_messages = test_sample['messages']
user_message_content = next((msg['content'].strip() for msg in all_messages if msg['role'] == 'user'), "Not Found")
dataset_emoji = next((msg['content'].strip() for msg in all_messages if msg['role'] == 'assistant'), "Not Found")
prompt_messages = [
    {"role": "system", "content": "Translate this text to emoji: "},
    {"role": "user", "content": user_message_content}
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

# Generate the output
output = pipe(prompt, max_new_tokens=64)
model_output_only = output[0]['generated_text'][len(prompt):].strip()

print(f"\nDataset text: {user_message_content}")
print(f"\nDataset emoji: {dataset_emoji}")
print(f"\nModel generated output: {model_output_only}")
