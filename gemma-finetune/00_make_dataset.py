"""
Step 0: Make Dataset
Create a JSON file with training data from Huggingface dataset
"""

import emoji
from datasets import load_dataset
import json

# Load the text-to-emoji dataset from Hugging Face
general_dataset_path = load_dataset("kr15t3n/text2emoji", encoding="utf-8", split="train")

# Clean dataset to only use examples where 'emoji' field contains only emoji characters
def is_only_emoji(sample):
    emoji_string = sample['emoji']
    if not emoji_string:
        return False
    return all(emoji.is_emoji(char) for char in emoji_string)

dataset = general_dataset_path.filter(is_only_emoji)

print(f"\nDataset loaded with {len(dataset)} examples")
print(f"\nHere's the 10th example from the dataset: {dataset[10]}")

# Convert to messages format
def translate_to_messages(sample):
    return {
        "messages": [
            {"role": "system", "content": "Translate this text to emoji: "},
            {"role": "user", "content": f"{sample['text']}"},
            {"role": "assistant", "content": f"{sample['emoji']}"}
        ]
    }

# Convert all samples to messages format
training_data = [translate_to_messages(sample) for sample in dataset]

# Save as JSON file
with open('training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\nTraining data saved to training_data.json with {len(training_data)} examples")
print("\nHere's the 40th example from the formatted training data:")
print(training_data[40])
