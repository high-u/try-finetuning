"""
Step 2: Load Dataset
Load the text-to-emoji dataset from Hugging Face
"""

import emoji
from datasets import load_dataset
import pickle

# Use the first 2000 samples for efficient training
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

# Save dataset for next steps
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("\nDataset saved to dataset.pkl")
