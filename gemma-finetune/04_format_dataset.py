"""
Step 4: Format Dataset
Format the training dataset into conversational format
"""

import pickle
from transformers import AutoTokenizer

# Load dataset
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

def translate(sample):
    return {
        "messages": [
            {"role": "system", "content": "Translate this text to emoji: "},
            {"role": "user", "content": f"{sample['text']}"},
            {"role": "assistant", "content": f"{sample['emoji']}"}
        ]
    }

training_dataset = dataset.map(translate, remove_columns=dataset.features.keys())
training_dataset_splits = training_dataset.train_test_split(test_size=0.1, shuffle=True)

print(f"\nTraining samples: {len(training_dataset_splits['train'])}")
print(f"Test samples: {len(training_dataset_splits['test'])}")
print("\nHere's the 40th example from the formatted training dataset:")
print(training_dataset[40])

# Save formatted dataset
with open('training_dataset_splits.pkl', 'wb') as f:
    pickle.dump(training_dataset_splits, f)
print("\nFormatted dataset saved to training_dataset_splits.pkl")
