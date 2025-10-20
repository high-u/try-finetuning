"""
Step 0: Make Dataset
Create a JSON file with training data from Huggingface dataset
"""

from datasets import load_dataset
import json

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

# Load the text-to-sql dataset from Hugging Face
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")

print(f"\nDataset loaded with {len(dataset)} examples")
print(f"\nHere's the first example from the dataset:")
print(f"  sql_prompt: {dataset[0]['sql_prompt']}")
print(f"  sql_context: {dataset[0]['sql_context'][:100]}...")
print(f"  sql: {dataset[0]['sql'][:100]}...")

# Convert to messages format
def create_conversation(sample):
    return {
        "messages": [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
            {"role": "assistant", "content": sample["sql"]}
        ]
    }

# Convert all samples to messages format
training_data = [create_conversation(sample) for sample in dataset]

# Save as JSON file
with open('training_data_sql.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\nTraining data saved to training_data_sql.json with {len(training_data)} examples")
print("\nHere's the first example from the formatted training data:")
print(json.dumps(training_data[0], ensure_ascii=False, indent=2))
