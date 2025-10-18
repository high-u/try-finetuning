"""
Step 3: Load Base Model
Load Gemma 3 270M instruction-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

gemma_model = "google/gemma-3-270m-it"
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map="auto",
    attn_implementation="eager",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(gemma_model)

print(f"Device: {base_model.device}")
print(f"DType: {base_model.dtype}")

# Save for next steps
torch.save(base_model.state_dict(), 'base_model_state.pt')
tokenizer.save_pretrained('tokenizer')
with open('model_config.pkl', 'wb') as f:
    pickle.dump({'gemma_model': gemma_model}, f)
print("\nModel and tokenizer saved")
