"""
Step 1: Hugging Face Authentication
Login to Hugging Face Hub with your token
"""

from huggingface_hub import login
import os

# Get HF token from environment variable
# Set it with: export HF_TOKEN=your_token_here
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    print("ERROR: HF_TOKEN environment variable not set")
    print("Please set it with: export HF_TOKEN=your_token_here")
    print("Get your token from: https://huggingface.co/settings/tokens")
    exit(1)

# Login to Hugging Face Hub
login(hf_token)
print("Successfully logged in to Hugging Face Hub")
