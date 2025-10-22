"""
Step 8: Plot Training Results
Visualize training and validation loss
"""

import matplotlib.pyplot as plt
import json
import os

# Environment variables
FINETUNING_NAME = os.getenv("FINETUNING_NAME", "default")
BASE_DIR = f"./finetunings/{FINETUNING_NAME}"

# Load training log history
log_path = f'{BASE_DIR}/log.json'
with open(log_path, 'r', encoding='utf-8') as f:
    log_history = json.load(f)

# Extract training / validation loss
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(epoch_train, train_losses, label="Training Loss", marker='o')
plt.plot(epoch_eval, eval_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plot_path = f'{BASE_DIR}/loss.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to {plot_path}")
print(f"\nFinal training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {eval_losses[-1]:.4f}")
