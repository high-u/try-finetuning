"""
Step 8: Plot Training Results
Visualize training and validation loss
"""

import matplotlib.pyplot as plt
import json

# Load training log history
with open('training_log.json', 'r', encoding='utf-8') as f:
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
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to training_loss.png")
print(f"\nFinal training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {eval_losses[-1]:.4f}")
