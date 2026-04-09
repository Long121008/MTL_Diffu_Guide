import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
log_path = "results/20251128_032340"  # Change this to your actual log path
df = pd.read_csv(f"{log_path}/train_log.csv")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot Score
ax1.plot(df['epoch'], df['score'], linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Score')
ax1.set_title('Training Score over Time')
ax1.grid(True, alpha=0.3)

# Plot Loss
ax2.plot(df['epoch'], df['loss'], linewidth=2, color='orange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{log_path}/training_curves.png", dpi=300)
plt.show()

print(f"Plot saved to {log_path}/training_curves.png")