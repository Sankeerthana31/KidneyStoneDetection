import matplotlib.pyplot as plt
from pathlib import Path

# ========================================
# PROJECT PATH
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "visualization"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ========================================
# TRAINING RESULTS
# ========================================

epochs = [1, 2, 3, 4, 5]

training_accuracy = [
    0.7094017,
    0.7418804,
    0.7333333,
    0.7418804,
    0.7299145
]

validation_accuracy = [
    0.7094017,
    0.7418804,
    0.7333333,
    0.7418804,
    0.7299145
]

training_loss = [
    0.5193270,
    0.3510230,
    0.2646192,
    0.2075522,
    0.1637555
]

validation_loss = [
    0.5554801,
    0.5384623,
    0.5703182,
    0.5785719,
    0.6670682
]

# ========================================
# ACCURACY GRAPH
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ClfNet Training and Validation Accuracy")

plt.xticks(epochs)
plt.ylim(0, 1)

plt.legend()
plt.grid(True)

accuracy_path = OUTPUT_DIR / "clfnet_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ========================================
# LOSS GRAPH
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    validation_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ClfNet Training and Validation Loss")

plt.xticks(epochs)

plt.legend()
plt.grid(True)

loss_path = OUTPUT_DIR / "clfnet_loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ========================================
# COMPLETE
# ========================================

print("\n========================================")
print("       VISUALIZATION COMPLETE")
print("========================================")

print("\nAccuracy graph:")
print(accuracy_path)

print("\nLoss graph:")
print(loss_path)