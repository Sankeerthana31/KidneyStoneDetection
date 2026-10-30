
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Project visualization folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Five-fold cross-validation accuracy (%)
folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
fold_accuracies = [76.41, 81.02, 75.00, 82.19, 81.04]

# Latest holdout evaluation
holdout_accuracy = 78.55

# Rows = actual classes; columns = predicted classes
# Class order: Negative, Positive
confusion_matrix = np.array([
    [184, 108],
    [49, 391]
])

# 1. Accuracy across five folds
plt.figure(figsize=(9, 5))
bars = plt.bar(folds, fold_accuracies)

plt.axhline(
    np.mean(fold_accuracies),
    linestyle="--",
    label=f"Mean fold accuracy: {np.mean(fold_accuracies):.2f}%"
)

plt.ylim(0, 100)
plt.ylabel("Test Accuracy (%)")
plt.xlabel("Cross-validation Fold")
plt.title("ClfNet Test Accuracy Across Five Folds")
plt.legend()

for bar, accuracy in zip(bars, fold_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fold_accuracy.png", dpi=300)
plt.close()

# 2. Latest holdout confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix, cmap="Blues")
plt.colorbar(label="Number of Images")

plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Holdout Confusion Matrix")

for row in range(2):
    for col in range(2):
        plt.text(
            col,
            row,
            str(confusion_matrix[row, col]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "holdout_confusion_matrix.png", dpi=300)
plt.close()

# 3. Cross-validation versus holdout accuracy
labels = folds + ["Holdout"]
accuracies = fold_accuracies + [holdout_accuracy]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels, accuracies)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Cross-validation and Holdout Accuracy Comparison")

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
plt.close()

print("All graphs generated successfully!")
print(f"Graphs saved in: {OUTPUT_DIR}")
print(f"Mean five-fold accuracy: {np.mean(fold_accuracies):.2f}%")
print(f"Holdout accuracy: {holdout_accuracy:.2f}%")
print("Holdout images evaluated: 732")
