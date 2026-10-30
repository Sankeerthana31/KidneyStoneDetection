
import matplotlib.pyplot as plt
from pathlib import Path

# Project output directory
PROJECT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT / "visualization" / "localization" / "threshold_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Actual validation results
thresholds = [0.25, 0.35, 0.40, 0.45, 0.50]

dice = [0.9538, 0.9543, 0.9544, 0.9545, 0.9549]
precision = [0.9804, 0.9816, 0.9822, 0.9828, 0.9834]
recall = [0.9768, 0.9759, 0.9752, 0.9747, 0.9744]

# Plot 1: Threshold vs Dice
plt.figure(figsize=(9, 5))
plt.plot(thresholds, dice, marker="o", linewidth=2)

for x, y in zip(thresholds, dice):
    plt.annotate(f"{y:.4f}", (x, y), xytext=(0, 9),
                 textcoords="offset points", ha="center")

plt.title("Validation Dice Score vs Prediction Threshold")
plt.xlabel("Prediction Threshold")
plt.ylabel("Mean Dice Score")
plt.xticks(thresholds)
plt.ylim(0.95, 0.956)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dice_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Plot 2: Precision and Recall
plt.figure(figsize=(9, 5))
plt.plot(thresholds, precision, marker="o",
         linewidth=2, label="Precision")
plt.plot(thresholds, recall, marker="s",
         linewidth=2, label="Recall")

plt.title("Validation Precision and Recall vs Threshold")
plt.xlabel("Prediction Threshold")
plt.ylabel("Score")
plt.xticks(thresholds)
plt.ylim(0.97, 0.99)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Graphs generated successfully!")
print(f"Output folder: {OUTPUT_DIR}")
print("1. dice_vs_threshold.png")
print("2. precision_recall_vs_threshold.png")
