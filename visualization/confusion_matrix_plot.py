import matplotlib.pyplot as plt
import numpy as np

# Confusion Matrix
cm = np.array([
    [194, 108],
    [43, 240]
])

classes = ["Negative", "Positive"]

fig, ax = plt.subplots(figsize=(6, 5))

ax.imshow(cm)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix - ClfNet")

# Display values inside cells
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j],
                ha="center",
                va="center",
                fontsize=16)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Confusion matrix saved successfully!")
plt.show()