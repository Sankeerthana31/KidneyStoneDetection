from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt

data_dir = Path(
    r"D:\KidneyStoneDetection\data\ct_localization\KSSD2025\data"
)

image = tifffile.imread(str(data_dir / "image" / "1.tif"))
mask = tifffile.imread(str(data_dir / "label" / "1.tif"))

# Display the original mask and a thresholded version for inspection.
binary_mask = mask >= 247

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original CT Image")

axes[1].imshow(binary_mask, cmap="gray")
axes[1].set_title("Original Label Mask")

axes[2].imshow(image, cmap="gray")
overlay = np.zeros((*binary_mask.shape, 4), dtype=float)
overlay[binary_mask] = [1, 0, 0, 0.8]
axes[2].imshow(overlay)
axes[2].set_title("CT Image + Thresholded Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path(
    r"D:\KidneyStoneDetection\visualization\localization\stone_mask_preview.png"
)
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", np.unique(mask))
print("Foreground pixels:", np.count_nonzero(binary_mask))
print("Preview saved to:", output)