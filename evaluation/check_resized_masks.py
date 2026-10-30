
from pathlib import Path
import numpy as np
import tifffile
import tensorflow as tf

PROJECT = Path(r"D:\KidneyStoneDetection")
TEST_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test"
SIZE = 256

target_counts = []
original_counts = []

for image_path in sorted((TEST_DIR / "images").glob("*.tif")):
    mask_path = TEST_DIR / "masks" / image_path.name
    mask = tifffile.imread(str(mask_path))

    original = mask >= 247

    resized = tf.image.resize(
        original.astype(np.float32)[..., None],
        (SIZE, SIZE),
        method="nearest"
    ).numpy()[..., 0] >= 0.5

    original_counts.append(int(original.sum()))
    target_counts.append(int(resized.sum()))

print("Test images:", len(target_counts))
print("Original mask pixels - min:", min(original_counts))
print("Original mask pixels - median:", int(np.median(original_counts)))
print("Original mask pixels - max:", max(original_counts))
print("Resized mask pixels - min:", min(target_counts))
print("Resized mask pixels - median:", int(np.median(target_counts)))
print("Resized mask pixels - max:", max(target_counts))
print("Resized masks with fewer than 5 pixels:",
      sum(n < 5 for n in target_counts))
print("Resized masks with fewer than 10 pixels:",
      sum(n < 10 for n in target_counts))
