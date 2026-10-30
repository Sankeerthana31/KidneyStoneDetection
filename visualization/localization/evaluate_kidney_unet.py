
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path("data/kidney_segmentation")
MODEL_PATH = Path("models/localization/kidney_unet_best.keras")
IMAGE_SIZE = 256

# Recreate the validation split used during training
pairs = [
    (p, BASE / "masks" / p.name)
    for p in sorted((BASE / "images").glob("*.png"))
    if (BASE / "masks" / p.name).exists()
]

random.seed(42)
random.shuffle(pairs)
validation_pairs = pairs[int(len(pairs) * 0.8):]

# Evaluate the same image as the previous prediction
image_path, mask_path = next(
    (img, mask) for img, mask in validation_pairs
    if img.name == "01322.png"
)

image = np.array(
    Image.open(image_path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    dtype=np.float32,
) / 255.0

truth = np.array(
    Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
    )
) > 127

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
probabilities = model.predict(image[None, ...], verbose=0)[0, ..., 0]
prediction = probabilities > 0.5

intersection = np.logical_and(truth, prediction).sum()
union = np.logical_or(truth, prediction).sum()

dice = 2 * intersection / (truth.sum() + prediction.sum() + 1e-8)
iou = intersection / (union + 1e-8)

print("Image:", image_path.name)
print(f"Dice score: {dice:.4f} ({dice * 100:.2f}%)")
print(f"IoU score:  {iou:.4f} ({iou * 100:.2f}%)")
