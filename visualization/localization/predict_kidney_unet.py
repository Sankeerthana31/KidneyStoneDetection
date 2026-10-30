
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

BASE = Path("data/kidney_segmentation")
MODEL_PATH = Path("models/localization/kidney_unet_best.keras")
OUTPUT = Path("visualization/localization/kidney_unet_prediction.png")

# Recreate the same split used during training
pairs = [
    (p, BASE / "masks" / p.name)
    for p in sorted((BASE / "images").glob("*.png"))
    if (BASE / "masks" / p.name).exists()
]

random.seed(42)
random.shuffle(pairs)
validation_pairs = pairs[int(len(pairs) * 0.8):]

# Choose a validation image
image_path, mask_path = validation_pairs[0]

image = np.array(
    Image.open(image_path).convert("RGB").resize((256, 256)),
    dtype=np.uint8,
)

mask = np.array(
    Image.open(mask_path).convert("L").resize(
        (256, 256), Image.Resampling.NEAREST
    )
) > 127

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

input_image = image.astype(np.float32) / 255.0
prediction = model.predict(input_image[None, ...], verbose=0)[0, ..., 0]
predicted_mask = prediction > 0.5

overlay = image.copy()
overlay[predicted_mask] = (
    0.5 * image[predicted_mask]
    + 0.5 * np.array([0, 255, 0])
).astype(np.uint8)

canvas = Image.new("RGB", (768, 290), "white")
draw = ImageDraw.Draw(canvas)

panels = [
    Image.fromarray(image),
    Image.fromarray(mask.astype(np.uint8) * 255),
    Image.fromarray(overlay),
]

labels = ["Validation image", "Ground-truth mask", "Predicted overlay"]

for i, (panel, label) in enumerate(zip(panels, labels)):
    canvas.paste(panel, (i * 256, 25))
    draw.text((i * 256 + 5, 5), label, fill="black")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT)

print("Validation image:", image_path.name)
print("Prediction saved to:", OUTPUT.resolve())
print("Predicted foreground pixels:", int(predicted_mask.sum()))
print("Ground-truth foreground pixels:", int(mask.sum()))
