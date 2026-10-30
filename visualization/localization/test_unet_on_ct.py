
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = Path(
    r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
    r"\cross_validation\Test\Fold_1\Positive\1.jpg"
)

MODEL_PATH = PROJECT_ROOT / "models" / "localization" / "kidney_unet_best.keras"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "localization" / "ct_unet_experiment.png"

def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"CT image not found: {IMAGE_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"U-Net model not found: {MODEL_PATH}")

    # Load the CT image and resize it to the U-Net's expected input size.
    image = Image.open(IMAGE_PATH).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.asarray(image, dtype=np.uint8)

    # This preprocessing matches the ultrasound model's training code.
    input_array = image_array.astype(np.float32) / 255.0

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    prediction = model.predict(input_array[None, ...], verbose=0)[0, ..., 0]
    predicted_mask = prediction > 0.5

    # Create a green overlay for visual inspection only.
    overlay = image_array.copy()
    overlay[predicted_mask] = (
        0.5 * overlay[predicted_mask]
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    canvas = Image.new("RGB", (512, 290), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(image_array), (0, 25))
    canvas.paste(Image.fromarray(overlay), (256, 25))
    draw.text((5, 5), "CT image", fill="black")
    draw.text((261, 5), "Experimental U-Net overlay", fill="black")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)

    print("Input CT image:", IMAGE_PATH)
    print("Output saved to:", OUTPUT_PATH)
    print("Predicted foreground pixels:", int(predicted_mask.sum()))
    print("WARNING: This ultrasound-trained model is not validated for CT.")
    print("The green overlay is experimental, not a reliable kidney localization.")

if __name__ == "__main__":
    main()
