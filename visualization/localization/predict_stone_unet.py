
from pathlib import Path

import numpy as np
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT = Path(r"D:\KidneyStoneDetection")
DATA_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test"
MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"
OUTPUT_PATH = PROJECT / "visualization" / "localization" / "stone_unet_512_prediction.png"

IMAGE_SIZE = 512
MASK_THRESHOLD = 247
PREDICTION_THRESHOLD = 0.5


def main():
    image_dir = DATA_DIR / "images"
    mask_dir = DATA_DIR / "masks"

    image_paths = sorted(image_dir.glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No test TIFF images found in {image_dir}")

    image_path = next(
        (p for p in image_paths if p.name == "1012.tif"),
        image_paths[0],
    )
    mask_path = mask_dir / image_path.name

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Matching mask not found: {mask_path}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    original = tifffile.imread(image_path)
    ground_truth = tifffile.imread(mask_path)

    if original.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Expected grayscale CT images and 2D masks.")

    # The trained model uses CT values scaled to [0, 1].
    resized_image = tf.image.resize(
        original[..., np.newaxis].astype(np.float32) / 255.0,
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
    ).numpy()

    probability = model.predict(resized_image[np.newaxis, ...], verbose=0)[0, ..., 0]
    predicted_small = probability >= PREDICTION_THRESHOLD

    # Resize the predicted mask to the original image dimensions for display.
    predicted = tf.image.resize(
        predicted_small.astype(np.uint8)[..., np.newaxis],
        original.shape,
        method="nearest",
    ).numpy()[..., 0] > 0

    truth = ground_truth >= MASK_THRESHOLD

    intersection = np.logical_and(predicted, truth).sum()
    dice = (2.0 * intersection) / (predicted.sum() + truth.sum() + 1e-7)

    print(f"Model: {MODEL_PATH.name}")
    print(f"Test image: {image_path.name}")
    print(f"Prediction shape: {probability.shape}")
    print(f"Predicted foreground pixels: {predicted.sum()}")
    print(f"Ground-truth foreground pixels: {truth.sum()}")
    print(f"Single-image Dice: {dice:.4f}")

    overlay = np.stack([original] * 3, axis=-1).astype(np.float32)
    overlay[predicted] = (
        0.5 * overlay[predicted] + 0.5 * np.array([255, 0, 0])
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original CT")

    axes[1].imshow(truth, cmap="gray")
    axes[1].set_title("Ground truth")

    axes[2].imshow(predicted, cmap="gray")
    axes[2].set_title("512×512 model prediction")

    axes[3].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[3].set_title("Prediction overlay (red)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Preview saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
