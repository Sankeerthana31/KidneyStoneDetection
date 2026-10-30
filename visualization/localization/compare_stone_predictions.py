
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile
import matplotlib.pyplot as plt

# Project paths
PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test" / "images"
MASK_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test" / "masks"
MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"

OUTPUT_DIR = PROJECT / "visualization" / "localization" / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
THRESHOLD = 0.5
NUMBER_OF_IMAGES = 10

# Images requiring closer inspection
FOCUS_IMAGES = ["740.tif", "875.tif"]
ZOOM_SIZE = 128


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_mask(model, image):
    normalized = image.astype(np.float32) / 255.0

    resized = tf.image.resize(
        normalized[..., np.newaxis],
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear"
    ).numpy()

    probability = model.predict(
        resized[np.newaxis, ...],
        verbose=0
    )[0, ..., 0]

    return probability >= THRESHOLD


def make_overlay(image, mask):
    overlay = np.stack([image] * 3, axis=-1).astype(np.float32)

    overlay[mask] = (
        0.5 * overlay[mask]
        + 0.5 * np.array([255, 0, 0])
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_zoom_crop(ground_truth, predicted_mask, zoom_size=128):
    # Find the combined foreground region
    combined = np.logical_or(ground_truth, predicted_mask)
    coordinates = np.argwhere(combined)

    height, width = ground_truth.shape

    if len(coordinates) == 0:
        center_y, center_x = height // 2, width // 2
    else:
        center_y, center_x = coordinates.mean(axis=0).astype(int)

    half = zoom_size // 2

    y1 = max(0, center_y - half)
    y2 = min(height, y1 + zoom_size)
    x1 = max(0, center_x - half)
    x2 = min(width, x1 + zoom_size)

    # Adjust crop when it reaches the image boundary
    y1 = max(0, y2 - zoom_size)
    x1 = max(0, x2 - zoom_size)

    return y1, y2, x1, x2


def save_zoom_comparison(
    image_path,
    image,
    ground_truth,
    predicted_mask,
    dice
):
    y1, y2, x1, x2 = get_zoom_crop(
        ground_truth,
        predicted_mask,
        ZOOM_SIZE
    )

    image_crop = image[y1:y2, x1:x2]
    gt_crop = ground_truth[y1:y2, x1:x2]
    pred_crop = predicted_mask[y1:y2, x1:x2]

    overlay = make_overlay(image, predicted_mask)
    overlay_crop = overlay[y1:y2, x1:x2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image_crop, cmap="gray")
    axes[0].set_title("Original CT - Zoomed")

    axes[1].imshow(gt_crop, cmap="gray", interpolation="nearest")
    axes[1].set_title("Ground Truth - Zoomed")

    axes[2].imshow(pred_crop, cmap="gray", interpolation="nearest")
    axes[2].set_title("Prediction - Zoomed")

    axes[3].imshow(overlay_crop)
    axes[3].set_title("Overlay - Zoomed")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_path.name} | Dice Score: {dice:.4f} "
        f"| Crop: {ZOOM_SIZE}x{ZOOM_SIZE}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        image_path.stem + "_zoom_comparison.png"
    )

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Zoom comparison saved: {output_path}")


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(
            f"No TIFF test images found in: {IMAGE_DIR}"
        )

    count = min(NUMBER_OF_IMAGES, len(image_paths))

    indices = np.linspace(
        0, len(image_paths) - 1, count, dtype=int
    )

    selected_images = [image_paths[i] for i in indices]

    model = load_model()

    print(f"Loaded model: {MODEL_PATH.name}")
    print(f"Selected images: {len(selected_images)}")

    fig, axes = plt.subplots(
        len(selected_images),
        4,
        figsize=(16, 4 * len(selected_images))
    )

    if len(selected_images) == 1:
        axes = np.expand_dims(axes, axis=0)

    results = []

    for row, image_path in enumerate(selected_images):
        mask_path = MASK_DIR / image_path.name

        if not mask_path.exists():
            print(f"Skipping missing mask: {mask_path.name}")
            continue

        image = tifffile.imread(image_path)
        ground_truth_raw = tifffile.imread(mask_path)

        if image.ndim != 2 or ground_truth_raw.ndim != 2:
            print(f"Skipping non-grayscale image: {image_path.name}")
            continue

        if image.dtype != np.uint8:
            print(f"Skipping non-uint8 image: {image_path.name}")
            continue

        ground_truth = ground_truth_raw >= 247

        predicted_mask = predict_mask(model, image)

        if predicted_mask.shape != ground_truth.shape:
            raise ValueError(
                f"Mask shape mismatch for {image_path.name}"
            )

        intersection = np.logical_and(
            ground_truth, predicted_mask
        ).sum()

        denominator = ground_truth.sum() + predicted_mask.sum()

        dice = (
            2.0 * intersection / denominator
            if denominator > 0
            else 1.0
        )

        overlay = make_overlay(image, predicted_mask)

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title("Original CT")

        axes[row, 1].imshow(ground_truth, cmap="gray")
        axes[row, 1].set_title("Ground Truth Mask")

        axes[row, 2].imshow(predicted_mask, cmap="gray")
        axes[row, 2].set_title("Predicted Mask")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction Overlay")

        for col in range(4):
            axes[row, col].axis("off")

        results.append(
            (
                image_path.name,
                dice,
                int(predicted_mask.sum()),
                int(ground_truth.sum())
            )
        )

        print(
            f"{image_path.name}: Dice={dice:.4f}, "
            f"Predicted pixels={predicted_mask.sum()}, "
            f"Ground-truth pixels={ground_truth.sum()}"
        )

        # Generate enlarged comparisons for selected images
        if image_path.name in FOCUS_IMAGES:
            save_zoom_comparison(
                image_path,
                image,
                ground_truth,
                predicted_mask,
                dice
            )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "stone_prediction_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nComparison saved successfully:")
    print(output_path)

    if results:
        mean_dice = np.mean([result[1] for result in results])
        print(f"\nMean Dice for displayed images: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
