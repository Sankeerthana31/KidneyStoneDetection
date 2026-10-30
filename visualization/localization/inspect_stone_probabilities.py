
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test" / "images"
MASK_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test" / "masks"
MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"

OUTPUT_DIR = PROJECT / "visualization" / "localization" / "probability_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
THRESHOLD = 0.5
FOCUS_IMAGES = ["740.tif", "875.tif"]


def main():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.\n")

    for filename in FOCUS_IMAGES:
        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = tifffile.imread(image_path)
        ground_truth_raw = tifffile.imread(mask_path)

        ground_truth = ground_truth_raw >= 247

        normalized = image.astype(np.float32) / 255.0

        model_input = tf.image.resize(
            normalized[..., np.newaxis],
            (IMAGE_SIZE, IMAGE_SIZE),
            method="bilinear"
        ).numpy()

        probability = model.predict(
            model_input[np.newaxis, ...],
            verbose=0
        )[0, ..., 0]

        predicted_mask = probability >= THRESHOLD

        # Probability values in the annotated region
        gt_probabilities = probability[ground_truth]

        # Statistics
        print("=" * 50)
        print(f"Image: {filename}")
        print(f"Ground-truth pixels: {ground_truth.sum()}")
        print(f"Predicted pixels at threshold 0.5: {predicted_mask.sum()}")
        print(f"Maximum probability: {probability.max():.6f}")
        print(f"Mean probability over entire image: {probability.mean():.6f}")

        if gt_probabilities.size > 0:
            print(
                "Mean probability inside ground truth: "
                f"{gt_probabilities.mean():.6f}"
            )
            print(
                "Maximum probability inside ground truth: "
                f"{gt_probabilities.max():.6f}"
            )
            print(
                "Ground-truth pixels with probability >= 0.5: "
                f"{np.sum(gt_probabilities >= THRESHOLD)} "
                f"/ {gt_probabilities.size}"
            )

            print(
                "Ground-truth pixels with probability >= 0.25: "
                f"{np.sum(gt_probabilities >= 0.25)} "
                f"/ {gt_probabilities.size}"
            )
        else:
            print("No ground-truth foreground pixels found.")

        # Find crop around ground truth and predicted regions
        combined = np.logical_or(ground_truth, predicted_mask)
        coords = np.argwhere(combined)

        if len(coords) > 0:
            cy, cx = coords.mean(axis=0).astype(int)
        else:
            cy, cx = image.shape[0] // 2, image.shape[1] // 2

        size = 128
        half = size // 2

        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        y2 = min(image.shape[0], y1 + size)
        x2 = min(image.shape[1], x1 + size)

        y1 = max(0, y2 - size)
        x1 = max(0, x2 - size)

        image_crop = image[y1:y2, x1:x2]
        gt_crop = ground_truth[y1:y2, x1:x2]
        prob_crop = probability[y1:y2, x1:x2]
        pred_crop = predicted_mask[y1:y2, x1:x2]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        axes[0].imshow(image_crop, cmap="gray")
        axes[0].set_title("Original CT")

        axes[1].imshow(gt_crop, cmap="gray", interpolation="nearest")
        axes[1].set_title("Ground Truth")

        im = axes[2].imshow(
            prob_crop,
            cmap="hot",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )
        axes[2].set_title("Prediction Probability")
        fig.colorbar(im, ax=axes[2], fraction=0.046)

        axes[3].imshow(pred_crop, cmap="gray", interpolation="nearest")
        axes[3].set_title("Predicted Mask (0.5)")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(filename)
        plt.tight_layout()

        output_path = OUTPUT_DIR / (
            Path(filename).stem + "_probability.png"
        )

        plt.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Probability map saved: {output_path}\n")


if __name__ == "__main__":
    main()
