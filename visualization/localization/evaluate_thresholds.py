
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile

PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "valid" / "images"
MASK_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "valid" / "masks"
MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"

IMAGE_SIZE = 512
THRESHOLDS = [0.25, 0.35, 0.40, 0.45, 0.50]


def main():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    image_paths = sorted(IMAGE_DIR.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(
            f"No validation images found in {IMAGE_DIR}"
        )

    print("Model loaded successfully.")
    print(f"Validation images: {len(image_paths)}")

    # Store totals for each threshold
    results = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "dice_sum": 0.0,
            "images": 0
        }
        for threshold in THRESHOLDS
    }

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = MASK_DIR / image_path.name

        if not mask_path.exists():
            print(f"Missing mask: {image_path.name}")
            continue

        image = tifffile.imread(image_path)
        ground_truth_raw = tifffile.imread(mask_path)

        if image.ndim != 2 or ground_truth_raw.ndim != 2:
            print(f"Skipping invalid image: {image_path.name}")
            continue

        if image.dtype != np.uint8:
            print(f"Skipping non-uint8 image: {image_path.name}")
            continue

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

        for threshold in THRESHOLDS:
            prediction = probability >= threshold

            tp = np.logical_and(
                prediction, ground_truth
            ).sum()

            fp = np.logical_and(
                prediction, ~ground_truth
            ).sum()

            fn = np.logical_and(
                ~prediction, ground_truth
            ).sum()

            denominator = prediction.sum() + ground_truth.sum()

            dice = (
                2.0 * tp / denominator
                if denominator > 0
                else 1.0
            )

            results[threshold]["tp"] += int(tp)
            results[threshold]["fp"] += int(fp)
            results[threshold]["fn"] += int(fn)
            results[threshold]["dice_sum"] += float(dice)
            results[threshold]["images"] += 1

        if index % 25 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print("\n========== VALIDATION THRESHOLD RESULTS ==========")

    print(
        f"{'Threshold':<12}"
        f"{'Dice':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'TP':<10}"
        f"{'FP':<10}"
        f"{'FN':<10}"
    )

    for threshold in THRESHOLDS:
        r = results[threshold]

        tp = r["tp"]
        fp = r["fp"]
        fn = r["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        mean_dice = (
            r["dice_sum"] / r["images"]
            if r["images"]
            else 0.0
        )

        print(
            f"{threshold:<12.2f}"
            f"{mean_dice:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{tp:<10}"
            f"{fp:<10}"
            f"{fn:<10}"
        )

    print("\nNote: Dice is averaged across images.")
    print("Precision and recall are calculated from total pixel counts.")
    print("The trained model has not been modified.")


if __name__ == "__main__":
    main()
