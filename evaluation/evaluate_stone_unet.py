
from pathlib import Path
import numpy as np
import tifffile
import tensorflow as tf

PROJECT = Path(r"D:\KidneyStoneDetection")
TEST_DIR = PROJECT / "data" / "ct_localization" / "stone_splits" / "test"
MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"

IMAGE_SIZE = 512
MASK_THRESHOLD = 247
PREDICTION_THRESHOLD = 0.5
EPSILON = 1e-7


def main():
    image_dir = TEST_DIR / "images"
    mask_dir = TEST_DIR / "masks"

    image_paths = sorted(image_dir.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No test images found: {image_dir}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    dice_scores = []
    empty_predictions = 0
    empty_targets = 0

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Matching mask not found: {mask_path}")

        image = tifffile.imread(str(image_path)).astype(np.float32)
        mask = tifffile.imread(str(mask_path))

        image = tf.image.resize(
            (image / 255.0)[..., None],
            (IMAGE_SIZE, IMAGE_SIZE),
            method="bilinear",
        ).numpy()

        target = mask >= MASK_THRESHOLD

        prediction_probability = model.predict(
            image[None, ...], verbose=0
        )[0, ..., 0]

        prediction = prediction_probability >= PREDICTION_THRESHOLD

        if prediction.sum() == 0:
            empty_predictions += 1

        if target.sum() == 0:
            empty_targets += 1

        intersection = np.logical_and(target, prediction).sum()
        denominator = target.sum() + prediction.sum()

        dice = (
            2.0 * intersection / (denominator + EPSILON)
            if denominator > 0
            else 1.0
        )

        dice_scores.append(dice)

        if index % 10 == 0 or index == len(image_paths):
            print(f"Evaluated {index}/{len(image_paths)} images")

    print("\n--- Stone U-Net 512x512 Test Evaluation ---")
    print("Model:", MODEL_PATH.name)
    print("Test images:", len(image_paths))
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Median Dice: {np.median(dice_scores):.4f}")
    print("Empty predictions:", empty_predictions)
    print("Empty ground-truth masks:", empty_targets)


if __name__ == "__main__":
    main()
