from pathlib import Path
import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


def create_roi_target(image_path, mask_path):
    """
    Create the ground-truth ROI image.

    ROI = Kidney mask × Original CT image
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(IMAGE_SIZE)

    image_array = np.array(image).astype(np.float32) / 255.0
    mask_array = np.array(mask).astype(np.float32) / 255.0

    # Convert mask to binary
    mask_array = (mask_array > 0.5).astype(np.float32)

    # Add channel dimension
    mask_array = np.expand_dims(mask_array, axis=-1)

    # Mask × original image
    roi = image_array * mask_array

    return roi


def check_mask_exists(mask_path):
    """
    Check whether a kidney mask exists.
    """

    return Path(mask_path).exists()


def validate_roi_pair(image_path, mask_path):
    """
    Validate that both the CT image and kidney mask exist.
    """

    image_exists = Path(image_path).exists()
    mask_exists = Path(mask_path).exists()

    return image_exists and mask_exists


if __name__ == "__main__":

    print("========================================")
    print("       ROI DATASET PIPELINE TEST")
    print("========================================")

    print("\nExpected training pair:")
    print("1. CT image")
    print("2. Kidney mask")

    print("\nROI generation:")
    print("ROI = Kidney Mask × Original CT Image")

    print("\nChecking current dataset...")

    dataset_path = Path(
        r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
    )

    mask_files = []

    for file in dataset_path.rglob("*"):
        if file.is_file():
            filename = file.name.lower()

            if any(
                keyword in filename
                for keyword in [
                    "mask",
                    "roi",
                    "annotation",
                    "annot",
                    "contour"
                ]
            ):
                mask_files.append(file)

    print("\nDataset path:")
    print(dataset_path)

    print("\nPotential mask/annotation files found:")
    print(len(mask_files))

    if len(mask_files) == 0:

        print("\nSTATUS: MASKS NOT FOUND")

        print("\nActual I2IRegNet training cannot start yet.")
        print("The paper requires kidney ROI masks to")
        print("create the regression ground-truth images.")

    else:

        print("\nSTATUS: POTENTIAL MASK FILES FOUND")

        for file in mask_files[:10]:
            print(file)

    print("\n========================================")
    print("      ROI PIPELINE TEST COMPLETE")
    print("========================================")