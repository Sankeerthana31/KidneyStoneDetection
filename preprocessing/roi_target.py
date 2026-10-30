from PIL import Image
import numpy as np


IMAGE_SIZE = (224, 224)


def create_roi_target(image_path, mask_path):
    """
    Create the regression ground-truth ROI.

    According to the project paper:
    ROI ground truth = kidney mask × original CT image.

    Parameters:
        image_path : path to original CT image
        mask_path  : path to binary kidney mask

    Returns:
        ROI image as a NumPy array.
    """

    # Load original CT image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Load kidney mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(IMAGE_SIZE)

    # Convert to NumPy
    image_array = np.array(image).astype(np.float32) / 255.0
    mask_array = np.array(mask).astype(np.float32) / 255.0

    # Make mask binary
    mask_array = (mask_array > 0.5).astype(np.float32)

    # Add channel dimension
    mask_array = np.expand_dims(mask_array, axis=-1)

    # Element-wise multiplication
    roi = image_array * mask_array

    return roi


if __name__ == "__main__":
    print("========================================")
    print("        ROI TARGET GENERATOR")
    print("========================================")

    print("\nROI target formula:")
    print("ROI = Kidney Mask × Original CT Image")

    print("\nCurrent dataset does not contain")
    print("kidney mask files.")
    print("\nTherefore, real ROI target generation")
    print("will be performed when the original")
    print("annotation/mask files are available.")

    print("\n========================================")
    print("       ROI PIPELINE READY")
    print("========================================")