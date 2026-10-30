from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
    r"\cross_validation\Test\Fold_1\Negative\284.jpg"
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    print("Original image:")
    print("Path:", image_path)

    # Open image
    image = Image.open(image_path)

    print("Original size:", image.size)
    print("Original mode:", image.mode)

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Resize to model input size
    # --------------------------------------------------------

    image = image.resize((224, 224))

    print("\nAfter resizing:")
    print("Size:", image.size)

    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    image_array = np.array(image)

    print("\nNumPy information:")
    print("Shape:", image_array.shape)
    print("Data type:", image_array.dtype)

    # --------------------------------------------------------
    # Normalize pixel values
    # --------------------------------------------------------

    image_array = image_array / 255.0

    print("\nAfter normalization:")
    print("Minimum pixel value:", image_array.min())
    print("Maximum pixel value:", image_array.max())
    print("Data type:", image_array.dtype)

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(image_array, axis=0)

    print("\nFinal model input:")
    print("Shape:", image_array.shape)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("     PREPROCESSING TEST")
    print("========================================")

    preprocess_image(IMAGE_PATH)