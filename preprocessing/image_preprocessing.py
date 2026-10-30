from PIL import Image
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Load and preprocess a CT image for MobileNetV2.

    Steps:
    1. Open image
    2. Convert to RGB
    3. Resize to 224 x 224
    4. Convert to float32
    5. Apply MobileNetV2 ImageNet preprocessing
    """

    image = Image.open(image_path)

    # Convert grayscale/RGBA images to RGB
    image = image.convert("RGB")

    # Resize to the input size expected by the model
    image = image.resize(IMAGE_SIZE)

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Convert to float32
    image_array = image_array.astype(np.float32)

    # MobileNetV2 ImageNet preprocessing
    # Converts pixel values approximately from [0, 255]
    # to [-1, 1]
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# TEST THE PREPROCESSING PIPELINE
# ============================================================

if __name__ == "__main__":

    from pathlib import Path

    print("========================================")
    print("       IMAGE PREPROCESSING TEST")
    print("========================================")

    # Dataset location
    dataset_path = Path(
        r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
    )

    # Find one JPG image
    image_files = list(dataset_path.rglob("*.jpg"))

    print("\nDataset path:")
    print(dataset_path)

    print("\nNumber of JPG images found:")
    print(len(image_files))

    if len(image_files) == 0:

        print("\nERROR: No JPG images found.")

    else:

        # Select first image for testing
        test_image = image_files[0]

        print("\nTest image:")
        print(test_image)

        # Preprocess image
        image = preprocess_image(str(test_image))

        print("\n========================================")
        print("       PREPROCESSING RESULT")
        print("========================================")

        print("\nImage shape:")
        print(image.shape)

        print("\nImage dtype:")
        print(image.dtype)

        print("\nMinimum pixel value:")
        print(image.min())

        print("\nMaximum pixel value:")
        print(image.max())

        print("\n========================================")
        print("       PREPROCESSING TEST COMPLETE")
        print("========================================")