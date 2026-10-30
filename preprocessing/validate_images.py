from pathlib import Path
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
)


def validate_images():

    images = list(DATASET_PATH.rglob("*.jpg"))

    print("Total JPG images:", len(images))
    print("\nChecking all images...\n")

    size_counts = {}
    mode_counts = {}
    format_counts = {}

    errors = []

    for index, image_path in enumerate(images, start=1):

        try:

            with Image.open(image_path) as image:

                size = image.size
                mode = image.mode
                image_format = image.format

                size_counts[size] = size_counts.get(size, 0) + 1
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                format_counts[image_format] = format_counts.get(image_format, 0) + 1

        except Exception as e:

            errors.append((str(image_path), str(e)))

        # Progress every 1000 images
        if index % 1000 == 0:
            print(f"Checked {index} images...")


    print("\n========================================")
    print("IMAGE VALIDATION RESULTS")
    print("========================================")

    print("\nImage sizes:")
    for size, count in size_counts.items():
        print(size, ":", count)

    print("\nImage modes:")
    for mode, count in mode_counts.items():
        print(mode, ":", count)

    print("\nImage formats:")
    for image_format, count in format_counts.items():
        print(image_format, ":", count)

    print("\nImage errors:", len(errors))

    if errors:
        print("\nFirst 10 errors:")

        for path, error in errors[:10]:
            print("----------------------------------------")
            print("Image:", path)
            print("Error:", error)


if __name__ == "__main__":
    validate_images()