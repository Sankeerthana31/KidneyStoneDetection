from pathlib import Path
from PIL import Image


# Dataset location
DATASET_PATH = Path(
    r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
)


def inspect_images():

    # Find JPG images
    images = list(DATASET_PATH.rglob("*.jpg"))

    print("Total JPG images:", len(images))

    if len(images) == 0:
        print("No images found!")
        return

    print("\nInspecting first 10 images...\n")

    for image_path in images[:10]:

        try:

            with Image.open(image_path) as image:

                print("----------------------------------------")
                print("Image:", image_path.name)
                print("Path:", image_path)
                print("Size:", image.size)
                print("Mode:", image.mode)
                print("Format:", image.format)

        except Exception as e:

            print("ERROR:", image_path)
            print(e)


if __name__ == "__main__":
    inspect_images()