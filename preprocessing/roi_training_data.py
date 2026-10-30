from pathlib import Path
import pandas as pd


def find_mask_for_image(image_path, mask_folder):
    """
    Find the kidney mask corresponding to a CT image.

    The exact filename relationship depends on the
    annotation dataset supplied with the project.
    """

    image_path = Path(image_path)
    mask_folder = Path(mask_folder)

    possible_names = [
        image_path.stem + ".png",
        image_path.stem + ".jpg",
        image_path.stem + ".jpeg",
        image_path.stem + ".bmp",
        image_path.stem + ".tif",
        image_path.stem + ".tiff",
    ]

    for name in possible_names:

        mask_path = mask_folder / name

        if mask_path.exists():
            return str(mask_path)

    return None


def create_roi_training_dataframe(image_paths, mask_folder):
    """
    Create a dataframe containing:

        CT image
        Kidney mask
        ROI target availability
    """

    data = []

    for image_path in image_paths:

        mask_path = find_mask_for_image(
            image_path,
            mask_folder
        )

        data.append({
            "image_path": str(image_path),
            "mask_path": mask_path,
            "mask_available": mask_path is not None
        })

    return pd.DataFrame(data)


if __name__ == "__main__":

    print("========================================")
    print("      ROI TRAINING DATA LOADER")
    print("========================================")

    # Example paths
    image_folder = Path(
        r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
    )

    # This folder will be replaced with the actual
    # annotation/mask folder when it is available.
    mask_folder = Path(
        r"C:\Users\sanke\Downloads\Dataset\Dataset\masks"
    )

    print("\nCT dataset:")
    print(image_folder)

    print("\nExpected mask folder:")
    print(mask_folder)

    if not image_folder.exists():

        print("\nERROR: CT image dataset not found.")

    else:

        image_paths = list(
            image_folder.rglob("*.jpg")
        )

        print("\nCT images found:", len(image_paths))

        if not mask_folder.exists():

            print("\nMask folder does not exist yet.")
            print("No ROI training pairs can be created.")

        else:

            df = create_roi_training_dataframe(
                image_paths,
                mask_folder
            )

            print(
                "\nImages with masks:",
                df["mask_available"].sum()
            )

            print(
                "Images without masks:",
                (~df["mask_available"]).sum()
            )

            print("\nFirst 5 records:")
            print(df.head())

    print("\n========================================")
    print("       ROI LOADER TEST COMPLETE")
    print("========================================")