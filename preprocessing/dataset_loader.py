from pathlib import Path
import pandas as pd


# ============================================================
# DATASET LOCATION
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\sanke\Downloads\Dataset\Dataset\dataset_integrated_task"
)


# ============================================================
# FIND JPG IMAGES
# ============================================================

def find_images(folder_path):
    """
    Find all JPG images inside a folder and its subfolders.
    """

    images = []

    for file in Path(folder_path).rglob("*.jpg"):
        images.append(str(file))

    return images


# ============================================================
# ADD IMAGES TO DATASET
# ============================================================

def add_images(data, class_folder, dataset_type, split, fold):
    """
    Add Positive/Negative images to our dataset list.
    """

    class_name = class_folder.name

    # Only accept Positive and Negative
    if class_name not in ["Positive", "Negative"]:
        return

    # Positive = 1
    # Negative = 0
    if class_name == "Positive":
        label = 1
    else:
        label = 0

    # Find images
    images = find_images(class_folder)

    # Add every image
    for image_path in images:

        data.append({
            "image_path": image_path,
            "label": label,
            "class_name": class_name,
            "dataset_type": dataset_type,
            "split": split,
            "fold": fold
        })


# ============================================================
# LOAD DATASET
# ============================================================

def get_classification_data(base_path):
    """
    Load cross-validation and holdout datasets.
    """

    data = []

    base_path = Path(base_path)

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    cross_validation_path = base_path / "cross_validation"

    if cross_validation_path.exists():

        # ----------------------------------------------------
        # Training and Test
        # ----------------------------------------------------

        for split in ["Training", "Test"]:

            split_path = cross_validation_path / split

            if not split_path.exists():
                continue

            # Find every Positive/Negative folder
            for class_folder in split_path.rglob("*"):

                if not class_folder.is_dir():
                    continue

                if class_folder.name not in ["Positive", "Negative"]:
                    continue

                # ------------------------------------------------
                # Find the folder containing Positive/Negative
                # ------------------------------------------------

                fold = class_folder.parent.name

                # If Positive/Negative is directly inside Test,
                # there is no fold.
                if fold == split:
                    fold = None

                add_images(
                    data,
                    class_folder,
                    "cross_validation",
                    split,
                    fold
                )

    # ========================================================
    # HOLDOUT
    # ========================================================

    holdout_path = base_path / "holdout"

    if holdout_path.exists():

        # Holdout contains Positive and Negative directly
        for class_folder in holdout_path.iterdir():

            if not class_folder.is_dir():
                continue

            if class_folder.name not in ["Positive", "Negative"]:
                continue

            add_images(
                data,
                class_folder,
                "holdout",
                "holdout",
                None
            )

    # Convert list to DataFrame
    return pd.DataFrame(data)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("     KIDNEY STONE DATASET LOADER")
    print("========================================")

    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    print("\nDataset path:")
    print(DATASET_PATH)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        print("\nERROR: Dataset path not found!")

    else:

        print("\nDataset path found!")

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        print("\nLoading dataset...")

        df = get_classification_data(DATASET_PATH)

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print("\n========================================")
        print("DATASET SUMMARY")
        print("========================================")

        print("\nTotal images:", len(df))

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        print("\nClass distribution:")

        if not df.empty:
            print(df["class_name"].value_counts())

        # ----------------------------------------------------
        # Dataset type
        # ----------------------------------------------------

        print("\nDataset type:")

        if not df.empty:
            print(df["dataset_type"].value_counts())

        # ----------------------------------------------------
        # Split distribution
        # ----------------------------------------------------

        print("\nSplit distribution:")

        if not df.empty:
            print(df["split"].value_counts())

        # ----------------------------------------------------
        # Fold distribution
        # ----------------------------------------------------

        print("\nFold distribution:")

        if not df.empty:
            print(df["fold"].value_counts(dropna=False))

        # ----------------------------------------------------
        # First 10 records
        # ----------------------------------------------------

        print("\nFirst 10 records:")

        if not df.empty:
            print(df.head(10))