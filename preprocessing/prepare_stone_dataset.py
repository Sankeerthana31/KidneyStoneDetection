from pathlib import Path
import random
import shutil

# Source dataset
SOURCE = Path(
    r"D:\KidneyStoneDetection\data\ct_localization\KSSD2025\data"
)

# New split folders
OUTPUT = Path(
    r"D:\KidneyStoneDetection\data\ct_localization\stone_splits"
)

IMAGE_DIR = SOURCE / "image"
MASK_DIR = SOURCE / "label"

SEED = 42
random.seed(SEED)

# Match images and masks by filename
images = sorted(IMAGE_DIR.glob("*.tif"))
masks = {p.name: p for p in MASK_DIR.glob("*.tif")}

pairs = [(img, masks[img.name]) for img in images if img.name in masks]

if len(pairs) != len(images):
    raise RuntimeError("Some images do not have matching masks.")

if len(pairs) != 838:
    raise RuntimeError(
        f"Expected 838 image-mask pairs, found {len(pairs)}."
    )

random.shuffle(pairs)

n_train = int(len(pairs) * 0.70)
n_val = int(len(pairs) * 0.15)

splits = {
    "train": pairs[:n_train],
    "valid": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:],
}

for split_name, split_pairs in splits.items():
    image_out = OUTPUT / split_name / "images"
    mask_out = OUTPUT / split_name / "masks"

    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in split_pairs:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(mask_path, mask_out / mask_path.name)

    print(f"{split_name}: {len(split_pairs)} image-mask pairs")

print("Dataset splits saved to:", OUTPUT)
print("Original dataset was not modified.")