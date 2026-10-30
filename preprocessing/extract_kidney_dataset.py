from pathlib import Path
import io
import pyarrow.parquet as pq
from PIL import Image

parquet_path = Path(
    r"data\kidney_segmentation\data\train-00000-of-00001.parquet"
)
image_dir = Path(r"data\kidney_segmentation\images")
mask_dir = Path(r"data\kidney_segmentation\masks")

image_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

table = pq.read_table(parquet_path)
success = 0
errors = []

for row in table.to_pylist():
    idx = row["id"]
    image_bytes = row["image"]["bytes"]
    mask_bytes = row["mask"]["bytes"]

    image_path = image_dir / f"{idx:05d}.png"
    mask_path = mask_dir / f"{idx:05d}.png"

    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.convert("RGB").save(image_path)

        with Image.open(io.BytesIO(mask_bytes)) as mask:
            mask.convert("L").save(mask_path)

        success += 1
    except Exception as exc:
        errors.append((idx, str(exc)))

print("Total rows:", len(table))
print("Successfully extracted pairs:", success)
print("Errors:", len(errors))
if errors:
    print("First errors:", errors[:5])
print("Images:", image_dir.resolve())
print("Masks:", mask_dir.resolve())
