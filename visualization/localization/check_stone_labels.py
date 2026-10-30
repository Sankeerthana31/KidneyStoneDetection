from pathlib import Path
from collections import Counter

import numpy as np
import tifffile

data_dir = Path(
    r"D:\KidneyStoneDetection\data\ct_localization\KSSD2025\data"
)
label_dir = data_dir / "label"

value_counts = Counter()
foreground_counts = []
shapes = Counter()

for path in label_dir.glob("*.tif"):
    mask = tifffile.imread(str(path))
    shapes[mask.shape] += 1

    values, counts = np.unique(mask, return_counts=True)
    value_counts.update(dict(zip(values.tolist(), counts.tolist())))

    foreground_counts.append(
        (path.name, int(np.count_nonzero(mask > 127)))
    )

print("Total masks:", len(list(label_dir.glob("*.tif"))))
print("Mask dimensions:", shapes)
print("Pixel values:", sorted(value_counts.keys()))
print("Pixels <= 5:", sum(value_counts[v] for v in value_counts if v <= 5))
print("Pixels >= 251:", sum(value_counts[v] for v in value_counts if v >= 251))
print("Masks with no pixels above 127:", sum(n == 0 for _, n in foreground_counts))
print("Smallest foreground count:", min(n for _, n in foreground_counts))
print("Largest foreground count:", max(n for _, n in foreground_counts))
print("First five foreground counts:", foreground_counts[:5])