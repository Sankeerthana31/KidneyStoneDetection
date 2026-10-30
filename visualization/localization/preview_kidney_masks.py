from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

base = Path(r"data\kidney_segmentation")
out = Path(r"visualization\localization\kidney_mask_preview.png")
names = ["00002.png", "00003.png", "00020.png", "00021.png"]

panels = []

for name in names:
    image = np.array(Image.open(base / "images" / name).convert("RGB"))
    gray = np.array(Image.open(base / "masks" / name).convert("L"))
    binary = gray > 127

    mask_rgb = np.zeros_like(image)
    mask_rgb[binary] = [255, 255, 255]

    overlay = image.copy()
    overlay[binary] = (
        0.5 * image[binary] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    panels.extend([
        Image.fromarray(image),
        Image.fromarray(mask_rgb),
        Image.fromarray(overlay)
    ])

canvas = Image.new("RGB", (256 * 3, 280 * len(names)), "white")
draw = ImageDraw.Draw(canvas)

for i, panel in enumerate(panels):
    row, col = divmod(i, 3)
    canvas.paste(panel, (col * 256, row * 280 + 24))
    if col == 0:
        title = names[row]
        draw.text((col * 256 + 5, row * 280 + 5), title, fill="black")
    draw.text(
        (col * 256 + 5, row * 280 + 5),
        ["CT image", "Binary mask", "Overlay"][col],
        fill="black"
    )

out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print("Preview saved to:", out.resolve())
