"""Robustness-check variant of the occluded images: every occluder shape is
filled BLUE with a YELLOW border, regardless of the colors stored in the
annotations. Geometry (bbox/polygon positions) is taken verbatim from
annotations.json so only the occluder color changes vs. the original
`_occluded.png` set."""
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

SRC_DIR = Path("data/combined_output")
ANNOTATIONS_PATH = SRC_DIR / "annotations.json"
OUT_DIR = Path("data/combined_output_blue")

FILL_COLOR = "#0000ff"    # blue interior
BORDER_COLOR = "#eeff00"  # yellow border
BORDER_WIDTH = 2

OUT_DIR.mkdir(exist_ok=True)
# eval_script.py reads annotations.json from the data_dir
shutil.copy(ANNOTATIONS_PATH, OUT_DIR / "annotations.json")

with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)

made = 0
for entry in data:
    original_file = SRC_DIR / entry["original_file"]
    if not original_file.exists():
        print(f"Missing original: {original_file}")
        continue

    img = Image.open(original_file).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in entry["annotations"]:
        if ann["type"] == "bbox":
            draw.rectangle(
                [ann["x1"], ann["y1"], ann["x2"], ann["y2"]],
                outline=BORDER_COLOR,
                fill=FILL_COLOR,
                width=BORDER_WIDTH,
            )
        elif ann["type"] == "polygon":
            points = [(p["x"], p["y"]) for p in ann["points"]]
            draw.polygon(points, outline=BORDER_COLOR, fill=FILL_COLOR)
            # thicken the outline to match the border width used elsewhere
            draw.line(points + [points[0]], fill=BORDER_COLOR, width=BORDER_WIDTH)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out_path = OUT_DIR / f"{original_file.stem}_occluded.png"
    result.save(out_path)
    made += 1

print(f"Done. Generated {made} blue-occluded images in {OUT_DIR}/")
