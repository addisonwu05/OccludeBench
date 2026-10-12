import json
from pathlib import Path
from PIL import Image, ImageDraw

ANNOTATIONS_PATH = Path("data/combined_output/annotations.json")
OUTPUT_DIR = Path("data/combined_output")
BORDER_COLOR = "#f5ec00"
BORDER_WIDTH = 2

with open(ANNOTATIONS_PATH) as f:
    data = json.load(f)

for entry in data:
    original_file = OUTPUT_DIR / entry["original_file"]
    if not original_file.exists():
        print(f"Missing: {original_file}")
        continue

    img = Image.open(original_file).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in entry["annotations"]:
        if ann["type"] == "bbox":
            draw.rectangle(
                [ann["x1"], ann["y1"], ann["x2"], ann["y2"]],
                outline=BORDER_COLOR,
                fill=None,
                width=BORDER_WIDTH,
            )
        elif ann["type"] == "polygon":
            points = [(p["x"], p["y"]) for p in ann["points"]]
            draw.polygon(points, outline=BORDER_COLOR, fill=None)
            # thicken the outline
            draw.line(points + [points[0]], fill=BORDER_COLOR, width=BORDER_WIDTH)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out_path = OUTPUT_DIR / f"{original_file.stem}_transparent.png"
    result.save(out_path)

print(f"Done. Generated {len(data)} transparent images.")
