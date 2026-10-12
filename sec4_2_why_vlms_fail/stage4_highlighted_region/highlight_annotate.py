"""
Single-highlight images for the identify-from-highlighted-surface question.

For each object-occluded image (object hidden, reflection/shadow visible), draw
ONE pink box around the correct reflection/shadow region (the user's crop). The
model is then asked to identify the occluded object from that highlighted surface.

Only object-occluded images (clue crop visible, not blacked out) are used.

Outputs:
    sec4_2_why_vlms_fail/stage4_highlighted_region/annotated_highlight/<id>.png
    sec4_2_why_vlms_fail/stage4_highlighted_region/regions_highlight.json
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG_DIR = os.path.join(ROOT, "data/combined_output")
CROP_META = os.path.join(ROOT, "sec4_2_why_vlms_fail", "stage2_zoomed_recognition", "crop_meta.json")
ANN_DIR = os.path.join(HERE, "annotated_highlight")
OUT = os.path.join(HERE, "regions_highlight.json")
os.makedirs(ANN_DIR, exist_ok=True)
PINK = (255, 0, 170)


def main():
    meta = json.load(open(CROP_META))
    seen = set()
    regions = []
    n = n_excl = 0
    for r in meta:
        iid = str(r["id"])
        if iid in seen:
            continue
        seen.add(iid)
        base = os.path.splitext(r["original_file"])[0]
        occ = os.path.join(IMG_DIR, f"{base}_occluded.png")
        if not os.path.exists(occ):
            continue
        img = Image.open(occ).convert("RGB")
        W, H = img.size
        b = {k: int(r["crop_box"][k]) for k in ("x", "y", "w", "h")}
        # object-occluded only (highlighted surface must be visible)
        cr = np.asarray(img.crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])).convert("L"))
        if cr.size and (cr < 25).mean() > 0.40:
            n_excl += 1
            continue
        draw = ImageDraw.Draw(img)
        lw = max(3, int(min(W, H) / 200))
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=PINK, width=lw)
        img.save(os.path.join(ANN_DIR, f"{iid}.png"))
        regions.append({"id": iid, "original_file": r["original_file"],
                        "ground_truth": r["ground_truth"], "box": b})
        n += 1
    with open(OUT, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"highlighted (object-occluded): {n}  |  excluded (cue-occluded): {n_excl}")


if __name__ == "__main__":
    main()
