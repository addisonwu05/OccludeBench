"""
REVERSE mapping test: the shadow/reflection is visible; the OBJECT is occluded.
Four candidate regions are ALL blacked out (the real object occluder + 3 decoy
occluders elsewhere). The model must use the visible shadow/reflection to pick
which occluded region actually hides the object.

To avoid a shape giveaway, all four regions are drawn as uniform black rectangles
(the real occluder's bbox is filled to match the decoys), outlined pink + lettered.
Decoys never cover the shadow crop (kept visible).

Only object-occluded images are used (clue crop must be visible, not blacked out).

Outputs:
    sec4_2_why_vlms_fail/stage3_source_mapping/annotated_reverse/<id>.png
    sec4_2_why_vlms_fail/stage3_source_mapping/regions_reverse.json
"""
import hashlib
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG_DIR = os.path.join(ROOT, "data/combined_output")
CROP_META = os.path.join(ROOT, "sec4_2_why_vlms_fail", "stage2_zoomed_recognition", "crop_meta.json")
ANN_JSON = os.path.join(IMG_DIR, "annotations.json")
ANN_DIR = os.path.join(HERE, "annotated_reverse")
OUT = os.path.join(HERE, "regions_reverse.json")
os.makedirs(ANN_DIR, exist_ok=True)

# original hand-annotated occluder shapes (polygons/bboxes), indexed by id
ANN = {str(e["id"]): e for e in json.load(open(ANN_JSON))}

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
LETTERS = ["A", "B", "C", "D"]
PINK = (237, 28, 36)     # red markings for the reverse test


def intersects(a, b, margin):
    return not (a["x"] + a["w"] + margin <= b["x"] or
                b["x"] + b["w"] + margin <= a["x"] or
                a["y"] + a["h"] + margin <= b["y"] or
                b["y"] + b["h"] + margin <= a["y"])


def occluder_bbox(iid):
    """Bounding box of the ORIGINAL hand-annotated occluder shape(s) in
    annotations.json (polygons and/or bboxes), unioned."""
    e = ANN.get(iid)
    if not e:
        return None
    xs, ys = [], []
    for a in e.get("annotations", []):
        if a.get("type") == "polygon":
            for p in a["points"]:
                xs.append(p["x"]); ys.append(p["y"])
        elif a.get("type") == "bbox":
            xs += [a["x1"], a["x2"]]; ys += [a["y1"], a["y2"]]
    if not xs:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}


def place_decoys(real, shadow, W, H, seed, n=3):
    rng = random.Random(seed)
    margin = max(8, int(0.02 * min(W, H)))
    placed = []
    for _ in range(n):
        got = None
        for shrink in (1.0, 0.85, 0.7, 0.55):
            cw = max(14, int(real["w"] * shrink))
            ch = max(14, int(real["h"] * shrink))
            if cw >= W or ch >= H:
                continue
            for _ in range(500):
                w = min(W - 2, max(14, int(cw * rng.uniform(0.85, 1.15))))
                h = min(H - 2, max(14, int(ch * rng.uniform(0.85, 1.15))))
                x = rng.randint(0, W - w); y = rng.randint(0, H - h)
                cand = {"x": x, "y": y, "w": w, "h": h}
                if intersects(cand, real, margin):
                    continue
                if intersects(cand, shadow, margin):        # keep shadow visible
                    continue
                if any(intersects(cand, p, margin) for p in placed):
                    continue
                got = cand; break
            if got:
                break
        if not got:
            return None
        placed.append(got)
    return placed


def render(iid, occ_img, boxes, real_letter):
    img = occ_img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    # fill EVERY candidate box solid black so all four look identical
    for b in boxes.values():
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], fill=(0, 0, 0, 255))
    lw = max(3, int(min(W, H) / 200))
    fs = max(18, int(min(W, H) / 16))
    try:
        font = ImageFont.truetype(FONT_PATH, fs)
    except Exception:
        font = ImageFont.load_default()
    for letter, b in boxes.items():
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                       outline=PINK + (255,), width=lw)
        tw, th = fs * 0.85, fs
        tx, ty = b["x"], max(0, b["y"] - th - 2)
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 4], fill=PINK + (255,))
        draw.text((tx + 4, ty + 2), letter, fill=(255, 255, 255), font=font)
    img.convert("RGB").save(os.path.join(ANN_DIR, f"{iid}.png"))


def main():
    meta = json.load(open(CROP_META))
    from collections import Counter
    counts = Counter(str(r["id"]) for r in meta)
    seen = set()
    regions = []
    n_auto = n_excluded = n_fail = 0
    for r in meta:
        iid = str(r["id"])
        if iid in seen:
            continue
        seen.add(iid)
        base = os.path.splitext(r["original_file"])[0]
        occ = os.path.join(IMG_DIR, f"{base}_occluded.png")
        if not os.path.exists(occ):
            continue
        occ_img = Image.open(occ).convert("RGB")
        W, H = occ_img.size
        shadow = {k: int(r["crop_box"][k]) for k in ("x", "y", "w", "h")}

        # object-occluded only (shadow crop must be visible)
        cr = np.asarray(occ_img.crop((shadow["x"], shadow["y"],
                        shadow["x"] + shadow["w"], shadow["y"] + shadow["h"])).convert("L"))
        if cr.size and (cr < 25).mean() > 0.40:
            regions.append({"id": iid, "status": "excluded_cue_occluded"})
            n_excluded += 1
            continue

        real = occluder_bbox(iid)
        seed = int(hashlib.md5(iid.encode()).hexdigest(), 16) % (2**32)
        decoys = None
        if real and counts[iid] == 1:
            decoys = place_decoys(real, shadow, W, H, seed)
        if not real or decoys is None:
            regions.append({"id": iid, "status": "needs_manual",
                            "reason": "no occluder" if not real else "cant place decoys"})
            n_fail += 1
            continue

        boxes4 = [real] + decoys
        rng = random.Random(seed + 11)
        order = LETTERS[:]; rng.shuffle(order)
        boxes = {order[i]: boxes4[i] for i in range(4)}
        real_letter = order[0]
        render(iid, occ_img, boxes, real_letter)
        regions.append({"id": iid, "original_file": r["original_file"], "W": W, "H": H,
                        "ground_truth": r["ground_truth"], "status": "auto",
                        "boxes": boxes, "correct_letter": real_letter,
                        "real_occluder": real, "shadow_box": shadow})
        n_auto += 1

    with open(OUT, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"Total unique images: {len(regions)}")
    print(f"  auto (reverse) annotated: {n_auto}")
    print(f"  excluded (cue-occluded):  {n_excluded}")
    print(f"  needs_manual/failed:      {n_fail}")


if __name__ == "__main__":
    main()
