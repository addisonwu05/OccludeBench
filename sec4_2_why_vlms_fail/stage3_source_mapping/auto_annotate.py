"""
Auto-generate the mapping/localization test for the physical (direct) images.

Each image already has ONE correct region = the manually-cropped reflection/shadow
box (from the cropper's crop_meta.json). We add 3 non-overlapping distractor boxes
of similar size elsewhere in the image, label all four A-D in randomized order,
render an annotated PNG, and record the answer key.

Images where 3 distractors cannot be placed (box too large, or multi-crop) are
flagged status="needs_manual" for the manual platform.

Outputs:
    sec4_2_why_vlms_fail/stage3_source_mapping/annotated/<id>.png     annotated image (boxes + letters)
    sec4_2_why_vlms_fail/stage3_source_mapping/regions.json           boxes, correct letter, status per image
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
ANN_DIR = os.path.join(HERE, "annotated")
OUT = os.path.join(HERE, "regions.json")
os.makedirs(ANN_DIR, exist_ok=True)

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
LETTERS = ["A", "B", "C", "D"]
PINK = (255, 0, 170)                                   # all candidate boxes pink


def intersects(a, b, margin):
    return not (a["x"] + a["w"] + margin <= b["x"] or
                b["x"] + b["w"] + margin <= a["x"] or
                a["y"] + a["h"] + margin <= b["y"] or
                b["y"] + b["h"] + margin <= a["y"])


def dark_fraction(dark_small, sW, sH, W, H, b):
    """Fraction of near-black (occluder) pixels inside box b, via downscaled map."""
    x1 = int(b["x"] / W * sW); x2 = int((b["x"] + b["w"]) / W * sW)
    y1 = int(b["y"] / H * sH); y2 = int((b["y"] + b["h"]) / H * sH)
    x2 = max(x2, x1 + 1); y2 = max(y2, y1 + 1)
    region = dark_small[y1:y2, x1:x2]
    return region.mean() if region.size else 0.0


def place_distractors(correct, W, H, seed, dark_small, n=3):
    sH, sW = dark_small.shape
    rng = random.Random(seed)
    margin = max(8, int(0.02 * min(W, H)))
    placed = []
    for _ in range(n):
        got = None
        # shrink attempts: try same size, then progressively smaller
        for shrink in (1.0, 0.85, 0.7, 0.55):
            cw = max(12, int(correct["w"] * shrink))
            ch = max(12, int(correct["h"] * shrink))
            if cw >= W or ch >= H:
                continue
            for _ in range(400):
                jx = rng.uniform(0.85, 1.15)          # mild size jitter
                jy = rng.uniform(0.85, 1.15)
                w = min(W - 2, max(12, int(cw * jx)))
                h = min(H - 2, max(12, int(ch * jy)))
                x = rng.randint(0, W - w)
                y = rng.randint(0, H - h)
                cand = {"x": x, "y": y, "w": w, "h": h}
                if intersects(cand, correct, margin):
                    continue
                if any(intersects(cand, p, margin) for p in placed):
                    continue
                # avoid the black occluder: decoy must be mostly real content
                if dark_fraction(dark_small, sW, sH, W, H, cand) > 0.30:
                    continue
                got = cand
                break
            if got:
                break
        if not got:
            return None            # couldn't fit -> needs_manual
        placed.append(got)
    return placed


def render(item, W, H, occ_path):
    """Draw all candidate regions as PINK outlines (interior kept transparent so
    the shadow/reflection content stays fully visible) on the hand-occluded image."""
    img = Image.open(occ_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lw = max(3, int(min(W, H) / 200))
    fs = max(18, int(min(W, H) / 16))
    try:
        font = ImageFont.truetype(FONT_PATH, fs)
    except Exception:
        font = ImageFont.load_default()
    for letter, b in item["boxes"].items():
        x1, y1, x2, y2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        # pink border only; NO interior fill -> region stays transparent/visible
        draw.rectangle([x1, y1, x2, y2], outline=PINK + (255,), width=lw)
        # letter tag in a solid pink chip at the top-left corner
        tw, th = fs * 0.85, fs
        tx, ty = x1, max(0, y1 - th - 2)
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 4], fill=PINK + (255,))
        draw.text((tx + 4, ty + 2), letter, fill=(255, 255, 255), font=font)
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(os.path.join(ANN_DIR, f"{item['id']}.png"))


def main():
    meta = json.load(open(CROP_META))
    # one correct box per image (first crop); note multi-crop for manual review
    from collections import Counter
    counts = Counter(r["id"] for r in meta)
    seen = set()
    regions = []
    n_auto = n_manual = n_excluded = 0
    for r in meta:
        iid = str(r["id"])
        if iid in seen:
            continue
        seen.add(iid)
        base = os.path.splitext(r["original_file"])[0]
        occ = os.path.join(IMG_DIR, f"{base}_occluded.png")   # HAND-OCCLUDED version
        if not os.path.exists(occ):
            continue
        occ_img = Image.open(occ).convert("RGB")
        W, H = occ_img.size
        # downscaled near-black map to keep decoys off the black occluder
        small = occ_img.resize((min(W, 300), max(1, int(min(W, 300) * H / W)))).convert("L")
        dark_small = (np.asarray(small) < 25).astype(np.float32)
        correct = {k: int(r["crop_box"][k]) for k in ("x", "y", "w", "h")}
        seed = int(hashlib.md5(iid.encode()).hexdigest(), 16) % (2**32)

        # EXCLUDE cue-occluded images: if the clue crop is itself mostly blacked
        # out, the shadow/reflection is the occluded target -> answer is trivially
        # "the black box". Keep only object-occluded (cue-visible) images.
        crop_region = np.asarray(occ_img.crop(
            (correct["x"], correct["y"], correct["x"] + correct["w"],
             correct["y"] + correct["h"])).convert("L"))
        crop_dark = float((crop_region < 25).mean()) if crop_region.size else 0.0
        if crop_dark > 0.40:
            regions.append({"id": iid, "original_file": r["original_file"], "W": W, "H": H,
                            "ground_truth": r["ground_truth"], "status": "excluded_cue_occluded",
                            "reason": f"clue crop {crop_dark:.0%} black", "boxes": {},
                            "correct_letter": None, "correct_box": correct})
            n_excluded += 1
            continue

        needs_manual = counts[iid] > 1
        distractors = None if needs_manual else place_distractors(correct, W, H, seed, dark_small)
        if distractors is None:
            status = "needs_manual"
            reason = "multi-crop" if counts[iid] > 1 else "cannot fit 3 distractors"
            # still record the correct box so the manual tool can start from it
            item = {"id": iid, "original_file": r["original_file"], "W": W, "H": H,
                    "ground_truth": r["ground_truth"], "status": status, "reason": reason,
                    "boxes": {}, "correct_letter": None,
                    "correct_box": correct}
            n_manual += 1
        else:
            boxes4 = [correct] + distractors
            rng = random.Random(seed + 7)
            order = LETTERS[:]
            rng.shuffle(order)
            boxes = {order[i]: boxes4[i] for i in range(4)}
            correct_letter = order[0]                 # correct is boxes4[0]
            item = {"id": iid, "original_file": r["original_file"], "W": W, "H": H,
                    "ground_truth": r["ground_truth"], "status": "auto", "reason": "",
                    "boxes": boxes, "correct_letter": correct_letter,
                    "correct_box": correct}
            render(item, W, H, occ)
            n_auto += 1
        regions.append(item)

    with open(OUT, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"Total images: {len(regions)}")
    print(f"  auto-annotated:       {n_auto}  (rendered to annotated/)")
    print(f"  excluded (cue-occl):  {n_excluded}")
    print(f"  needs_manual:         {n_manual}")
    print("  needs_manual ids:", [r["id"] for r in regions if r["status"] == "needs_manual"])
    print("  excluded ids:", [r["id"] for r in regions if r["status"] == "excluded_cue_occluded"])


if __name__ == "__main__":
    main()
