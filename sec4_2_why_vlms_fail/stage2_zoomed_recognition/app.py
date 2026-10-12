"""
Cropper platform (REBUTTAL).

Browse the ORIGINAL (un-occluded) direct/physical images and manually crop the
reflection- or shadow-bearing region. Each crop is saved to disk. No LLM here —
run sec4_2_why_vlms_fail/stage2_zoomed_recognition/ask_crops.py separately over the saved crops when ready.

Run:
    cd sec4_2_why_vlms_fail/stage2_zoomed_recognition
    python app.py           # then open http://127.0.0.1:5005
"""
import base64
import io
import json
import os
import re

from flask import Flask, jsonify, request, send_file, render_template, abort
from PIL import Image

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # project root
IMG_DIR = os.path.join(ROOT, "data/combined_output")                 # originals live here
CAT_FILE = os.path.join(ROOT, "data/Image categories (3).xlsx")
ANN_FILE = os.path.join(IMG_DIR, "annotations.json")
META = os.path.join(HERE, "crop_meta.json")                     # crop index
CROP_DIR = os.path.join(HERE, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

app = Flask(__name__)


# ---- data: the direct/physical originals -----------------------------------
def load_items():
    import pandas as pd
    cat = pd.read_excel(CAT_FILE)[["image id", "direct or inferential"]]
    cat["image id"] = cat["image id"].astype(str)
    direct = set(cat[cat["direct or inferential"] == "direct"]["image id"])
    ann = json.load(open(ANN_FILE))
    items = []
    for e in ann:
        iid = str(e["id"])
        if iid not in direct:
            continue
        of = e["original_file"]
        if not os.path.exists(os.path.join(IMG_DIR, of)):
            continue
        items.append({
            "id": iid,
            "original_file": of,
            "ground_truth": (e.get("answer") or "").strip(),
        })
    items.sort(key=lambda x: (len(x["id"]), x["id"]))
    return items


ITEMS = load_items()
ITEM_BY_ID = {it["id"]: it for it in ITEMS}


# ---- crop metadata persistence ---------------------------------------------
def load_meta():
    if os.path.exists(META):
        return json.load(open(META))
    return []


def save_meta(rows):
    with open(META, "w") as f:
        json.dump(rows, f, indent=2)


# ---- routes ----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", total=len(ITEMS))


@app.route("/api/items")
def api_items():
    meta = load_meta()
    counts = {}
    for r in meta:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    return jsonify([{**it, "crops": counts.get(it["id"], 0)} for it in ITEMS])


@app.route("/img/<iid>")
def img(iid):
    it = ITEM_BY_ID.get(iid)
    if not it:
        abort(404)
    return send_file(os.path.join(IMG_DIR, it["original_file"]))


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json()
    iid = str(data["id"])
    it = ITEM_BY_ID.get(iid)
    if not it:
        return jsonify({"error": "unknown id"}), 400

    m = re.match(r"data:image/\w+;base64,(.*)", data["crop"], re.DOTALL)
    if not m:
        return jsonify({"error": "bad crop payload"}), 400
    png = base64.b64decode(m.group(1))
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(buf, format="PNG")
    png = buf.getvalue()

    meta = load_meta()
    k = sum(1 for r in meta if r["id"] == iid) + 1     # per-image crop index
    crop_name = f"{iid}-{k}.png"
    with open(os.path.join(CROP_DIR, crop_name), "wb") as f:
        f.write(png)

    meta.append({
        "id": iid,
        "ground_truth": it["ground_truth"],
        "label": (data.get("label") or "").strip(),   # optional: "shadow"/"reflection"
        "crop_box": data.get("box"),                   # {x,y,w,h} native px
        "crop_file": os.path.join("crops", crop_name),
        "original_file": it["original_file"],
    })
    save_meta(meta)
    return jsonify({"crop_file": crop_name, "crops_for_id": k, "total_saved": len(meta)})


@app.route("/api/delete_last", methods=["POST"])
def api_delete_last():
    """Undo the most recent crop for a given image id."""
    iid = str(request.get_json()["id"])
    meta = load_meta()
    for i in range(len(meta) - 1, -1, -1):
        if meta[i]["id"] == iid:
            path = os.path.join(HERE, meta[i]["crop_file"])
            if os.path.exists(path):
                os.remove(path)
            meta.pop(i)
            save_meta(meta)
            return jsonify({"ok": True, "crops_for_id": sum(1 for r in meta if r["id"] == iid)})
    return jsonify({"ok": False, "crops_for_id": 0})


if __name__ == "__main__":
    print(f"Loaded {len(ITEMS)} direct/physical originals from {IMG_DIR}")
    print("Open http://127.0.0.1:5005")
    app.run(port=5005, debug=False)
