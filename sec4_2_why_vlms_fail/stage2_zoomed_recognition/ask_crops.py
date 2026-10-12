"""
Mass eval over the manually-saved crops (run AFTER cropping in the web tool).

Reads sec4_2_why_vlms_fail/stage2_zoomed_recognition/crop_meta.json, sends each saved crop image to the given
model, and writes sec4_2_why_vlms_fail/stage2_zoomed_recognition/crop_eval_<model>_results.json.

Usage:
    python ask_crops.py --model gpt-5.4
    python ask_crops.py --model claude-opus-4-6
    python ask_crops.py --model gemini-3.1-pro-preview
    # or all three:
    ./run_crop_eval.sh
"""
import argparse
import base64
import io
import json
import os

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
META = os.path.join(HERE, "crop_meta.json")

load_dotenv(os.path.join(ROOT, ".env"))
KEYS = {"openai": os.getenv("OpenAI_API_KEY"),
        "anthropic": os.getenv("Anthropic_API_KEY"),
        "gemini": os.getenv("Gemini_API_KEY")}

PROMPT = ("Depicted in this image is a shadow or reflection of another object. "
          "What is the original object that it is a shadow or reflection of? "
          "Be as specific as you can.")


def init_client(model):
    if "gemini" in model:
        return OpenAI(api_key=KEYS["gemini"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    if "claude" in model:
        return anthropic.Anthropic(api_key=KEYS["anthropic"])
    return OpenAI(api_key=KEYS["openai"])


def encode(path, max_bytes=3.5 * 1024 * 1024):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) <= max_bytes:
        return base64.b64encode(data).decode("utf-8")
    img = Image.open(io.BytesIO(data)); scale = 0.9
    while True:
        buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            break
        w, h = img.size; img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask(model, client, path, prompt):
    b64 = encode(path)
    if "claude" in model:
        return client.messages.create(
            model=model, max_tokens=500, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}]}],
        ).content[0].text
    return client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}]}],
    ).choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    meta = json.load(open(META))
    model_name = args.model.split("/")[-1]
    out_path = os.path.join(HERE, f"crop_eval_{model_name}_results.json")

    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done = {r["crop_file"] for r in results}
    else:
        results, done = [], set()

    client = init_client(args.model)
    for i, m in enumerate(meta):
        if m["crop_file"] in done:
            continue
        path = os.path.join(HERE, m["crop_file"])
        if not os.path.exists(path):
            print(f"  [skip] missing {m['crop_file']}")
            continue
        print(f"[{i+1}/{len(meta)}] {m['crop_file']} (gt={m['ground_truth']!r})")
        resp = ask(args.model, client, path, args.prompt)
        print(f"    -> {resp[:100]}")
        results.append({
            "id": m["id"], "crop_file": m["crop_file"], "label": m.get("label", ""),
            "ground_truth": m["ground_truth"], "crop_box": m.get("crop_box"),
            "prompt": args.prompt, "response": resp,
        })
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Done. {len(results)} results -> {out_path}")


if __name__ == "__main__":
    main()
