"""
Identify-from-highlighted-surface eval.
Shows the occluded image with the correct reflection/shadow region highlighted
(pink box) and asks the model to name the occluded object.

Usage: python identify_highlight_eval.py --model gpt-5.4
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
ANN_DIR = os.path.join(HERE, "annotated_highlight")
REGIONS = os.path.join(HERE, "regions_highlight.json")

load_dotenv(os.path.join(ROOT, ".env"))
KEYS = {"openai": os.getenv("OpenAI_API_KEY"),
        "anthropic": os.getenv("Anthropic_API_KEY"),
        "gemini": os.getenv("Gemini_API_KEY")}

PROMPT = ("The occluded object is portrayed in the highlighted reflected surface. "
          "What is this occluded object? Be as specific as you can.")


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
    img = Image.open(io.BytesIO(data)); s = 0.9
    while True:
        buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            break
        w, h = img.size; img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask(model, client, path):
    b64 = encode(path)
    if "claude" in model:
        return client.messages.create(
            model=model, max_tokens=500, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT}]}]).content[0].text
    return client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}]).choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    items = json.load(open(REGIONS))
    model_name = args.model.split("/")[-1]
    out = os.path.join(HERE, f"identify_highlight_{model_name}_results.json")
    if os.path.exists(out):
        results = json.load(open(out)); done = {r["id"] for r in results}
    else:
        results, done = [], set()
    client = init_client(args.model)
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        resp = ask(args.model, client, os.path.join(ANN_DIR, f"{it['id']}.png"))
        print(f"[{i+1}/{len(items)}] {it['id']} (gt={it['ground_truth']!r}) -> {resp[:80]}")
        results.append({"id": it["id"], "ground_truth": it["ground_truth"],
                        "prompt": PROMPT, "response": resp})
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    print(f"Done. {len(results)} -> {out}")


if __name__ == "__main__":
    main()
