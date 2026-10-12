"""
Mapping / localization eval on the auto-annotated physical images.

Each image shows an object that is physically occluded; its shadow/reflection is
visible in exactly one of four labeled regions (A-D). The model must pick the
region that is the shadow/reflection of the hidden object. Scored vs the answer
key (correct_letter) in regions.json. Chance = 25%.

Usage:
    python map_eval.py --model gpt-5.4
"""
import argparse
import base64
import io
import json
import os
import re

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ANN_DIR = os.path.join(HERE, "annotated")
REGIONS = os.path.join(HERE, "regions.json")

load_dotenv(os.path.join(ROOT, ".env"))
KEYS = {"openai": os.getenv("OpenAI_API_KEY"),
        "anthropic": os.getenv("Anthropic_API_KEY"),
        "gemini": os.getenv("Gemini_API_KEY")}

PROMPT = (
    "This image contains an object that is completely occluded (hidden from view). "
    "Its shadow or reflection is visible in exactly ONE of the four labeled regions "
    "A, B, C, or D. Which labeled region contains the shadow or reflection of the "
    "hidden object? Answer with only the single letter: A, B, C, or D."
)


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
            model=model, max_tokens=50, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT}]}]).content[0].text
    return client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}]).choices[0].message.content


def parse_letter(text):
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text.upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    items = [r for r in json.load(open(REGIONS)) if r["status"] == "auto"]
    model_name = args.model.split("/")[-1]
    out = os.path.join(HERE, f"map_eval_{model_name}_results.json")
    if os.path.exists(out):
        results = json.load(open(out)); done = {r["id"] for r in results}
    else:
        results, done = [], set()

    client = init_client(args.model)
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        path = os.path.join(ANN_DIR, f"{it['id']}.png")
        resp = ask(args.model, client, path)
        letter = parse_letter(resp)
        correct = (letter == it["correct_letter"])
        print(f"[{i+1}/{len(items)}] {it['id']}: picked {letter} (key {it['correct_letter']}) "
              f"{'✓' if correct else '✗'}")
        results.append({
            "id": it["id"], "ground_truth": it["ground_truth"],
            "correct_letter": it["correct_letter"], "response": resp,
            "parsed_letter": letter, "correct": correct,
        })
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    n = len(results); c = sum(r["correct"] for r in results)
    print(f"\n{model_name}: {c}/{n} = {c/n:.1%}  (chance 25%)")


if __name__ == "__main__":
    main()
