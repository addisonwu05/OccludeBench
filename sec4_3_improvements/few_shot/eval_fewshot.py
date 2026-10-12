import argparse
import base64
import io
import json
import os
from openai import OpenAI
from together import Together
import anthropic
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")

PROMPT = "What is the primary object that is completely occluded in this image? Be as specific as you can."
FOLLOWUP_PROMPT = "What is it a reflection of?"

JUDGE_RESPONSE_PROMPT = (
    "A vision model was asked to identify the primary occluded object in an image. "
    "Here is its response:\n\n\"{response}\"\n\n"
    "Does this response identify a reflective surface (such as a mirror, puddle, or similar) "
    "WITHOUT specifying what that surface is reflecting? "
    "Answer only 'yes' or 'no'."
)

JUDGE_GT_PROMPT = (
    "The ground truth answer for an occluded object identification task is:\n\n\"{ground_truth}\"\n\n"
    "Is this ground truth answer itself just a reflective surface (e.g. a mirror or puddle), "
    "with no information about what it is reflecting? "
    "Answer only 'yes' or 'no'."
)

# Few-shot example IDs — excluded from evaluation
FEWSHOT_IDS = {"120", "img1032"}

FEWSHOT_TRACES = {
    "120": (
        "The answer is not the sky, as the sky is not completely occluded, nor is it the bird, "
        "as the bird is not completely occluded. The only object that is completely occluded is "
        "the bird's wings. It is the bird's wings because, since a bird has wings and it is in "
        "the sky, the only possible object that could be flapping is its wings. Hence the correct "
        "answer is wings."
    ),
    "img1032": (
        "The answer is not drink, as there are many possible drinks. It's also not just mug, as "
        "there are many different colors of mugs. It is specifically a dark mug. We can arrive at "
        "this conclusion by observing that the teapot is casting a reflection. In the reflection "
        "we observe the white mug and the plate, which are present in the image, but we see that "
        "the dark mug next to the white mug is not present, hence occluded. Thus the answer is "
        "dark mug."
    ),
}


def init_client(model):
    if "gpt" in model or model[0] == 'o':
        return OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in model:
        return OpenAI(api_key=Gemini_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif "claude" in model:
        return anthropic.Anthropic(api_key=Anthropic_API_KEY)
    else:
        return Together(api_key=Together_API_KEY)

def judge(prompt, judge_model):
    client = OpenAI(api_key=OpenAI_API_KEY)
    verdict = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().lower()
    return verdict.startswith("yes")

def encode_image(image_path, max_bytes=3.5 * 1024 * 1024):
    with open(image_path, "rb") as f:
        data = f.read()
    if len(data) <= max_bytes:
        return base64.b64encode(data).decode("utf-8")
    img = Image.open(io.BytesIO(data))
    scale = 0.9
    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            break
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_fewshot_messages_openai(fewshot_image_paths, media_type="image/png"):
    """Build few-shot turns for OpenAI-compatible APIs."""
    messages = []
    for fs_id in ["120", "img1032"]:
        b64 = encode_image(fewshot_image_paths[fs_id])
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": FEWSHOT_TRACES[fs_id]
        })
    return messages

def build_fewshot_messages_claude(fewshot_image_paths, media_type="image/png"):
    """Build few-shot turns for the Anthropic API."""
    messages = []
    for fs_id in ["120", "img1032"]:
        b64 = encode_image(fewshot_image_paths[fs_id])
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": PROMPT}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": FEWSHOT_TRACES[fs_id]
        })
    return messages

def chat_completion(model, client, image_path, fewshot_image_paths):
    b64 = encode_image(image_path)
    media_type = "image/png"

    if "claude" in model:
        messages = build_fewshot_messages_claude(fewshot_image_paths, media_type)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": PROMPT}
            ]
        })
        return client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=messages
        ).content[0].text
    else:
        messages = build_fewshot_messages_openai(fewshot_image_paths, media_type)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        })
        return client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        ).choices[0].message.content

def followup_completion(model, client, image_path, initial_response, fewshot_image_paths):
    b64 = encode_image(image_path)
    media_type = "image/png"

    if "claude" in model:
        messages = build_fewshot_messages_claude(fewshot_image_paths, media_type)
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": PROMPT}
                ]
            },
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": FOLLOWUP_PROMPT}
        ]
        return client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=messages
        ).content[0].text
    else:
        messages = build_fewshot_messages_openai(fewshot_image_paths, media_type)
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": PROMPT}
                ]
            },
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": FOLLOWUP_PROMPT}
        ]
        return client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        ).choices[0].message.content

def run_eval(args, client):
    with open(os.path.join(args.data_dir, "annotations.json")) as f:
        entries = json.load(f)

    fewshot_image_paths = {
        "120": os.path.join(args.data_dir, "120_occluded.png"),
        "img1032": os.path.join(args.data_dir, "img1032_occluded.png"),
    }

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_name}_results.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done, {len(entries) - len(done_ids)} remaining.")
    else:
        results = []
        done_ids = set()

    for entry in entries:
        if entry["id"] in done_ids:
            continue
        if str(entry["id"]) in FEWSHOT_IDS:
            print(f"Skipping few-shot example {entry['id']}...")
            continue

        base = os.path.splitext(entry["original_file"])[0]
        occluded_path = os.path.join(args.data_dir, f"{base}_occluded.png")
        ground_truth = entry.get("answer", "")

        print(f"Running {entry['id']}...")
        response = chat_completion(args.model, client, occluded_path, fewshot_image_paths)
        print(f"  Response: {response}")

        result = {
            "id": entry["id"],
            "ground_truth": ground_truth,
            "response": response,
            "followup_response": None
        }

        gt_is_bare_reflective = judge(
            JUDGE_GT_PROMPT.format(ground_truth=ground_truth),
            args.judge_model
        )

        if not gt_is_bare_reflective:
            response_is_bare_reflective = judge(
                JUDGE_RESPONSE_PROMPT.format(response=response),
                args.judge_model
            )
            if response_is_bare_reflective:
                print(f"  [Judge] Reflective surface without specifics — sending follow-up...")
                followup = followup_completion(args.model, client, occluded_path, response, fewshot_image_paths)
                print(f"  Follow-up: {followup}")
                result["followup_response"] = followup

        results.append(result)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--data_dir", type=str, default="data/combined_output")
    parser.add_argument("--output_dir", type=str, default="data/eval_results_fewshot")
    parser.add_argument("--judge_model", type=str, default="gpt-5.4")
    return parser.parse_args()

def main(args):
    client = init_client(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    results = run_eval(args, client)
    model_name = args.model.split("/")[-1]
    print(f"Done. {len(results)} results in {os.path.join(args.output_dir, f'{model_name}_results.json')}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
