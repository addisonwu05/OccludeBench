import argparse
import base64
import io
import json
import os
import anthropic
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")

OCC_PROMPT = "What is the primary object that is completely occluded in this image? Be as specific as you can."
NUDGE_SUFFIX = " Pay close attention to any reflective surfaces or shadows in the image as you will need it to answer the question."
NUDGE_OCCLUSION = (
    "You are answering visual questions about images where a primary object may be completely occluded by a visible mask, box, or polygon.\n\n"
    "Your task is to identify the hidden object, not the occluder and not merely a shadow, reflection, surface, or visible nearby object.\n\n"
    "Before answering, carefully inspect the whole image. Pay special attention to:\n"
    "- shadows, reflections, mirrors, glass, water, shiny surfaces, and other indirect visual cues;\n"
    "- whether those cues physically correspond to the hidden object given position, lighting, orientation, scale, and scene geometry;\n"
    "- surrounding context, but only as secondary evidence when direct visual cues are insufficient.\n\n"
    "Do not rely mainly on common-sense priors if a shadow or reflection provides more specific evidence. Do not answer \"shadow\" or \"reflection\" unless the question specifically asks for the visual phenomenon itself; answer the object casting the shadow or being reflected.\n\n"
    "Give your best single answer. Be specific enough to distinguish the object when the image supports it, but do not over-specify uncertain details."
)
TRANS_PROMPT = "What is the object outlined by the yellow border in this image? Be as specific as you can."

MAX_API_ROUNDS = 5  # max LLM calls per question

# ---------------------------------------------------------------------------
# Tool definitions (sent to the API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "crop",
        "description": (
            "Crop a rectangular region of the current image for closer inspection. "
            "Coordinates are in pixels (origin = top-left)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x1": {"type": "integer", "description": "Left boundary"},
                "y1": {"type": "integer", "description": "Top boundary"},
                "x2": {"type": "integer", "description": "Right boundary"},
                "y2": {"type": "integer", "description": "Bottom boundary"},
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
    {
        "name": "edge_detect",
        "description": "Run edge detection on the image to highlight object boundaries.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "zoom",
        "description": (
            "Zoom into a rectangular region of the image, scaling it back up to the original "
            "dimensions. Coordinates are in pixels (origin = top-left)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x1": {"type": "integer", "description": "Left boundary"},
                "y1": {"type": "integer", "description": "Top boundary"},
                "x2": {"type": "integer", "description": "Right boundary"},
                "y2": {"type": "integer", "description": "Bottom boundary"},
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
    {
        "name": "flip_horizontal",
        "description": "Flip the image horizontally.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "flip_vertical",
        "description": "Flip the image vertically.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rotate",
        "description": "Rotate the image by a given angle in degrees (counter-clockwise).",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle": {"type": "number", "description": "Degrees to rotate counter-clockwise (e.g. 90, 180, 270)"},
            },
            "required": ["angle"],
        },
    },
    {
        "name": "final_answer",
        "description": "Submit your final answer. Call this when you are ready to give your answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your final answer"},
            },
            "required": ["answer"],
        },
    },
]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_image(path: str, max_bytes: int = 3_500_000) -> Image.Image:
    with open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if len(data) > max_bytes:
        scale = 0.9
        while True:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            if buf.tell() <= max_bytes:
                break
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def encode_pil_compressed(img: Image.Image, max_dim: int = 768) -> str:
    """Encode image at reduced size for tool result payloads."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def serialize_claude_content(content) -> list:
    """Convert SDK response content blocks to plain dicts safe for re-use in messages."""
    result = []
    for block in content:
        if block.type == "thinking":
            result.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})
        elif block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return result


def execute_tool(tool_name: str, tool_input: dict, current_img: Image.Image):
    """Run a tool on current_img, return (new_img, description)."""
    if tool_name == "crop":
        x1, y1, x2, y2 = tool_input["x1"], tool_input["y1"], tool_input["x2"], tool_input["y2"]
        w, h = current_img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        new_img = current_img.crop((x1, y1, x2, y2))
        desc = f"Cropped to ({x1},{y1})-({x2},{y2}), size {new_img.size}"
    elif tool_name == "zoom":
        x1, y1, x2, y2 = tool_input["x1"], tool_input["y1"], tool_input["x2"], tool_input["y2"]
        w, h = current_img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        new_img = current_img.crop((x1, y1, x2, y2)).resize((w, h), Image.LANCZOS)
        desc = f"Zoomed into ({x1},{y1})-({x2},{y2}), scaled back to {new_img.size}"
    elif tool_name == "enhance_contrast":
        factor = tool_input.get("factor", 2.0)
        new_img = ImageEnhance.Contrast(current_img).enhance(factor)
        desc = f"Contrast enhanced by factor {factor}"
    elif tool_name == "edge_detect":
        new_img = current_img.filter(ImageFilter.FIND_EDGES)
        desc = "Edge detection applied"
    elif tool_name == "flip_horizontal":
        new_img = current_img.transpose(Image.FLIP_LEFT_RIGHT)
        desc = "Flipped horizontally"
    elif tool_name == "flip_vertical":
        new_img = current_img.transpose(Image.FLIP_TOP_BOTTOM)
        desc = "Flipped vertically"
    elif tool_name == "rotate":
        angle = tool_input.get("angle", 90)
        new_img = current_img.rotate(angle, expand=True)
        desc = f"Rotated {angle} degrees counter-clockwise"
    else:
        new_img = current_img
        desc = "Unknown tool — image unchanged"
    return new_img, desc


# ---------------------------------------------------------------------------
# Claude ReAct loop
# ---------------------------------------------------------------------------

def run_claude(client: anthropic.Anthropic, model: str, image_path: str, prompt: str) -> tuple:
    img = load_image(image_path)
    w, h = img.size
    full_prompt = f"Image dimensions: {w}x{h} pixels (width x height, origin top-left). You have at most {MAX_API_ROUNDS} rounds to examine the image. Call exactly one tool per round, then submit a final_answer when ready.\n\n{prompt}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encode_pil(img)}},
            {"type": "text", "text": full_prompt},
        ],
    }]
    tool_log = []

    for _ in range(MAX_API_ROUNDS):
        resp = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        )

        # Append full assistant turn as plain dicts (avoids Pydantic re-serialization bug)
        messages.append({"role": "assistant", "content": serialize_claude_content(resp.content)})

        if resp.stop_reason == "end_turn":
            for block in reversed(resp.content):
                if block.type == "text":
                    return block.text, tool_log
            return "", tool_log

        # Process tool calls
        tool_results = []
        final = None

        all_tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        # Log any extra tool calls the model returned beyond the first (not executed)
        for block in all_tool_use_blocks[1:]:
            tool_log.append({"tool": block.name, "input": block.input, "skipped": True})

        tool_use_blocks = all_tool_use_blocks[:1]  # execute only one tool per round
        for block in tool_use_blocks:
            name, inp, tid = block.name, block.input, block.id

            if name == "final_answer":
                final = inp.get("answer", "")
                tool_log.append({"tool": name, "input": inp})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "Answer recorded.",
                })
            else:
                result_img, desc = execute_tool(name, inp, img)
                tool_log.append({"tool": name, "input": inp, "result": desc})
                print(f"    [tool:{name}] {desc}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": [
                        {"type": "text", "text": desc},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_pil_compressed(result_img)}},
                    ],
                })

        messages.append({"role": "user", "content": tool_results})

        if final is not None:
            return final, tool_log

    # Fallback: ask for a plain answer with no tools
    messages.append({"role": "user", "content": "Please give your final answer now."})
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=messages,
    )
    for block in reversed(resp.content):
        if block.type == "text":
            return block.text, tool_log
    return "", tool_log


# ---------------------------------------------------------------------------
# GPT ReAct loop
# ---------------------------------------------------------------------------

TOOLS_RESPONSES = [
    {
        "type": "function",
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }
    for t in TOOLS
]


def run_gpt(client: OpenAI, model: str, image_path: str, prompt: str) -> tuple:
    img = load_image(image_path)
    w, h = img.size
    full_prompt = f"Image dimensions: {w}x{h} pixels (width x height, origin top-left). You have at most {MAX_API_ROUNDS} rounds to examine the image. Call exactly one tool per round, then submit a final_answer when ready.\n\n{prompt}"
    tool_log = []

    # Responses API uses a flat input list; previous response output items are appended each turn
    input_items = [{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": f"data:image/png;base64,{encode_pil(img)}"},
            {"type": "input_text", "text": full_prompt},
        ],
    }]

    for _ in range(MAX_API_ROUNDS):
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "high"},
            tools=TOOLS_RESPONSES,
            input=input_items,
        )

        # Accumulate output items for next turn
        input_items += resp.output

        fn_calls = [item for item in resp.output if item.type == "function_call"]

        if not fn_calls:
            for item in resp.output:
                if item.type == "message":
                    for block in item.content:
                        if hasattr(block, "text"):
                            return block.text, tool_log
            return "", tool_log

        final = None
        # Log any extra tool calls the model returned beyond the first (not executed)
        for tc in fn_calls[1:]:
            tool_log.append({"tool": tc.name, "input": json.loads(tc.arguments), "skipped": True})

        for tc in fn_calls[:1]:  # execute only one tool per round
            name = tc.name
            inp = json.loads(tc.arguments)

            if name == "final_answer":
                final = inp.get("answer", "")
                tool_log.append({"tool": name, "input": inp})
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": "Answer recorded.",
                })
            else:
                result_img, desc = execute_tool(name, inp, img)
                tool_log.append({"tool": name, "input": inp, "result": desc})
                print(f"    [tool:{name}] {desc}")
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps({
                        "description": desc,
                        "image": f"data:image/jpeg;base64,{encode_pil_compressed(result_img)}",
                    }),
                })

        if final is not None:
            return final, tool_log

    # Fallback
    input_items.append({"role": "user", "content": [{"type": "input_text", "text": "Please give your final answer now."}]})
    resp = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=input_items,
    )
    for item in resp.output:
        if item.type == "message":
            for block in item.content:
                if hasattr(block, "text"):
                    return block.text, tool_log
    return "", tool_log


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(args, client, eval_type: str):
    """eval_type: 'occluded' or 'transparent'"""
    with open(os.path.join(args.data_dir, "annotations.json")) as f:
        entries = json.load(f)

    model_name = args.model.split("/")[-1]
    suffix = "results" if eval_type == "occluded" else "transparent_results"
    output_path = os.path.join(args.output_dir, f"{model_name}_{suffix}.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming {eval_type}: {len(done_ids)} done, {len(entries) - len(done_ids)} remaining.")
    else:
        results = []
        done_ids = set()

    is_claude = "claude" in args.model

    entries = [e for e in entries if e.get("physical", False)]
    print(f"Filtered to {len(entries)} physical entries.")

    for entry in entries:
        eid = entry["id"]
        if eid in done_ids:
            continue

        base = os.path.splitext(entry["original_file"])[0]
        img_path = os.path.join(args.data_dir, f"{base}_{'occluded' if eval_type == 'occluded' else 'transparent'}.png")

        if not os.path.exists(img_path):
            print(f"Missing image for {eid}, skipping.")
            continue

        base_prompt = OCC_PROMPT if eval_type == "occluded" else TRANS_PROMPT
        if eval_type == "occluded" and args.nudge == "reflective":
            prompt = base_prompt + NUDGE_SUFFIX
        elif eval_type == "occluded" and args.nudge == "occlusion":
            prompt = NUDGE_OCCLUSION
        else:
            prompt = base_prompt
        print(f"[{eval_type}] {eid}...")

        if is_claude:
            response, tool_log = run_claude(client, args.model, img_path, prompt)
        else:
            response, tool_log = run_gpt(client, args.model, img_path, prompt)

        print(f"  -> {response[:120]}")

        row = {"id": eid, "ground_truth": entry.get("answer", ""), "response": response, "tool_calls": tool_log}
        if eval_type == "occluded":
            row["followup_response"] = None

        results.append(row)
        done_ids.add(eid)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Done {eval_type}: {len(results)} results -> {output_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--data_dir", type=str, default="data/combined_output")
    parser.add_argument("--output_dir", type=str, default="data/eval_results_tool_use_thinking")
    parser.add_argument("--eval_type", type=str, default="occluded",
                        choices=["occluded"])
    parser.add_argument("--nudge", type=str, default=None, choices=["reflective", "occlusion"],
                        help="Nudge variant for occluded prompts: 'reflective' appends the reflective-surface hint; 'occlusion' uses a detailed occlusion-aware prompt.")
    return parser.parse_args()


def main(args):
    if "claude" in args.model:
        client = anthropic.Anthropic(api_key=Anthropic_API_KEY)
    else:
        client = OpenAI(api_key=OpenAI_API_KEY)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_type in ("occluded", "both"):
        run_eval(args, client, "occluded")
    if args.eval_type in ("transparent", "both"):
        run_eval(args, client, "transparent")


if __name__ == "__main__":
    main(parse_args())
