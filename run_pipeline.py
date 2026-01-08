#!/usr/bin/env python3
"""
infer_ade_affordance.py

Prompt-free instance discovery (SAM) + LLM affordance reasoning, exported in ADE-Affordance-style files:
  - *_relationship.txt  (instance_id # sit_code # run_code # grasp_code)
  - *_exco.json         (explanation+consequence for exception relations)
  - *_instances.json    (full structured output for analysis)
  - *_overlay.png       (mask overlay for sanity checking)

Why this design is "publication-friendly":
  - No affordance training required: segmentation is SOTA foundation model (SAM); reasoning is SOTA LLM.
  - Reproducible: stable instance ids, deterministic ordering, caching of LLM calls.
  - Comparable: exports the same artifact types as ADE-Affordance (relationship/exco), plus richer JSON.

Notes on relationship codes:
  The ADE-Affordance paper defines 7 relationship categories (Table 1) :contentReference[oaicite:2]{index=2}
  but the integer mapping (e.g., which category == 6) is not specified in the paper text we have.
  Your sample relationship file uses codes {0,1,5,6} :contentReference[oaicite:3]{index=3} and the exco file
  shows explanations/consequences for some of those ids :contentReference[oaicite:4]{index=4}.
  Therefore, this script makes the mapping configurable via JSON. Default mapping is a reasonable guess.

Usage:
  python3 infer_ade_affordance.py \
      --image example.jpg \
      --outdir outputs \
      --llm_backend openai \
      --llm_model gpt-4.1-mini \
      --actions sit run grasp

Requirements:
  - segment-anything (Meta SAM) + torch + torchvision
  - opencv-python, pillow, numpy

LLM backends (optional, choose one):
  - OpenAI:    pip install openai
  - Anthropic: pip install anthropic
  - Gemini:    pip install google-generativeai
  - Ollama:    no pip required; needs local ollama server

  
HOW TO RUN MINIMAL:
# 1) Install SAM + basics
pip install opencv-python pillow numpy
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2) Download a SAM checkpoint (example: sam_vit_h_4b8939.pth) and place it locally

# 3) Choose ONE LLM backend
pip install openai        # or: pip install anthropic
# or: pip install google-generativeai
# or: use ollama locally

# 4) Run
python3 infer_ade_affordance.py \
  --image example.jpg \
  --sam_ckpt sam_vit_h_4b8939.pth \
  --llm_backend openai \
  --llm_model gpt-4.1-mini \
  --outdir outputs
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from urllib.request import urlopen, Request

# SAM CHECKPOINT (MODEL) URLS
SAM_CKPT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}  # official links :contentReference[oaicite:1]{index=1}

# ----------------------------
# Relationship categories (Table 1 in paper) :contentReference[oaicite:5]{index=5}
# ----------------------------
REL_CATEGORIES = [
    "Positive",
    "Firmly Negative",
    "Object Non-functional",
    "Physical Obstacle",
    "Socially Awkward",
    "Socially Forbidden",
    "Dangerous to ourselves/others",
]

# Default numeric mapping (CONFIGURABLE).
# If you later confirm the dataset's true numeric ids, just update this JSON (or pass --relmap_json).
DEFAULT_REL_CODEMAP = {
    "Positive": 0,
    "Firmly Negative": 1,
    "Object Non-functional": 2,
    "Physical Obstacle": 3,
    "Socially Awkward": 4,
    "Socially Forbidden": 5,
    "Dangerous to ourselves/others": 6,
}

# Which categories require explanation/consequence (the "exception" types)
EXCEPTION_CATEGORIES = {
    "Object Non-functional",
    "Physical Obstacle",
    "Socially Awkward",
    "Socially Forbidden",
    "Dangerous to ourselves/others",
}


# ----------------------------
# Utilities
# ----------------------------
def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pil_to_png_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_png(img: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(img)).decode("utf-8")


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def crop_with_mask(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.12) -> Image.Image:
    """
    Crop an object region with a little context padding. Background is kept (not transparent),
    because many VLMs do better with context.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = mask_to_bbox(mask)
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px)
    y1 = min(h - 1, y1 + py)
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    return Image.fromarray(crop)


def overlay_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    """
    Overlay instance masks on an RGB image with deterministic pseudo-colors.
    Uses safe per-channel blending (no boolean flattening issues).
    """
    out = rgb.astype(np.float32).copy()
    h, w, c = out.shape
    assert c == 3, "Expected RGB image with 3 channels"

    for i, m in enumerate(masks):
        if m.dtype != np.bool_:
            m = m.astype(bool)
        if m.shape != (h, w):
            raise ValueError(f"Mask shape {m.shape} does not match image shape {(h, w)}")

        rng = np.random.default_rng(i + 123)
        color = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)

        # Blend only where mask is True
        for ch in range(3):
            out[..., ch][m] = (1.0 - alpha) * out[..., ch][m] + alpha * color[ch]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def download_with_progress(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download URL -> dst with a simple progress indicator.
    Uses only stdlib (works well on SSH).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        total = r.headers.get("Content-Length")
        total = int(total) if total is not None else None

        tmp = dst.with_suffix(dst.suffix + ".partial")
        downloaded = 0

        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = 100.0 * downloaded / total
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    print(f"\rDownloading {dst.name}: {pct:5.1f}% ({mb:,.1f}/{tot_mb:,.1f} MB)", end="")
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\rDownloading {dst.name}: {mb:,.1f} MB", end="")

        print()  # newline
        os.replace(tmp, dst)


def ensure_sam_checkpoint(
    sam_type: str,
    sam_ckpt: str | None,
    cache_dir: str | None = None,
) -> str:
    """
    Returns a local path to a SAM checkpoint, downloading it if needed.
    Priority:
      1) user-provided --sam_ckpt (if exists)
      2) cache_dir/<filename>  (download if missing)
      3) ~/.cache/segment_anything/<filename>
    """
    if sam_type not in SAM_CKPT_URLS:
        raise ValueError(f"Unknown sam_type={sam_type}. Choose from {list(SAM_CKPT_URLS.keys())}")

    if sam_ckpt:
        p = Path(sam_ckpt)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"--sam_ckpt provided but not found: {p}")

    url = SAM_CKPT_URLS[sam_type]
    fname = url.split("/")[-1]

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "segment_anything")
    dst = Path(cache_dir) / fname

    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)

    print(f"[INFO] SAM checkpoint not found. Downloading to: {dst}")
    print(f"[INFO] URL: {url}")
    download_with_progress(url, dst)
    return str(dst)


# ----------------------------
# LLM client abstraction
# ----------------------------
class VisionLLMClient:
    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIVisionClient(VisionLLMClient):
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except Exception as e:
            die(f"OpenAI backend requires `pip install openai`. Import failed: {e}")
        self._OpenAI = OpenAI
        self.model = model
        self.client = OpenAI()

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        # OpenAI Responses API style, but compatible with "openai" python package.
        # If you use a different OpenAI SDK version, you may need to adjust field names slightly.
        content = [{"type": "input_text", "text": user}]
        for b in images_b64png:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b}"})

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        # SDK returns structured output in different places depending on version.
        # This is robust across common variants:
        try:
            txt = resp.output_text
            return json.loads(txt)
        except Exception:
            # fallback: search any text
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        return json.loads(getattr(c, "text"))
        die("OpenAI response parsing failed (unexpected SDK response shape).")


class AnthropicVisionClient(VisionLLMClient):
    def __init__(self, model: str):
        try:
            import anthropic
        except Exception as e:
            die(f"Anthropic backend requires `pip install anthropic`. Import failed: {e}")
        self.model = model
        self.client = anthropic.Anthropic()

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        # Anthropic messages API with vision input blocks.
        content = [{"type": "text", "text": user}]
        for b in images_b64png:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b},
                }
            )

        msg = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": content}],
            max_tokens=1200,
        )
        # Extract the first text block
        text = ""
        for block in msg.content:
            if block.type == "text":
                text += block.text
        return json.loads(text)


class GeminiVisionClient(VisionLLMClient):
    def __init__(self, model: str):
        try:
            import google.generativeai as genai
        except Exception as e:
            die(f"Gemini backend requires `pip install google-generativeai`. Import failed: {e}")
        self.model = model
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.genai = genai
        self.client = genai.GenerativeModel(model)

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        # Gemini expects "parts"; we include system+user as text.
        parts: List[Any] = [{"text": system + "\n\n" + user}]
        for b in images_b64png:
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64decode(b)}})

        resp = self.client.generate_content(parts, generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)


class OllamaVisionClient(VisionLLMClient):
    """
    Works with vision-capable Ollama models (e.g., llava variants).
    Requires an Ollama server (default http://localhost:11434).
    """
    def __init__(self, model: str, host: str):
        import urllib.request

        self.model = model
        self.host = host.rstrip("/")
        self.urllib = urllib.request

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user,
                    "images": images_b64png,
                },
            ],
            "format": "json",
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read().decode("utf-8"))
        # ollama returns {"message": {"content": "...json..."}}
        content = out.get("message", {}).get("content", "")
        return json.loads(content)


def make_llm_client(backend: str, model: str, ollama_host: str) -> VisionLLMClient:
    backend = backend.lower()
    if backend == "openai":
        return OpenAIVisionClient(model)
    if backend == "anthropic":
        return AnthropicVisionClient(model)
    if backend == "gemini":
        return GeminiVisionClient(model)
    if backend == "ollama":
        return OllamaVisionClient(model=model, host=ollama_host)
    die(f"Unknown --llm_backend: {backend}. Choose from: openai|anthropic|gemini|ollama.")


# ----------------------------
# SAM loader
# ----------------------------
def load_sam(ckpt: str, model_type: str, device: str):
    """
    Uses Meta's segment-anything package.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        die(
            "SAM not installed. Install with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Import error: {e}"
        )

    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=device)
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=256,
    )
    return mask_gen


# ----------------------------
# Core schema
# ----------------------------
@dataclasses.dataclass
class InstancePrediction:
    instance_id: str
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    object_name: str
    actions: Dict[str, Dict[str, Any]]  # action -> {relationship_label, relationship_code, explanation?, consequence?}


# ----------------------------
# LLM prompt
# ----------------------------
def build_system_prompt(actions: List[str]) -> str:
    # Keep it strict: structured + conservative (avoid hallucinating social/legal claims too strongly).
    return f"""You are a careful vision-language reasoning system for action-object affordances.

Given:
- a full scene image
- an object crop (same image, focused on one segmented instance)

Your job:
For each action in {actions}, decide the relationship category from this fixed set:
{REL_CATEGORIES}

Definitions (from the dataset paper Table 1):
- Positive: action can be taken with the object in this scene
- Firmly Negative: action can never be taken with this object class (in general)
- Exception types (need explanation+consequence if you choose them):
  Object Non-functional, Physical Obstacle, Socially Awkward, Socially Forbidden, Dangerous to ourselves/others

You must return STRICT JSON only, matching the schema below.
Be conservative: if uncertain, prefer "Firmly Negative" or "Physical Obstacle" over wild social claims.
Explanations/consequences must be short, factual, third-person, and grounded in the visible scene.

JSON schema:
{{
  "object_name": "short noun phrase",
  "per_action": {{
    "<action>": {{
      "relationship_label": "one of the fixed labels exactly",
      "explanation": "string, required only if relationship_label is an exception type",
      "consequence": "string, required only if relationship_label is an exception type"
    }}
  }}
}}
"""


def build_user_prompt(action_list: List[str]) -> str:
    acts = ", ".join(action_list)
    return f"""Analyze the object instance in the crop (use the full image for context).

Return JSON for:
- object_name
- per_action for actions: {acts}

Remember: explanation+consequence only if relationship_label is one of the exception types.
"""


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image.")
    ap.add_argument("--outdir", default="outputs", help="Output directory.")
    ap.add_argument("--actions", nargs="+", default=["sit", "run", "grasp"], help="Actions to evaluate.")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")

    # SAM config
    ap.add_argument("--sam_ckpt", default="sam_vit_h_4b8939.pth", help="SAM checkpoint path.")
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type.")
    ap.add_argument("--max_instances", type=int, default=30, help="Max instances to process (largest areas first).")
    ap.add_argument("--sam_cache_dir", default="", help="Directory to cache SAM checkpoints (optional).")
    ap.add_argument("--sam_ckpt", default="", help="Path to SAM checkpoint. If empty, auto-download.")

    # LLM config
    ap.add_argument("--llm_backend", default="openai", help="openai|anthropic|gemini|ollama")
    ap.add_argument("--llm_model", default="gpt-4.1-mini", help="Model name for the selected backend.")
    ap.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama server base URL.")

    # Relationship code mapping
    ap.add_argument("--relmap_json", default="", help="Optional JSON file mapping label->int code.")

    # Caching
    ap.add_argument("--cache", default="outputs/cache_llm.json", help="Cache file for LLM calls.")

    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        die(f"Image not found: {image_path}")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load relationship mapping
    relmap = DEFAULT_REL_CODEMAP.copy()
    if args.relmap_json:
        p = Path(args.relmap_json)
        if not p.exists():
            die(f"--relmap_json not found: {p}")
        relmap = json.loads(p.read_text(encoding="utf-8"))
    # Validate relmap
    for k in REL_CATEGORIES:
        if k not in relmap:
            die(f"relmap_json missing label: {k}")
        if not isinstance(relmap[k], int):
            die(f"relmap_json value for {k} must be int")

    # Load cache
    cache_path = Path(args.cache)
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    # Load image
    pil = Image.open(image_path).convert("RGB")
    rgb = np.array(pil)

    # Run SAM
    ckpt_path = ensure_sam_checkpoint(
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt if args.sam_ckpt else None,
        cache_dir=args.sam_cache_dir if args.sam_cache_dir else None,
    )
    mask_gen = load_sam(ckpt_path, args.sam_type, args.device)
    sam_masks = mask_gen.generate(rgb)  # list of dicts with "segmentation" boolean mask
    if not sam_masks:
        die("SAM produced no masks. Try different thresholds or a different image.")

    # Sort instances by descending area for stable ids and to prioritize salient objects
    sam_masks = sorted(sam_masks, key=lambda d: int(d.get("area", 0)), reverse=True)
    sam_masks = sam_masks[: args.max_instances]

    masks_bool: List[np.ndarray] = [m["segmentation"].astype(bool) for m in sam_masks]
    overlay = overlay_masks(rgb, masks_bool)
    overlay_path = outdir / f"{image_path.stem}_overlay.png"
    overlay.save(overlay_path)

    # Init LLM
    client = make_llm_client(args.llm_backend, args.llm_model, args.ollama_host)
    sys_prompt = build_system_prompt(args.actions)
    usr_prompt = build_user_prompt(args.actions)

    full_img_b64 = b64_png(pil)

    predictions: List[InstancePrediction] = []

    for idx, m in enumerate(masks_bool):
        inst_id = f"{idx:03d}"  # ADE-style 3-digit ids in your sample :contentReference[oaicite:6]{index=6}
        bbox = mask_to_bbox(m)
        area = int(m.sum())

        crop = crop_with_mask(rgb, m, pad=0.12)
        crop_b64 = b64_png(crop)

        # Cache key: image hash + mask hash + action list + model
        key_material = (
            f"{args.llm_backend}|{args.llm_model}|{','.join(args.actions)}|"
            f"{sha256_bytes(pil_to_png_bytes(pil))}|{sha256_bytes(pil_to_png_bytes(crop))}"
        ).encode("utf-8")
        cache_key = sha256_bytes(key_material)

        if cache_key in cache:
            llm_out = cache[cache_key]
        else:
            llm_out = client.complete_json(
                system=sys_prompt,
                user=usr_prompt,
                images_b64png=[full_img_b64, crop_b64],
            )
            cache[cache_key] = llm_out
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

        # Validate / normalize LLM output
        obj_name = str(llm_out.get("object_name", "object")).strip()[:80]
        per_action = llm_out.get("per_action", {}) or {}
        actions_out: Dict[str, Dict[str, Any]] = {}

        for a in args.actions:
            a_out = per_action.get(a, {}) or {}
            label = str(a_out.get("relationship_label", "Firmly Negative")).strip()

            if label not in REL_CATEGORIES:
                # Defensive fallback: keep pipeline running
                label = "Firmly Negative"

            entry: Dict[str, Any] = {
                "relationship_label": label,
                "relationship_code": int(relmap[label]),
            }

            if label in EXCEPTION_CATEGORIES:
                exp = str(a_out.get("explanation", "")).strip()
                con = str(a_out.get("consequence", "")).strip()
                # Enforce non-empty strings for exceptions
                if not exp:
                    exp = "The action is not appropriate for this object in the current scene."
                if not con:
                    con = "Taking the action may cause a negative outcome."
                entry["explanation"] = exp
                entry["consequence"] = con

            actions_out[a] = entry

        predictions.append(
            InstancePrediction(
                instance_id=inst_id,
                bbox_xyxy=bbox,
                area=area,
                object_name=obj_name,
                actions=actions_out,
            )
        )

    # ----------------------------
    # Write outputs
    # ----------------------------
    # 1) Full JSON (analysis-friendly)
    instances_json = {
        "image": str(image_path),
        "actions": args.actions,
        "relationship_categories": REL_CATEGORIES,
        "relationship_code_map": relmap,
        "instances": [
            {
                "instance_id": p.instance_id,
                "object_name": p.object_name,
                "bbox_xyxy": p.bbox_xyxy,
                "area": p.area,
                "actions": p.actions,
            }
            for p in predictions
        ],
    }
    (outdir / f"{image_path.stem}_instances.json").write_text(
        json.dumps(instances_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2) relationship.txt (ADE-Affordance-style: id # sit # run # grasp)
    # We preserve the "id # code # code # code" shape seen in your example :contentReference[oaicite:7]{index=7}.
    # If you use different actions, columns follow your --actions order.
    rel_lines = []
    for p in predictions:
        codes = [str(p.actions[a]["relationship_code"]) for a in args.actions]
        rel_lines.append(f"{p.instance_id} # " + " # ".join(codes))
    (outdir / f"{image_path.stem}_relationship.txt").write_text("\n".join(rel_lines) + "\n", encoding="utf-8")

    # 3) exco.json (store explanations+consequences for exception relations)
    # Your sample is {"grasp": {"209": {...}, ...}} :contentReference[oaicite:8]{index=8}.
    exco: Dict[str, Dict[str, Dict[str, str]]] = {}
    for a in args.actions:
        exco[a] = {}
    for p in predictions:
        for a in args.actions:
            info = p.actions[a]
            if info["relationship_label"] in EXCEPTION_CATEGORIES:
                exco[a][p.instance_id] = {
                    "explanation": info["explanation"],
                    "consequence": info["consequence"],
                }
    # Drop empty action keys to match the sample style (it had only "grasp") :contentReference[oaicite:9]{index=9}
    exco = {a: d for a, d in exco.items() if len(d) > 0}

    (outdir / f"{image_path.stem}_exco.json").write_text(
        json.dumps(exco, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] Wrote:")
    print(f"  - {overlay_path}")
    print(f"  - {outdir / (image_path.stem + '_instances.json')}")
    print(f"  - {outdir / (image_path.stem + '_relationship.txt')}")
    print(f"  - {outdir / (image_path.stem + '_exco.json')}")
    print()
    print("Tip: If you later confirm the dataset's true integer IDs, pass --relmap_json to match exactly.")


if __name__ == "__main__":
    main()
