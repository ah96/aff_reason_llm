#!/usr/bin/env python3
"""
run_pipeline_saliency.py

Pipeline (same spirit as run_pipeline.py, but saliency-first):
  1) SAM: prompt-free instance proposals (masks)
  2) Affordance saliency maps: for each queried affordance label a, run a saliency/segmentation model f(I, a)->S_a
     - supports TWO checkpoints/models: SEEN_AFF and UNSEEN_AFF
     - supports label routing: seen|unseen|auto|both
  3) Score each SAM instance for each action: score(o_i,a)=mean(S_a over mask o_i)
  4) Select top-K (instance, action) pairs -> call LLM (multi-backend) with evidence:
       full image, instance crop, saliency overlay (for that action)
     LLM outputs:
       relationship_label (+ explanation/consequence for exception categories)
  5) Export:
       *_overlay.png
       *_instances.json (full structured output)
       *_relationship.txt (ADE-Affordance-like: id # code # code # ...)
       *_exco.json (explanations+consequences for exception relations)

IMPORTANT: Affordance saliency model API is provided via an ADAPTER module you write:
  - load_model(ckpt_path: str, device: str) -> Any
  - predict_saliency(model: Any, pil_rgb: PIL.Image, label: str) -> np.ndarray (H,W float in [0,1] OR logits)
See "adapter_affordance_example.py" stub at bottom of this file.

python3 run_pipeline_saliency.py --adapter adapter_affordance_example.py
"""

import argparse
import base64
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ----------------------------
# Label sets (yours)
# ----------------------------
SEEN_AFF = [
    "beat", "boxing", "brush_with", "carry", "catch",
    "cut", "cut_with", "drag", "drink_with", "eat",
    "hit", "hold", "jump", "kick", "lie_on", "lift",
    "look_out", "open", "pack", "peel", "pick_up",
    "pour", "push", "ride", "sip", "sit_on", "stick",
    "stir", "swing", "take_photo", "talk_on", "text_on",
    "throw", "type_on", "wash", "write"
]
UNSEEN_AFF = [
    "carry", "catch", "cut", "cut_with", "drink_with",
    "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
    "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
    "swing", "take_photo", "throw", "type_on", "wash"
]

# ----------------------------
# Relationship categories (ADE-Affordance Table 1 style)
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

EXCEPTION_CATEGORIES = {
    "Object Non-functional",
    "Physical Obstacle",
    "Socially Awkward",
    "Socially Forbidden",
    "Dangerous to ourselves/others",
}

DEFAULT_REL_CODEMAP = {
    "Positive": 0,
    "Firmly Negative": 1,
    "Object Non-functional": 2,
    "Physical Obstacle": 3,
    "Socially Awkward": 4,
    "Socially Forbidden": 5,
    "Dangerous to ourselves/others": 6,
}

# ----------------------------
# SAM checkpoint auto-download
# ----------------------------
SAM_CKPT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_png(img: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(img)).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_with_progress(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    from urllib.request import Request, urlopen

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
        print()
        os.replace(tmp, dst)


def ensure_sam_checkpoint(sam_type: str, sam_ckpt: str, cache_dir: str) -> str:
    if sam_ckpt:
        p = Path(sam_ckpt)
        if p.exists():
            return str(p)
        die(f"--sam_ckpt provided but not found: {p}")

    if sam_type not in SAM_CKPT_URLS:
        die(f"Unknown --sam_type {sam_type}. Choose from {list(SAM_CKPT_URLS.keys())}")

    url = SAM_CKPT_URLS[sam_type]
    fname = url.split("/")[-1]
    dst = Path(cache_dir) / fname
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)

    print(f"[INFO] SAM checkpoint missing. Downloading to: {dst}")
    download_with_progress(url, dst)
    return str(dst)


# ----------------------------
# Image / mask helpers
# ----------------------------
def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def crop_with_mask(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.12) -> Image.Image:
    h, w = mask.shape
    x0, y0, x1, y1 = mask_to_bbox(mask)
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px)
    y1 = min(h - 1, y1 + py)
    return Image.fromarray(rgb[y0:y1 + 1, x0:x1 + 1])


def overlay_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    h, w, c = out.shape
    assert c == 3
    for i, m in enumerate(masks):
        if m.dtype != np.bool_:
            m = m.astype(bool)
        if m.shape != (h, w):
            raise ValueError(f"Mask shape {m.shape} != image shape {(h, w)}")
        rng = np.random.default_rng(i + 123)
        color = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)
        for ch in range(3):
            out[..., ch][m] = (1.0 - alpha) * out[..., ch][m] + alpha * color[ch]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def saliency_overlay_image(rgb: np.ndarray, sal: np.ndarray, mask: Optional[np.ndarray] = None) -> Image.Image:
    """
    Create an RGB overlay image showing saliency as a heatmap blended over the original.
    - sal: HxW float
    - mask (optional): restrict overlay to mask area
    """
    # Normalize sal to [0,1]
    sal = np.asarray(sal, dtype=np.float32)
    if sal.ndim != 2:
        raise ValueError(f"saliency must be HxW, got {sal.shape}")
    smin, smax = float(sal.min()), float(sal.max())
    if smax > smin:
        saln = (sal - smin) / (smax - smin)
    else:
        saln = np.zeros_like(sal)

    # Simple heat mapping (no external deps): red intensity
    heat = np.zeros_like(rgb, dtype=np.float32)
    heat[..., 0] = saln * 255.0  # R
    heat[..., 1] = (saln ** 0.5) * 80.0
    heat[..., 2] = (1.0 - saln) * 40.0

    out = rgb.astype(np.float32).copy()
    alpha = 0.45
    if mask is not None:
        m = mask.astype(bool)
        for ch in range(3):
            out[..., ch][m] = (1.0 - alpha) * out[..., ch][m] + alpha * heat[..., ch][m]
    else:
        out = (1.0 - alpha) * out + alpha * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ----------------------------
# SAM loader
# ----------------------------
def load_sam_mask_generator(ckpt: str, sam_type: str, device: str):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        die(
            "SAM not installed. Install:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Import error: {e}"
        )

    sam = sam_model_registry[sam_type](checkpoint=ckpt)
    sam.to(device=device)

    # Same default-ish settings as before
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
# LLM clients (multi-backend)
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
        self.model = model
        self.client = OpenAI()

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        content = [{"type": "input_text", "text": user}]
        for b in images_b64png:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b}"})

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )

        text = getattr(resp, "output_text", None)
        if not text:
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if isinstance(c, dict):
                        if c.get("type") in ("output_text", "text"):
                            if c.get("text"):
                                parts.append(c["text"])
                    else:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(t)
            text = "\n".join(parts).strip()

        if not text:
            die("OpenAI response parsing failed: no output text.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                die("Model did not return JSON. Strengthen prompt or inspect output.")
            return json.loads(m.group(0))


class AnthropicVisionClient(VisionLLMClient):
    def __init__(self, model: str):
        try:
            import anthropic
        except Exception as e:
            die(f"Anthropic backend requires `pip install anthropic`. Import failed: {e}")
        self.model = model
        self.client = anthropic.Anthropic()

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        content = [{"type": "text", "text": user}]
        for b in images_b64png:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}})

        msg = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": content}],
            max_tokens=1200,
        )
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
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.client = genai.GenerativeModel(model)

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        import base64 as b64
        parts: List[Any] = [{"text": system + "\n\n" + user}]
        for b in images_b64png:
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64.b64decode(b)}})
        resp = self.client.generate_content(parts, generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)


class OllamaVisionClient(VisionLLMClient):
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
                {"role": "user", "content": user, "images": images_b64png},
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
        content = out.get("message", {}).get("content", "")
        return json.loads(content)


def make_llm_client(backend: str, model: str, ollama_host: str) -> VisionLLMClient:
    b = backend.lower()
    if b == "openai":
        return OpenAIVisionClient(model)
    if b == "anthropic":
        return AnthropicVisionClient(model)
    if b == "gemini":
        return GeminiVisionClient(model)
    if b == "ollama":
        return OllamaVisionClient(model, ollama_host)
    die("Unknown --llm_backend. Use openai|anthropic|gemini|ollama")


# ----------------------------
# Affordance saliency model adapter loader
# ----------------------------
@dataclasses.dataclass
class AffAdapter:
    load_model: Any
    predict_saliency: Any


def load_adapter(py_path: str) -> AffAdapter:
    p = Path(py_path)
    if not p.exists():
        die(f"Adapter file not found: {p}")

    spec = importlib.util.spec_from_file_location("aff_adapter", str(p))
    if spec is None or spec.loader is None:
        die(f"Failed to import adapter: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "load_model") or not hasattr(mod, "predict_saliency"):
        die(
            "Adapter must define:\n"
            "  load_model(ckpt_path: str, device: str) -> Any\n"
            "  predict_saliency(model: Any, pil_rgb: PIL.Image, label: str) -> np.ndarray (H,W float)\n"
        )
    return AffAdapter(load_model=mod.load_model, predict_saliency=mod.predict_saliency)


# ----------------------------
# Prompts
# ----------------------------
def build_system_prompt(actions: List[str]) -> str:
    return f"""You are a careful vision-language reasoning system for action-object affordances.

You are given:
- full scene image
- an object instance crop (from SAM)
- an affordance saliency overlay for a specific action (perception evidence)

Your job:
For each requested action in {actions}, decide ONE relationship label from:
{REL_CATEGORIES}

Rules:
- Output STRICT JSON only (no markdown, no commentary).
- Be conservative. Use the saliency overlay as evidence, but do not blindly trust it.
- If relationship_label is one of the exception types:
  {sorted(list(EXCEPTION_CATEGORIES))}
  then provide short grounded "explanation" and "consequence".
- If not an exception type, do NOT include explanation/consequence keys.

JSON schema:
{{
  "object_name": "short noun phrase",
  "per_action": {{
    "<action>": {{
      "relationship_label": "one of the fixed labels exactly",
      "explanation": "only for exception labels",
      "consequence": "only for exception labels"
    }}
  }}
}}
"""


def build_user_prompt(actions: List[str]) -> str:
    acts = ", ".join(actions)
    return f"""Analyze the object instance in the crop using the full image for context.
Also use the saliency overlay as evidence for the given action(s).

Return JSON for:
- object_name
- per_action for actions: {acts}
"""


# ----------------------------
# Scoring + selection
# ----------------------------
def mean_saliency_over_mask(sal: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(sal[m]))


def select_topk_pairs(
    scores: Dict[Tuple[str, str], float],
    topk: int,
    min_score: float,
) -> List[Tuple[str, str]]:
    """
    scores keys: (instance_id, action)
    returns topk pairs by score, filtered by min_score
    """
    items = [(k, v) for k, v in scores.items() if v >= min_score]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:topk]]


# ----------------------------
# Main
# ----------------------------
@dataclasses.dataclass
class InstancePred:
    instance_id: str
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    object_name: str
    actions: Dict[str, Dict[str, Any]]  # action -> {relationship_label, relationship_code, ...}


def main() -> None:
    ap = argparse.ArgumentParser()

    # I/O
    ap.add_argument("--image", required=True, help="Path to input image.")
    ap.add_argument("--outdir", default="outputs", help="Output directory.")
    ap.add_argument("--cache", default="outputs/cache_llm.json", help="Cache file for LLM calls.")

    # SAM
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam_ckpt", default="", help="If empty, auto-download official SAM checkpoint.")
    ap.add_argument("--sam_cache_dir", default=str(Path.home() / ".cache" / "segment_anything"))
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--max_instances", type=int, default=30)

    # Affordance saliency models
    ap.add_argument("--adapter", required=True, help="Path to adapter .py implementing load_model/predict_saliency.")
    ap.add_argument("--seen_ckpt", required=True, help="Checkpoint for SEEN_AFF model.")
    ap.add_argument("--unseen_ckpt", required=True, help="Checkpoint for UNSEEN_AFF model.")
    ap.add_argument("--label_mode", default="auto", choices=["seen", "unseen", "auto", "both"],
                    help="Which label set/model(s) to use.")
    ap.add_argument("--prefer_on_overlap", default="unseen", choices=["seen", "unseen"],
                    help="In auto mode, which model to prefer if a label is in both sets.")

    # Actions to query (defaults to union of both sets in auto/both)
    ap.add_argument("--actions", nargs="*", default=[],
                    help="Actions/affordance labels to query. If empty, inferred from label_mode.")

    # Pair selection (LLM cost control)
    ap.add_argument("--topk_pairs", type=int, default=20,
                    help="How many (instance,action) pairs to send to LLM (global top-k).")
    ap.add_argument("--min_pair_score", type=float, default=0.15,
                    help="Minimum mean-saliency score to consider a pair for LLM reasoning.")
    ap.add_argument("--default_low_score_label", default="Firmly Negative", choices=REL_CATEGORIES,
                    help="Relationship label assigned to non-selected pairs.")

    # LLM
    ap.add_argument("--llm_backend", default="openai", help="openai|anthropic|gemini|ollama")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--ollama_host", default="http://localhost:11434")

    # Relationship code map
    ap.add_argument("--relmap_json", default="", help="Optional JSON label->int code map.")

    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        die(f"Image not found: {image_path}")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load relmap
    relmap = dict(DEFAULT_REL_CODEMAP)
    if args.relmap_json:
        p = Path(args.relmap_json)
        if not p.exists():
            die(f"--relmap_json not found: {p}")
        relmap = json.loads(p.read_text(encoding="utf-8"))
        for k in REL_CATEGORIES:
            if k not in relmap or not isinstance(relmap[k], int):
                die(f"relmap_json missing/invalid: {k}")

    # Cache
    cache_path = Path(args.cache)
    ensure_dir(cache_path.parent)
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    # Image
    pil = Image.open(image_path).convert("RGB")
    rgb = np.array(pil)

    # SAM
    sam_ckpt = ensure_sam_checkpoint(args.sam_type, args.sam_ckpt, args.sam_cache_dir)
    mask_gen = load_sam_mask_generator(sam_ckpt, args.sam_type, args.device)
    sam_masks = mask_gen.generate(rgb)
    if not sam_masks:
        die("SAM produced no masks.")

    sam_masks = sorted(sam_masks, key=lambda d: int(d.get("area", 0)), reverse=True)[:args.max_instances]
    masks_bool: List[np.ndarray] = [m["segmentation"].astype(bool) for m in sam_masks]

    overlay_img = overlay_masks(rgb, masks_bool)
    overlay_path = outdir / f"{image_path.stem}_overlay.png"
    overlay_img.save(overlay_path)

    # Adapter + load models
    adapter = load_adapter(args.adapter)
    print("[INFO] Loading affordance saliency models...")
    aff_seen = adapter.load_model(args.seen_ckpt, args.device)
    aff_unseen = adapter.load_model(args.unseen_ckpt, args.device)

    seen_set = set(SEEN_AFF)
    unseen_set = set(UNSEEN_AFF)

    # Determine actions to query
    if args.actions:
        actions = args.actions
    else:
        if args.label_mode == "seen":
            actions = list(SEEN_AFF)
        elif args.label_mode == "unseen":
            actions = list(UNSEEN_AFF)
        elif args.label_mode in ("auto", "both"):
            actions = sorted(list(seen_set | unseen_set))
        else:
            actions = sorted(list(seen_set | unseen_set))

    # Route label -> model(s)
    def route_models_for_label(label: str) -> List[Tuple[str, Any]]:
        in_seen = label in seen_set
        in_unseen = label in unseen_set
        if args.label_mode == "seen":
            if not in_seen:
                return []
            return [("seen", aff_seen)]
        if args.label_mode == "unseen":
            if not in_unseen:
                return []
            return [("unseen", aff_unseen)]
        if args.label_mode == "auto":
            if in_seen and in_unseen:
                chosen = args.prefer_on_overlap
                return [("unseen", aff_unseen)] if chosen == "unseen" else [("seen", aff_seen)]
            if in_unseen:
                return [("unseen", aff_unseen)]
            if in_seen:
                return [("seen", aff_seen)]
            return []
        if args.label_mode == "both":
            out = []
            if in_seen:
                out.append(("seen", aff_seen))
            if in_unseen:
                out.append(("unseen", aff_unseen))
            return out
        return []

    # Compute saliency maps (label -> (mode->saliency))
    print("[INFO] Computing saliency maps...")
    saliency_maps: Dict[Tuple[str, str], np.ndarray] = {}  # (label, model_tag) -> HxW float

    for a in actions:
        routes = route_models_for_label(a)
        if not routes:
            continue
        for tag, model in routes:
            s = adapter.predict_saliency(model, pil, a)
            s = np.asarray(s, dtype=np.float32)
            if s.shape != (rgb.shape[0], rgb.shape[1]):
                die(f"Saliency for action '{a}' ({tag}) has shape {s.shape}, expected {(rgb.shape[0], rgb.shape[1])}")
            # normalize to [0,1] for scoring stability
            smin, smax = float(s.min()), float(s.max())
            if smax > smin:
                s = (s - smin) / (smax - smin)
            else:
                s = np.zeros_like(s)
            saliency_maps[(a, tag)] = s

    if not saliency_maps:
        die("No saliency maps computed. Check label_mode/actions and adapter implementation.")

    # For the rest of the pipeline we want ONE saliency per action.
    # If label_mode==both and we have two maps, we pick one for scoring:
    #   - prefer args.prefer_on_overlap
    # but we still *can* reason with both later if you want (extension).
    chosen_sal: Dict[str, np.ndarray] = {}
    for a in actions:
        # pick in priority: preferred overlap tag, else any available
        preferred_tag = args.prefer_on_overlap
        if (a, preferred_tag) in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, preferred_tag)]
        elif (a, "seen") in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, "seen")]
        elif (a, "unseen") in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, "unseen")]

    # Score all (instance, action)
    instance_ids = [f"{i:03d}" for i in range(len(masks_bool))]
    scores: Dict[Tuple[str, str], float] = {}
    bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    areas: Dict[str, int] = {}

    for i, m in enumerate(masks_bool):
        iid = instance_ids[i]
        bboxes[iid] = mask_to_bbox(m)
        areas[iid] = int(m.sum())
        for a, s in chosen_sal.items():
            scores[(iid, a)] = mean_saliency_over_mask(s, m)

    # Select top pairs for LLM reasoning
    top_pairs = select_topk_pairs(scores, topk=args.topk_pairs, min_score=args.min_pair_score)
    print(f"[INFO] Selected {len(top_pairs)} pairs for LLM reasoning (topk={args.topk_pairs}, min_score={args.min_pair_score}).")

    # Init LLM client
    llm = make_llm_client(args.llm_backend, args.llm_model, args.ollama_host)

    # Prepare outputs per instance
    preds: Dict[str, InstancePred] = {}
    for iid in instance_ids:
        preds[iid] = InstancePred(
            instance_id=iid,
            bbox_xyxy=bboxes[iid],
            area=areas[iid],
            object_name="object",
            actions={},
        )
        # default actions (all queried)
        for a in actions:
            preds[iid].actions[a] = {
                "relationship_label": args.default_low_score_label,
                "relationship_code": int(relmap[args.default_low_score_label]),
                "score": float(scores.get((iid, a), 0.0)),
                "selected_for_llm": False,
            }

    # LLM reasoning for selected pairs
    # We call LLM per pair (keeps behavior closest to original run_pipeline.py).
    # (You can later batch per-image for cost.)
    sys_prompt = build_system_prompt(actions=["<single_action>"])  # placeholder, we rewrite per call

    full_b64 = b64_png(pil)

    for (iid, a) in top_pairs:
        m = masks_bool[int(iid)]
        crop = crop_with_mask(rgb, m, pad=0.12)
        crop_b64 = b64_png(crop)

        sal = chosen_sal[a]
        sal_overlay = saliency_overlay_image(rgb, sal, mask=m)
        # crop overlay to the instance crop bbox for clarity
        x0, y0, x1, y1 = bboxes[iid]
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(rgb.shape[1]-1, x1); y1 = min(rgb.shape[0]-1, y1)
        overlay_crop = Image.fromarray(np.array(sal_overlay)[y0:y1+1, x0:x1+1])
        overlay_b64 = b64_png(overlay_crop)

        # Cache key includes: model + image + action + instance crop + overlay crop
        key_material = (
            f"{args.llm_backend}|{args.llm_model}|{a}|"
            f"{sha256_bytes(pil_to_png_bytes(pil))}|"
            f"{sha256_bytes(pil_to_png_bytes(crop))}|"
            f"{sha256_bytes(pil_to_png_bytes(overlay_crop))}"
        ).encode("utf-8")
        cache_key = sha256_bytes(key_material)

        if cache_key in cache:
            llm_out = cache[cache_key]
        else:
            system = build_system_prompt(actions=[a])
            user = build_user_prompt(actions=[a])
            try:
                llm_out = llm.complete_json(
                    system=system,
                    user=user,
                    images_b64png=[full_b64, crop_b64, overlay_b64],
                )
            except Exception as e:
                # Graceful fallback (important for long runs / quota issues)
                msg = str(e)
                if "insufficient_quota" in msg or "429" in msg:
                    print("[WARN] LLM quota issue; using fallback for this pair.")
                    llm_out = {"object_name": "object", "per_action": {a: {"relationship_label": args.default_low_score_label}}}
                else:
                    raise

            cache[cache_key] = llm_out
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

        # Normalize LLM output
        obj_name = str(llm_out.get("object_name", "object")).strip()[:80]
        per_action = llm_out.get("per_action", {}) or {}
        a_out = per_action.get(a, {}) or {}
        label = str(a_out.get("relationship_label", args.default_low_score_label)).strip()
        if label not in REL_CATEGORIES:
            label = args.default_low_score_label

        preds[iid].object_name = obj_name
        entry = preds[iid].actions[a]
        entry["relationship_label"] = label
        entry["relationship_code"] = int(relmap[label])
        entry["selected_for_llm"] = True

        if label in EXCEPTION_CATEGORIES:
            exp = str(a_out.get("explanation", "")).strip() or "The action is not appropriate in the current scene."
            con = str(a_out.get("consequence", "")).strip() or "Taking the action may cause a negative outcome."
            entry["explanation"] = exp
            entry["consequence"] = con
        else:
            entry.pop("explanation", None)
            entry.pop("consequence", None)

    # Write outputs (same as original run_pipeline.py style)
    instances_json = {
        "image": str(image_path),
        "label_mode": args.label_mode,
        "prefer_on_overlap": args.prefer_on_overlap,
        "actions": actions,
        "relationship_categories": REL_CATEGORIES,
        "relationship_code_map": relmap,
        "topk_pairs": args.topk_pairs,
        "min_pair_score": args.min_pair_score,
        "instances": [
            {
                "instance_id": p.instance_id,
                "object_name": p.object_name,
                "bbox_xyxy": p.bbox_xyxy,
                "area": p.area,
                "actions": p.actions,
            }
            for p in preds.values()
        ],
    }
    (outdir / f"{image_path.stem}_instances.json").write_text(
        json.dumps(instances_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # relationship.txt: id # code # code # ... for actions in order
    rel_lines = []
    for iid in instance_ids:
        codes = []
        for a in actions:
            codes.append(str(preds[iid].actions[a]["relationship_code"]))
        rel_lines.append(f"{iid} # " + " # ".join(codes))
    (outdir / f"{image_path.stem}_relationship.txt").write_text("\n".join(rel_lines) + "\n", encoding="utf-8")

    # exco.json: action -> instance_id -> {explanation, consequence} only for exception relations
    exco: Dict[str, Dict[str, Dict[str, str]]] = {}
    for a in actions:
        exco[a] = {}
    for iid in instance_ids:
        for a in actions:
            info = preds[iid].actions[a]
            if info["relationship_label"] in EXCEPTION_CATEGORIES:
                exco[a][iid] = {"explanation": info.get("explanation", ""), "consequence": info.get("consequence", "")}
    exco = {a: d for a, d in exco.items() if len(d) > 0}
    (outdir / f"{image_path.stem}_exco.json").write_text(
        json.dumps(exco, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] Wrote:")
    print(f"  - {overlay_path}")
    print(f"  - {outdir / (image_path.stem + '_instances.json')}")
    print(f"  - {outdir / (image_path.stem + '_relationship.txt')}")
    print(f"  - {outdir / (image_path.stem + '_exco.json')}")


if __name__ == "__main__":
    main()

