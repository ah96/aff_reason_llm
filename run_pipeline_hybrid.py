#!/usr/bin/env python3
"""
run_pipeline_saliency.py

A saliency-first (but still SAM-compatible) affordance reasoning pipeline.

It keeps the SAME output artifacts and overall spirit as your original run_pipeline.py:
  - <stem>_overlay.png
  - <stem>_instances.json
  - <stem>_relationship.txt
  - <stem>_exco.json
  - cache file for LLM calls

NEW IDEA (integrated end-to-end, no manual patching required):
  - Uses TWO affordance saliency/segmentation models (SEEN_AFF and UNSEEN_AFF) via label routing.
  - Computes saliency maps S_a(x,y) for queried actions.
  - Supports THREE instance modes:
      1) sam_auto         : SAM automatic masks (baseline, stable object IDs)
      2) saliency_prompted: saliency peaks -> SAM point prompting -> masks (action-focused)
      3) hybrid_refine    : SAM auto masks -> score w/ saliency -> refine top pairs via saliency point prompts

LLM is used ONLY for top-K (instance, action) pairs (cheap) and uses saliency overlay as evidence.

IMPORTANT: You must provide an adapter module for your affordance saliency model:
  - load_model(ckpt_path: str, device: str) -> Any
  - predict_saliency(model: Any, pil_rgb: PIL.Image, label: str) -> np.ndarray(H,W)

Example adapter stub is included at the bottom of this file (copy it into a separate .py).
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
# Label sets (provided by you)
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
# Relationship categories
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


# ----------------------------
# Utils
# ----------------------------
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
# Mask / image helpers
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
    Overlay saliency as a simple heatmap on top of the RGB image.
    No extra deps. sal: HxW float (any range) -> normalized to [0,1] for display.
    If mask is provided, overlay is restricted to mask area.
    """
    sal = np.asarray(sal, dtype=np.float32)
    if sal.ndim != 2:
        raise ValueError(f"saliency must be HxW, got {sal.shape}")
    smin, smax = float(sal.min()), float(sal.max())
    if smax > smin:
        saln = (sal - smin) / (smax - smin)
    else:
        saln = np.zeros_like(sal)

    heat = np.zeros_like(rgb, dtype=np.float32)
    heat[..., 0] = saln * 255.0          # R
    heat[..., 1] = (saln ** 0.5) * 80.0  # G
    heat[..., 2] = (1.0 - saln) * 40.0   # B

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


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def dedup_masks_by_iou(masks: List[np.ndarray], iou_thr: float) -> List[np.ndarray]:
    kept: List[np.ndarray] = []
    for m in masks:
        ok = True
        for k in kept:
            if mask_iou(m, k) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(m)
    return kept


def topk_saliency_points(
    sal: np.ndarray,
    k: int,
    min_val: float,
    mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Return up to k (x,y) peak points from normalized saliency in [0,1].
    Greedy selection with a small suppression radius (simple NMS).
    """
    s = np.asarray(sal, dtype=np.float32).copy()

    # sal is assumed normalized [0,1] by our pipeline, but we guard anyway
    s = np.clip(s, 0.0, 1.0)

    if mask is not None:
        s[~mask.astype(bool)] = 0.0

    pts: List[Tuple[int, int]] = []
    h, w = s.shape
    r = max(4, int(0.02 * max(h, w)))  # suppression radius

    for _ in range(k):
        idx = int(np.argmax(s))
        y, x = np.unravel_index(idx, s.shape)
        if float(s[y, x]) < min_val:
            break
        pts.append((int(x), int(y)))

        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        s[y0:y1, x0:x1] = 0.0

    return pts


# ----------------------------
# SAM loaders
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


def load_sam_predictor(ckpt: str, sam_type: str, device: str):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        die(
            "SAM predictor requires segment-anything. Install:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Import error: {e}"
        )

    sam = sam_model_registry[sam_type](checkpoint=ckpt)
    sam.to(device=device)
    return SamPredictor(sam)


def sam_prompt_masks_from_points(
    predictor,
    rgb: np.ndarray,
    points_xy: List[Tuple[int, int]],
    dedup_iou_thr: float,
) -> List[np.ndarray]:
    """
    Prompt SAM with positive point prompts. For each point, take the best mask.
    Deduplicate by IoU across all prompted masks.
    """
    predictor.set_image(rgb)
    all_masks: List[np.ndarray] = []

    for (x, y) in points_xy:
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        all_masks.append(masks[best].astype(bool))

    all_masks = dedup_masks_by_iou(all_masks, iou_thr=dedup_iou_thr)
    return all_masks


# ----------------------------
# LLM clients (multi-backend)
# ----------------------------
class VisionLLMClient:
    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIVisionClient(VisionLLMClient):
    """
    Uses the OpenAI Responses API, but avoids response_format because older SDKs may not support it.
    """
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
                        if c.get("type") in ("output_text", "text") and c.get("text"):
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
        self.model = model
        self.host = host.rstrip("/")

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
# Affordance saliency adapter
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
# Prompt builders
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


def select_topk_pairs(scores: Dict[Tuple[str, str], float], topk: int, min_score: float) -> List[Tuple[str, str]]:
    items = [(k, v) for k, v in scores.items() if v >= min_score]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:topk]]


# ----------------------------
# Output structure
# ----------------------------
@dataclasses.dataclass
class InstancePred:
    instance_id: str
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    object_name: str
    actions: Dict[str, Dict[str, Any]]  # action -> data


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # I/O
    ap.add_argument("--image", required=True, help="Path to input image.")
    ap.add_argument("--outdir", default="outputs", help="Output directory.")
    ap.add_argument("--cache", default="outputs/cache_llm.json", help="Cache file for LLM calls.")

    # Device
    ap.add_argument("--device", default="cuda", help="cuda or cpu")

    # SAM
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam_ckpt", default="", help="If empty, auto-download official SAM checkpoint.")
    ap.add_argument("--sam_cache_dir", default=str(Path.home() / ".cache" / "segment_anything"))
    ap.add_argument("--max_instances", type=int, default=30)

    # NEW: instance source modes
    ap.add_argument(
        "--instance_source",
        default="sam_auto",
        choices=["sam_auto", "saliency_prompted", "hybrid_refine"],
        help="How to obtain instance masks."
    )
    ap.add_argument("--saliency_points_per_action", type=int, default=6,
                    help="For saliency_prompted/hybrid_refine: number of peak points per action.")
    ap.add_argument("--saliency_peak_min", type=float, default=0.65,
                    help="For saliency_prompted/hybrid_refine: minimum normalized saliency for a peak.")
    ap.add_argument("--dedup_iou", type=float, default=0.85,
                    help="For prompted modes: deduplicate masks by IoU threshold.")
    ap.add_argument("--refine_topk_pairs", type=int, default=15,
                    help="For hybrid_refine: refine only these top scoring (instance, action) pairs.")

    # Affordance saliency adapter + checkpoints
    ap.add_argument("--adapter", required=True, help="Path to adapter .py implementing load_model/predict_saliency.")
    ap.add_argument("--seen_ckpt", required=True, help="Checkpoint for SEEN_AFF model.")
    ap.add_argument("--unseen_ckpt", required=True, help="Checkpoint for UNSEEN_AFF model.")
    ap.add_argument("--label_mode", default="auto", choices=["seen", "unseen", "auto", "both"],
                    help="Which label set/model(s) to use.")
    ap.add_argument("--prefer_on_overlap", default="unseen", choices=["seen", "unseen"],
                    help="In auto/both mode, prefer this model's saliency when a label is in both sets.")

    # Actions
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

    # Load relationship code map
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

    # Load image
    pil = Image.open(image_path).convert("RGB")
    rgb = np.array(pil)

    # SAM setup
    sam_ckpt = ensure_sam_checkpoint(args.sam_type, args.sam_ckpt, args.sam_cache_dir)
    mask_gen = load_sam_mask_generator(sam_ckpt, args.sam_type, args.device)
    predictor = None
    if args.instance_source in ("saliency_prompted", "hybrid_refine"):
        predictor = load_sam_predictor(sam_ckpt, args.sam_type, args.device)

    # Affordance adapter + models
    adapter = load_adapter(args.adapter)
    print("[INFO] Loading affordance saliency models...")
    aff_seen = adapter.load_model(args.seen_ckpt, args.device)
    aff_unseen = adapter.load_model(args.unseen_ckpt, args.device)

    seen_set = set(SEEN_AFF)
    unseen_set = set(UNSEEN_AFF)

    # Determine actions
    if args.actions:
        actions = args.actions
    else:
        if args.label_mode == "seen":
            actions = list(SEEN_AFF)
        elif args.label_mode == "unseen":
            actions = list(UNSEEN_AFF)
        else:
            actions = sorted(list(seen_set | unseen_set))

    def route_models_for_label(label: str) -> List[Tuple[str, Any]]:
        in_seen = label in seen_set
        in_unseen = label in unseen_set

        if args.label_mode == "seen":
            return [("seen", aff_seen)] if in_seen else []
        if args.label_mode == "unseen":
            return [("unseen", aff_unseen)] if in_unseen else []
        if args.label_mode == "auto":
            if in_seen and in_unseen:
                return [("unseen", aff_unseen)] if args.prefer_on_overlap == "unseen" else [("seen", aff_seen)]
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

    # Compute saliency maps (label, tag) -> HxW in [0,1]
    print("[INFO] Computing saliency maps...")
    saliency_maps: Dict[Tuple[str, str], np.ndarray] = {}
    for a in actions:
        routes = route_models_for_label(a)
        if not routes:
            continue
        for tag, model in routes:
            s = adapter.predict_saliency(model, pil, a)
            s = np.asarray(s, dtype=np.float32)
            if s.shape != (rgb.shape[0], rgb.shape[1]):
                die(f"Saliency for action '{a}' ({tag}) has shape {s.shape}, expected {(rgb.shape[0], rgb.shape[1])}")

            # Normalize to [0,1] (stable scoring)
            smin, smax = float(s.min()), float(s.max())
            if smax > smin:
                s = (s - smin) / (smax - smin)
            else:
                s = np.zeros_like(s, dtype=np.float32)

            saliency_maps[(a, tag)] = s

    if not saliency_maps:
        die("No saliency maps computed. Check label_mode/actions and adapter implementation.")

    # Choose ONE saliency per action for downstream scoring (unless label_mode==both and you later want comparison)
    chosen_sal: Dict[str, np.ndarray] = {}
    for a in actions:
        pref = args.prefer_on_overlap
        if (a, pref) in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, pref)]
        elif (a, "seen") in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, "seen")]
        elif (a, "unseen") in saliency_maps:
            chosen_sal[a] = saliency_maps[(a, "unseen")]

    # ----------------------------
    # Instance generation modes
    # ----------------------------
    masks_bool: List[np.ndarray] = []

    if args.instance_source == "sam_auto":
        sam_masks = mask_gen.generate(rgb)
        if not sam_masks:
            die("SAM produced no masks.")
        sam_masks = sorted(sam_masks, key=lambda d: int(d.get("area", 0)), reverse=True)[:args.max_instances]
        masks_bool = [m["segmentation"].astype(bool) for m in sam_masks]

    elif args.instance_source == "saliency_prompted":
        if predictor is None:
            die("Internal error: predictor missing.")
        prompted: List[np.ndarray] = []
        for a, s in chosen_sal.items():
            pts = topk_saliency_points(
                s, k=args.saliency_points_per_action, min_val=args.saliency_peak_min, mask=None
            )
            if not pts:
                continue
            prompted.extend(sam_prompt_masks_from_points(predictor, rgb, pts, dedup_iou_thr=args.dedup_iou))

        prompted = dedup_masks_by_iou(prompted, iou_thr=args.dedup_iou)
        prompted.sort(key=lambda m: int(m.sum()), reverse=True)
        masks_bool = prompted[:args.max_instances]
        if not masks_bool:
            die("saliency_prompted produced no masks. Lower --saliency_peak_min or increase points.")

    elif args.instance_source == "hybrid_refine":
        sam_masks = mask_gen.generate(rgb)
        if not sam_masks:
            die("SAM produced no masks.")
        sam_masks = sorted(sam_masks, key=lambda d: int(d.get("area", 0)), reverse=True)[:args.max_instances]
        masks_bool = [m["segmentation"].astype(bool) for m in sam_masks]

    else:
        die(f"Unknown --instance_source: {args.instance_source}")

    if not masks_bool:
        die("No instance masks available after instance generation.")

    # Overlay of instances (always produced)
    overlay_img = overlay_masks(rgb, masks_bool)
    overlay_path = outdir / f"{image_path.stem}_overlay.png"
    overlay_img.save(overlay_path)

    # Prepare instance IDs and stats
    instance_ids = [f"{i:03d}" for i in range(len(masks_bool))]
    bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    areas: Dict[str, int] = {}
    for i, m in enumerate(masks_bool):
        iid = instance_ids[i]
        bboxes[iid] = mask_to_bbox(m)
        areas[iid] = int(m.sum())

    # Score all (instance, action) pairs
    scores: Dict[Tuple[str, str], float] = {}
    for i, m in enumerate(masks_bool):
        iid = instance_ids[i]
        for a, s in chosen_sal.items():
            scores[(iid, a)] = mean_saliency_over_mask(s, m)

    # Hybrid refinement: refine only top pairs, updating masks_bool in-place (stable IDs preserved)
    if args.instance_source == "hybrid_refine":
        if predictor is None:
            die("Internal error: predictor missing.")
        refine_pairs = select_topk_pairs(scores, topk=args.refine_topk_pairs, min_score=args.min_pair_score)
        if refine_pairs:
            print(f"[INFO] Hybrid refine: refining {len(refine_pairs)} top pairs using saliency point prompts...")
        for (iid, a) in refine_pairs:
            i = int(iid)
            base_mask = masks_bool[i]
            s = chosen_sal.get(a, None)
            if s is None:
                continue

            pts = topk_saliency_points(
                s,
                k=max(2, args.saliency_points_per_action // 2),
                min_val=args.saliency_peak_min,
                mask=base_mask,  # constrain peaks inside the object
            )
            if not pts:
                continue

            refined = sam_prompt_masks_from_points(predictor, rgb, pts, dedup_iou_thr=args.dedup_iou)
            if not refined:
                continue

            refined.sort(key=lambda rm: mask_iou(rm, base_mask), reverse=True)
            best = refined[0]

            # accept only if it doesn't "jump" to another object
            if mask_iou(best, base_mask) >= 0.25:
                masks_bool[i] = best

        # Recompute bboxes/areas/scores after refinement
        for i, m in enumerate(masks_bool):
            iid = instance_ids[i]
            bboxes[iid] = mask_to_bbox(m)
            areas[iid] = int(m.sum())
        scores.clear()
        for i, m in enumerate(masks_bool):
            iid = instance_ids[i]
            for a, s in chosen_sal.items():
                scores[(iid, a)] = mean_saliency_over_mask(s, m)

        # Update overlay to reflect refined masks (keeps output aligned)
        overlay_img = overlay_masks(rgb, masks_bool)
        overlay_img.save(overlay_path)

    # Select top-K pairs for LLM reasoning
    top_pairs = select_topk_pairs(scores, topk=args.topk_pairs, min_score=args.min_pair_score)
    print(f"[INFO] Selected {len(top_pairs)} pairs for LLM reasoning (topk={args.topk_pairs}, min_score={args.min_pair_score}).")

    # Init LLM client
    llm = make_llm_client(args.llm_backend, args.llm_model, args.ollama_host)

    # Initialize per-instance predictions with defaults
    preds: Dict[str, InstancePred] = {}
    for iid in instance_ids:
        preds[iid] = InstancePred(
            instance_id=iid,
            bbox_xyxy=bboxes[iid],
            area=areas[iid],
            object_name="object",
            actions={},
        )
        for a in actions:
            if a in chosen_sal:
                default_score = float(scores.get((iid, a), 0.0))
            else:
                default_score = 0.0
            preds[iid].actions[a] = {
                "relationship_label": args.default_low_score_label,
                "relationship_code": int(relmap[args.default_low_score_label]),
                "score": default_score,
                "selected_for_llm": False,
            }

    # LLM reasoning for selected pairs (per-pair call to keep behavior closest to original)
    full_b64 = b64_png(pil)

    for (iid, a) in top_pairs:
        if a not in chosen_sal:
            continue

        i = int(iid)
        m = masks_bool[i]
        crop = crop_with_mask(rgb, m, pad=0.12)
        crop_b64 = b64_png(crop)

        # Saliency overlay restricted to this instance, then cropped to bbox for clarity
        sal = chosen_sal[a]
        overlay_full = saliency_overlay_image(rgb, sal, mask=m)
        x0, y0, x1, y1 = bboxes[iid]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(rgb.shape[1] - 1, x1); y1 = min(rgb.shape[0] - 1, y1)
        overlay_crop = Image.fromarray(np.array(overlay_full)[y0:y1 + 1, x0:x1 + 1])
        overlay_b64 = b64_png(overlay_crop)

        # Cache key includes model + image + action + crop + overlay crop
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

    # Write instances.json
    instances_json = {
        "image": str(image_path),
        "instance_source": args.instance_source,
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

    # Write relationship.txt: instance_id # code # code # ...
    rel_lines = []
    for iid in instance_ids:
        codes = []
        for a in actions:
            codes.append(str(preds[iid].actions[a]["relationship_code"]))
        rel_lines.append(f"{iid} # " + " # ".join(codes))
    (outdir / f"{image_path.stem}_relationship.txt").write_text("\n".join(rel_lines) + "\n", encoding="utf-8")

    # Write exco.json: only exception relations
    exco: Dict[str, Dict[str, Dict[str, str]]] = {a: {} for a in actions}
    for iid in instance_ids:
        for a in actions:
            info = preds[iid].actions[a]
            if info["relationship_label"] in EXCEPTION_CATEGORIES:
                exco[a][iid] = {
                    "explanation": info.get("explanation", ""),
                    "consequence": info.get("consequence", "")
                }
    exco = {a: d for a, d in exco.items() if len(d) > 0}
    (outdir / f"{image_path.stem}_exco.json").write_text(
        json.dumps(exco, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] Wrote:")
    print(f"  - {overlay_path}")
    print(f"  - {outdir / (image_path.stem + '_instances.json')}")
    print(f"  - {outdir / (image_path.stem + '_relationship.txt')}")
    print(f"  - {outdir / (image_path.stem + '_exco.json')}")
    print(f"[INFO] LLM cache: {cache_path}")


if __name__ == "__main__":
    main()


"""
------------------------------------------------------------
adapter_affordance_example.py  (SAVE AS A SEPARATE FILE)
------------------------------------------------------------
# Save the following as: adapter_affordance_example.py
# Then run:
#   python3 run_pipeline_saliency.py --adapter adapter_affordance_example.py ...

from typing import Any
import numpy as np
from PIL import Image

def load_model(ckpt_path: str, device: str) -> Any:
    # TODO: load your torch model checkpoint here.
    # Example:
    #   import torch
    #   model = YourNet(...)
    #   sd = torch.load(ckpt_path, map_location=device)
    #   model.load_state_dict(sd, strict=False)
    #   model.to(device).eval()
    #   return model
    raise NotImplementedError

def predict_saliency(model: Any, pil_rgb: Image.Image, label: str) -> np.ndarray:
    # TODO: run your model to produce a saliency map for (image,label).
    # Must return np.ndarray of shape (H,W), float (any range is fine; script normalizes).
    # If your model works at a different resolution, resize back to original H,W BEFORE returning.
    raise NotImplementedError
"""
