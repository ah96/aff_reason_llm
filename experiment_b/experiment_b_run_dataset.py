#!/usr/bin/env python3
"""
Experiment B runner WITHOUT SAM: uses dataset-provided segmentation masks.

Supports:
- Semantic segmentation masks (e.g., ImageNet-S): single-channel PNG of class IDs.
  -> converts to instance candidates via connected components per class.
- Instance segmentation masks: single-channel PNG of instance IDs (0=background).
  -> extracts per-instance masks; optionally infers class_id from a semantic mask
     by majority vote over instance pixels.

Regimes:
- dataset: select top-K by area
- dataset_saliency: select top-K by saliency score (optional affordance adapter)

Outputs per image x LLM:
- <stem>_<llm>_instances.json    (main output; evaluator-friendly)
- <stem>_<llm>_relationship.txt  (ADE-like relationship codes)
- <stem>_<llm>_exco.json         (only exception ExCo entries)
- <stem>_overlay.png             (overlay of selected masks)
- <stem>_<llm>_timings.json      (paper-grade timing + diagnostics)

Outputs per regime/K:
- run_config.json
"""

import argparse
import base64
import hashlib
import io
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from vision_llm_clients import load_llms, make_client

# Optional torch for metadata + seeding
try:
    import torch  # type: ignore
except Exception:
    torch = None

# Optional scipy connected components
try:
    from scipy.ndimage import label as cc_label  # type: ignore
except Exception:
    cc_label = None


# ----------------------------
# Relationship taxonomy (keep consistent with your paper)
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
REL_CODEMAP = {name: i for i, name in enumerate(REL_CATEGORIES)}
EXCEPTION_CATEGORIES = set(REL_CATEGORIES[2:])


# ----------------------------
# Prompts (strict JSON)
# ----------------------------
def system_prompt(actions: List[str]) -> str:
    return f"""You are a careful vision-language reasoning system for action-object affordances.

For each action in {actions}, output ONE relationship label from this fixed set:
{REL_CATEGORIES}

Exception labels (require explanation+consequence):
{sorted(list(EXCEPTION_CATEGORIES))}

Rules (very important):
- Return STRICT JSON only (no markdown, no code fences, no extra text).
- Use ONLY the keys defined in the schema below. Do not add extra keys.
- relationship_label must be exactly one of the fixed labels.
- For NON-exception labels: set explanation="" and consequence="" (empty strings).
- For exception labels: explanation and consequence must each be exactly ONE short sentence, grounded in the images.

JSON schema (must match exactly):
{{
  "object_name": "short noun phrase",
  "per_action": {{
    "<action>": {{
      "relationship_label": "one of the fixed labels exactly",
      "explanation": "one sentence or empty string",
      "consequence": "one sentence or empty string"
    }}
  }}
}}
"""


def user_prompt(actions: List[str]) -> str:
    return f"""Analyze the object in the crop using the full image for context.

Return STRICT JSON for:
- object_name (short noun phrase)
- per_action for actions: {", ".join(actions)}

Remember:
- Always include relationship_label, explanation, consequence for every action.
- Use empty strings for explanation/consequence unless the label is an exception.
"""


# ----------------------------
# Optional saliency adapter (same contract as your SAM-saliency setup)
# ----------------------------
def load_adapter(py_path: str):
    import importlib.util
    p = Path(py_path)
    spec = importlib.util.spec_from_file_location("aff_adapter", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import adapter: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "load_model") or not hasattr(mod, "predict_saliency"):
        raise RuntimeError("Adapter must define load_model() and predict_saliency().")
    return mod


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)


def score_mask_by_saliency(mask: np.ndarray, sal_maps: Dict[str, np.ndarray]) -> float:
    if mask.sum() == 0:
        return 0.0
    best = 0.0
    for S in sal_maps.values():
        best = max(best, float(np.mean(S[mask])))
    return best


# ----------------------------
# Utilities
# ----------------------------
@contextmanager
def timer():
    t0 = time.perf_counter()
    out = {"s": 0.0}
    try:
        yield out
    finally:
        out["s"] = float(time.perf_counter() - t0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_png(img: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(img)).decode("utf-8")


# ----------------------------
# Mask helpers
# ----------------------------
def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_with_mask(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.12) -> Image.Image:
    h, w = mask.shape
    x0, y0, x1, y1 = mask_to_bbox(mask)
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px)
    y1 = min(h - 1, y1 + py)
    return Image.fromarray(rgb[y0 : y1 + 1, x0 : x1 + 1])


def overlay_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    for i, m in enumerate(masks):
        m = m.astype(bool)
        rng = np.random.default_rng(i + 123)
        color = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ----------------------------
# Connected components
# ----------------------------
def connected_components(binary: np.ndarray) -> List[np.ndarray]:
    binary = binary.astype(bool)
    if binary.sum() == 0:
        return []

    if cc_label is not None:
        labeled, n = cc_label(binary)
        return [(labeled == k) for k in range(1, n + 1)]

    # Fallback BFS (4-connected)
    H, W = binary.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: List[np.ndarray] = []
    for y in range(H):
        for x in range(W):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            cm = np.zeros((H, W), dtype=bool)
            ys, xs = zip(*coords)
            cm[np.array(ys), np.array(xs)] = True
            comps.append(cm)
    return comps


# ----------------------------
# Dataset mask -> candidates
# ----------------------------
def semantic_to_candidates(
    sem: np.ndarray,
    background_id: int,
    ignore_id: Optional[int],
    min_area: int,
) -> List[Dict[str, Any]]:
    sem = sem.astype(np.int32)
    ids = np.unique(sem)
    out: List[Dict[str, Any]] = []
    for cid in ids:
        if cid == background_id:
            continue
        if ignore_id is not None and cid == ignore_id:
            continue
        binm = sem == int(cid)
        for comp in connected_components(binm):
            area = int(comp.sum())
            if area < min_area:
                continue
            out.append({"mask": comp, "area": area, "class_id": int(cid), "source": "semantic_cc"})
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def instance_id_to_candidates(
    inst_id: np.ndarray,
    background_id: int,
    ignore_id: Optional[int],
    min_area: int,
    sem_for_class: Optional[np.ndarray] = None,
    class_vote_ignore: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    inst_id: single-channel mask where each pixel is an instance id (0=background).
    sem_for_class: optional semantic class-id mask; used to assign class_id by majority vote.
    """
    inst_id = inst_id.astype(np.int32)
    ids = np.unique(inst_id)
    out: List[Dict[str, Any]] = []
    for iid in ids:
        if iid == background_id:
            continue
        if ignore_id is not None and iid == ignore_id:
            continue
        m = inst_id == int(iid)
        area = int(m.sum())
        if area < min_area:
            continue

        cid = -1
        if sem_for_class is not None:
            vals = sem_for_class[m].astype(np.int32)
            if class_vote_ignore is not None:
                vals = vals[vals != int(class_vote_ignore)]
            if vals.size > 0:
                # majority vote
                uniq, cnt = np.unique(vals, return_counts=True)
                cid = int(uniq[int(np.argmax(cnt))])

        out.append({"mask": m, "area": area, "instance_id": int(iid), "class_id": cid, "source": "instance_id"})
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--masks_dir", required=True, help="Mask folder. For semantic: class-id PNG. For instance: instance-id PNG.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--llms", required=True)
    ap.add_argument("--cache_dir", default="cache_b_dataset")

    # Mask naming convention
    ap.add_argument("--mask_ext", default=".png")
    ap.add_argument("--mask_suffix", default="", help="Mask filename = <image_stem><mask_suffix><mask_ext>")

    # Segmentation type
    ap.add_argument("--seg_type", choices=["semantic", "instance"], required=True)

    # If seg_type=instance: optional semantic mask directory to infer class_id by vote
    ap.add_argument("--semantic_masks_dir", default="", help="Optional: semantic masks for class voting (same naming).")

    # Common ids and filters
    ap.add_argument("--background_id", type=int, default=0)
    ap.add_argument("--ignore_id", type=int, default=255)
    ap.add_argument("--min_region_area", type=int, default=256)
    ap.add_argument("--max_candidates", type=int, default=200, help="Cap candidates before selection (speed).")

    # Regimes
    ap.add_argument("--mode", choices=["dataset", "dataset_saliency"], default="dataset")
    ap.add_argument("--Ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--min_mask_score", type=float, default=0.0)

    # Actions
    ap.add_argument("--actions", nargs="+", default=["sit", "run", "grasp"])

    # Optional saliency adapter
    ap.add_argument("--adapter", default="", help="Required for dataset_saliency")
    ap.add_argument("--seen_ckpt", default="")
    ap.add_argument("--unseen_ckpt", default="")
    ap.add_argument("--device", default="cuda")

    # Repro/logging
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Seed
    np.random.seed(args.seed)
    if torch is not None:
        try:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    semantic_masks_dir = Path(args.semantic_masks_dir) if args.semantic_masks_dir else None

    # LLMs
    llm_cfgs = load_llms(args.llms)
    llms = []
    for cfg in llm_cfgs:
        if not getattr(cfg, "supports_vision", True):
            continue
        llms.append((cfg.name, make_client(cfg)))
    if not llms:
        raise RuntimeError("No vision-capable LLMs configured in llms.json.")

    # Optional adapter
    use_saliency = args.mode == "dataset_saliency"
    adapter = None
    model_seen = None
    model_unseen = None
    if use_saliency:
        if not args.adapter:
            raise RuntimeError("--adapter is required for dataset_saliency")
        adapter = load_adapter(args.adapter)
        model_seen = adapter.load_model(args.seen_ckpt, args.device)
        model_unseen = adapter.load_model(args.unseen_ckpt, args.device)

    # Images
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    Ks = sorted(set(int(k) for k in args.Ks))
    if any(k <= 0 for k in Ks):
        raise ValueError("All K values must be > 0.")

    # Prompt hash for cache safety
    sys_p = system_prompt(args.actions)
    usr_p = user_prompt(args.actions)
    prompt_hash = sha1_short(sys_p + "\n" + usr_p + "\n" + ",".join(args.actions), n=10)

    # Run per K
    for K in Ks:
        run_root = outdir / f"{args.mode}_{args.seg_type}_K{K}"
        ensure_dir(run_root)

        run_config = {
            "timestamp_utc": now_utc_iso(),
            "mode": args.mode,
            "seg_type": args.seg_type,
            "K": K,
            "Ks_all": Ks,
            "actions": args.actions,
            "seed": args.seed,
            "prompt_hash": prompt_hash,
            "dataset": {
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
                "mask_suffix": args.mask_suffix,
                "mask_ext": args.mask_ext,
                "semantic_masks_dir": str(semantic_masks_dir) if semantic_masks_dir else "",
                "background_id": args.background_id,
                "ignore_id": args.ignore_id,
                "min_region_area": args.min_region_area,
                "max_candidates": args.max_candidates,
            },
            "saliency": {
                "enabled": bool(use_saliency),
                "adapter": args.adapter if use_saliency else "",
                "seen_ckpt": args.seen_ckpt if use_saliency else "",
                "unseen_ckpt": args.unseen_ckpt if use_saliency else "",
                "min_mask_score": float(args.min_mask_score),
            },
            "llms": [n for n, _ in llms],
            "platform": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": git_commit(),
                "cuda_available": bool(torch and torch.cuda.is_available()),
                "gpu_name": (torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else "none"),
                "torch_version": (getattr(torch, "__version__", None) if torch else None),
            },
        }
        (run_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        for img_path in tqdm(img_paths, desc=f"{args.mode}/{args.seg_type}/K={K}"):
            pil = Image.open(img_path).convert("RGB")
            rgb = np.array(pil)
            H, W = rgb.shape[0], rgb.shape[1]

            # Load mask
            mask_path = masks_dir / f"{img_path.stem}{args.mask_suffix}{args.mask_ext}"
            if not mask_path.exists():
                # skip (or raise)
                continue
            mask_img = Image.open(mask_path)
            mask_arr = np.array(mask_img)

            # Optional semantic for class voting if instance
            sem_vote = None
            if args.seg_type == "instance" and semantic_masks_dir is not None:
                sem_path = semantic_masks_dir / f"{img_path.stem}{args.mask_suffix}{args.mask_ext}"
                if sem_path.exists():
                    sem_vote = np.array(Image.open(sem_path))

            # Candidate extraction timing
            with timer() as t_cand:
                if args.seg_type == "semantic":
                    if mask_arr.ndim != 2:
                        # if mask is RGB-coded, take first channel (user must ensure IDs are in channel 0)
                        mask_arr = mask_arr[..., 0]
                    candidates = semantic_to_candidates(
                        mask_arr,
                        background_id=args.background_id,
                        ignore_id=args.ignore_id,
                        min_area=args.min_region_area,
                    )
                else:
                    if mask_arr.ndim != 2:
                        mask_arr = mask_arr[..., 0]
                    candidates = instance_id_to_candidates(
                        mask_arr,
                        background_id=args.background_id,
                        ignore_id=args.ignore_id,
                        min_area=args.min_region_area,
                        sem_for_class=sem_vote,
                        class_vote_ignore=args.ignore_id,
                    )
            candidates_s = float(t_cand["s"])

            num_candidates_total = len(candidates)
            candidates = candidates[: args.max_candidates]
            num_candidates_kept = len(candidates)

            # Optional saliency scoring for selection
            saliency_total_s = 0.0
            saliency_per_action_s: Dict[str, float] = {}
            saliency_fallbacks_to_unseen = 0

            if use_saliency:
                assert adapter is not None and model_seen is not None and model_unseen is not None
                with timer() as t_sal:
                    sal_maps: Dict[str, np.ndarray] = {}
                    for a in args.actions:
                        with timer() as t_a:
                            s = np.asarray(adapter.predict_saliency(model_seen, pil, a), dtype=np.float32)
                            # fallback if shape mismatch
                            if s.shape[:2] != (H, W):
                                saliency_fallbacks_to_unseen += 1
                                s = np.asarray(adapter.predict_saliency(model_unseen, pil, a), dtype=np.float32)
                        saliency_per_action_s[a] = float(t_a["s"])
                        # ensure HxW
                        if s.ndim == 3:
                            s = s[..., 0]
                        sal_maps[a] = normalize01(s)
                saliency_total_s = float(t_sal["s"])

                for c in candidates:
                    c["mask_score"] = score_mask_by_saliency(c["mask"], sal_maps)

                candidates = [c for c in candidates if float(c.get("mask_score", 0.0)) >= args.min_mask_score]
                candidates.sort(key=lambda d: float(d.get("mask_score", 0.0)), reverse=True)
            else:
                candidates.sort(key=lambda d: int(d["area"]), reverse=True)

            selected = candidates[:K]
            masks = [c["mask"] for c in selected]

            img_out = run_root / img_path.stem
            ensure_dir(img_out)

            if not masks:
                (img_out / f"{img_path.stem}_nomasks.json").write_text(
                    json.dumps(
                        {
                            "image": str(img_path),
                            "mode": args.mode,
                            "seg_type": args.seg_type,
                            "K": K,
                            "timestamp_utc": now_utc_iso(),
                            "seed": args.seed,
                            "prompt_hash": prompt_hash,
                            "candidates": {
                                "extract_s": candidates_s,
                                "num_total": num_candidates_total,
                                "num_kept": num_candidates_kept,
                                "num_selected": 0,
                            },
                            "saliency": {
                                "enabled": bool(use_saliency),
                                "saliency_total_s": saliency_total_s,
                                "saliency_per_action_s": saliency_per_action_s,
                                "fallbacks_to_unseen": int(saliency_fallbacks_to_unseen),
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                continue

            # Overlay
            overlay_masks(rgb, masks).save(img_out / f"{img_path.stem}_overlay.png")

            # Full-image b64
            with timer() as t_full:
                full_b64 = b64_png(pil)
            full_b64_s = float(t_full["s"])

            # Per LLM
            inst_ids = [f"{i:03d}" for i in range(len(masks))]

            for llm_name, client in llms:
                llm_cached = 0
                llm_called = 0
                llm_call_s_total = 0.0
                llm_cache_read_s_total = 0.0
                crop_encode_s_total = 0.0

                cache_base = Path(args.cache_dir) / llm_name / f"{args.mode}_{args.seg_type}_K{K}" / f"p{prompt_hash}"
                ensure_dir(cache_base)

                instances: List[Dict[str, Any]] = []
                per_instance_diag: List[Dict[str, Any]] = []

                for i, c in enumerate(selected):
                    iid = inst_ids[i]
                    m = c["mask"].astype(bool)

                    with timer() as t_crop:
                        crop = crop_with_mask(rgb, m, pad=0.12)
                        crop_b64 = b64_png(crop)
                    crop_encode_s = float(t_crop["s"])
                    crop_encode_s_total += crop_encode_s

                    cpath = cache_base / f"{img_path.stem}_{iid}_p{prompt_hash}.json"

                    used_cache = False
                    llm_call_s = 0.0

                    if cpath.exists():
                        used_cache = True
                        with timer() as t_cache:
                            out = json.loads(cpath.read_text(encoding="utf-8"))
                        llm_cached += 1
                        llm_cache_read_s_total += float(t_cache["s"])
                    else:
                        with timer() as t_call:
                            try:
                                out = client.complete_json(
                                    system=sys_p,
                                    user=usr_p,
                                    images_b64png=[full_b64, crop_b64],
                                )
                            except Exception as e:
                                out = {"object_name": "object", "per_action": {}, "error": str(e)}
                        llm_call_s = float(t_call["s"])
                        llm_called += 1
                        llm_call_s_total += llm_call_s
                        cpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

                    obj_name = str(out.get("object_name", "object")).strip()[:80]
                    pa = out.get("per_action", {}) or {}

                    acts: Dict[str, Any] = {}
                    for a in args.actions:
                        aout = pa.get(a, {}) or {}
                        label = str(aout.get("relationship_label", "Firmly Negative")).strip()
                        if label not in REL_CATEGORIES:
                            label = "Firmly Negative"

                        entry: Dict[str, Any] = {
                            "relationship_label": label,
                            "relationship_code": REL_CODEMAP[label],
                        }

                        # add optional selection diagnostics
                        if use_saliency:
                            entry["mask_score"] = float(c.get("mask_score", 0.0))

                        if label in EXCEPTION_CATEGORIES:
                            entry["explanation"] = str(aout.get("explanation", "")).strip() or \
                                "The action is not appropriate in the current scene."
                            entry["consequence"] = str(aout.get("consequence", "")).strip() or \
                                "Taking the action may cause a negative outcome."
                        else:
                            entry["explanation"] = ""
                            entry["consequence"] = ""

                        acts[a] = entry

                    bbox = mask_to_bbox(m)
                    area_px = int(m.sum())

                    inst_record = {
                        "instance_id": iid,
                        "object_name": obj_name,
                        "bbox_xyxy": list(bbox),
                        "area": area_px,
                        "actions": acts,
                        # dataset provenance (optional, helpful for paper)
                        "seg_source": str(c.get("source", "")),
                    }
                    if "class_id" in c:
                        inst_record["class_id"] = int(c["class_id"])
                    if "instance_id" in c:
                        inst_record["dataset_instance_id"] = int(c["instance_id"])

                    instances.append(inst_record)

                    per_instance_diag.append({
                        "instance_id": iid,
                        "bbox_xyxy": list(bbox),
                        "mask_area_px": area_px,
                        "mask_area_ratio": float(area_px) / float(H * W),
                        "crop_encode_s": crop_encode_s,
                        "llm_used_cache": used_cache,
                        "llm_call_s": llm_call_s,
                        "class_id": int(c.get("class_id", -1)) if "class_id" in c else None,
                        "dataset_instance_id": int(c.get("instance_id")) if "instance_id" in c else None,
                        "mask_score": float(c.get("mask_score", 0.0)) if use_saliency else None,
                        "seg_source": str(c.get("source", "")),
                    })

                # Main output
                out_json = {
                    "image": str(img_path),
                    "mode": args.mode,
                    "seg_type": args.seg_type,
                    "K": K,
                    "actions": args.actions,
                    "relationship_categories": REL_CATEGORIES,
                    "relationship_code_map": REL_CODEMAP,
                    "instances": instances,
                }
                (img_out / f"{img_path.stem}_{llm_name}_instances.json").write_text(
                    json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                # Timings + diagnostics (paper-grade)
                timings = {
                    "image": str(img_path),
                    "mode": args.mode,
                    "seg_type": args.seg_type,
                    "K": K,
                    "llm": llm_name,
                    "timestamp_utc": now_utc_iso(),
                    "seed": args.seed,
                    "prompt_hash": prompt_hash,
                    "image_shape": {"H": H, "W": W},

                    "dataset_segmentation": {
                        "mask_path": str(mask_path),
                        "semantic_vote_path": str((semantic_masks_dir / mask_path.name)) if (args.seg_type == "instance" and semantic_masks_dir is not None) else "",
                        "extract_candidates_s": candidates_s,
                        "num_candidates_total": int(num_candidates_total),
                        "num_candidates_kept": int(num_candidates_kept),
                        "num_selected": int(len(selected)),
                        "background_id": int(args.background_id),
                        "ignore_id": int(args.ignore_id),
                        "min_region_area": int(args.min_region_area),
                        "max_candidates": int(args.max_candidates),
                    },

                    "saliency": {
                        "enabled": bool(use_saliency),
                        "adapter": str(args.adapter) if use_saliency else "",
                        "seen_ckpt": str(args.seen_ckpt) if use_saliency else "",
                        "unseen_ckpt": str(args.unseen_ckpt) if use_saliency else "",
                        "min_mask_score": float(args.min_mask_score),
                        "saliency_total_s": float(saliency_total_s) if use_saliency else 0.0,
                        "saliency_per_action_s": saliency_per_action_s if use_saliency else {},
                        "fallbacks_to_unseen": int(saliency_fallbacks_to_unseen) if use_saliency else 0,
                    },

                    "llm_stats": {
                        "llm_called": int(llm_called),
                        "llm_cached": int(llm_cached),
                        "llm_call_s_total": float(llm_call_s_total),
                        "llm_cache_read_s_total": float(llm_cache_read_s_total),
                        "avg_llm_call_s": float(llm_call_s_total) / float(max(llm_called, 1)),
                        "crop_encode_s_total": float(crop_encode_s_total),
                        "full_b64_s": float(full_b64_s),
                    },

                    "instances": per_instance_diag,
                }
                (img_out / f"{img_path.stem}_{llm_name}_timings.json").write_text(
                    json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                # Convenience outputs
                rel_lines = []
                exco = {a: {} for a in args.actions}
                for inst in instances:
                    iid = inst["instance_id"]
                    codes = [str(inst["actions"][a]["relationship_code"]) for a in args.actions]
                    rel_lines.append(f"{iid} # " + " # ".join(codes))
                    for a in args.actions:
                        info = inst["actions"][a]
                        if info["relationship_label"] in EXCEPTION_CATEGORIES:
                            exco[a][iid] = {
                                "explanation": info.get("explanation", ""),
                                "consequence": info.get("consequence", ""),
                            }
                exco = {a: d for a, d in exco.items() if d}

                (img_out / f"{img_path.stem}_{llm_name}_relationship.txt").write_text(
                    "\n".join(rel_lines) + "\n", encoding="utf-8"
                )
                (img_out / f"{img_path.stem}_{llm_name}_exco.json").write_text(
                    json.dumps(exco, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    print(f"[OK] Finished. Outputs under: {outdir}")


if __name__ == "__main__":
    main()
