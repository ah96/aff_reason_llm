#!/usr/bin/env python3
"""
Experiment B Runner (SAM-only vs SAM+Saliency), paper-grade logging.

Drop-in replacement for your current experiment_b_run.py:
- Same main outputs consumed by experiment_b_eval_metrics.py
- Adds robust logging (run config, timing, counts, per-instance diagnostics)
- Keeps your dependencies: segment_anything, vision_llm_clients, adapter.load_model/predict_saliency

Outputs per image and per LLM:
  - <stem>_<llm>_instances.json      (main output consumed by evaluator)
  - <stem>_<llm>_relationship.txt    (ADE-style convenience dump)
  - <stem>_<llm>_exco.json           (only exception entries)
  - <stem>_overlay.png              (SAM mask overlay for quick inspection)
  - <stem>_<llm>_timings.json        (paper-grade timing + diagnostics)

Run-level output per <mode>_K<k> folder:
  - run_config.json
"""

import io
import os
import json
import base64
import hashlib
import argparse
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from vision_llm_clients import load_llms, make_client

# Optional torch (only for seeding + device metadata)
try:
    import torch  # type: ignore
except Exception:
    torch = None


# ----------------------------
# Optional saliency adapter (your affordance saliency models)
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


@contextmanager
def timer():
    t0 = time.perf_counter()
    out = {"s": 0.0}
    try:
        yield out
    finally:
        out["s"] = float(time.perf_counter() - t0)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


# ----------------------------
# Relationship taxonomy (fixed)
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
EXCEPTION_CATEGORIES = set(REL_CATEGORIES[2:])  # codes 2..6


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    """Crop a padded bounding box around the binary mask."""
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
    """Simple random-color mask overlay for inspection."""
    out = rgb.astype(np.float32).copy()
    for i, m in enumerate(masks):
        m = m.astype(bool)
        rng = np.random.default_rng(i + 123)
        color = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ----------------------------
# SAM loader
# ----------------------------
def load_sam_mask_generator(ckpt: str, sam_type: str, device: str, sam_params: Dict[str, Any]):
    """Loads SAM automatic mask generator. Requires `segment_anything`."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[sam_type](checkpoint=ckpt)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(model=sam, **sam_params)


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
# Saliency scoring (mask selection only)
# ----------------------------
def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)


def score_mask_by_saliency(mask: np.ndarray, sal_maps: Dict[str, np.ndarray]) -> float:
    """
    Implements the paper scoring:
      score(mask) = max_a mean( S_a over pixels in mask )
    """
    if mask.sum() == 0:
        return 0.0
    best = 0.0
    for _, S in sal_maps.items():
        best = max(best, float(np.mean(S[mask])))
    return best


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_dir", required=True, help="Folder with RGB images to process.")
    ap.add_argument("--outdir", required=True, help="Output folder.")
    ap.add_argument("--llms", required=True, help="Path to llms.json")

    # Locked regimes
    ap.add_argument("--mode", choices=["sam", "sam_saliency"], default="sam")

    # Region budgets (run all in one go)
    ap.add_argument("--Ks", nargs="+", type=int, default=[5, 10, 20], help="Region budgets K to evaluate.")

    # Actions
    ap.add_argument("--actions", nargs="+", default=["sit", "run", "grasp"])

    # SAM
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_sam_masks", type=int, default=50,
                    help="How many raw SAM proposals to keep before selecting top-K.")

    # Saliency (only for sam_saliency)
    ap.add_argument("--adapter", default="", help="Path to affordance adapter .py (required for sam_saliency)")
    ap.add_argument("--seen_ckpt", default="")
    ap.add_argument("--unseen_ckpt", default="")
    ap.add_argument("--min_mask_score", type=float, default=0.0,
                    help="Optional threshold on mask saliency score before top-K selection.")

    # Caching
    ap.add_argument("--cache_dir", default="cache_b", help="Cache LLM outputs to avoid re-calls")

    # Repro / logging
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Seeding (best-effort)
    np.random.seed(args.seed)
    if torch is not None:
        try:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    images_dir = Path(args.images_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # LLMs
    llm_cfgs = load_llms(args.llms)
    llms = []
    for cfg in llm_cfgs:
        if not getattr(cfg, "supports_vision", True):
            continue
        llms.append((cfg.name, make_client(cfg)))
    if not llms:
        raise RuntimeError("No vision-capable LLMs configured. Set supports_vision=true in llms.json.")

    # SAM params (locked for the paper; log verbatim)
    SAM_PARAMS: Dict[str, Any] = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 256,
    }
    mask_gen = load_sam_mask_generator(args.sam_ckpt, args.sam_type, args.device, SAM_PARAMS)

    # Saliency models if needed
    use_saliency = args.mode == "sam_saliency"
    adapter = None
    model_seen = None
    model_unseen = None
    if use_saliency:
        if not args.adapter:
            raise RuntimeError("--adapter is required for mode sam_saliency")
        adapter = load_adapter(args.adapter)
        model_seen = adapter.load_model(args.seen_ckpt, args.device)
        model_unseen = adapter.load_model(args.unseen_ckpt, args.device)

    # Enumerate images
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    # Validate K values
    Ks = sorted(set(int(k) for k in args.Ks))
    for k in Ks:
        if k <= 0:
            raise ValueError("All K values must be > 0.")

    # Precompute prompt hash (prompt-safe caching)
    sys_p = system_prompt(args.actions)
    usr_p = user_prompt(args.actions)
    prompt_hash = sha1_short(sys_p + "\n" + usr_p + "\n" + ",".join(args.actions), n=10)

    # Per-K runs
    for k in Ks:
        run_root = outdir / f"{args.mode}_K{k}"
        ensure_dir(run_root)

        # Run config (per mode/K)
        run_config = {
            "timestamp_utc": now_utc_iso(),
            "mode": args.mode,
            "K": k,
            "Ks_all": Ks,
            "actions": args.actions,
            "seed": args.seed,
            "prompt_hash": prompt_hash,
            "sam_type": args.sam_type,
            "sam_ckpt": str(args.sam_ckpt),
            "sam_params": SAM_PARAMS,
            "device": args.device,
            "max_sam_masks": int(args.max_sam_masks),
            "saliency_enabled": bool(use_saliency),
            "adapter": str(args.adapter) if use_saliency else "",
            "seen_ckpt": str(args.seen_ckpt) if use_saliency else "",
            "unseen_ckpt": str(args.unseen_ckpt) if use_saliency else "",
            "min_mask_score": float(args.min_mask_score),
            "llms": [name for name, _ in llms],
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

        for img_path in tqdm(img_paths, desc=f"Experiment B ({args.mode}, K={k})"):
            pil = Image.open(img_path).convert("RGB")
            rgb = np.array(pil)
            H, W = rgb.shape[0], rgb.shape[1]

            img_out = run_root / img_path.stem
            ensure_dir(img_out)

            # 1) Raw SAM proposals
            with timer() as t_sam:
                raw = mask_gen.generate(rgb)
            sam_generate_s = float(t_sam["s"])

            num_raw_masks_total = len(raw)
            raw = sorted(raw, key=lambda d: int(d.get("area", 0)), reverse=True)
            raw_areas = [int(d.get("area", 0)) for d in raw]
            raw = raw[: args.max_sam_masks]
            num_raw_masks_kept = len(raw)

            raw_masks = [m["segmentation"].astype(bool) for m in raw]
            if args.verbose:
                print(f"[SAM] {img_path.name}: total={num_raw_masks_total} kept={num_raw_masks_kept} gen_s={sam_generate_s:.3f}")

            # 2) Optional saliency ranking (selection only)
            scores: List[Optional[float]]
            kept_areas: List[int]
            saliency_total_s = 0.0
            saliency_per_action_s: Dict[str, float] = {}
            saliency_fallbacks_to_unseen = 0

            if use_saliency:
                assert adapter is not None and model_seen is not None and model_unseen is not None

                # If SAM returned nothing, skip safely
                if not raw_masks:
                    (img_out / f"{img_path.stem}_nomasks.json").write_text(
                        json.dumps(
                            {
                                "image": str(img_path),
                                "mode": args.mode,
                                "K": k,
                                "sam_generate_s": sam_generate_s,
                                "num_raw_masks_total": num_raw_masks_total,
                                "num_raw_masks_kept": num_raw_masks_kept,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    continue

                with timer() as t_sal_total:
                    sal_maps: Dict[str, np.ndarray] = {}
                    for a in args.actions:
                        with timer() as t_a:
                            s = np.asarray(adapter.predict_saliency(model_seen, pil, a), dtype=np.float32)
                            if s.shape != raw_masks[0].shape:
                                saliency_fallbacks_to_unseen += 1
                                s = np.asarray(adapter.predict_saliency(model_unseen, pil, a), dtype=np.float32)
                        saliency_per_action_s[a] = float(t_a["s"])
                        sal_maps[a] = normalize01(s)
                saliency_total_s = float(t_sal_total["s"])

                mask_scores = [score_mask_by_saliency(m, sal_maps) for m in raw_masks]
                keep = [(m, sc, ar) for (m, sc, ar) in zip(raw_masks, mask_scores, raw_areas[:len(raw_masks)]) if sc >= args.min_mask_score]
                keep.sort(key=lambda x: x[1], reverse=True)
                keep = keep[:k]
                masks = [m for (m, _, _) in keep]
                scores = [float(sc) for (_, sc, _) in keep]
                kept_areas = [int(ar) for (_, _, ar) in keep]
            else:
                masks = raw_masks[:k]
                scores = [None for _ in masks]
                kept_areas = raw_areas[: len(masks)]

            num_selected_masks = len(masks)

            # If no masks, just write marker and continue
            if not masks:
                (img_out / f"{img_path.stem}_nomasks.json").write_text(
                    json.dumps(
                        {
                            "image": str(img_path),
                            "mode": args.mode,
                            "K": k,
                            "timestamp_utc": now_utc_iso(),
                            "seed": args.seed,
                            "prompt_hash": prompt_hash,
                            "sam": {
                                "sam_type": args.sam_type,
                                "sam_ckpt": str(args.sam_ckpt),
                                "sam_params": SAM_PARAMS,
                                "device": args.device,
                                "max_sam_masks": int(args.max_sam_masks),
                                "sam_generate_s": sam_generate_s,
                                "num_raw_masks_total": num_raw_masks_total,
                                "num_raw_masks_kept": num_raw_masks_kept,
                                "num_selected_masks": 0,
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

            # 3) Save overlay for quick inspection
            overlay = overlay_masks(rgb, masks)
            overlay.save(img_out / f"{img_path.stem}_overlay.png")

            # Full image b64 (timed)
            with timer() as t_full:
                full_b64 = b64_png(pil)
            full_b64_s = float(t_full["s"])

            # 4) Query LLMs for each selected region
            inst_ids = [f"{i:03d}" for i in range(len(masks))]

            for llm_name, client in llms:
                llm_cached = 0
                llm_called = 0
                llm_call_s_total = 0.0
                llm_cache_read_s_total = 0.0
                crop_encode_s_total = 0.0

                # Cache base includes prompt hash (prevents stale collisions)
                cache_base = Path(args.cache_dir) / llm_name / f"{args.mode}_K{k}" / f"p{prompt_hash}"
                ensure_dir(cache_base)

                instances: List[Dict[str, Any]] = []
                per_instance_diag: List[Dict[str, Any]] = []

                for i, m in enumerate(masks):
                    iid = inst_ids[i]

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

                        if use_saliency:
                            entry["mask_score"] = float(scores[i]) if scores[i] is not None else 0.0

                        if label in EXCEPTION_CATEGORIES:
                            entry["explanation"] = str(aout.get("explanation", "")).strip() or \
                                "The action is not appropriate in the current scene."
                            entry["consequence"] = str(aout.get("consequence", "")).strip() or \
                                "Taking the action may cause a negative outcome."
                        else:
                            # enforce empty strings per your schema
                            entry["explanation"] = ""
                            entry["consequence"] = ""

                        acts[a] = entry

                    bbox = mask_to_bbox(m)
                    area_px = int(m.sum())

                    instances.append({
                        "instance_id": iid,
                        "object_name": obj_name,
                        "bbox_xyxy": list(bbox),
                        "area": area_px,
                        "actions": acts,
                    })

                    per_instance_diag.append({
                        "instance_id": iid,
                        "bbox_xyxy": list(bbox),
                        "mask_area_px": area_px,
                        "mask_area_ratio": float(area_px) / float(H * W),
                        "sam_area_reported": int(kept_areas[i]) if i < len(kept_areas) else None,
                        "crop_encode_s": crop_encode_s,
                        "llm_used_cache": used_cache,
                        "llm_call_s": llm_call_s,
                    })

                # 5) Save main JSON (unchanged)
                out_json = {
                    "image": str(img_path),
                    "mode": args.mode,
                    "K": k,
                    "actions": args.actions,
                    "relationship_categories": REL_CATEGORIES,
                    "relationship_code_map": REL_CODEMAP,
                    "instances": instances,
                }
                (img_out / f"{img_path.stem}_{llm_name}_instances.json").write_text(
                    json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                # 6) Save timings + diagnostics (paper-grade)
                timings = {
                    "image": str(img_path),
                    "mode": args.mode,
                    "K": k,
                    "llm": llm_name,
                    "timestamp_utc": now_utc_iso(),
                    "seed": args.seed,
                    "prompt_hash": prompt_hash,
                    "image_shape": {"H": H, "W": W},

                    "sam": {
                        "sam_type": args.sam_type,
                        "sam_ckpt": str(args.sam_ckpt),
                        "sam_params": SAM_PARAMS,
                        "device": args.device,
                        "max_sam_masks": int(args.max_sam_masks),
                        "sam_generate_s": sam_generate_s,
                        "num_raw_masks_total": int(num_raw_masks_total),
                        "num_raw_masks_kept": int(num_raw_masks_kept),
                        "num_selected_masks": int(num_selected_masks),
                        "raw_area_stats": {
                            "mean": float(np.mean(raw_areas)) if raw_areas else 0.0,
                            "median": float(np.median(raw_areas)) if raw_areas else 0.0,
                            "max": int(np.max(raw_areas)) if raw_areas else 0,
                        },
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

                # 7) Convenience dumps (unchanged)
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
