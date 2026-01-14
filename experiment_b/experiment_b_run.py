#!/usr/bin/env python3
"""
Experiment B Runner (SAM-only vs SAM+Saliency), revised to match the paper.

This script produces *only predictions* (no ground-truth evaluation). Evaluation is done
offline by experiment_b_eval_metrics.py from the saved JSON outputs.

Key design choices (locked for the IJCAI paper):
  - Two regimes:
      (1) sam           : SAM-only region proposals
      (2) sam_saliency  : SAM proposals ranked by affordance saliency overlap (selection only)
  - SAME downstream LLM prompting in both regimes:
      - LLM sees: full image + object crop (no saliency overlay), to isolate the effect of region selection.
  - Region budgets: K in {5, 10, 20}. You can run all in one call via --Ks 5 10 20.
  - Outputs are written to: <outdir>/<mode>_K<k>/<image_stem>/...

Outputs per image and per LLM:
  - <stem>_<llm>_instances.json      (main output consumed by evaluator)
  - <stem>_<llm>_relationship.txt    (ADE-style convenience dump)
  - <stem>_<llm>_exco.json           (only exception entries)
  - <stem>_overlay.png              (SAM mask overlay for quick inspection)

NOTE:
  - This script assumes the LLM endpoints support vision inputs.
  - If a model is configured with supports_vision=false in llms.json, it will be skipped.

# only SAM
python experiment_b_run.py \
  --images_dir /path/to/images \
  --outdir /path/to/out_b \
  --llms /path/to/llms_updated.json \
  --mode sam \
  --Ks 5 10 20 \
  --sam_ckpt /path/to/sam_vit_h_4b8939.pth \
  --device cuda

  # SAM + Saliency
python experiment_b_run.py \
--images_dir /path/to/images \
--outdir /path/to/out_b \
--llms /path/to/llms_updated.json \
--mode sam_saliency \
--Ks 5 10 20 \
--sam_ckpt /path/to/sam_vit_h_4b8939.pth \
--device cuda \
--adapter /path/to/your_adapter.py \
--seen_ckpt /path/to/seen.ckpt \
--unseen_ckpt /path/to/unseen.ckpt


"""

import io
import json
import base64
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from vision_llm_clients import load_llms, make_client

# Optional saliency adapter (your affordance saliency models)
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
    """
    Crop a padded bounding box around the binary mask.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = mask_to_bbox(mask)
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px); y1 = min(h - 1, y1 + py)
    return Image.fromarray(rgb[y0:y1+1, x0:x1+1])

def overlay_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    """
    Simple random-color mask overlay for inspection.
    """
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
def load_sam_mask_generator(ckpt: str, sam_type: str, device: str):
    """
    Loads SAM automatic mask generator. Requires `segment_anything`.
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[sam_type](checkpoint=ckpt)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=256,
    )


# ----------------------------
# Prompts (strict JSON)
# ----------------------------
def system_prompt(actions: List[str]) -> str:
    return f"""You are a careful vision-language reasoning system for action-object affordances.

For each action in {actions}, output ONE relationship label from this fixed set:
{REL_CATEGORIES}

Rules:
- Return STRICT JSON only.
- relationship_label must be exactly one of the fixed labels.
- If relationship_label is an exception type (one of {sorted(list(EXCEPTION_CATEGORIES))}),
  include short grounded 'explanation' and 'consequence' (one sentence each).
- Otherwise omit explanation/consequence.

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

def user_prompt(actions: List[str]) -> str:
    return f"""Analyze the object in the crop using the full image for context.
Return JSON for:
- object_name
- per_action for actions: {", ".join(actions)}
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
    for a, S in sal_maps.items():
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
                    help="How many raw SAM proposals to generate before selecting top-K.")

    # Saliency (only for sam_saliency)
    ap.add_argument("--adapter", default="", help="Path to affordance adapter .py (required for sam_saliency)")
    ap.add_argument("--seen_ckpt", default="")
    ap.add_argument("--unseen_ckpt", default="")
    ap.add_argument("--min_mask_score", type=float, default=0.0,
                    help="Optional threshold on mask saliency score before top-K selection.")

    # Caching
    ap.add_argument("--cache_dir", default="cache_b", help="Cache LLM outputs to avoid re-calls")

    args = ap.parse_args()

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

    # SAM
    mask_gen = load_sam_mask_generator(args.sam_ckpt, args.sam_type, args.device)

    # Saliency models if needed
    use_saliency = (args.mode == "sam_saliency")
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

    for k in Ks:
        run_root = outdir / f"{args.mode}_K{k}"
        ensure_dir(run_root)

        for img_path in tqdm(img_paths, desc=f"Experiment B ({args.mode}, K={k})"):
            pil = Image.open(img_path).convert("RGB")
            rgb = np.array(pil)

            # 1) Raw SAM proposals
            raw = mask_gen.generate(rgb)
            raw = sorted(raw, key=lambda d: int(d.get("area", 0)), reverse=True)[:args.max_sam_masks]
            raw_masks = [m["segmentation"].astype(bool) for m in raw]

            # 2) Optional saliency ranking (selection only)
            mask_scores: List[float] = [0.0 for _ in raw_masks]
            if use_saliency:
                assert adapter is not None and model_seen is not None and model_unseen is not None
                sal_maps: Dict[str, np.ndarray] = {}
                for a in args.actions:
                    s = np.asarray(adapter.predict_saliency(model_seen, pil, a), dtype=np.float32)
                    # If adapter returns incompatible shape for seen, try unseen checkpoint
                    if s.shape != raw_masks[0].shape:
                        s = np.asarray(adapter.predict_saliency(model_unseen, pil, a), dtype=np.float32)
                    sal_maps[a] = normalize01(s)

                for i, m in enumerate(raw_masks):
                    mask_scores[i] = score_mask_by_saliency(m, sal_maps)

                # Apply optional threshold before top-K
                keep = [(m, sc) for (m, sc) in zip(raw_masks, mask_scores) if sc >= args.min_mask_score]
                keep.sort(key=lambda x: x[1], reverse=True)
                keep = keep[:k]
                masks = [m for (m, _) in keep]
                scores = [float(sc) for (_, sc) in keep]
            else:
                masks = raw_masks[:k]
                scores = [None for _ in masks]  # type: ignore

            if not masks:
                # still create dir, but skip if no masks
                img_out = run_root / img_path.stem
                ensure_dir(img_out)
                continue

            # 3) Save overlay for quick inspection
            img_out = run_root / img_path.stem
            ensure_dir(img_out)
            overlay = overlay_masks(rgb, masks)
            overlay.save(img_out / f"{img_path.stem}_overlay.png")

            # Precompute full image b64
            full_b64 = b64_png(pil)

            # 4) Query LLMs for each selected region (same prompt in both regimes)
            inst_ids = [f"{i:03d}" for i in range(len(masks))]

            for llm_name, client in llms:
                cache_base = Path(args.cache_dir) / llm_name / f"{args.mode}_K{k}"
                ensure_dir(cache_base)

                instances = []
                for i, m in enumerate(masks):
                    iid = inst_ids[i]
                    crop = crop_with_mask(rgb, m, pad=0.12)
                    crop_b64 = b64_png(crop)

                    # cache key depends on image, iid, mode, and K
                    cpath = cache_base / f"{img_path.stem}_{iid}.json"
                    if cpath.exists():
                        out = json.loads(cpath.read_text(encoding="utf-8"))
                    else:
                        out = client.complete_json(
                            system=system_prompt(args.actions),
                            user=user_prompt(args.actions),
                            images_b64png=[full_b64, crop_b64],
                        )
                        cpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

                    obj_name = str(out.get("object_name", "object")).strip()[:80]
                    pa = out.get("per_action", {}) or {}

                    acts = {}
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
                            entry["mask_score"] = float(scores[i])  # same for all actions (selection score)
                        if label in EXCEPTION_CATEGORIES:
                            entry["explanation"] = str(aout.get("explanation", "")).strip() or \
                                "The action is not appropriate in the current scene."
                            entry["consequence"] = str(aout.get("consequence", "")).strip() or \
                                "Taking the action may cause a negative outcome."
                        acts[a] = entry

                    instances.append({
                        "instance_id": iid,
                        "object_name": obj_name,
                        "bbox_xyxy": list(mask_to_bbox(m)),
                        "area": int(m.sum()),
                        "actions": acts,
                    })

                # 5) Save main JSON
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

                # 6) Convenience: ADE-style relationship.txt and exco.json
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

        # small run marker
        (run_root / "run_config.json").write_text(
            json.dumps({
                "mode": args.mode,
                "K": k,
                "actions": args.actions,
                "sam_type": args.sam_type,
                "max_sam_masks": args.max_sam_masks,
                "saliency_enabled": bool(use_saliency),
            }, indent=2),
            encoding="utf-8"
        )

    print(f"[OK] Finished. Outputs under: {outdir}")


if __name__ == "__main__":
    main()
