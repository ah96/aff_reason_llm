import os
import re
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
from experiment_b_metrics import agreement_rate, saliency_positive_gap

# NEW: for perturbations
import cv2

# ----------------------------
# Relationship taxonomy (same as your pipelines)
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
DEFAULT_REL_CODEMAP = {name: i for i, name in enumerate(REL_CATEGORIES)}
EXCEPTION_CATEGORIES = set(REL_CATEGORIES[2:])

# ----------------------------
# Utils
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64_png(img: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(img)).decode("utf-8")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

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
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(w - 1, x1 + px); y1 = min(h - 1, y1 + py)
    return Image.fromarray(rgb[y0:y1+1, x0:x1+1])

def overlay_masks(rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    h, w, _ = out.shape
    for i, m in enumerate(masks):
        m = m.astype(bool)
        rng = np.random.default_rng(i + 123)
        color = rng.integers(0, 255, size=(3,), dtype=np.int32).astype(np.float32)
        for ch in range(3):
            out[..., ch][m] = (1 - alpha) * out[..., ch][m] + alpha * color[ch]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def saliency_overlay(rgb: np.ndarray, sal: np.ndarray, mask: np.ndarray) -> Image.Image:
    sal = sal.astype(np.float32)
    smin, smax = float(sal.min()), float(sal.max())
    if smax > smin:
        sal = (sal - smin) / (smax - smin)
    else:
        sal = np.zeros_like(sal)
    heat = np.zeros_like(rgb, dtype=np.float32)
    heat[..., 0] = sal * 255.0
    heat[..., 1] = (sal ** 0.5) * 80.0
    heat[..., 2] = (1.0 - sal) * 40.0

    out = rgb.astype(np.float32).copy()
    alpha = 0.45
    m = mask.astype(bool)
    for ch in range(3):
        out[..., ch][m] = (1 - alpha) * out[..., ch][m] + alpha * heat[..., ch][m]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

# ----------------------------
# Faithfulness perturbations
# ----------------------------
def perturb_rgb(rgb: np.ndarray, mask: np.ndarray, method: str = "inpaint") -> np.ndarray:
    """
    Perturb the pixels inside `mask` in the full image.
    method:
      - "gray": fill with mean color
      - "blur": strong blur within bbox
      - "inpaint": OpenCV Telea inpainting (recommended default)
    """
    out = rgb.copy()
    m = mask.astype(bool)
    if not m.any():
        return out

    if method == "gray":
        mean_color = out.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        out[m] = mean_color
        return out

    if method == "blur":
        x0, y0, x1, y1 = mask_to_bbox(mask)
        roi = out[y0:y1+1, x0:x1+1].copy()
        if roi.size == 0:
            return out
        roi_blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=12, sigmaY=12)
        mroi = mask[y0:y1+1, x0:x1+1].astype(bool)
        roi[mroi] = roi_blur[mroi]
        out[y0:y1+1, x0:x1+1] = roi
        return out

    if method == "inpaint":
        mask_u8 = (mask.astype(np.uint8) * 255)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        bgr_inp = cv2.inpaint(bgr, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(bgr_inp, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unknown perturb method: {method}")

# ----------------------------
# SAM
# ----------------------------
def load_sam_mask_generator(ckpt: str, sam_type: str, device: str):
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
# Optional saliency adapter
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

# ----------------------------
# Prompts
# ----------------------------
def system_prompt(actions: List[str], use_saliency: bool) -> str:
    extra = ""
    if use_saliency:
        extra = "You are also given an affordance saliency overlay as evidence."

    return f"""You are a careful vision-language reasoning system for action-object affordances.
{extra}

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
    return f"""Analyze the object instance in the crop using the full image for context.
Return JSON for:
- object_name
- per_action for actions: {", ".join(actions)}
"""

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder with RGB images to process.")
    ap.add_argument("--outdir", required=True, help="Output folder.")
    ap.add_argument("--llms", required=True, help="configs/llms.json")

    # Mode
    ap.add_argument("--mode", choices=["sam", "sam_saliency"], default="sam")

    # Actions
    ap.add_argument("--actions", nargs="+", default=["sit", "run", "grasp"])

    # SAM
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_instances", type=int, default=25)

    # Saliency (only for sam_saliency)
    ap.add_argument("--adapter", default="", help="Path to affordance adapter .py (required for sam_saliency)")
    ap.add_argument("--seen_ckpt", default="")
    ap.add_argument("--unseen_ckpt", default="")
    ap.add_argument("--topk_pairs", type=int, default=20)
    ap.add_argument("--min_pair_score", type=float, default=0.15)

    # Caching
    ap.add_argument("--cache_dir", default="cache_b", help="Cache LLM outputs to avoid re-calls")

    # Faithfulness test (NEW)
    ap.add_argument("--faithfulness", action="store_true",
                    help="If set, runs perturbation-based faithfulness test (extra LLM calls).")
    ap.add_argument("--perturb_method", default="inpaint", choices=["inpaint", "gray", "blur"])
    ap.add_argument("--faith_pairs_per_image", type=int, default=10,
                    help="How many (instance,action) pairs to test per image per LLM (sam mode heuristic).")
    ap.add_argument("--faith_high_low", type=int, default=5,
                    help="In sam_saliency mode: test this many HIGH and LOW saliency pairs per image.")

    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    llm_cfgs = load_llms(args.llms)
    llms = [(cfg.name, make_client(cfg)) for cfg in llm_cfgs]

    # Load SAM
    mask_gen = load_sam_mask_generator(args.sam_ckpt, args.sam_type, args.device)

    # Load saliency models if needed
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

    run_summary = []

    for img_path in tqdm(img_paths, desc="Experiment B images"):
        pil = Image.open(img_path).convert("RGB")
        rgb = np.array(pil)

        # SAM masks
        sam_masks = mask_gen.generate(rgb)
        sam_masks = sorted(sam_masks, key=lambda d: int(d.get("area", 0)), reverse=True)[:args.max_instances]
        masks = [m["segmentation"].astype(bool) for m in sam_masks]
        inst_ids = [f"{i:03d}" for i in range(len(masks))]

        # Save overlay
        overlay = overlay_masks(rgb, masks)
        img_out = outdir / img_path.stem
        ensure_dir(img_out)
        overlay.save(img_out / f"{img_path.stem}_overlay.png")

        full_b64 = b64_png(pil)

        # Prepare saliency scores if enabled
        pair_scores: Dict[Tuple[str, str], float] = {}
        chosen_pairs: List[Tuple[str, str]] = []

        sal_maps = None
        if use_saliency:
            sal_maps = {}
            for a in args.actions:
                s = adapter.predict_saliency(model_seen, pil, a)
                s = np.asarray(s, dtype=np.float32)
                if s.shape != masks[0].shape:
                    s = adapter.predict_saliency(model_unseen, pil, a)
                    s = np.asarray(s, dtype=np.float32)
                smin, smax = float(s.min()), float(s.max())
                if smax > smin:
                    s = (s - smin) / (smax - smin)
                else:
                    s = np.zeros_like(s)
                sal_maps[a] = s

            for i, m in enumerate(masks):
                iid = inst_ids[i]
                for a in args.actions:
                    sc = float(np.mean(sal_maps[a][m])) if m.sum() else 0.0
                    pair_scores[(iid, a)] = sc

            items = [(k, v) for k, v in pair_scores.items() if v >= args.min_pair_score]
            items.sort(key=lambda kv: kv[1], reverse=True)
            chosen_pairs = [k for k, _ in items[:args.topk_pairs]]

        # Decide which (instance,action) pairs to perturb for faithfulness
        all_pairs = [(iid, a) for iid in inst_ids for a in args.actions]
        faith_pairs: List[Tuple[str, str]] = []
        if args.faithfulness and all_pairs:
            if use_saliency:
                scored = [((iid, a), float(pair_scores.get((iid, a), 0.0))) for (iid, a) in all_pairs]
                scored.sort(key=lambda x: x[1], reverse=True)
                k = min(args.faith_high_low, len(scored) // 2) if len(scored) >= 2 else 1
                high = [p for (p, _) in scored[:k]]
                low = [p for (p, _) in scored[-k:]]
                faith_pairs = high + low
            else:
                areas = [(iid, int(masks[int(iid)].sum())) for iid in inst_ids]
                areas.sort(key=lambda x: x[1], reverse=True)
                per_action = max(1, args.faith_pairs_per_image // max(1, len(args.actions)))
                top_iids = [iid for (iid, _) in areas[:per_action]]
                faith_pairs = [(iid, a) for iid in top_iids for a in args.actions]
                faith_pairs = faith_pairs[:args.faith_pairs_per_image]

        # For SAM-only, query every instance once (all actions).
        # For saliency mode, query only chosen pairs (instance-action) and default others.
        per_model_outputs = {}

        for llm_name, client in llms:
            cache_base = Path(args.cache_dir) / llm_name
            ensure_dir(cache_base)

            instances = []

            if not use_saliency:
                # one call per instance for all actions
                for i, m in enumerate(masks):
                    iid = inst_ids[i]
                    crop = crop_with_mask(rgb, m, pad=0.12)
                    crop_b64 = b64_png(crop)

                    cpath = cache_base / f"{img_path.stem}_{iid}.json"
                    if cpath.exists():
                        out = json.loads(cpath.read_text(encoding="utf-8"))
                    else:
                        out = client.complete_json(
                            system=system_prompt(args.actions, use_saliency=False),
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
                        entry = {"relationship_label": label, "relationship_code": DEFAULT_REL_CODEMAP[label]}
                        if label in EXCEPTION_CATEGORIES:
                            entry["explanation"] = str(aout.get("explanation", "")).strip() or "The action is not appropriate in the current scene."
                            entry["consequence"] = str(aout.get("consequence", "")).strip() or "Taking the action may cause a negative outcome."
                        acts[a] = entry

                    instances.append({
                        "instance_id": iid,
                        "object_name": obj_name,
                        "bbox_xyxy": list(mask_to_bbox(m)),
                        "area": int(m.sum()),
                        "actions": acts,
                    })

            else:
                # initialize defaults for all instance-action
                defaults = {}
                for i, m in enumerate(masks):
                    iid = inst_ids[i]
                    defaults[iid] = {
                        "instance_id": iid,
                        "object_name": "object",
                        "bbox_xyxy": list(mask_to_bbox(m)),
                        "area": int(m.sum()),
                        "actions": {
                            a: {
                                "relationship_label": "Firmly Negative",
                                "relationship_code": DEFAULT_REL_CODEMAP["Firmly Negative"],
                                "score": float(pair_scores.get((iid, a), 0.0)),
                                "selected_for_llm": False,
                            }
                            for a in args.actions
                        },
                    }

                # query only selected pairs
                for (iid, a) in chosen_pairs:
                    m = masks[int(iid)]
                    crop = crop_with_mask(rgb, m, pad=0.12)
                    crop_b64 = b64_png(crop)

                    # overlay for this action
                    assert sal_maps is not None
                    s = sal_maps[a]
                    ov = saliency_overlay(rgb, s, m)
                    x0, y0, x1, y1 = mask_to_bbox(m)
                    ov_crop = Image.fromarray(np.array(ov)[y0:y1+1, x0:x1+1])
                    ov_b64 = b64_png(ov_crop)

                    cpath = cache_base / f"{img_path.stem}_{iid}_{a}.json"
                    if cpath.exists():
                        out = json.loads(cpath.read_text(encoding="utf-8"))
                    else:
                        out = client.complete_json(
                            system=system_prompt([a], use_saliency=True),
                            user=user_prompt([a]),
                            images_b64png=[full_b64, crop_b64, ov_b64],
                        )
                        cpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

                    obj_name = str(out.get("object_name", "object")).strip()[:80]
                    pa = out.get("per_action", {}) or {}
                    aout = pa.get(a, {}) or {}
                    label = str(aout.get("relationship_label", "Firmly Negative")).strip()
                    if label not in REL_CATEGORIES:
                        label = "Firmly Negative"

                    defaults[iid]["object_name"] = obj_name
                    defaults[iid]["actions"][a]["relationship_label"] = label
                    defaults[iid]["actions"][a]["relationship_code"] = DEFAULT_REL_CODEMAP[label]
                    defaults[iid]["actions"][a]["selected_for_llm"] = True

                    if label in EXCEPTION_CATEGORIES:
                        defaults[iid]["actions"][a]["explanation"] = str(aout.get("explanation", "")).strip() or "The action is not appropriate in the current scene."
                        defaults[iid]["actions"][a]["consequence"] = str(aout.get("consequence", "")).strip() or "Taking the action may cause a negative outcome."

                instances = [defaults[i] for i in inst_ids]

            # Save per-model outputs
            out_json = {
                "image": str(img_path),
                "mode": args.mode,
                "actions": args.actions,
                "relationship_categories": REL_CATEGORIES,
                "relationship_code_map": DEFAULT_REL_CODEMAP,
                "instances": instances,
            }
            (img_out / f"{img_path.stem}_{llm_name}_instances.json").write_text(
                json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Save ADE-style relationship.txt and exco.json per model
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

            (img_out / f"{img_path.stem}_{llm_name}_relationship.txt").write_text("\n".join(rel_lines) + "\n", encoding="utf-8")
            (img_out / f"{img_path.stem}_{llm_name}_exco.json").write_text(json.dumps(exco, indent=2, ensure_ascii=False), encoding="utf-8")

            per_model_outputs[llm_name] = instances

            # ----------------------------
            # Faithfulness test (NEW)
            # ----------------------------
            faith_report = {
                "enabled": bool(args.faithfulness),
                "mode": args.mode,
                "perturb_method": args.perturb_method,
                "pairs_tested": [],
                "flip_rate": None,
                "high_flip_rate": None,
                "low_flip_rate": None,
            }

            if args.faithfulness and faith_pairs:
                # map instance_id -> instance record for original predictions
                idx_inst = {inst["instance_id"]: inst for inst in instances}

                flips = []
                for (iid, a) in faith_pairs:
                    i_int = int(iid)
                    m = masks[i_int]

                    # original prediction code
                    y = int(idx_inst[iid]["actions"][a]["relationship_code"])

                    # perturbed full + crop
                    rgb_pert = perturb_rgb(rgb, m, method=args.perturb_method)
                    pil_pert = Image.fromarray(rgb_pert)
                    crop_pert = crop_with_mask(rgb_pert, m, pad=0.12)

                    # cache for faith calls
                    cpath = cache_base / f"{img_path.stem}_faith_{iid}_{a}_{args.perturb_method}.json"
                    if cpath.exists():
                        outp = json.loads(cpath.read_text(encoding="utf-8"))
                    else:
                        outp = client.complete_json(
                            system=system_prompt([a], use_saliency=False),
                            user=user_prompt([a]),
                            images_b64png=[b64_png(pil_pert), b64_png(crop_pert)],
                        )
                        cpath.write_text(json.dumps(outp, indent=2, ensure_ascii=False), encoding="utf-8")

                    pa = outp.get("per_action", {}) or {}
                    aout = pa.get(a, {}) or {}
                    label = str(aout.get("relationship_label", "Firmly Negative")).strip()
                    if label not in REL_CATEGORIES:
                        label = "Firmly Negative"
                    y2 = int(DEFAULT_REL_CODEMAP[label])

                    flip = int(y2 != y)
                    flips.append(flip)

                    entry = {
                        "instance_id": iid,
                        "action": a,
                        "orig_code": y,
                        "pert_code": y2,
                        "flip": flip,
                    }
                    if use_saliency:
                        entry["score"] = float(pair_scores.get((iid, a), 0.0))
                    faith_report["pairs_tested"].append(entry)

                if flips:
                    faith_report["flip_rate"] = float(sum(flips) / len(flips))

                # For saliency mode compute median split high/low flip rates
                if use_saliency and faith_report["pairs_tested"]:
                    scores = [p.get("score", 0.0) for p in faith_report["pairs_tested"] if "score" in p]
                    if scores:
                        med = float(np.median(scores))
                        hi = [p["flip"] for p in faith_report["pairs_tested"] if p.get("score", 0.0) >= med]
                        lo = [p["flip"] for p in faith_report["pairs_tested"] if p.get("score", 0.0) < med]
                        if hi:
                            faith_report["high_flip_rate"] = float(sum(hi) / len(hi))
                        if lo:
                            faith_report["low_flip_rate"] = float(sum(lo) / len(lo))

            (img_out / f"{img_path.stem}_{llm_name}_faithfulness.json").write_text(
                json.dumps(faith_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Run-level summary metrics for this image
        labels_per_model = {}
        flat_keys = [(iid, a) for iid in inst_ids for a in args.actions]

        for llm_name, instances in per_model_outputs.items():
            idx = {inst["instance_id"]: inst for inst in instances}
            labs = [int(idx[iid]["actions"][a]["relationship_code"]) for (iid, a) in flat_keys]
            labels_per_model[llm_name] = labs

        agree = agreement_rate(labels_per_model)

        sal_gap = 0.0
        if use_saliency:
            first_llm = list(per_model_outputs.keys())[0]
            pairs = []
            idx = {inst["instance_id"]: inst for inst in per_model_outputs[first_llm]}
            for (iid, a) in flat_keys:
                pairs.append({
                    "score": float(pair_scores.get((iid, a), 0.0)),
                    "relationship_code": int(idx[iid]["actions"][a]["relationship_code"]),
                })
            sal_gap = saliency_positive_gap(pairs)

        # Aggregate faithfulness across LLMs
        avg_flip = None
        avg_hi = None
        avg_lo = None
        if args.faithfulness:
            flips = []
            his = []
            los = []
            for llm_name in per_model_outputs.keys():
                fpath = img_out / f"{img_path.stem}_{llm_name}_faithfulness.json"
                if fpath.exists():
                    fd = json.loads(fpath.read_text(encoding="utf-8"))
                    if fd.get("flip_rate") is not None:
                        flips.append(float(fd["flip_rate"]))
                    if fd.get("high_flip_rate") is not None:
                        his.append(float(fd["high_flip_rate"]))
                    if fd.get("low_flip_rate") is not None:
                        los.append(float(fd["low_flip_rate"]))
            if flips:
                avg_flip = float(np.mean(flips))
            if his:
                avg_hi = float(np.mean(his))
            if los:
                avg_lo = float(np.mean(los))

        run_summary.append({
            "image": img_path.name,
            "mode": args.mode,
            "num_instances": len(inst_ids),
            "num_llms": len(llms),
            "agreement_rate": agree,
            "saliency_positive_gap": sal_gap,
            "faithfulness_flip_rate_avg": avg_flip,
            "faithfulness_flip_rate_high_avg": avg_hi,
            "faithfulness_flip_rate_low_avg": avg_lo,
            "faithfulness_gap_high_minus_low": (avg_hi - avg_lo) if (avg_hi is not None and avg_lo is not None) else None,
            "faith_pairs_tested": len(faith_pairs) if args.faithfulness else 0,
            "perturb_method": args.perturb_method if args.faithfulness else None,
        })

    # Save run summary
    (outdir / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote run summary -> {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
