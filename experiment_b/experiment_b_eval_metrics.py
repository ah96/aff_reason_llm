#!/usr/bin/env python3
"""
Experiment B evaluation (JSON-only).

Reads per-image outputs produced by experiment_b_run.py:
  - <outdir>/<image_stem>/<image_stem>_<llm>_instances.json
  - <outdir>/<image_stem>/<image_stem>_<llm>_faithfulness.json (optional)

Writes:
  - <outdir>/metrics_llms.csv      (row = image × llm)
  - <outdir>/metrics_images.csv    (row = image, aggregated across llms)
  - <outdir>/metrics_run.json      (run-wide aggregates)

Design goal:
  - No access to images
  - No re-running SAM or LLM calls
  - Easy to extend metrics later via registry
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
from collections import defaultdict, Counter

import numpy as np


# -----------------------------
# Helpers: IO
# -----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # stable header ordering
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# -----------------------------
# Helpers: parsing instances
# -----------------------------
def iter_instances(instances_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for inst in instances_json.get("instances", []) or []:
        yield inst

def get_actions(instances_json: Dict[str, Any]) -> List[str]:
    return list(instances_json.get("actions", []) or [])

def flatten_labels(instances_json: Dict[str, Any]) -> List[int]:
    """
    Returns list of relationship_code for each instance-action in deterministic order:
      instances sorted by instance_id, then actions in instances_json['actions'] order.
    """
    actions = get_actions(instances_json)
    insts = list(iter_instances(instances_json))
    insts.sort(key=lambda x: x.get("instance_id", ""))
    out = []
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            out.append(int((acts.get(a, {}) or {}).get("relationship_code", -1)))
    return out

def flatten_pairs(instances_json: Dict[str, Any]) -> List[Tuple[str, str]]:
    actions = get_actions(instances_json)
    insts = list(iter_instances(instances_json))
    insts.sort(key=lambda x: x.get("instance_id", ""))
    keys = []
    for inst in insts:
        iid = str(inst.get("instance_id", ""))
        for a in actions:
            keys.append((iid, a))
    return keys

def label_distribution(instances_json: Dict[str, Any]) -> Dict[int, int]:
    labels = flatten_labels(instances_json)
    c = Counter(labels)
    return dict(sorted(c.items(), key=lambda kv: kv[0]))

def exception_rate(instances_json: Dict[str, Any]) -> float:
    """
    Exception = relationship_code in {2..6} per your taxonomy.
    """
    labels = flatten_labels(instances_json)
    if not labels:
        return 0.0
    exc = sum(1 for y in labels if 2 <= y <= 6)
    return float(exc / len(labels))

def selected_rate_saliency(instances_json: Dict[str, Any]) -> Optional[float]:
    """
    Only available for sam_saliency: each action entry may have selected_for_llm bool.
    Returns fraction of instance-action entries that were selected.
    """
    insts = list(iter_instances(instances_json))
    if not insts:
        return None
    actions = get_actions(instances_json)
    total = 0
    sel = 0
    any_field = False
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            total += 1
            info = acts.get(a, {}) or {}
            if "selected_for_llm" in info:
                any_field = True
                if bool(info.get("selected_for_llm")):
                    sel += 1
    if not any_field or total == 0:
        return None
    return float(sel / total)

def avg_saliency_score(instances_json: Dict[str, Any]) -> Optional[float]:
    """
    Only available for sam_saliency: action entries may have 'score' float.
    Returns mean score over all instance-action entries.
    """
    insts = list(iter_instances(instances_json))
    if not insts:
        return None
    actions = get_actions(instances_json)
    scores = []
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            info = acts.get(a, {}) or {}
            if "score" in info:
                try:
                    scores.append(float(info["score"]))
                except Exception:
                    pass
    if not scores:
        return None
    return float(np.mean(scores))


# -----------------------------
# Helpers: parsing faithfulness
# -----------------------------
def faithfulness_flip_rate(faith_json: Dict[str, Any]) -> Optional[float]:
    v = faith_json.get("flip_rate", None)
    return None if v is None else float(v)

def faithfulness_gap_high_low(faith_json: Dict[str, Any]) -> Optional[float]:
    hi = faith_json.get("high_flip_rate", None)
    lo = faith_json.get("low_flip_rate", None)
    if hi is None or lo is None:
        return None
    return float(hi) - float(lo)

def directional_flip_stats(faith_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Directional stats: how flips move relative to "Positive" (code 0).
    This is alignment-free but more interpretable:
      - pos_to_nonpos_rate: among pairs with orig==0, fraction pert!=0
      - nonpos_to_pos_rate: among pairs with orig!=0, fraction pert==0
    """
    pairs = faith_json.get("pairs_tested", []) or []
    if not pairs:
        return {
            "pos_to_nonpos_rate": None,
            "nonpos_to_pos_rate": None,
            "n_pos": 0,
            "n_nonpos": 0,
        }

    pos = [p for p in pairs if int(p.get("orig_code", -1)) == 0]
    nonpos = [p for p in pairs if int(p.get("orig_code", -1)) != 0]

    def rate_pos_to_nonpos(ps):
        if not ps:
            return None
        return float(sum(1 for p in ps if int(p.get("pert_code", -1)) != 0) / len(ps))

    def rate_nonpos_to_pos(ps):
        if not ps:
            return None
        return float(sum(1 for p in ps if int(p.get("pert_code", -1)) == 0) / len(ps))

    return {
        "pos_to_nonpos_rate": rate_pos_to_nonpos(pos),
        "nonpos_to_pos_rate": rate_nonpos_to_pos(nonpos),
        "n_pos": len(pos),
        "n_nonpos": len(nonpos),
    }


# -----------------------------
# Agreement metrics across LLMs (JSON-only)
# -----------------------------
def majority_vote(labels: List[int]) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[int(np.argmax(counts))])

def agreement_rate_across_llms(instances_per_llm: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """
    For a single image: instances_per_llm = {llm_name: instances.json_data}
    Returns average fraction of models agreeing with majority per item.
    """
    if not instances_per_llm:
        return None
    llms = list(instances_per_llm.keys())
    keys = flatten_pairs(instances_per_llm[llms[0]])
    if not keys:
        return None

    # build matrix [num_llms, num_items]
    mat = []
    for llm in llms:
        mat.append(flatten_labels(instances_per_llm[llm]))
    mat = np.array(mat, dtype=int)
    if mat.size == 0:
        return None

    fracs = []
    for j in range(mat.shape[1]):
        col = mat[:, j].tolist()
        maj = majority_vote(col)
        fracs.append(sum(1 for x in col if x == maj) / len(col))
    return float(np.mean(fracs))


# -----------------------------
# Metric registry
# -----------------------------
MetricFn = Callable[[Dict[str, Any]], Dict[str, Any]]

@dataclass
class Metric:
    name: str
    fn: MetricFn
    scope: str  # "llm" or "image"


METRICS: List[Metric] = []

def register_metric(name: str, scope: str):
    def deco(fn: MetricFn):
        METRICS.append(Metric(name=name, fn=fn, scope=scope))
        return fn
    return deco


# -----------------------------
# LLM-level metrics (computed from one instances.json and optional faithfulness.json)
# -----------------------------
@register_metric("label_dist", scope="llm")
def metric_label_dist(ctx: Dict[str, Any]) -> Dict[str, Any]:
    inst = ctx["instances_json"]
    dist = label_distribution(inst)
    # flatten into columns dist_0 .. dist_6 (and dist_-1 if present)
    out = {}
    for k, v in dist.items():
        out[f"dist_{k}"] = v
    return out

@register_metric("exception_rate", scope="llm")
def metric_exception_rate(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"exception_rate": exception_rate(ctx["instances_json"])}

@register_metric("saliency_selected_rate", scope="llm")
def metric_saliency_selected_rate(ctx: Dict[str, Any]) -> Dict[str, Any]:
    v = selected_rate_saliency(ctx["instances_json"])
    return {"selected_rate": v if v is not None else ""}

@register_metric("saliency_avg_score", scope="llm")
def metric_saliency_avg_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    v = avg_saliency_score(ctx["instances_json"])
    return {"avg_saliency_score": v if v is not None else ""}

@register_metric("faithfulness_flip_rate", scope="llm")
def metric_faith_flip(ctx: Dict[str, Any]) -> Dict[str, Any]:
    fj = ctx.get("faith_json", None)
    return {"faith_flip_rate": (faithfulness_flip_rate(fj) if fj else "")}

@register_metric("faithfulness_gap", scope="llm")
def metric_faith_gap(ctx: Dict[str, Any]) -> Dict[str, Any]:
    fj = ctx.get("faith_json", None)
    return {"faith_gap_high_minus_low": (faithfulness_gap_high_low(fj) if fj else "")}

@register_metric("faithfulness_directional", scope="llm")
def metric_faith_dir(ctx: Dict[str, Any]) -> Dict[str, Any]:
    fj = ctx.get("faith_json", None)
    if not fj:
        return {
            "pos_to_nonpos_rate": "",
            "nonpos_to_pos_rate": "",
            "n_pos_faith": "",
            "n_nonpos_faith": "",
        }
    st = directional_flip_stats(fj)
    return {
        "pos_to_nonpos_rate": st["pos_to_nonpos_rate"] if st["pos_to_nonpos_rate"] is not None else "",
        "nonpos_to_pos_rate": st["nonpos_to_pos_rate"] if st["nonpos_to_pos_rate"] is not None else "",
        "n_pos_faith": st["n_pos"],
        "n_nonpos_faith": st["n_nonpos"],
    }


# -----------------------------
# Image-level metrics (computed across all LLMs for an image)
# -----------------------------
@register_metric("agreement", scope="image")
def metric_agreement(ctx: Dict[str, Any]) -> Dict[str, Any]:
    insts = ctx["instances_per_llm"]
    ar = agreement_rate_across_llms(insts)
    return {"agreement_rate": ar if ar is not None else ""}

@register_metric("faithfulness_avg_across_llms", scope="image")
def metric_faith_avg(ctx: Dict[str, Any]) -> Dict[str, Any]:
    faith_per_llm = ctx.get("faith_per_llm", {})
    vals = []
    gaps = []
    for llm, fj in faith_per_llm.items():
        fr = faithfulness_flip_rate(fj)
        if fr is not None:
            vals.append(fr)
        gg = faithfulness_gap_high_low(fj)
        if gg is not None:
            gaps.append(gg)

    out = {}
    out["faith_flip_rate_avg"] = float(np.mean(vals)) if vals else ""
    out["faith_gap_high_minus_low_avg"] = float(np.mean(gaps)) if gaps else ""
    return out


# -----------------------------
# Discovery: find files
# -----------------------------
def discover_image_dirs(outdir: Path) -> List[Path]:
    """
    Image directories are immediate children that contain *_instances.json.
    """
    img_dirs = []
    for p in sorted(outdir.iterdir()):
        if not p.is_dir():
            continue
        if any(x.name.endswith("_instances.json") for x in p.iterdir()):
            img_dirs.append(p)
    return img_dirs

def parse_llm_name_from_instances_file(fname: str, image_stem: str) -> Optional[str]:
    """
    Expected: <stem>_<llm>_instances.json
    """
    prefix = image_stem + "_"
    suffix = "_instances.json"
    if not (fname.startswith(prefix) and fname.endswith(suffix)):
        return None
    return fname[len(prefix):-len(suffix)]

def parse_llm_name_from_faith_file(fname: str, image_stem: str) -> Optional[str]:
    """
    Expected: <stem>_<llm>_faithfulness.json
    """
    prefix = image_stem + "_"
    suffix = "_faithfulness.json"
    if not (fname.startswith(prefix) and fname.endswith(suffix)):
        return None
    return fname[len(prefix):-len(suffix)]


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Experiment B output dir produced by experiment_b_run.py")
    ap.add_argument("--write_prefix", default="metrics", help="Output filename prefix (default: metrics)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    img_dirs = discover_image_dirs(outdir)
    if not img_dirs:
        raise RuntimeError(f"No image directories found in {outdir}. Did you run experiment_b_run.py?")

    llm_rows: List[Dict[str, Any]] = []
    img_rows: List[Dict[str, Any]] = []

    # For run-wide aggregation
    run_accum = {
        "num_images": 0,
        "num_image_llm_pairs": 0,
        "agreement_rates": [],
        "faith_flip_rates": [],
        "faith_gaps": [],
        "exception_rates": [],
        "selected_rates": [],
    }

    for d in img_dirs:
        image_stem = d.name

        # load all llm instances.json for this image
        instances_per_llm: Dict[str, Dict[str, Any]] = {}
        faith_per_llm: Dict[str, Dict[str, Any]] = {}

        for f in d.iterdir():
            if f.name.endswith("_instances.json"):
                llm = parse_llm_name_from_instances_file(f.name, image_stem)
                if llm:
                    instances_per_llm[llm] = read_json(f)

            if f.name.endswith("_faithfulness.json"):
                llm = parse_llm_name_from_faith_file(f.name, image_stem)
                if llm:
                    faith_per_llm[llm] = read_json(f)

        if not instances_per_llm:
            continue

        # --------------------------
        # LLM-level rows
        # --------------------------
        for llm_name, inst_json in instances_per_llm.items():
            faith_json = faith_per_llm.get(llm_name, None)

            row = {
                "image": image_stem,
                "llm": llm_name,
                "mode": inst_json.get("mode", ""),
                "num_instances": len(inst_json.get("instances", []) or []),
                "num_actions": len(inst_json.get("actions", []) or []),
            }

            ctx = {"instances_json": inst_json, "faith_json": faith_json}
            for m in METRICS:
                if m.scope != "llm":
                    continue
                vals = m.fn(ctx)
                # prefix metric name only if necessary; here we just merge
                row.update(vals)

            llm_rows.append(row)

            # accumulate run-wide stats from llm-level metrics
            run_accum["num_image_llm_pairs"] += 1
            # exception rate
            er = exception_rate(inst_json)
            run_accum["exception_rates"].append(er)
            # selected rate
            sr = selected_rate_saliency(inst_json)
            if sr is not None:
                run_accum["selected_rates"].append(sr)
            # faithfulness
            if faith_json:
                fr = faithfulness_flip_rate(faith_json)
                if fr is not None:
                    run_accum["faith_flip_rates"].append(fr)
                gg = faithfulness_gap_high_low(faith_json)
                if gg is not None:
                    run_accum["faith_gaps"].append(gg)

        # --------------------------
        # Image-level row
        # --------------------------
        img_row = {
            "image": image_stem,
            "mode": list(instances_per_llm.values())[0].get("mode", ""),
            "num_llms": len(instances_per_llm),
        }

        ctx_img = {"instances_per_llm": instances_per_llm, "faith_per_llm": faith_per_llm}
        for m in METRICS:
            if m.scope != "image":
                continue
            img_row.update(m.fn(ctx_img))

        img_rows.append(img_row)
        run_accum["num_images"] += 1

        # agreement accumulation
        ar = agreement_rate_across_llms(instances_per_llm)
        if ar is not None:
            run_accum["agreement_rates"].append(ar)

    # --------------------------
    # Write outputs
    # --------------------------
    llm_csv = outdir / f"{args.write_prefix}_llms.csv"
    img_csv = outdir / f"{args.write_prefix}_images.csv"
    run_json = outdir / f"{args.write_prefix}_run.json"

    write_csv(llm_csv, llm_rows)
    write_csv(img_csv, img_rows)

    run_report = {
        "num_images": run_accum["num_images"],
        "num_image_llm_pairs": run_accum["num_image_llm_pairs"],
        "agreement_rate_mean": float(np.mean(run_accum["agreement_rates"])) if run_accum["agreement_rates"] else None,
        "exception_rate_mean": float(np.mean(run_accum["exception_rates"])) if run_accum["exception_rates"] else None,
        "selected_rate_mean": float(np.mean(run_accum["selected_rates"])) if run_accum["selected_rates"] else None,
        "faith_flip_rate_mean": float(np.mean(run_accum["faith_flip_rates"])) if run_accum["faith_flip_rates"] else None,
        "faith_gap_high_minus_low_mean": float(np.mean(run_accum["faith_gaps"])) if run_accum["faith_gaps"] else None,
        "notes": "All metrics computed from JSON outputs only. Add new metrics by registering functions in METRICS registry.",
    }
    write_json(run_json, run_report)

    print(f"[OK] wrote: {llm_csv}")
    print(f"[OK] wrote: {img_csv}")
    print(f"[OK] wrote: {run_json}")


if __name__ == "__main__":
    main()
