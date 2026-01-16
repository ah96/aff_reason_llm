#!/usr/bin/env python3
"""experiment_b_eval_extra_metrics.py

Compute additional metrics for Experiment B runs.

This script scans the output directory produced by experiment_b_run.py and
aggregates metrics per (mode, K, llm) as well as per-image.

Expected structure (as produced by experiment_b_run.py):
  <outdir>/<mode>_K<k>/<image_stem>/
      <image_stem>_<llm>_instances.json
      <image_stem>_<llm>_timings.json

Outputs:
  - metrics_extra_summary.csv  (one row per mode/K/llm)
  - metrics_extra_per_image.csv (one row per image/mode/K/llm)
  - metrics_extra_summary.json (same info, nested)

Usage:
  python3 experiment_b_eval_extra_metrics.py --outdir ./out_b
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import experiment_b_metrics_plus as mx


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def find_runs(outdir: Path) -> List[Tuple[str, int, Path]]:
    runs: List[Tuple[str, int, Path]] = []
    for d in sorted(outdir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "_K" not in name:
            continue
        mode, k_str = name.rsplit("_K", 1)
        try:
            k = int(k_str)
        except Exception:
            continue
        runs.append((mode, k, d))
    return runs


def iter_instance_files(run_root: Path):
    # Yields (image_stem, llm, instances_path, timings_path_or_none)
    for img_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
        stem = img_dir.name
        for inst_path in sorted(img_dir.glob(f"{stem}_*_instances.json")):
            # stem_<llm>_instances.json  (llm may contain underscores)
            llm = inst_path.name[len(stem) + 1 : -len("_instances.json")]
            t_path = img_dir / f"{stem}_{llm}_timings.json"
            yield stem, llm, inst_path, (t_path if t_path.exists() else None)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    # stable column order
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Root output directory from experiment_b_run.py")
    ap.add_argument("--summary_csv", default="metrics_extra_summary.csv")
    ap.add_argument("--per_image_csv", default="metrics_extra_per_image.csv")
    ap.add_argument("--summary_json", default="metrics_extra_summary.json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs = find_runs(outdir)
    if not runs:
        raise SystemExit(f"No run folders found under: {outdir}")

    per_image_rows: List[Dict[str, Any]] = []

    # summary accumulators keyed by (mode, K, llm)
    acc: Dict[Tuple[str, int, str], Dict[str, Any]] = defaultdict(lambda: {
        "n_images": 0,
        "n_instances": 0,
        "label_codes": [],
        "label_codes_per_action": defaultdict(list),
        "mask_scores": [],
        "sal_pairs": [],
        "bboxes": [],
        "schema": [],
        "timings": [],
    })

    for mode, k, run_root in runs:
        for stem, llm, inst_path, t_path in iter_instance_files(run_root):
            j = read_json(inst_path)
            instances = j.get("instances", []) or []
            actions = j.get("actions", []) or []
            rel_map = j.get("relationship_code_map", {}) or {}
            num_labels = len(j.get("relationship_categories", []) or [])
            # in runner: exception labels are codes 2..6
            exception_codes = list(range(2, num_labels))

            # Collect label codes
            all_codes: List[int] = []
            codes_by_action: Dict[str, List[int]] = {a: [] for a in actions}
            bboxes: List[List[int]] = []

            sal_pairs: List[Dict[str, Any]] = []
            mask_scores: List[float] = []

            for inst in instances:
                b = inst.get("bbox_xyxy", None)
                if isinstance(b, list) and len(b) == 4:
                    bboxes.append([int(x) for x in b])
                acts = inst.get("actions", {}) or {}
                for a in actions:
                    e = acts.get(a, {}) or {}
                    code = int(e.get("relationship_code", -1))
                    all_codes.append(code)
                    codes_by_action[a].append(code)
                    if "mask_score" in e:
                        sc = float(e.get("mask_score", 0.0))
                        mask_scores.append(sc)
                        sal_pairs.append({"score": sc, "relationship_code": code})

            schema = mx.compute_schema_stats(instances, actions, num_labels, exception_codes=exception_codes)

            # Per-image metrics
            per_img = {
                "mode": mode,
                "K": k,
                "llm": llm,
                "image_stem": stem,
                "n_instances": len(instances),
                "n_actions": len(actions),
                "label_entropy": mx.label_entropy(all_codes, num_labels) if num_labels else 0.0,
                "bbox_pairwise_iou": mx.mean_pairwise_bbox_iou(bboxes),
                "schema_missing_action_entries": schema.missing_action_entries,
                "schema_invalid_label_entries": schema.invalid_label_entries,
                "schema_exception_entries": schema.exception_label_entries,
                "schema_exception_missing_text": schema.exception_missing_text,
                "schema_nonexception_has_text": schema.nonexception_has_text,
                "mask_score_mean": mx.safe_mean(mask_scores),
                "mask_score_std": mx.safe_std(mask_scores),
                "saliency_pos_gap": mx.saliency_positive_gap(sal_pairs) if sal_pairs else 0.0,
                "saliency_auc_like": mx.saliency_auc_like(sal_pairs) if sal_pairs else 0.0,
            }

            # Per-action entropies
            for a, codes in codes_by_action.items():
                per_img[f"entropy.{a}"] = mx.label_entropy(codes, num_labels) if num_labels else 0.0

            # Optional timings
            if t_path is not None:
                tj = read_json(t_path)
                llm_stats = tj.get("llm_stats", {}) or {}
                sam = tj.get("sam", {}) or {}
                sal = tj.get("saliency", {}) or {}
                per_img.update({
                    "sam_generate_s": float(sam.get("sam_generate_s", 0.0)),
                    "num_raw_masks_total": int(sam.get("num_raw_masks_total", 0)),
                    "num_selected_masks": int(sam.get("num_selected_masks", 0)),
                    "saliency_total_s": float(sal.get("saliency_total_s", 0.0)),
                    "llm_called": int(llm_stats.get("llm_called", 0)),
                    "llm_cached": int(llm_stats.get("llm_cached", 0)),
                    "llm_call_s_total": float(llm_stats.get("llm_call_s_total", 0.0)),
                    "avg_llm_call_s": float(llm_stats.get("avg_llm_call_s", 0.0)),
                    "cache_hit_rate": float(llm_stats.get("llm_cached", 0)) / float(max(int(llm_stats.get("llm_called", 0)) + int(llm_stats.get("llm_cached", 0)), 1)),
                })

            per_image_rows.append(per_img)

            key = (mode, k, llm)
            A = acc[key]
            A["n_images"] += 1
            A["n_instances"] += len(instances)
            A["label_codes"].extend(all_codes)
            for a, codes in codes_by_action.items():
                A["label_codes_per_action"][a].extend(codes)
            A["bboxes"].extend(bboxes)
            A["mask_scores"].extend(mask_scores)
            A["sal_pairs"].extend(sal_pairs)
            A["schema"].append(schema)

            if t_path is not None:
                A["timings"].append(read_json(t_path))

    # Build summary rows
    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {}

    for (mode, k, llm), A in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        labels = A["label_codes"]

        # label hist / entropy
        # infer num labels from max label or default 7
        max_code = max(labels) if labels else 6
        num_labels = int(max(max_code + 1, 7))
        hist = mx.label_histogram(labels, num_labels)

        # schema aggregations
        schemas: List[mx.SchemaStats] = A["schema"]
        sch = {
            "missing_action_entries": int(sum(s.missing_action_entries for s in schemas)),
            "invalid_label_entries": int(sum(s.invalid_label_entries for s in schemas)),
            "exception_entries": int(sum(s.exception_label_entries for s in schemas)),
            "exception_missing_text": int(sum(s.exception_missing_text for s in schemas)),
            "nonexception_has_text": int(sum(s.nonexception_has_text for s in schemas)),
        }

        # timings aggregations
        cache_hit_rates: List[float] = []
        avg_llm_calls: List[float] = []
        sam_s: List[float] = []
        sal_s: List[float] = []
        llm_s_total: List[float] = []
        for tj in A["timings"]:
            llm_stats = tj.get("llm_stats", {}) or {}
            sam = tj.get("sam", {}) or {}
            sal = tj.get("saliency", {}) or {}
            called = int(llm_stats.get("llm_called", 0))
            cached = int(llm_stats.get("llm_cached", 0))
            cache_hit_rates.append(cached / float(max(called + cached, 1)))
            avg_llm_calls.append(float(llm_stats.get("avg_llm_call_s", 0.0)))
            sam_s.append(float(sam.get("sam_generate_s", 0.0)))
            sal_s.append(float(sal.get("saliency_total_s", 0.0)))
            llm_s_total.append(float(llm_stats.get("llm_call_s_total", 0.0)))

        row: Dict[str, Any] = {
            "mode": mode,
            "K": k,
            "llm": llm,
            "n_images": int(A["n_images"]),
            "n_instances_total": int(A["n_instances"]),
            "labels_entropy": mx.label_entropy(labels, num_labels),
            "bbox_pairwise_iou_mean": mx.mean_pairwise_bbox_iou(A["bboxes"]),
            "mask_score_mean": mx.safe_mean(A["mask_scores"]),
            "mask_score_std": mx.safe_std(A["mask_scores"]),
            "saliency_pos_gap": mx.saliency_positive_gap(A["sal_pairs"]) if A["sal_pairs"] else 0.0,
            "saliency_auc_like": mx.saliency_auc_like(A["sal_pairs"]) if A["sal_pairs"] else 0.0,
            "cache_hit_rate_mean": mx.safe_mean(cache_hit_rates),
            "avg_llm_call_s_mean": mx.safe_mean(avg_llm_calls),
            "sam_generate_s_mean": mx.safe_mean(sam_s),
            "saliency_total_s_mean": mx.safe_mean(sal_s),
            "llm_call_s_total_mean": mx.safe_mean(llm_s_total),
            **{f"label_frac.{i}": float(hist[i] / max(int(hist.sum()), 1)) for i in range(num_labels)},
            **{f"schema.{k}": v for k, v in sch.items()},
        }

        # Per-action entropies
        for a, codes in A["label_codes_per_action"].items():
            row[f"entropy.{a}"] = mx.label_entropy(codes, num_labels)

        summary_rows.append(row)

        summary_json.setdefault(mode, {}).setdefault(str(k), {})[llm] = row

    # Save
    write_csv(outdir / args.per_image_csv, per_image_rows)
    write_csv(outdir / args.summary_csv, summary_rows)
    (outdir / args.summary_json).write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote: {outdir / args.summary_csv}")
    print(f"[OK] Wrote: {outdir / args.per_image_csv}")
    print(f"[OK] Wrote: {outdir / args.summary_json}")


if __name__ == "__main__":
    main()
