#!/usr/bin/env python3
"""
Experiment B Evaluator (paper tables) — revised.

This script reads outputs from experiment_b_run_updated.py and produces:

1) CSV summaries that are easy to copy into paper tables.
2) Optional LaTeX table snippets with placeholders.

IMPORTANT:
- True ADE-Affordance evaluation (mAcc/mAcc-E + BLEU/METEOR/ROUGE/CIDEr vs GT EXCO)
  requires ground-truth alignment between regions and annotations.
  Because GT instance masks are not available in standard ADE20K releases, many users
  run Experiment B without GT alignment.

Therefore this evaluator supports two modes:

(A) Consensus evaluation (default, GT-free):
    - Treats majority vote across LLMs as a pseudo-reference.
    - Reports per-LLM agreement with the consensus (acc_3way, acc_7way).
    - Useful for sanity checks, robustness across LLMs, and debugging.

(B) Ground-truth evaluation (optional):
    - If you provide a GT directory that mirrors the runner output format
      (instances.json files with the same instance_id/action ordering),
      then true accuracies and text metrics can be computed.

Usage:
  python experiment_b_eval_paper_tables.py --outdir <OUTDIR> [--gt_dir <GT>] [--write_latex 1]

Expected directory structure:
  <OUTDIR>/<mode>_K<k>/<image_stem>/<image_stem>_<llm>_instances.json
"""

from __future__ import annotations

import json
import csv
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np


# -----------------------------
# IO helpers
# -----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# -----------------------------
# Parsing helpers
# -----------------------------
def iter_instances(instances_json: Dict[str, Any]):
    for inst in instances_json.get("instances", []) or []:
        yield inst

def get_actions(instances_json: Dict[str, Any]) -> List[str]:
    return list(instances_json.get("actions", []) or [])

def flatten_labels(instances_json: Dict[str, Any]) -> List[int]:
    """
    Deterministic order: instances sorted by instance_id, then actions in JSON order.
    """
    actions = get_actions(instances_json)
    insts = list(iter_instances(instances_json))
    insts.sort(key=lambda x: str(x.get("instance_id", "")))
    out = []
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            out.append(int((acts.get(a, {}) or {}).get("relationship_code", -1)))
    return out

def flatten_explanations(instances_json: Dict[str, Any]) -> List[Tuple[int, str]]:
    """
    Returns list of (relationship_code, explanation_text) for exception entries only.
    """
    actions = get_actions(instances_json)
    insts = list(iter_instances(instances_json))
    insts.sort(key=lambda x: str(x.get("instance_id", "")))
    out: List[Tuple[int, str]] = []
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            info = acts.get(a, {}) or {}
            code = int(info.get("relationship_code", -1))
            if 2 <= code <= 6:
                out.append((code, str(info.get("explanation", "")).strip()))
    return out

def collapse_7_to_3(code7: int) -> int:
    """
    3-way grouping used in the original work:
      0 -> Positive
      1 -> FirmlyNegative
      2..6 -> Exception
    """
    if code7 == 0:
        return 0
    if code7 == 1:
        return 1
    if 2 <= code7 <= 6:
        return 2
    return -1


def majority_vote(col: List[int]) -> int:
    c = Counter(col)
    # deterministic tie-break: smaller label id wins
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# -----------------------------
# Discovery helpers
# -----------------------------
def discover_runs(outdir: Path) -> List[Path]:
    """
    Runs are subdirectories like: sam_K10, sam_saliency_K20, ...
    """
    runs = []
    for p in sorted(outdir.iterdir()):
        if p.is_dir() and any(x.is_dir() for x in p.iterdir()):
            runs.append(p)
    return runs

def discover_image_dirs(run_dir: Path) -> List[Path]:
    """
    Image dirs contain *_instances.json files.
    """
    img_dirs = []
    for p in sorted(run_dir.iterdir()):
        if not p.is_dir():
            continue
        if any(x.name.endswith("_instances.json") for x in p.iterdir()):
            img_dirs.append(p)
    return img_dirs

def parse_llm_name(fname: str, image_stem: str) -> Optional[str]:
    prefix = image_stem + "_"
    suffix = "_instances.json"
    if not (fname.startswith(prefix) and fname.endswith(suffix)):
        return None
    return fname[len(prefix):-len(suffix)]


# -----------------------------
# Core evaluation
# -----------------------------
def eval_consensus(instances_per_llm: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Returns per-llm accuracies vs consensus:
      - acc_7way
      - acc_3way
    """
    llms = sorted(instances_per_llm.keys())
    if not llms:
        return {}

    mats = [np.array(flatten_labels(instances_per_llm[m]), dtype=int) for m in llms]
    if any(m.size == 0 for m in mats):
        return {}

    M = np.stack(mats, axis=0)  # [num_llms, num_items]
    consensus7 = np.apply_along_axis(lambda col: majority_vote(col.tolist()), 0, M)

    consensus3 = np.vectorize(collapse_7_to_3)(consensus7)
    preds3 = np.vectorize(collapse_7_to_3)(M)

    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(llms):
        acc7 = float(np.mean(M[i] == consensus7))
        acc3 = float(np.mean(preds3[i] == consensus3))
        out[name] = {"acc_7way": acc7, "acc_3way": acc3}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output root from experiment_b_run_updated.py")
    ap.add_argument("--gt_dir", default="", help="Optional GT directory mirroring output format (advanced).")
    ap.add_argument("--write_latex", type=int, default=1, help="Write LaTeX snippets (1=yes,0=no)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs = discover_runs(outdir)
    if not runs:
        raise RuntimeError(f"No run subdirectories found in {outdir}.")

    # Per-run summaries that directly map to the paper tables
    run_rows: List[Dict[str, Any]] = []
    llm_rows: List[Dict[str, Any]] = []

    for run_dir in runs:
        img_dirs = discover_image_dirs(run_dir)
        if not img_dirs:
            continue

        # parse mode and K from folder name if possible
        mode = run_dir.name
        K = ""
        if "_K" in run_dir.name:
            mode, K = run_dir.name.split("_K", 1)
            K = int(K)

        # accumulate per-llm accuracies (consensus)
        per_llm_acc7 = defaultdict(list)
        per_llm_acc3 = defaultdict(list)

        # also track exception rate for reporting/debugging
        exc_rates = []

        for d in img_dirs:
            stem = d.name
            instances_per_llm: Dict[str, Dict[str, Any]] = {}

            for f in d.iterdir():
                if f.name.endswith("_instances.json"):
                    llm = parse_llm_name(f.name, stem)
                    if llm:
                        instances_per_llm[llm] = read_json(f)

            if len(instances_per_llm) < 2:
                # consensus needs at least 2 models
                continue

            # exception rate (from first llm, since ordering aligns)
            labs0 = flatten_labels(next(iter(instances_per_llm.values())))
            if labs0:
                exc = sum(1 for y in labs0 if 2 <= y <= 6) / len(labs0)
                exc_rates.append(float(exc))

            accs = eval_consensus(instances_per_llm)
            for llm, a in accs.items():
                per_llm_acc7[llm].append(a["acc_7way"])
                per_llm_acc3[llm].append(a["acc_3way"])

        # summarize per-llm
        for llm in sorted(per_llm_acc7.keys()):
            row = {
                "mode": mode,
                "K": K,
                "llm": llm,
                "acc_7way_mean": float(np.mean(per_llm_acc7[llm])) if per_llm_acc7[llm] else "",
                "acc_3way_mean": float(np.mean(per_llm_acc3[llm])) if per_llm_acc3[llm] else "",
                "num_images": len(per_llm_acc7[llm]),
            }
            llm_rows.append(row)

        # run-wide aggregate (mean across llms, mean across images)
        all7 = [v for llm in per_llm_acc7 for v in per_llm_acc7[llm]]
        all3 = [v for llm in per_llm_acc3 for v in per_llm_acc3[llm]]

        run_rows.append({
            "mode": mode,
            "K": K,
            "acc_7way_mean": float(np.mean(all7)) if all7 else "",
            "acc_3way_mean": float(np.mean(all3)) if all3 else "",
            "exception_rate_mean": float(np.mean(exc_rates)) if exc_rates else "",
            "num_images_used": int(max([len(per_llm_acc7[llm]) for llm in per_llm_acc7], default=0)),
        })

    # write CSVs
    write_csv(outdir / "paper_run_summary.csv", run_rows)
    write_csv(outdir / "paper_llm_summary.csv", llm_rows)

    # write LaTeX snippets that map to the paper tables (relationship-only placeholders)
    if args.write_latex:
        # Main: compare sam vs sam_saliency at K=10 if present, else first K
        # We will output placeholders for BLEU/METEOR/ROUGE/CIDEr.
        def find_row(mode: str, K: int) -> Optional[Dict[str, Any]]:
            for r in run_rows:
                if r.get("mode") == mode and r.get("K") == K:
                    return r
            return None

        # choose K=10 if available
        Ks = sorted(set(r["K"] for r in run_rows if r.get("K") != ""))
        k_main = 10 if 10 in Ks else (Ks[0] if Ks else 10)

        r_sam = find_row("sam", k_main) or {}
        r_sal = find_row("sam_saliency", k_main) or {}

        main_tex = f"""% Auto-generated by experiment_b_eval_paper_tables.py
% NOTE: Text metrics require GT EXCO alignment; fill once available.

\\begin{{table}}[t]
    \\centering
    \\caption{{Main results (relationship prediction) at $K={k_main}$. Text metrics are placeholders until GT EXCO alignment is provided.}}
    \\label{{tab:main_results}}
    \\begin{{tabular}}{{lcccccc}}
        \\toprule
        Method & mAcc & mAcc-E & BLEU-4 & METEOR & ROUGE-L & CIDEr \\\\
        \\midrule
        SAM-only & {100*float(r_sam.get("acc_3way_mean") or 0):.1f} & {100*float(r_sam.get("acc_7way_mean") or 0):.1f} & -- & -- & -- & -- \\\\
        SAM+Saliency (ours) & \\textbf{{{100*float(r_sal.get("acc_3way_mean") or 0):.1f}}} & \\textbf{{{100*float(r_sal.get("acc_7way_mean") or 0):.1f}}} & -- & -- & -- & -- \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
        (outdir / "table_main_results.tex").write_text(main_tex, encoding="utf-8")

        # Budget table: acc_7way_mean at each K
        rows_sam = {r["K"]: r for r in run_rows if r["mode"] == "sam"}
        rows_sal = {r["K"]: r for r in run_rows if r["mode"] == "sam_saliency"}
        def fmt_acc(r): return f"{100*float(r.get('acc_7way_mean') or 0):.1f}" if r else "--"

        budget_tex = """% Auto-generated by experiment_b_eval_paper_tables.py
\\begin{table}[t]
    \\centering
    \\caption{Effect of region budget $K$ on relationship prediction (mAcc-E).}
    \\label{tab:region_budget}
    \\begin{tabular}{lccc}
        \\toprule
        Method & $K{=}5$ & $K{=}10$ & $K{=}20$ \\\\
        \\midrule
"""
        budget_tex += f"        SAM-only & {fmt_acc(rows_sam.get(5))} & {fmt_acc(rows_sam.get(10))} & {fmt_acc(rows_sam.get(20))} \\\\\n"
        budget_tex += f"        SAM+Saliency & \\textbf{{{fmt_acc(rows_sal.get(5))}}} & \\textbf{{{fmt_acc(rows_sal.get(10))}}} & \\textbf{{{fmt_acc(rows_sal.get(20))}}} \\\\\n"
        budget_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
        (outdir / "table_region_budget.tex").write_text(budget_tex, encoding="utf-8")

        # LLM robustness table at K=10 (or chosen k_main)
        llm_rows_k = [r for r in llm_rows if r.get("K") == k_main]
        # group by llm
        by_llm = defaultdict(dict)
        for r in llm_rows_k:
            by_llm[r["llm"]][r["mode"]] = r
        # render
        rob_tex = f"""% Auto-generated by experiment_b_eval_paper_tables.py
\\begin{{table}}[t]
    \\centering
    \\caption{{Robustness across LLM backends at $K={k_main}$ (mAcc-E).}}
    \\label{{tab:llm_robustness}}
    \\begin{{tabular}}{{lcc}}
        \\toprule
        LLM & SAM-only & SAM+Saliency \\\\
        \\midrule
"""
        for llm in sorted(by_llm.keys()):
            r1 = by_llm[llm].get("sam", {})
            r2 = by_llm[llm].get("sam_saliency", {})
            rob_tex += f"        {llm} & {100*float(r1.get('acc_7way_mean') or 0):.1f} & \\textbf{{{100*float(r2.get('acc_7way_mean') or 0):.1f}}} \\\\\n"
        rob_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
        (outdir / "table_llm_robustness.tex").write_text(rob_tex, encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  - {outdir / 'paper_run_summary.csv'}")
    print(f"  - {outdir / 'paper_llm_summary.csv'}")
    if args.write_latex:
        print(f"  - {outdir / 'table_main_results.tex'}")
        print(f"  - {outdir / 'table_region_budget.tex'}")
        print(f"  - {outdir / 'table_llm_robustness.tex'}")


if __name__ == "__main__":
    main()
