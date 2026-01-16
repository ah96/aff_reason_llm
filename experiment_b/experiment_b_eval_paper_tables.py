#!/usr/bin/env python3
"""experiment_b_eval_paper_tables.py

Experiment B Evaluator (paper tables) — updated.

Reads outputs from the Experiment B runner and produces:
  1) CSV summaries that map directly to paper tables.
  2) LaTeX table snippets (optional).

IMPORTANT about "mAcc" and text metrics:
- True ADE-Affordance evaluation (mAcc/mAcc-E + BLEU/METEOR/ROUGE/CIDEr vs GT EXCO)
  requires aligning each predicted region to GT instances.
- Since ADE20K instance masks are not publicly released in standard distributions,
  many users run Experiment B in a GT-free setting.

Therefore, this script supports:

(A) Consensus evaluation (default, GT-free):
    - Uses majority vote across LLMs as a pseudo-reference.
    - Reports per-LLM agreement with the consensus (acc_3way, acc_7way).
    - In LaTeX tables, these are reported in the mAcc/mAcc-E columns with a note
      that they reflect consensus agreement (not GT accuracy).

(B) Ground-truth evaluation (optional, advanced):
    - Placeholder hook only (not enabled by default).
    - If you provide a GT directory mirroring runner outputs (same ordering),
      you can compute true accuracies and text metrics.

Extra diagnostic metrics:
- If <OUTDIR>/metrics_extra_summary.csv exists (from experiment_b_eval_extra_metrics.py),
  we merge in additional metrics such as schema validity, entropy, redundancy IoU,
  saliency rank score, and efficiency.

Usage:
  python3 experiment_b_eval_paper_tables.py --outdir <OUTDIR> [--write_latex 1]

Expected structure:
  <OUTDIR>/<mode>_K<k>/<image_stem>/<image_stem>_<llm>_instances.json

Outputs (written into <OUTDIR>):
  - paper_run_summary.csv
  - paper_llm_summary.csv
  - table_main_results.tex
  - table_region_budget.tex
  - table_llm_robustness.tex
  - table_diagnostics.tex
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


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _to_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Parsing helpers
# -----------------------------

def iter_instances(instances_json: Dict[str, Any]):
    for inst in instances_json.get("instances", []) or []:
        yield inst


def get_actions(instances_json: Dict[str, Any]) -> List[str]:
    return list(instances_json.get("actions", []) or [])


def flatten_labels(instances_json: Dict[str, Any]) -> List[int]:
    """Deterministic order: instances sorted by instance_id, then actions in JSON order."""
    actions = get_actions(instances_json)
    insts = list(iter_instances(instances_json))
    insts.sort(key=lambda x: str(x.get("instance_id", "")))
    out: List[int] = []
    for inst in insts:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            out.append(_to_int((acts.get(a, {}) or {}).get("relationship_code", -1), -1))
    return out


def collapse_7_to_3(code7: int) -> int:
    """3-way grouping used in the original work:
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
    """Runs are subdirectories like: sam_K10, sam_saliency_K20, ..."""
    runs: List[Path] = []
    for p in sorted(outdir.iterdir()):
        if p.is_dir() and any(x.is_dir() for x in p.iterdir()):
            runs.append(p)
    return runs


def discover_image_dirs(run_dir: Path) -> List[Path]:
    """Image dirs contain *_instances.json files."""
    img_dirs: List[Path] = []
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
    return fname[len(prefix) : -len(suffix)]


# -----------------------------
# Core evaluation (consensus)
# -----------------------------

def eval_consensus(instances_per_llm: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Returns per-LLM accuracies vs consensus:
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


# -----------------------------
# Extra-metric merge
# -----------------------------

def load_extra_metrics(outdir: Path):
    """Loads metrics_extra_summary.csv (if present) and returns maps.

    Returns:
      by_run[(mode, K)] -> dict (aggregated across llms)
      by_llm[(mode, K, llm)] -> dict
    """
    path = outdir / "metrics_extra_summary.csv"
    rows = read_csv_dicts(path)
    by_run: Dict[Tuple[str, int], Dict[str, Any]] = {}
    by_llm: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for r in rows:
        mode = (r.get("mode") or "").strip()
        K = _to_int(r.get("K"), -1)
        llm = (r.get("llm") or "").strip()

        # store full row (string->best effort float)
        rr: Dict[str, Any] = dict(r)
        for k, v in list(rr.items()):
            # keep mode/K/llm as strings/ints
            if k in {"mode", "llm"}:
                continue
            if k == "K":
                rr[k] = K
                continue
            # attempt float
            rr[k] = _to_float(v, default=float("nan")) if isinstance(v, str) else v

        if llm:
            by_llm[(mode, K, llm)] = rr
        else:
            # some summaries might omit llm; treat as run-level
            by_run[(mode, K)] = rr

    # If file includes llm-level only, compute run-level means.
    if rows and not by_run:
        buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for (mode, K, llm), rr in by_llm.items():
            buckets[(mode, K)].append(rr)
        for key, rs in buckets.items():
            agg: Dict[str, Any] = {"mode": key[0], "K": key[1]}
            # numeric mean for shared numeric keys
            keys = set().union(*[set(x.keys()) for x in rs])
            for k in keys:
                if k in {"mode", "K", "llm"}:
                    continue
                vals = [x.get(k) for x in rs]
                vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
                if vals:
                    agg[k] = float(np.mean(vals))
            by_run[key] = agg

    return by_run, by_llm


# -----------------------------
# LaTeX helper
# -----------------------------

def f_pct(x: Any) -> str:
    v = _to_float(x, default=float("nan"))
    if np.isnan(v):
        return "--"
    return f"{100.0 * v:.1f}"


def f_num(x: Any, nd: int = 3) -> str:
    v = _to_float(x, default=float("nan"))
    if np.isnan(v):
        return "--"
    return f"{v:.{nd}f}"


def f_sec(x: Any) -> str:
    v = _to_float(x, default=float("nan"))
    if np.isnan(v):
        return "--"
    # keep as seconds (not %)
    return f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output root from experiment_b_run.py")
    ap.add_argument("--gt_dir", default="", help="Optional GT directory mirroring output format (advanced).")
    ap.add_argument("--write_latex", type=int, default=1, help="Write LaTeX snippets (1=yes,0=no)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs = discover_runs(outdir)
    if not runs:
        raise RuntimeError(f"No run subdirectories found in {outdir}.")

    extra_by_run, extra_by_llm = load_extra_metrics(outdir)

    # Per-run summaries that directly map to the paper tables
    run_rows: List[Dict[str, Any]] = []
    llm_rows: List[Dict[str, Any]] = []

    for run_dir in runs:
        img_dirs = discover_image_dirs(run_dir)
        if not img_dirs:
            continue

        # parse mode and K from folder name if possible
        mode = run_dir.name
        K: int = -1
        if "_K" in run_dir.name:
            mode, k_str = run_dir.name.split("_K", 1)
            K = _to_int(k_str, -1)

        # accumulate per-LLM accuracies (consensus)
        per_llm_acc7 = defaultdict(list)
        per_llm_acc3 = defaultdict(list)
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
                continue  # consensus needs at least 2 LLMs

            labs0 = flatten_labels(next(iter(instances_per_llm.values())))
            if labs0:
                exc = sum(1 for y in labs0 if 2 <= y <= 6) / len(labs0)
                exc_rates.append(float(exc))

            accs = eval_consensus(instances_per_llm)
            for llm, a in accs.items():
                per_llm_acc7[llm].append(a["acc_7way"])
                per_llm_acc3[llm].append(a["acc_3way"])

        # summarize per-LLM
        for llm in sorted(per_llm_acc7.keys()):
            row: Dict[str, Any] = {
                "mode": mode,
                "K": K,
                "llm": llm,
                "acc_7way_mean": float(np.mean(per_llm_acc7[llm])) if per_llm_acc7[llm] else "",
                "acc_3way_mean": float(np.mean(per_llm_acc3[llm])) if per_llm_acc3[llm] else "",
                "num_images": len(per_llm_acc7[llm]),
            }

            # merge in extra metrics if present
            extra = extra_by_llm.get((mode, K, llm))
            if extra:
                for k, v in extra.items():
                    if k in {"mode", "K", "llm"}:
                        continue
                    row[f"extra_{k}"] = v
            llm_rows.append(row)

        # run-wide aggregate (mean across llms, mean across images)
        all7 = [v for llm in per_llm_acc7 for v in per_llm_acc7[llm]]
        all3 = [v for llm in per_llm_acc3 for v in per_llm_acc3[llm]]

        run_row: Dict[str, Any] = {
            "mode": mode,
            "K": K,
            "acc_7way_mean": float(np.mean(all7)) if all7 else "",
            "acc_3way_mean": float(np.mean(all3)) if all3 else "",
            "exception_rate_mean": float(np.mean(exc_rates)) if exc_rates else "",
            "num_images_used": int(max([len(per_llm_acc7[llm]) for llm in per_llm_acc7], default=0)),
        }

        # merge run-level extra metrics
        extra_r = extra_by_run.get((mode, K))
        if extra_r:
            for k, v in extra_r.items():
                if k in {"mode", "K", "llm"}:
                    continue
                run_row[f"extra_{k}"] = v

        run_rows.append(run_row)

    # write CSVs
    write_csv(outdir / "paper_run_summary.csv", run_rows)
    write_csv(outdir / "paper_llm_summary.csv", llm_rows)

    # write LaTeX snippets
    if args.write_latex:
        # choose K=10 if available, else first K
        Ks = sorted({r["K"] for r in run_rows if isinstance(r.get("K"), int) and r.get("K") != -1})
        k_main = 10 if 10 in Ks else (Ks[0] if Ks else 10)

        def find_row(mode: str, K: int) -> Dict[str, Any]:
            for r in run_rows:
                if r.get("mode") == mode and r.get("K") == K:
                    return r
            return {}

        r_sam = find_row("sam", k_main)
        r_sal = find_row("sam_saliency", k_main)

        # Main table: keep original columns but clarify that mAcc/mAcc-E are consensus agreement in GT-free mode
        main_tex = f"""% Auto-generated by experiment_b_eval_paper_tables.py
% NOTE: If you are in GT-free mode, mAcc/mAcc-E below are consensus agreement (majority vote across LLMs),
% not accuracy vs ADE-Affordance ground truth.
% Text metrics require GT EXCO alignment; keep as placeholders until available.

\\begin{{table}}[t]
    \\centering
    \\caption{{Main results at $K={k_main}$. In GT-free mode, mAcc/mAcc-E report agreement with LLM consensus.}}
    \\label{{tab:main_results}}
    \\begin{{tabular}}{{lcccccc}}
        \\toprule
        Method & mAcc & mAcc-E & BLEU-4 & METEOR & ROUGE-L & CIDEr \\
        \\midrule
        SAM-only & {f_pct(r_sam.get('acc_3way_mean'))} & {f_pct(r_sam.get('acc_7way_mean'))} & -- & -- & -- & -- \\
        SAM+Saliency (ours) & \\textbf{{{f_pct(r_sal.get('acc_3way_mean'))}}} & \\textbf{{{f_pct(r_sal.get('acc_7way_mean'))}}} & -- & -- & -- & -- \\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
        (outdir / "table_main_results.tex").write_text(main_tex, encoding="utf-8")

        # Budget table: relationship (mAcc-E) across K
        rows_sam = {r["K"]: r for r in run_rows if r.get("mode") == "sam"}
        rows_sal = {r["K"]: r for r in run_rows if r.get("mode") == "sam_saliency"}

        def fmt_acc7(r: Optional[Dict[str, Any]]) -> str:
            if not r:
                return "--"
            return f_pct(r.get("acc_7way_mean"))

        budget_tex = """% Auto-generated by experiment_b_eval_paper_tables.py
\\begin{table}[t]
    \\centering
    \\caption{Effect of region budget $K$ on relationship prediction (mAcc-E). In GT-free mode this is consensus agreement.}
    \\label{tab:region_budget}
    \\begin{tabular}{lccc}
        \\toprule
        Method & $K{=}5$ & $K{=}10$ & $K{=}20$ \\
        \\midrule
"""
        budget_tex += f"        SAM-only & {fmt_acc7(rows_sam.get(5))} & {fmt_acc7(rows_sam.get(10))} & {fmt_acc7(rows_sam.get(20))} \\\n"
        budget_tex += f"        SAM+Saliency & \\textbf{{{fmt_acc7(rows_sal.get(5))}}} & \\textbf{{{fmt_acc7(rows_sal.get(10))}}} & \\textbf{{{fmt_acc7(rows_sal.get(20))}}} \\\n"
        budget_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
        (outdir / "table_region_budget.tex").write_text(budget_tex, encoding="utf-8")

        # LLM robustness table at k_main
        llm_rows_k = [r for r in llm_rows if r.get("K") == k_main]
        by_llm = defaultdict(dict)
        for r in llm_rows_k:
            by_llm[r["llm"]][r["mode"]] = r

        rob_tex = f"""% Auto-generated by experiment_b_eval_paper_tables.py
\\begin{{table}}[t]
    \\centering
    \\caption{{Robustness across LLM backends at $K={k_main}$ (mAcc-E). In GT-free mode this is consensus agreement.}}
    \\label{{tab:llm_robustness}}
    \\begin{{tabular}}{{lcc}}
        \\toprule
        LLM & SAM-only & SAM+Saliency \\
        \\midrule
"""
        for llm in sorted(by_llm.keys()):
            r1 = by_llm[llm].get("sam", {})
            r2 = by_llm[llm].get("sam_saliency", {})
            rob_tex += f"        {llm} & {f_pct(r1.get('acc_7way_mean'))} & \\textbf{{{f_pct(r2.get('acc_7way_mean'))}}} \\\n"
        rob_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
        (outdir / "table_llm_robustness.tex").write_text(rob_tex, encoding="utf-8")

        # Diagnostics table (merged from metrics_extra_summary.csv if available)
        # We keep it compact: schema validity, label entropy, redundancy IoU, saliency rank, runtime.
        def get_extra(r: Dict[str, Any], key: str) -> Any:
            return r.get(f"extra_{key}")

        # likely keys from experiment_b_eval_extra_metrics.py
        # (we keep fallbacks; missing values render as --)
        diag_tex = f"""% Auto-generated by experiment_b_eval_paper_tables.py
% Requires metrics_extra_summary.csv (from experiment_b_eval_extra_metrics.py).
\\begin{{table}}[t]
    \\centering
    \\caption{{Diagnostic metrics at $K={k_main}$ (higher is better except entropy/IoU/time; GT-free).}}
    \\label{{tab:diagnostics}}
    \\begin{{tabular}}{{lccccc}}
        \\toprule
        Method & SchemaValid & Entropy & OverlapIoU & SaliencyRank & TotalTime(s) \\
        \\midrule
        SAM-only & {f_pct(get_extra(r_sam,'schema_valid_rate'))} & {f_num(get_extra(r_sam,'label_entropy_overall'),3)} & {f_num(get_extra(r_sam,'mean_pairwise_iou'),3)} & -- & {f_sec(get_extra(r_sam,'time_total_s'))} \\
        SAM+Saliency (ours) & \\textbf{{{f_pct(get_extra(r_sal,'schema_valid_rate'))}}} & {f_num(get_extra(r_sal,'label_entropy_overall'),3)} & {f_num(get_extra(r_sal,'mean_pairwise_iou'),3)} & \\textbf{{{f_num(get_extra(r_sal,'saliency_rank_auc'),3)}}} & {f_sec(get_extra(r_sal,'time_total_s'))} \\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
        (outdir / "table_diagnostics.tex").write_text(diag_tex, encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  - {outdir / 'paper_run_summary.csv'}")
    print(f"  - {outdir / 'paper_llm_summary.csv'}")
    if args.write_latex:
        print(f"  - {outdir / 'table_main_results.tex'}")
        print(f"  - {outdir / 'table_region_budget.tex'}")
        print(f"  - {outdir / 'table_llm_robustness.tex'}")
        print(f"  - {outdir / 'table_diagnostics.tex'}")


if __name__ == "__main__":
    main()
