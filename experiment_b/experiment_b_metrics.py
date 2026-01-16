"""experiment_b_metrics.py

Base metrics for Experiment B.

Originally this module only contained a small set of metrics.
It now also re-exports a richer set from ``experiment_b_metrics_plus`` while
keeping the original function names intact.
"""

from typing import Dict, List, Any

import numpy as np

# Optional richer metrics (kept in a separate module so legacy scripts that
# only expect numpy keep working). If the module is missing, base metrics below
# still work.
try:
    from experiment_b_metrics_plus import (
        label_histogram,
        label_entropy,
        js_divergence,
        mean_pairwise_js,
        pairwise_agreement_rate,
        saliency_auc_like,
        SchemaStats,
        compute_schema_stats,
        bbox_iou_xyxy,
        mean_pairwise_bbox_iou,
        safe_mean,
        safe_std,
        safe_median,
    )
except Exception:  # pragma: no cover
    pass


def majority_vote(labels: List[int]) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def agreement_rate(labels_per_model: Dict[str, List[int]]) -> float:
    """
    labels_per_model: model_name -> [label for each item in the same order]
    returns average fraction of models agreeing with majority.
    """
    models = list(labels_per_model.keys())
    if not models:
        return 0.0
    n = len(labels_per_model[models[0]])
    if n == 0:
        return 0.0

    agree_fracs = []
    for i in range(n):
        ls = [labels_per_model[m][i] for m in models]
        maj = majority_vote(ls)
        agree_fracs.append(sum(1 for x in ls if x == maj) / len(ls))
    return float(np.mean(agree_fracs))


def saliency_positive_gap(pairs: List[Dict[str, Any]]) -> float:
    """
    pairs entries should include:
      - score (float)
      - relationship_code (int)
    Returns mean(score | Positive) - mean(score | not Positive)
    """
    pos = [p["score"] for p in pairs if p.get("relationship_code") == 0]
    neg = [p["score"] for p in pairs if p.get("relationship_code") != 0]
    if not pos or not neg:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))
