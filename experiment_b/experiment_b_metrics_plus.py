"""experiment_b_metrics_plus.py

Additional, paper-friendly metrics for Experiment B outputs.

This module is intentionally dependency-light (numpy only) and is meant to be
imported both by your evaluation script and by notebooks.

It complements the original experiment_b_metrics.py (agreement_rate,
saliency_positive_gap) with:
  - label distribution / entropy
  - per-action schema validity checks
  - exception text completeness checks
  - inter-model divergence (JS) over label distributions
  - pairwise label agreement / disagreement rates
  - simple bbox-overlap stats (proxy for region redundancy)

All metrics are designed to work *without* ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# Generic helpers
# -------------------------


def safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def safe_std(xs: Sequence[float]) -> float:
    return float(np.std(xs)) if len(xs) else 0.0


def safe_median(xs: Sequence[float]) -> float:
    return float(np.median(xs)) if len(xs) else 0.0


def normalize_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    s = float(p.sum())
    if s <= eps:
        return np.ones_like(p, dtype=np.float64) / float(len(p))
    return p / s


# -------------------------
# Label distribution metrics
# -------------------------


def label_histogram(labels: Sequence[int], num_labels: int) -> np.ndarray:
    """Counts labels into a fixed-length histogram."""
    h = np.zeros((num_labels,), dtype=np.int64)
    for x in labels:
        if 0 <= int(x) < num_labels:
            h[int(x)] += 1
    return h


def label_entropy(labels: Sequence[int], num_labels: int, eps: float = 1e-12) -> float:
    """Shannon entropy of the empirical distribution (nats)."""
    h = label_histogram(labels, num_labels).astype(np.float64)
    p = normalize_prob(h, eps=eps)
    return float(-(p * np.log(p + eps)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions (nats)."""
    p = normalize_prob(np.asarray(p, dtype=np.float64), eps=eps)
    q = normalize_prob(np.asarray(q, dtype=np.float64), eps=eps)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p + eps) - np.log(m + eps))).sum())
    kl_qm = float((q * (np.log(q + eps) - np.log(m + eps))).sum())
    return 0.5 * (kl_pm + kl_qm)


def mean_pairwise_js(hists: Sequence[np.ndarray], eps: float = 1e-12) -> float:
    """Mean pairwise JS divergence across a list of histograms."""
    if len(hists) < 2:
        return 0.0
    vals: List[float] = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            vals.append(js_divergence(hists[i], hists[j], eps=eps))
    return safe_mean(vals)


# -------------------------
# Agreement metrics (label-level)
# -------------------------


def majority_vote(labels: Sequence[int]) -> int:
    vals, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def agreement_rate(labels_per_model: Dict[str, List[int]]) -> float:
    """Average fraction of models agreeing with the majority label per item."""
    models = list(labels_per_model.keys())
    if not models:
        return 0.0
    n = len(labels_per_model[models[0]])
    if n == 0:
        return 0.0

    agree_fracs: List[float] = []
    for i in range(n):
        ls = [int(labels_per_model[m][i]) for m in models]
        maj = majority_vote(ls)
        agree_fracs.append(sum(1 for x in ls if x == maj) / len(ls))
    return safe_mean(agree_fracs)


def pairwise_agreement_rate(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Fraction of positions where two label sequences match."""
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    return float(np.mean(a[:n] == b[:n]))


# -------------------------
# Saliency metrics
# -------------------------


def saliency_positive_gap(pairs: List[Dict[str, Any]]) -> float:
    """Mean(score|Positive) - Mean(score|not Positive)."""
    pos = [float(p["score"]) for p in pairs if int(p.get("relationship_code", -1)) == 0]
    neg = [float(p["score"]) for p in pairs if int(p.get("relationship_code", -1)) != 0]
    if not pos or not neg:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))


def saliency_auc_like(pairs: List[Dict[str, Any]], pos_code: int = 0) -> float:
    """Probability that a random Positive has higher score than a random non-Positive.

    This is a rank-based, threshold-free proxy for "does saliency prioritize
    positives?" Equivalent to AUC under a binary label (Positive vs not).
    """
    pos = np.asarray([float(p["score"]) for p in pairs if int(p.get("relationship_code", -1)) == pos_code], dtype=np.float64)
    neg = np.asarray([float(p["score"]) for p in pairs if int(p.get("relationship_code", -1)) != pos_code], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    # Mann–Whitney U / AUC
    # P(pos > neg) + 0.5*P(pos==neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


# -------------------------
# Schema/quality checks
# -------------------------


@dataclass
class SchemaStats:
    n_instances: int = 0
    n_actions_expected: int = 0
    missing_action_entries: int = 0
    invalid_label_entries: int = 0
    exception_label_entries: int = 0
    exception_missing_text: int = 0
    nonexception_has_text: int = 0


def compute_schema_stats(
    instances: List[Dict[str, Any]],
    actions: Sequence[str],
    num_labels: int,
    exception_codes: Optional[Iterable[int]] = None,
) -> SchemaStats:
    """Checks structural and content validity of the per-instance action dicts."""

    exc = set(exception_codes) if exception_codes is not None else set()
    st = SchemaStats(n_instances=len(instances), n_actions_expected=len(actions))

    for inst in instances:
        acts = inst.get("actions", {}) or {}
        for a in actions:
            if a not in acts:
                st.missing_action_entries += 1
                continue
            entry = acts[a] or {}
            code = int(entry.get("relationship_code", -1))
            if code < 0 or code >= int(num_labels):
                st.invalid_label_entries += 1
            is_exc = code in exc
            if is_exc:
                st.exception_label_entries += 1
                expl = str(entry.get("explanation", "") or "").strip()
                cons = str(entry.get("consequence", "") or "").strip()
                if not expl or not cons:
                    st.exception_missing_text += 1
            else:
                expl = str(entry.get("explanation", "") or "").strip()
                cons = str(entry.get("consequence", "") or "").strip()
                if expl or cons:
                    st.nonexception_has_text += 1

    return st


# -------------------------
# Region redundancy proxy (bbox)
# -------------------------


def bbox_iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = [int(x) for x in a]
    bx0, by0, bx1, by1 = [int(x) for x in b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0, inter_x1 - inter_x0 + 1)
    ih = max(0, inter_y1 - inter_y0 + 1)
    inter = iw * ih
    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def mean_pairwise_bbox_iou(bboxes: Sequence[Sequence[int]]) -> float:
    if len(bboxes) < 2:
        return 0.0
    vals: List[float] = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            vals.append(bbox_iou_xyxy(bboxes[i], bboxes[j]))
    return safe_mean(vals)
