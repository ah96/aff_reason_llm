from typing import Dict, List, Any, Tuple
import numpy as np


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
