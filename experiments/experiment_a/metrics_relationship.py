from typing import List, Dict, Tuple
import numpy as np


REL7_NAMES = {
    0: "Positive",
    1: "FirmlyNegative",
    2: "ObjectNonFunctional",
    3: "PhysicalObstacle",
    4: "SociallyAwkward",
    5: "SociallyForbidden",
    6: "Dangerous",
}

def to_rel3(label7: int) -> int:
    """
    Map 7-way to 3-way: Positive / FirmlyNegative / Exception
    """
    if label7 == 0:
        return 0
    if label7 == 1:
        return 1
    return 2


def mean_class_accuracy(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    """
    mAcc: average over per-class accuracies.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    accs = []
    for c in range(num_classes):
        idx = (y_true == c)
        if idx.sum() == 0:
            continue
        accs.append(float((y_pred[idx] == y_true[idx]).mean()))
    if not accs:
        return 0.0
    return float(np.mean(accs))


def compute_macc_metrics(gt7: List[int], pred7: List[int]) -> Dict[str, float]:
    """
    Returns:
      mAcc (3-way),
      mAcc-E (7-way)
    """
    gt3 = [to_rel3(x) for x in gt7]
    pr3 = [to_rel3(x) for x in pred7]
    return {
        "mAcc": mean_class_accuracy(gt3, pr3, num_classes=3),
        "mAcc-E": mean_class_accuracy(gt7, pred7, num_classes=7),
    }
