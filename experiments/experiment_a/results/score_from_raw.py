"""
Recompute Experiment A metrics from the raw prediction files in this folder — self-contained,
no cache, no API, no heavy deps. Proves the raw results are enough to regenerate every statistic.

    python3 score_from_raw.py                 # mAcc-3 / mAcc-7 per model, over all instances
    python3 score_from_raw.py --exceptions_only   # over GT exception instances (codes 2..6) only
"""
import json
import glob
import argparse


def macc(gt, pr, K):
    accs = []
    for c in range(K):
        idx = [i for i, g in enumerate(gt) if g == c]
        if idx:
            accs.append(sum(pr[i] == c for i in idx) / len(idx))
    return sum(accs) / max(1, len(accs))


def to3(c):
    return 0 if c == 0 else (1 if c == 1 else 2)     # Positive / FirmlyNegative / Exception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--exceptions_only", action="store_true")
    args = ap.parse_args()
    for f in sorted(glob.glob(f"{args.dir}/raw_*.jsonl")):
        name = f.split("raw_")[-1][:-6]
        gt, pr = [], []
        for line in open(f):
            r = json.loads(line)
            g, p = r["gt"], r["pred"]
            if g is None or not (0 <= p <= 6):
                continue
            if args.exceptions_only and not (2 <= g <= 6):
                continue
            gt.append(g); pr.append(p)
        if not gt:
            continue
        m7 = macc(gt, pr, 7)
        m3 = macc([to3(x) for x in gt], [to3(x) for x in pr], 3)
        print(f"{name:16s} n={len(gt):6d}  mAcc-3 {m3:.3f}  mAcc-7 {m7:.3f}")


if __name__ == "__main__":
    main()
