"""
Score Experiment A from the cache only — no API keys, no network, instant.

The runner caches ONLY successful responses, so scoring the cache is inherently
error-free (failed/rate-limited calls are never stored). Use this to regenerate
clean relationship metrics at any time, even from a partial run.

    python3 score_from_cache.py --out clean_results.json

(Explanation/consequence text metrics come from the runner's own --out; this
scorer reports the robust relationship accuracy only.)
"""
import os
import re
import json
import glob
import argparse
from collections import Counter

from metrics_relationship import compute_macc_metrics

ACTION_POS = {"sit": 0, "run": 1, "grasp": 2}
FILE2CANON = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}   # (file+1)%7 — see eval_experiment_a_vision


def _agg(votes):
    code, n = Counter(votes).most_common(1)[0]
    return code if n >= 2 else max(votes)


def parse_relationship(path, actions):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        groups = [[int(x) for x in re.findall(r"-?\d+", s)] for s in line.split("|")]
        if len(groups) != 3 or not groups[0]:
            continue
        iid = groups[0][0]; groups[0] = groups[0][1:]
        if any(len(g) != 3 for g in groups):
            continue
        out[iid] = {a: FILE2CANON[_agg([groups[g][ACTION_POS[a]] for g in range(3)])]
                    for a in actions if a in ACTION_POS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache_a_vision")
    ap.add_argument("--bundle", default="../experiment_a_bundle")
    ap.add_argument("--actions", default="sit,run,grasp")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    actions = args.actions.split(",")
    lab_dir = os.path.join(args.bundle, "ade_affordance_test")
    rel_cache = {}

    def gt_of(img, action, iid):
        if img not in rel_cache:
            rel_cache[img] = parse_relationship(os.path.join(lab_dir, img + "_relationship.txt"), actions)
        r = rel_cache[img]
        return r[iid][action] if iid in r and action in r[iid] else None

    summary = {"models": {}}
    for model_dir in sorted(glob.glob(os.path.join(args.cache_dir, "*"))):
        name = os.path.basename(model_dir)
        gts, preds = [], []
        for f in glob.glob(os.path.join(model_dir, "**", "*.json"), recursive=True):
            parts = f.split(os.sep)
            img = parts[-2]
            action, iid = parts[-1][:-5].split("_")
            g = gt_of(img, action, int(iid))
            if g is None:
                continue
            pr = json.load(open(f)).get("relationship_id", -1)
            if not (0 <= pr <= 6):
                continue
            gts.append(g); preds.append(pr)
        if not gts:
            print(f"{name:16s}: no cached successes")
            continue
        m = compute_macc_metrics(gts, preds)
        summary["models"][name] = {"n": len(gts), "mAcc_7": m["mAcc-E"], "mAcc_3": m["mAcc"]}
        print(f"{name:16s} n={len(gts):6d}  mAcc-3 {m['mAcc']:.3f}  mAcc-7 {m['mAcc-E']:.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
