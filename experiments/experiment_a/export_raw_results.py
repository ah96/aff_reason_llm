"""
Consolidate the Experiment A prediction cache into compact, repo-friendly raw-result files.

The cache (`cache_a_vision/`) is 54k+ tiny JSON files — too many to commit. This flattens each
model into one JSONL of every prediction, joined with the ground-truth code, so the raw results
are preserved in the repo (a few MB) without the loose cache.

    python3 export_raw_results.py            # writes results/raw_<model>.jsonl
"""
import os
import re
import json
import glob
import argparse
from collections import Counter

# Self-contained GT parse (mirrors eval_experiment_a_vision) so results/ scripts need nothing else.
ACTION_POS = {"sit": 0, "run": 1, "grasp": 2}
FILE2CANON = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}   # canonical = (file+1)%7


def _agg(votes):
    code, n = Counter(votes).most_common(1)[0]
    return code if n >= 2 else max(votes)


def parse_relationship(path, actions):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        g = [[int(x) for x in re.findall(r"-?\d+", s)] for s in line.split("|")]
        if len(g) != 3 or not g[0]:
            continue
        iid = g[0][0]; g[0] = g[0][1:]
        if any(len(x) != 3 for x in g):
            continue
        out[iid] = {a: FILE2CANON[_agg([g[k][ACTION_POS[a]] for k in range(3)])]
                    for a in actions if a in ACTION_POS}
    return out


LAB = "../experiment_a_bundle/ade_affordance_test"
_rel = {}


def gt_of(img, action, iid):
    if img not in _rel:
        _rel[img] = parse_relationship(os.path.join(LAB, img + "_relationship.txt"), ["sit", "run", "grasp"])
    r = _rel[img]
    return r[iid][action] if iid in r and action in r[iid] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache_a_vision")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    combined = []
    for model_dir in sorted(glob.glob(os.path.join(args.cache_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        name = os.path.basename(model_dir)
        rows = []
        for f in glob.glob(os.path.join(model_dir, "**", "*.json"), recursive=True):
            parts = f.split(os.sep)
            img = parts[-2]
            action, iid = parts[-1][:-5].split("_")
            iid = int(iid)
            pred = json.load(open(f))
            rows.append({
                "model": name, "image": img, "action": action, "region_id": iid,
                "gt": gt_of(img, action, iid),                 # canonical 7-way code (0=Positive..6=Dangerous)
                "pred": pred.get("relationship_id", -1),
                "explanation": pred.get("explanation", ""),
                "consequence": pred.get("consequence", ""),
            })
        rows.sort(key=lambda r: (r["image"], r["action"], r["region_id"]))
        outp = os.path.join(args.out_dir, f"raw_{name}.jsonl")
        with open(outp, "w") as w:
            for r in rows:
                w.write(json.dumps(r) + "\n")
        print(f"{name:16s} {len(rows):6d} rows -> {outp}")
        combined.extend(rows)

    combined.sort(key=lambda r: (r["model"], r["image"], r["action"], r["region_id"]))
    allp = os.path.join(args.out_dir, "all_predictions.jsonl")
    with open(allp, "w") as w:
        for r in combined:
            w.write(json.dumps(r) + "\n")
    print(f"combined {len(combined):6d} rows -> {allp}")


if __name__ == "__main__":
    main()
