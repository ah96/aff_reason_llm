"""
Snapshot Experiment B raw results into experiment_b/results/ for reproducibility.

The runner already writes clean per-(image, region, action) predictions to
../experiment_b_bundle/out/<model>_<mode>_K<K>.jsonl (git-ignored, archived separately). This copies
those raw files plus the agreement summaries into experiment_b/results/, so every statistic, example,
and per-model answer can be regenerated later with experiment_b_agreement.py -- no cache, no API.

    cd experiment_b && python3 snapshot_results.py         # after the runs + agreement scoring
"""
import os
import glob
import json
import shutil

OUT = "../experiment_b_bundle/out"     # where the runner streams raw predictions
DST = "results"                         # tracked snapshot dir


def main():
    os.makedirs(DST, exist_ok=True)
    raw = sorted(glob.glob(os.path.join(OUT, "*.jsonl")))
    if not raw:
        print(f"no raw jsonl found in {OUT} -- run the experiment first."); return

    print("raw per-model predictions:")
    total = 0
    for f in raw:
        dst = os.path.join(DST, os.path.basename(f))
        shutil.copy2(f, dst)
        rows = sum(1 for _ in open(dst))
        exc = sum(1 for line in open(dst) if 2 <= json.loads(line).get("relationship_id", -1) <= 6)
        print(f"  {os.path.basename(f):42s} {rows:6d} rows  ({exc} exceptions)")
        total += rows

    print("\nagreement summaries:")
    summaries = sorted(glob.glob("agree_*.json"))
    for f in summaries:
        shutil.copy2(f, os.path.join(DST, os.path.basename(f)))
        print(f"  {f}")
    if not summaries:
        print("  (none yet -- run experiment_b_agreement.py --out agree_<mode>_K<K>.json)")

    print(f"\ncopied {len(raw)} raw files ({total} total rows) + {len(summaries)} summaries -> {DST}/")
    print("recompute anytime:  python3 experiment_b_agreement.py --outdir results --mode sam2_area --K 3 --models <STD>")


if __name__ == "__main__":
    main()
