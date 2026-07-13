#!/usr/bin/env python3
"""
fill_missing.py — surgically re-run ONLY the missing (image, action, instance) predictions
for specific models in Experiment A, without re-touching anything already cached.

Why this exists: some models finished with gaps (e.g. Gemini's free-tier rate limit dropped a
few hundred calls). This script:
  1. enumerates the exact expected jobs with the SAME logic as eval_experiment_a_vision.py,
  2. reports which are still missing, per model and per action,
  3. fills them by invoking the real runner (resumable, so it only calls the cache-misses) --
     guaranteeing the new cache entries are byte-identical to the original run,
  4. re-checks and reports the closed gaps.

It never re-calls anything already cached, and only runs the models that actually have gaps.

Usage (from experiments/experiment_a/, with the relevant keys exported):
  export GEMINI_API_KEY=...  OPENROUTER_API_KEY=...
  python3 fill_missing.py                        # gemini + llama, first 200 images
  python3 fill_missing.py --report_only          # just show the gaps, make no API calls
  python3 fill_missing.py --models gemini_3_5_flash --workers 8
"""
import os
import sys
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from PIL import Image

# Reuse the runner's exact GT parsing + taxonomy mapping so the job set is identical.
from eval_experiment_a_vision import parse_relationship

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_experiment_a_vision.py")


def cache_path(cache_dir, model, image_id, action, iid):
    # Must match eval_experiment_a_vision.py exactly.
    return os.path.join(cache_dir, model, image_id, f"{action}_{iid}.json")


def enumerate_jobs(bundle, actions, limit):
    """The (image_id, action, iid) triples the runner would query -- same selection logic."""
    seg_dir = os.path.join(bundle, "instance_seg")
    img_dir = os.path.join(bundle, "images_full")
    lab_dir = os.path.join(bundle, "ade_affordance_test")
    ids = sorted(
        f[: -len("_seg.png")] for f in os.listdir(seg_dir)
        if f.endswith("_seg.png") and os.path.exists(os.path.join(img_dir, f[: -len("_seg.png")] + ".jpg"))
    )
    if limit:
        ids = ids[:limit]
    jobs = []
    for image_id in ids:
        B = np.array(Image.open(os.path.join(seg_dir, image_id + "_seg.png")))[:, :, 2]
        present = {int(v) for v in np.unique(B)}
        rel0 = parse_relationship(os.path.join(lab_dir, image_id + "_relationship.txt"), actions)
        for iid, pa in rel0.items():
            if iid not in present:
                continue
            for a in actions:
                if a in pa:
                    jobs.append((image_id, a, iid))
    return len(ids), jobs


def missing_for(model, jobs, cache_dir):
    return [(img, a, iid) for (img, a, iid) in jobs
            if not os.path.exists(cache_path(cache_dir, model, img, a, iid))]


def report(models, jobs, cache_dir, actions):
    print(f"  expected jobs per model: {len(jobs)}")
    gaps = {}
    for m in models:
        miss = missing_for(m, jobs, cache_dir)
        gaps[m] = miss
        by_action = defaultdict(int)
        for _, a, _ in miss:
            by_action[a] += 1
        by_img = len({img for img, _, _ in miss})
        det = ", ".join(f"{a}:{by_action[a]}" for a in actions if by_action[a]) or "-"
        print(f"  {m:18} missing {len(miss):5}  (across {by_img} images)   [{det}]")
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini_3_5_flash,llama_4_maverick")
    ap.add_argument("--actions", default="sit,run,grasp")
    ap.add_argument("--bundle", default="../experiment_a_bundle")
    ap.add_argument("--cache_dir", default="cache_a_vision")
    ap.add_argument("--limit_images", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--llms", default="configs/llms.json")
    ap.add_argument("--out", default="results/results_a_gemllama.json")
    ap.add_argument("--report_only", action="store_true",
                    help="Only print the gaps; make no API calls.")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    n_imgs, jobs = enumerate_jobs(args.bundle, actions, args.limit_images)
    print(f"== Experiment A gap check ({n_imgs} images x actions {actions}) ==")
    gaps = report(models, jobs, args.cache_dir, actions)

    models_with_gaps = [m for m in models if gaps[m]]
    if not models_with_gaps:
        print("\nNothing missing -- all requested models are complete. Nothing to run.")
        return
    if args.report_only:
        print(f"\n[report_only] would fill: {', '.join(models_with_gaps)} "
              f"({sum(len(gaps[m]) for m in models_with_gaps)} calls). Re-run without --report_only.")
        return

    total = sum(len(gaps[m]) for m in models_with_gaps)
    print(f"\n== Filling {total} missing calls via the runner (resumable; cached calls skipped) ==")
    cmd = [sys.executable, RUNNER,
           "--actions", args.actions,
           "--models", ",".join(models_with_gaps),
           "--limit_images", str(args.limit_images),
           "--workers", str(args.workers),
           "--llms", args.llms,
           "--out", args.out]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\n== Re-checking after fill ==")
    _, jobs2 = enumerate_jobs(args.bundle, actions, args.limit_images)
    gaps2 = report(models, jobs2, args.cache_dir, actions)
    still = {m: len(v) for m, v in gaps2.items() if v}
    if still:
        print(f"\nStill missing (likely persistent API errors): {still}. Re-run to retry.")
    else:
        print("\nAll gaps closed. Now run:  python3 export_raw_results.py   to refresh results/raw_*.jsonl")


if __name__ == "__main__":
    main()
