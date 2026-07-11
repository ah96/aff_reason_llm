"""
Experiment A (vision) — typed affordance reasoning on ADE-Affordance ground truth.

For each ground-truth instance we send the VLM the FULL image plus a crop of that
instance (the same query style as Experiment B), and score the predicted 7-way code
and exception text against ADE-Affordance labels. Instances come from the ADE20K
object segmentation (blue channel = ADE-Affordance instance id); see
build_instance_masks.py for how the masks are produced.

Inputs (defaults point at ../experiment_a_bundle):
  images_full/<id>.jpg      full-res RGB (aligned with the seg)
  instance_seg/<id>_seg.png object seg; B channel encodes the instance id
  ade_affordance_test/<id>_relationship.txt , <id>_exco.json   ground truth

Run (4 standard vision models; set keys first):
  export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... TOGETHER_API_KEY=...
  python3 eval_experiment_a_vision.py --llms configs/llms.json --limit_images 200

Offline smoke test (no keys/network):
  python3 eval_experiment_a_vision.py --dry_run --limit_images 5
"""
import os
import io
import re
import sys
import json
import base64
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from ade_parsing import load_exco_json
from metrics_relationship import compute_macc_metrics
from metrics_caption import compute_caption_metrics

# ADE-Affordance relationship.txt format (verified):
#   <iid> # s # r # g  |  # s # r # g  |  # s # r # g
# Three '|'-separated groups are three ANNOTATORS; the three positions in each group
# are the actions in the order [sit, run, grasp] (pos 0=sit, 1=run, 2=grasp). The
# ground-truth code per (instance, action) is the majority vote over the three
# annotators; a 3-way tie is resolved to the most severe (max) code.
ACTION_POS = {"sit": 0, "run": 1, "grasp": 2}

# CRITICAL: ADE-Affordance's integer codes are NOT our taxonomy order. Verified empirically
# (non-exception instances only ever use file codes {0,6}; ~90% are code 0) and against the
# paper (Chuang et al. 2018): the file encodes
#   0=FirmlyNegative 1=ObjNonFunctional 2=PhysicalObstacle 3=SociallyAwkward
#   4=SociallyForbidden 5=Dangerous 6=Positive
# Our prompt/taxonomy is [0=Positive 1=FirmlyNegative 2..6=the five exceptions], i.e. a
# rotation: canonical = (file + 1) % 7. We map GT into our scheme so it matches model output.
FILE2CANON = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}


def _aggregate(votes: List[int]) -> int:
    code, n = Counter(votes).most_common(1)[0]
    return code if n >= 2 else max(votes)


def parse_relationship(path: str, actions: List[str]) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        groups = [[int(x) for x in re.findall(r"-?\d+", g)] for g in line.split("|")]
        if len(groups) != 3 or not groups[0]:
            continue
        iid = groups[0][0]
        groups[0] = groups[0][1:]                       # strip the leading instance id
        if any(len(g) != 3 for g in groups):
            continue
        rec = {}
        for a in actions:
            p = ACTION_POS.get(a)
            if p is None:
                continue
            rec[a] = FILE2CANON[_aggregate([groups[g][p] for g in range(3)])]
        out[iid] = rec
    return out

# Reuse Experiment B's provider-agnostic vision client.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiment_b"))
from vision_llm_clients import load_llms, make_client  # noqa: E402


SYSTEM_PROMPT = """You are evaluating affordances in images under a CLOSED ontology.
You must follow the label taxonomy EXACTLY and output STRICT JSON only.

Relationship label ids:
0: Positive
1: FirmlyNegative
2: ObjectNonFunctional
3: PhysicalObstacle
4: SociallyAwkward
5: SociallyForbidden
6: Dangerous

Return schema:
{"relationship_id": <int 0..6>, "explanation": <string>, "consequence": <string>}

Rules:
- Always output relationship_id.
- If relationship_id is 0 or 1, explanation and consequence must be empty strings.
- If relationship_id is 2..6, explanation and consequence must be ONE short sentence each.
"""

# ADE action -> natural phrasing shown to the model.
ACTION_PHRASE = {"sit": "sit on", "run": "run on", "grasp": "grasp"}


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def to_refs(x) -> List[str]:
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, list):
        return [v.strip() for v in x if isinstance(v, str) and v.strip()]
    return []


def b64png(img: Image.Image, max_px: Optional[int] = None) -> str:
    if max_px:
        w, h = img.size
        s = max_px / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_instance(img: Image.Image, mask: np.ndarray, pad: float = 0.12) -> Optional[Image.Image]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    px, py = int(w * pad), int(h * pad)
    W, H = img.size
    box = (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px + 1), min(H, y2 + py + 1))
    return img.crop(box)


class DryRunClient:
    """Returns deterministic valid JSON so the pipeline can be tested without APIs."""
    def complete_json(self, *, system, user, images_b64png):
        return {"relationship_id": 0, "explanation": "", "consequence": ""}


def build_user_prompt(action: str) -> str:
    phrase = ACTION_PHRASE.get(action, action)
    return (
        f"The first image is the full scene. The second image is a crop of ONE target object.\n"
        f"Question: for the target object, is the action \"{phrase}\" appropriate?\n"
        f"Classify the relationship (0..6) and, for exception categories (2..6), give a "
        f"one-sentence explanation and consequence. Output STRICT JSON only."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="../experiment_a_bundle")
    ap.add_argument("--llms", default="configs/llms.json")
    ap.add_argument("--actions", default="sit,run,grasp")
    ap.add_argument("--limit_images", type=int, default=None)
    ap.add_argument("--full_px", type=int, default=1024, help="Downscale the full image so its longest side <= this (speed/cost).")
    ap.add_argument("--crop_px", type=int, default=768, help="Downscale the instance crop so its longest side <= this.")
    ap.add_argument("--cache_dir", default="cache_a_vision")
    ap.add_argument("--models", default=None,
                    help="Comma-separated config names to run (default: all standard vision models). "
                         "e.g. --models gemini_3_1_flash for a single-key smoke test.")
    ap.add_argument("--dry_run", action="store_true", help="Use a fake client (no keys/network).")
    ap.add_argument("--gt_exceptions_only", action="store_true",
                    help="Only query (instance, action) pairs whose GT is an exception (codes 2-6). "
                         "Use for the o4-mini reasoning subset.")
    ap.add_argument("--out", default=None, help="Write per-model metrics JSON to this path.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent API calls (per image). Lower if you hit rate limits.")
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    img_dir = os.path.join(args.bundle, "images_full")
    seg_dir = os.path.join(args.bundle, "instance_seg")
    lab_dir = os.path.join(args.bundle, "ade_affordance_test")

    # Only images for which we have BOTH a seg mask and a full image.
    ids = sorted(
        f[:-len("_seg.png")] for f in os.listdir(seg_dir)
        if f.endswith("_seg.png") and os.path.exists(os.path.join(img_dir, f[:-len("_seg.png")] + ".jpg"))
    )
    if args.limit_images:
        ids = ids[: args.limit_images]
    print(f"Evaluating {len(ids)} images x {len(actions)} actions", flush=True)

    if args.dry_run:
        clients = [("dry_run", DryRunClient())]
    else:
        cfgs = [c for c in load_llms(args.llms) if getattr(c, "supports_vision", True)]
        if args.models:
            want = {m.strip() for m in args.models.split(",") if m.strip()}
            cfgs = [c for c in cfgs if c.name in want]
        clients = [(c.name, make_client(c)) for c in cfgs]
        print("Models:", [n for n, _ in clients], flush=True)

    results = {n: {"gt": [], "pred": [], "expl": [], "cons": [], "n_text": 0} for n, _ in clients}
    live = {n: {"ok": 0, "err": 0, "msg": None} for n, _ in clients}  # this-run API call tally
    disabled = set()   # models dropped mid-run after repeated failures (others keep going)

    # Pre-count total (instance, action) x model calls so the bar advances per call, not per image.
    total_calls = 0
    for image_id in ids:
        B = np.array(Image.open(os.path.join(seg_dir, image_id + "_seg.png")))[:, :, 2]
        present = {int(v) for v in np.unique(B)}
        rel0 = parse_relationship(os.path.join(lab_dir, image_id + "_relationship.txt"), actions=actions)
        total_calls += sum(1 for iid, pa in rel0.items() if iid in present for a in actions
                           if a in pa and (not args.gt_exceptions_only or 2 <= pa[a] <= 6))
    total_calls *= max(1, len(clients))
    pbar = tqdm(total=total_calls, desc="calls", smoothing=0.03)

    def process_image(image_id):
        seg = np.array(Image.open(os.path.join(seg_dir, image_id + "_seg.png")))
        inst_map = seg[:, :, 2]                         # B channel = instance id
        img = Image.open(os.path.join(img_dir, image_id + ".jpg")).convert("RGB")
        full_b64 = b64png(img, args.full_px)

        rel_map = parse_relationship(os.path.join(lab_dir, image_id + "_relationship.txt"), actions=actions)
        exco_path = os.path.join(lab_dir, image_id + "_exco.json")
        exco_map = load_exco_json(exco_path) if os.path.exists(exco_path) else {}

        # Collect this image's (iid, action, gt, crop) query tasks.
        tasks = []
        for iid, per_action in rel_map.items():
            mask = inst_map == iid
            if not mask.any():
                continue                                # instance id not present in this seg
            crop = crop_instance(img, mask)
            if crop is None:
                continue
            crop_b64 = b64png(crop, args.crop_px)
            for action in actions:
                if action not in per_action:
                    continue
                gt = int(per_action[action])
                if args.gt_exceptions_only and not (2 <= gt <= 6):
                    continue
                tasks.append((iid, action, gt, crop_b64))

        # One job per (task, active model); run this image's jobs concurrently. The API call and
        # cache I/O happen in worker threads (distinct cache files, so no collisions); all shared
        # state (results/live/pbar) is mutated only in the main thread as results come back.
        jobs = [(t, name, client) for t in tasks for name, client in clients if name not in disabled]

        def run_job(job):
            (iid, action, gt, crop_b64), name, client = job
            cache_path = os.path.join(args.cache_dir, name, image_id, f"{action}_{iid}.json")
            pred = read_json(cache_path)
            if pred is not None:
                return job, pred, False, None                 # cache hit
            try:
                pred = client.complete_json(system=SYSTEM_PROMPT, user=build_user_prompt(action),
                                            images_b64png=[full_b64, crop_b64])
                write_json(cache_path, pred)                  # cache successes only
                return job, pred, True, None
            except Exception as e:
                return job, {"relationship_id": -1, "explanation": "", "consequence": "", "_error": str(e)}, True, str(e)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for job, pred, is_live, err in ex.map(run_job, jobs):
                (iid, action, gt, crop_b64), name, client = job
                pbar.update(1)
                if name in disabled:
                    continue
                if is_live and err is not None:
                    live[name]["err"] += 1
                    live[name]["msg"] = err
                    if live[name]["ok"] == 0 and live[name]["err"] >= 6:
                        disabled.add(name)
                        print(f"\n[SKIP] '{name}': first {live[name]['err']} calls all failed -> "
                              f"dropping it (others continue).\n  Reason: {err}", flush=True)
                    continue                      # errored call: never counted in the metrics
                if is_live:
                    live[name]["ok"] += 1

                pred_rel = int(pred.get("relationship_id", -1))
                if not (0 <= pred_rel <= 6):
                    continue                      # malformed / missing label: exclude from metrics
                results[name]["gt"].append(gt)
                results[name]["pred"].append(pred_rel)
                if 2 <= gt <= 6 and action in exco_map and iid in exco_map[action]:
                    g = exco_map[action][iid]
                    er, cr = to_refs(g.get("explanation")), to_refs(g.get("consequence"))
                    if er:
                        results[name]["expl"].append(compute_caption_metrics(pred.get("explanation", ""), er))
                    if cr:
                        results[name]["cons"].append(compute_caption_metrics(pred.get("consequence", ""), cr))
                    results[name]["n_text"] += 1

    try:
        for image_id in ids:
            process_image(image_id)
    except KeyboardInterrupt:
        print("\n[interrupted] scoring the work completed so far; the cache is intact and the "
              "run resumes where it left off.", flush=True)

    pbar.close()
    print("\n=== Experiment A (vision) ===")
    if not args.dry_run:
        ok = [n for n, _ in clients if n not in disabled and live[n]["ok"] > 0]
        print(f"Models OK: {ok or 'NONE'}" + (f" | dropped: {sorted(disabled)}" if disabled else ""))
    avg = lambda ms, k: sum(x[k] for x in ms if k in x) / max(1, len(ms))
    summary = {"n_images": len(ids), "actions": actions, "gt_exceptions_only": args.gt_exceptions_only, "models": {}}
    for name, r in results.items():
        if name in disabled:
            print(f"\n{name}: DROPPED — all calls failed.\n  reason: {live[name]['msg']}")
            summary["models"][name] = {"dropped": True, "error": live[name]["msg"]}
            continue
        m = compute_macc_metrics(r["gt"], r["pred"])
        calls = "" if args.dry_run else f" (ok={live[name]['ok']} err={live[name]['err']})"
        print(f"\n{name}: n={len(r['gt'])} exc_text={r['n_text']}{calls}")
        print(f"  mAcc-7 {m['mAcc-E']:.4f} | mAcc-3 {m['mAcc']:.4f}")
        rec = {"n": len(r["gt"]), "n_exc_text": r["n_text"], "mAcc_7": m["mAcc-E"], "mAcc_3": m["mAcc"],
               "ok": live[name]["ok"], "err": live[name]["err"]}
        for lab, key in [("explanation", "expl"), ("consequence", "cons")]:
            if r[key]:
                vals = {k: avg(r[key], k) for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]}
                print(f"  {lab} " + " ".join(f"{k} {vals[k]:.3f}" for k in vals))
                rec[lab] = vals
        summary["models"][name] = rec

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote metrics -> {args.out}")


if __name__ == "__main__":
    main()
