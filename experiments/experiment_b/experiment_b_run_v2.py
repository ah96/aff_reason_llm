"""
Experiment B (v2) — GT-free typed-affordance agreement over the vision pipeline.

Two instance-selection strategies (the paper's selection ablation):
  * sam2_area   : SAM 2 automatic "segment everything" -> keep top-K instances by area,
                  then query every (instance, action). The naive baseline.
  * sam3_concept: SAM 3 text-prompted concept segmentation -> for each action, segment only
                  its knowledge-base object concepts (configs/action_concepts.json), keep the
                  top-K by confidence, and query those relevant (instance, action) pairs.

SAM runs ONCE per image, so all VLMs see the SAME regions -> aligned (region, action) pairs
for inter-model agreement. Both SAM providers run through **ultralytics**, which auto-downloads
its weights by name on first use (see download_sam.py). Each model's per-(image, region, action)
prediction is cached under cache_b/ and accumulated into outdir/<model>_<mode>_K<K>.jsonl.
Agreement/consensus is computed by experiment_b_agreement.py.

Output is APPEND-and-dedup keyed by (image, region_id, action): re-running with more images, a
larger K, or an added model tops up the same file without repeat spend or losing prior rows.

Heavy deps (SAM) run on the lab PC GPU; use --dry_run here to exercise everything else (crop,
VLM query, cache, output) with a mock segmenter and a mock client.
"""
import os
import io
import json
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

from vision_llm_clients import load_llms, make_client   # same hardened clients as Exp A

SYSTEM_PROMPT = """You are evaluating affordances in images under a CLOSED ontology.
Follow the taxonomy EXACTLY and output STRICT JSON only.

Relationship label ids:
0: Positive   1: FirmlyNegative   2: ObjectNonFunctional   3: PhysicalObstacle
4: SociallyAwkward   5: SociallyForbidden   6: Dangerous

Return: {"relationship_id": <int 0..6>, "explanation": <string>, "consequence": <string>}
Rules: always output relationship_id; explanation/consequence are one sentence each for ids 2..6, empty otherwise.
"""

ACTION_PHRASE = {"sit_on": "sit on", "hold": "hold", "carry": "carry",
                 "cut": "cut with", "throw": "throw", "ride": "ride"}


def b64png(img, max_px=None):
    if max_px:
        w, h = img.size
        s = max_px / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_bbox(img, bbox, pad=0.12):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad), int(h * pad)
    W, H = img.size
    return img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))


def user_prompt(action):
    return (f"The first image is the full scene; the second is a crop of ONE target object.\n"
            f"For the target object, is the action \"{ACTION_PHRASE.get(action, action)}\" appropriate?\n"
            f"Classify the relationship (0..6); for exceptions (2..6) give a one-sentence explanation "
            f"and consequence. STRICT JSON only.")


# ----------------------------------------------------------------------------------------
# Instance providers -> return list of dicts: {region_id, bbox (x1,y1,x2,y2), actions[list]}
# ----------------------------------------------------------------------------------------
def provider_mock(image, actions, K, **kw):
    W, H = image.size
    rng = [(int(W*0.1), int(H*0.1), int(W*0.4), int(H*0.4)),
           (int(W*0.5), int(H*0.5), int(W*0.9), int(H*0.9)),
           (int(W*0.2), int(H*0.6), int(W*0.5), int(H*0.95))]
    return [{"region_id": i, "bbox": b, "actions": list(actions)} for i, b in enumerate(rng[:K])]


def provider_sam2_area(image, actions, K, sam2=None, **kw):
    """SAM 2 automatic masks -> top-K by area; each instance queried for ALL actions."""
    masks = sam2.segment_all(image)                        # list of {area, bbox}
    masks.sort(key=lambda m: m["area"], reverse=True)
    return [{"region_id": i, "bbox": m["bbox"], "actions": list(actions)}
            for i, m in enumerate(masks[:K])]


def provider_sam3_concept(image, actions, K, sam3=None, concepts=None, **kw):
    """SAM 3 concept segmentation -> per action, its KB concepts; top-K by confidence.
    Each returned instance is tied to the action whose concepts matched it.
    The image is encoded ONCE and reused across all actions (SAM's encode-once / prompt-many design)."""
    out = []
    rid = 0
    sam3.set_image(image)                                   # encode once; reused by segment_text below
    for action in actions:
        prompts = concepts.get(action, [])
        if not prompts:
            continue
        dets = sam3.segment_text(prompts)                   # -> list of {bbox, score}, reuses image features
        for d in sorted(dets, key=lambda d: d["score"], reverse=True)[:K]:
            out.append({"region_id": rid, "bbox": d["bbox"], "actions": [action]})
            rid += 1
    return out


# ----------------------------------------------------------------------------------------
# SAM wrappers (ultralytics). Weights auto-download by name on first use; see download_sam.py.
# The exact ultralytics APIs are verified on the first real GPU run.
# ----------------------------------------------------------------------------------------
def _masks_to_boxes(r):
    """Extract (area, xyxy-bbox) for every mask in an ultralytics Results object."""
    out = []
    masks = getattr(r, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return out
    data = masks.data
    data = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
    for m in data:
        mb = m > 0.5
        ys, xs = np.where(mb)
        if xs.size:
            out.append({"area": int(mb.sum()),
                        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return out


class _Sam2:
    """ultralytics SAM 2 automatic ('segment everything') mask generator."""
    def __init__(self, ckpt, device):
        from ultralytics import SAM
        self.model = SAM(ckpt)                              # e.g. "sam2.1_l.pt" -> auto-download
        self.device = device

    def segment_all(self, image):
        res = self.model(np.array(image.convert("RGB")), device=self.device,
                         retina_masks=True, verbose=False)
        r = res[0] if isinstance(res, (list, tuple)) else res
        return _masks_to_boxes(r)


def _resolve_sam3_ckpt(ckpt):
    """SAM 3 weights are gated on HF and ultralytics does NOT auto-download them by name. If `ckpt`
    is a bare name (e.g. 'sam3.pt') rather than an existing file, resolve it from the gated repo
    facebook/sam3 (downloads once, then returns the cached path). Needs a prior `hf auth login`."""
    if os.path.exists(ckpt):
        return ckpt
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("facebook/sam3", os.path.basename(ckpt) or "sam3.pt")
    except Exception as e:
        raise FileNotFoundError(
            f"SAM 3 checkpoint '{ckpt}' not found and the HF download failed ({e}). Request access at "
            f"https://huggingface.co/facebook/sam3, run `hf auth login`, then `python download_sam.py "
            f"--only sam3` -- or pass --sam3_ckpt /full/path/to/sam3.pt."
        ) from e


class _Sam3:
    """ultralytics SAM 3 semantic (text-prompt) concept predictor."""
    def __init__(self, ckpt, device, conf=0.25):
        from ultralytics.models.sam import SAM3SemanticPredictor
        ckpt = _resolve_sam3_ckpt(ckpt)                      # bare 'sam3.pt' -> HF cache path
        # save=False etc.: we only need the boxes/masks in-memory; don't write annotated images to
        # runs/segment/ on every call (thousands of files + disk churn over the full run).
        self.pred = SAM3SemanticPredictor(overrides=dict(model=ckpt, conf=conf, device=device,
                                                         task="segment", mode="predict", verbose=False,
                                                         save=False, save_txt=False, save_crop=False))

    def set_image(self, image):
        """Encode the image ONCE (SAM 3 caches its features); reuse across every action's prompts."""
        self.pred.set_image(image)

    @staticmethod
    def _parse(res):
        r = res[0] if isinstance(res, (list, tuple)) else res
        out = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):                     # preferred: boxes + confidence
            conf = getattr(boxes, "conf", None)
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                out.append({"bbox": tuple(int(v) for v in xyxy),
                            "score": float(conf[i]) if conf is not None else 1.0})
        else:                                                    # fallback: bbox from mask
            for d in _masks_to_boxes(r):
                out.append({"bbox": d["bbox"], "score": 1.0})
        return out

    def segment_text(self, prompts):
        """Concept-segment for one prompt set, reusing the image set by set_image() (no re-encode)."""
        return self._parse(self.pred(text=list(prompts)))

    def segment(self, image, prompts):
        """Convenience: encode + segment in one call (single-shot / smoke path)."""
        self.set_image(image)
        return self.segment_text(prompts)


def read_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(d, open(p, "w"))


def load_existing(path):
    """Load a prior run's predictions keyed by (image, region_id, action) for append-and-dedup."""
    acc = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    acc[(r["image"], r["region_id"], r["action"])] = r
                except Exception:
                    continue
    return acc


def flush(path, acc):
    """Rewrite the accumulated (deduped) rows, sorted for stable diffs."""
    rows = sorted(acc.values(), key=lambda r: (r["image"], r["action"], r["region_id"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="../experiment_b_bundle/images")
    ap.add_argument("--outdir", default="../experiment_b_bundle/out")
    ap.add_argument("--llms", default="configs/llms.json")
    ap.add_argument("--mode", choices=["sam2_area", "sam3_concept", "mock"], default="sam2_area")
    ap.add_argument("--actions", nargs="+", default=["sit_on", "hold", "carry", "cut", "throw", "ride"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--concepts", default="configs/action_concepts.json")
    ap.add_argument("--sam2_ckpt", default="sam2.1_l.pt")     # ultralytics name -> auto-download
    ap.add_argument("--sam3_ckpt", default="sam3.pt")         # ultralytics name -> auto-download (gated)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full_px", type=int, default=1024)
    ap.add_argument("--crop_px", type=int, default=768)
    ap.add_argument("--cache_dir", default="cache_b")
    ap.add_argument("--models", default=None)
    ap.add_argument("--limit_images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ids = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit_images:
        ids = ids[: args.limit_images]
    print(f"{len(ids)} images | mode={args.mode} | K={args.K}", flush=True)

    # instance provider
    concepts = json.load(open(args.concepts)) if os.path.exists(args.concepts) else {}
    prov_kw = {}
    if args.dry_run or args.mode == "mock":
        provider = provider_mock
    elif args.mode == "sam2_area":
        prov_kw["sam2"] = _Sam2(args.sam2_ckpt, args.device)
        provider = provider_sam2_area
    else:
        prov_kw["sam3"] = _Sam3(args.sam3_ckpt, args.device)
        prov_kw["concepts"] = concepts
        provider = provider_sam3_concept

    # clients
    if args.dry_run:
        clients = [("dry_run", type("D", (), {"complete_json": lambda self, **k: {"relationship_id": 0, "explanation": "", "consequence": ""}})())]
    else:
        cfgs = [c for c in load_llms(args.llms) if getattr(c, "supports_vision", True)]
        if args.models:
            want = {m.strip() for m in args.models.split(",")}
            cfgs = [c for c in cfgs if c.name in want]
        clients = [(c.name, make_client(c)) for c in cfgs]
    print("Models:", [n for n, _ in clients], flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    # Append-and-dedup: seed each model's accumulator from any prior run of this (mode, K).
    paths = {n: os.path.join(args.outdir, f"{n}_{args.mode}_K{args.K}.jsonl") for n, _ in clients}
    acc = {n: load_existing(paths[n]) for n, _ in clients}
    seeded = sum(len(v) for v in acc.values())
    if seeded:
        print(f"Resuming: {seeded} existing rows loaded across {len(acc)} model file(s).", flush=True)

    # Per-CALL progress bar (like Experiment A): advances once per (region, action, model) call,
    # not once per image, and shows live cached/live/err counts. The total is an upper bound
    # (K regions x actions x models per image); it is corrected to the true count at the end.
    est_total = len(ids) * args.K * len(args.actions) * max(1, len(clients))
    stats = {"cached": 0, "live": 0, "err": 0, "exc": 0}
    pbar = tqdm(total=est_total, desc="calls", unit="call", smoothing=0.03)

    def run(job, image_id, full_b64, crops):
        (inst, action), name, client = job
        cp = os.path.join(args.cache_dir, args.mode, name, image_id, f"{action}_{inst['region_id']}.json")
        pred = read_json(cp)
        if pred is not None:
            return job, pred, "cached", None                  # cache hit -> no API call, no cost
        try:
            pred = client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(action),
                                        images_b64png=[full_b64, crops[inst["region_id"]]])
            write_json(cp, pred)                              # persist immediately -> resumable
            return job, pred, "live", None
        except Exception as e:
            return job, {"relationship_id": -1, "_error": str(e)}, "err", str(e)

    def _postfix(idx):
        pbar.set_postfix(img=f"{idx + 1}/{len(ids)}", cached=stats["cached"],
                         live=stats["live"], err=stats["err"], exc=stats["exc"])

    try:
        for idx, image_id in enumerate(ids):
            img = Image.open(os.path.join(args.images_dir, image_id)).convert("RGB")
            insts = provider(img, args.actions, args.K, **prov_kw)
            if not insts:
                _postfix(idx)
                continue
            full_b64 = b64png(img, args.full_px)
            pairs = [(inst, a) for inst in insts for a in inst["actions"]]
            crops = {inst["region_id"]: b64png(crop_bbox(img, inst["bbox"]), args.crop_px) for inst in insts}
            jobs = [(p, name, client) for p in pairs for name, client in clients]
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for job, pred, kind, err in ex.map(lambda j: run(j, image_id, full_b64, crops), jobs):
                    pbar.update(1)
                    stats[kind] += 1
                    (inst, action), name, client = job
                    pr = int(pred.get("relationship_id", -1))
                    if 0 <= pr <= 6:
                        if 2 <= pr <= 6:
                            stats["exc"] += 1
                        acc[name][(image_id, inst["region_id"], action)] = {
                            "image": image_id, "region_id": inst["region_id"], "bbox": inst["bbox"],
                            "action": action, "relationship_id": pr,
                            "explanation": pred.get("explanation", ""), "consequence": pred.get("consequence", ""),
                        }
                    _postfix(idx)
            # persist after every image so an interrupted run keeps its progress
            for n in acc:
                flush(paths[n], acc[n])
    except KeyboardInterrupt:
        for n in acc:
            flush(paths[n], acc[n])
        pbar.close()
        done = sum(len(v) for v in acc.values())
        print(f"\n[interrupted] progress saved: {done} rows | {stats['cached']} cached, "
              f"{stats['live']} live, {stats['err']} err.", flush=True)
        print("Re-run the SAME command to resume — cached calls are skipped (no repeat spend).", flush=True)
        return

    pbar.total = pbar.n            # correct the upper-bound estimate to the true count -> shows 100%
    pbar.refresh(); pbar.close()

    total = sum(len(v) for v in acc.values())
    print(f"\nWrote {total} rows across {len(acc)} model file(s) -> {args.outdir}/<model>_{args.mode}_K{args.K}.jsonl "
          f"| {stats['cached']} cached, {stats['live']} live calls, {stats['err']} errors, {stats['exc']} exceptions", flush=True)


if __name__ == "__main__":
    main()
