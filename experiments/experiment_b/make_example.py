"""
Build a qualitative Experiment B example for the paper: an image, its SAM-proposed region (bbox
drawn on the image), and every VLM's typed affordance output for a chosen action. Reads the raw
per-model jsonl in results/ (or ../experiment_b_bundle/out/); needs nothing else.

    # 1) list interesting candidates (models disagree, or an exception was predicted):
    python3 make_example.py --mode sam2_area --K 3 --list
    # 2) render one (draws the SAM bbox, prints each model's label + explanation + consequence):
    python3 make_example.py --mode sam2_area --K 3 --image ADE_val_00000013.jpg --region 0 --action sit_on
"""
import os
import glob
import json
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw

REL = {0: "Positive", 1: "FirmlyNegative", 2: "ObjectNonFunctional", 3: "PhysicalObstacle",
       4: "SociallyAwkward", 5: "SociallyForbidden", 6: "Dangerous"}


def load(results_dir, mode, K):
    idx, bbox = defaultdict(dict), {}
    suffix = f"_{mode}_K{K}.jsonl"
    for f in glob.glob(os.path.join(results_dir, f"*{suffix}")):
        name = os.path.basename(f)[:-len(suffix)]
        for line in open(f):
            r = json.loads(line)
            idx[(r["image"], r["region_id"], r["action"])][name] = r
            bbox[(r["image"], r["region_id"])] = r.get("bbox")
    return idx, bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--images_dir", default="../experiment_b_bundle/images")
    ap.add_argument("--mode", default="sam2_area")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--image"); ap.add_argument("--region", type=int); ap.add_argument("--action")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="example.png")
    args = ap.parse_args()

    idx, bbox = load(args.results_dir, args.mode, args.K)
    if not idx:
        print(f"no *_{args.mode}_K{args.K}.jsonl in {args.results_dir} "
              f"(snapshot first, or point --results_dir at ../experiment_b_bundle/out)"); return

    if args.list:
        cand = []
        for key, preds in idx.items():
            labels = {p["relationship_id"] for p in preds.values()}
            has_exc = any(2 <= p["relationship_id"] <= 6 for p in preds.values())
            if len(labels) > 1 or has_exc:
                cand.append((len(labels), has_exc, key))
        cand.sort(reverse=True)
        print(f"{len(cand)} candidates (models disagree or an exception was predicted); top 25:")
        for nlab, exc, key in cand[:25]:
            print(f"  {key[0]}  region={key[1]}  action={key[2]:8s}  distinct_labels={nlab}  exception={exc}")
        print("\nrender one with:  --image <img> --region <r> --action <a>")
        return

    key = (args.image, args.region, args.action)
    if key not in idx:
        print(f"no such (image, region, action): {key}. Use --list to see options."); return

    b = bbox.get((args.image, args.region))
    img = Image.open(os.path.join(args.images_dir, args.image)).convert("RGB")
    if b:
        d = ImageDraw.Draw(img)
        d.rectangle(list(b), outline=(255, 0, 0), width=max(3, img.size[0] // 200))
    img.save(args.out)

    print(f"\nimage: {args.image}   region: {args.region}   action: {args.action}")
    print(f"SAM region bbox (x1,y1,x2,y2): {b}   -> drawn on {args.out}\n")
    for name in sorted(idx[key]):
        p = idx[key][name]
        rid = p["relationship_id"]
        print(f"[{name}]  {rid} = {REL.get(rid, '?')}")
        if p.get("explanation"):
            print(f"      explanation: {p['explanation']}")
        if p.get("consequence"):
            print(f"      consequence: {p['consequence']}")
    print()


if __name__ == "__main__":
    main()
