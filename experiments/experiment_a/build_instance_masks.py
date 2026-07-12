"""
Build ADE-Affordance instance segmentation masks for Experiment A.

ADE-Affordance ships no masks; its instance IDs are the full-release ADE20K IDs,
encoded in the BLUE channel of the object-level segmentation. We pull those
segmentations from the HuggingFace mirror `1aurent/ADE20K` (full release: 25,574
train images, BSD, no registration) and save one `<id>_seg.png` per test image.

Encoding (verified): instance mask for ADE-Affordance id `v` = pixels where
seg[:, :, 2] == v, where seg = segmentations[0] of the matching row.

Run (lab PC or anywhere with internet):
    python3 build_instance_masks.py \
        --labels_dir ../experiment_a_bundle/ade_affordance_test \
        --out_dir    ../experiment_a_bundle/instance_seg

Downloads the train shards on the fly (~5.5 GB streamed) and stops early once all
requested masks are found. Re-running skips masks already saved.
"""
import os
import sys
import argparse
from datasets import load_dataset


def base(fn: str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="../experiment_a_bundle/ade_affordance_test",
                    help="Dir with *_relationship.txt (defines which image IDs we need).")
    ap.add_argument("--out_dir", default="../experiment_a_bundle/instance_seg",
                    help="Where to write <id>_seg.png object-segmentation masks.")
    ap.add_argument("--img_out", default="../experiment_a_bundle/images_full",
                    help="Where to write the matching full-res <id>.jpg (aligned with the seg).")
    ap.add_argument("--repo", default="1aurent/ADE20K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_scan", type=int, default=None,
                    help="Debug: stop after scanning this many rows.")
    ap.add_argument("--retries", type=int, default=8,
                    help="Restart the stream this many times on transient errors.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.img_out, exist_ok=True)
    want = {f[:-len("_relationship.txt")] for f in os.listdir(args.labels_dir)
            if f.endswith("_relationship.txt")}
    have = {f[:-len("_seg.png")] for f in os.listdir(args.out_dir)
            if f.endswith("_seg.png")}
    todo = want - have
    print(f"need {len(want)} masks | already have {len(have)} | to fetch {len(todo)}", flush=True)
    if not todo:
        print("nothing to do."); return

    # Pull filename + full-res image + object segmentation. We save the HF image too
    # because it matches the segmentation resolution (the SceneParse150 copy is downsized,
    # so bbox crops taken in seg coords would misalign on it). We skip the heavy per-part
    # `instances` list to keep bandwidth down.
    #
    # HF streaming can raise a transient "broken data stream" mid-pass; wrap each pass in a
    # retry that restarts the stream and skips whatever we've already saved.
    saved = 0
    for attempt in range(1, args.retries + 1):
        if not todo:
            break
        try:
            ds = load_dataset(args.repo, split=args.split, streaming=True).select_columns(
                ["filename", "image", "segmentations"])
            scanned = 0
            for row in ds:
                scanned += 1
                b = base(row["filename"])
                if b in todo:
                    try:
                        row["segmentations"][0].save(os.path.join(args.out_dir, f"{b}_seg.png"))
                        row["image"].convert("RGB").save(os.path.join(args.img_out, f"{b}.jpg"), quality=95)
                    except Exception as e:
                        print(f"  skip {b}: {e}", flush=True)
                        continue
                    todo.discard(b); saved += 1
                    if saved % 25 == 0:
                        print(f"  saved {saved} | scanned {scanned} | remaining {len(todo)}", flush=True)
                if not todo:
                    break
                if args.max_scan and scanned >= args.max_scan:
                    break
        except Exception as e:
            print(f"[pass {attempt}] stream error: {e} -- restarting, {len(todo)} remaining", flush=True)
            continue
        break  # a full pass completed without a fatal stream error

    print(f"DONE: saved {saved} | still missing {len(todo)}", flush=True)
    if todo:
        print("  (missing IDs were not found in this split — re-run to resume)", flush=True)
    sys.stdout.flush()
    os._exit(0)   # avoid a torch/GIL teardown crash in this env


if __name__ == "__main__":
    main()
