#!/usr/bin/env python3
"""
Extract ADE-Affordance *_exco.json and *_relationship.txt files from a zip,
preserving directory structure, with optional pairing enforcement and a manifest.

Usage examples:
  python extract_ade_affordance_exco_relationship.py \
      --zip /mnt/data/ADE-Affordance.zip \
      --out ./ADE-Affordance-extracted \
      --paired_only

  python extract_ade_affordance_exco_relationship.py \
      --zip /mnt/data/ADE-Affordance.zip \
      --out ./ADE-Affordance-extracted \
      --paired_only false
"""

import argparse
import csv
import json
import os
import posixpath
import zipfile
from collections import defaultdict


def split_and_scene_from_path(p: str):
    """
    Given a path like:
      ADE-Affordance/training/a/alcove/ADE_train_00001220_exco.json
    return:
      split='training', scene='alcove' (best-effort), subdir='a/alcove' (optional)
    """
    parts = p.split("/")
    split = parts[1] if len(parts) > 1 else ""
    scene = parts[-2] if len(parts) >= 2 else ""
    return split, scene


def safe_extract(zf: zipfile.ZipFile, member: str, out_root: str):
    """
    Extract a single member to out_root, preserving path, protecting against zip-slip.
    """
    # Zip names use POSIX paths; normalize.
    norm = posixpath.normpath(member)

    # Disallow absolute paths and parent traversal
    if norm.startswith("../") or norm.startswith("/") or "/../" in norm:
        raise ValueError(f"Unsafe zip path detected: {member}")

    out_path = os.path.join(out_root, *norm.split("/"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with zf.open(member) as src, open(out_path, "wb") as dst:
        dst.write(src.read())

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ADE-Affordance.zip")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--paired_only",
        default="true",
        help="If true, only extract samples that have BOTH exco and relationship (recommended).",
    )
    args = ap.parse_args()

    paired_only = str(args.paired_only).lower() in ("1", "true", "yes", "y")

    os.makedirs(args.out, exist_ok=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        exco = [n for n in names if n.endswith("_exco.json")]
        rel = [n for n in names if n.endswith("_relationship.txt")]

        # Base id is the filename without suffix, but keep full directory prefix.
        base_to_exco = {n.rsplit("_exco.json", 1)[0]: n for n in exco}
        base_to_rel = {n.rsplit("_relationship.txt", 1)[0]: n for n in rel}

        bases_ex = set(base_to_exco.keys())
        bases_rel = set(base_to_rel.keys())
        bases_paired = sorted(bases_ex & bases_rel)

        bases_ex_only = sorted(bases_ex - bases_rel)
        bases_rel_only = sorted(bases_rel - bases_ex)

        print(f"[INFO] exco files: {len(exco)}")
        print(f"[INFO] relationship files: {len(rel)}")
        print(f"[INFO] paired samples: {len(bases_paired)}")
        print(f"[INFO] exco-only samples: {len(bases_ex_only)}")
        print(f"[INFO] rel-only samples: {len(bases_rel_only)}")
        print(f"[INFO] paired_only = {paired_only}")

        manifest_rows = []
        extracted_counts = defaultdict(int)

        if paired_only:
            bases_to_extract = bases_paired
        else:
            # Extract everything we can, paired or not
            bases_to_extract = sorted(bases_ex | bases_rel)

        for base in bases_to_extract:
            exco_member = base_to_exco.get(base)
            rel_member = base_to_rel.get(base)

            exco_out = ""
            rel_out = ""

            if exco_member is not None:
                exco_out = safe_extract(zf, exco_member, args.out)
                extracted_counts["exco"] += 1

            if rel_member is not None:
                rel_out = safe_extract(zf, rel_member, args.out)
                extracted_counts["relationship"] += 1

            # Derive split/scene from whichever exists
            ref_path = exco_member or rel_member
            split, scene = split_and_scene_from_path(ref_path)

            manifest_rows.append(
                {
                    "base": base,
                    "split": split,
                    "scene": scene,
                    "exco_path": exco_out,
                    "relationship_path": rel_out,
                    "has_exco": int(exco_member is not None),
                    "has_relationship": int(rel_member is not None),
                }
            )

        # Write manifest
        manifest_csv = os.path.join(args.out, "manifest.csv")
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "base",
                    "split",
                    "scene",
                    "exco_path",
                    "relationship_path",
                    "has_exco",
                    "has_relationship",
                ],
            )
            w.writeheader()
            w.writerows(manifest_rows)

        stats = {
            "zip": os.path.abspath(args.zip),
            "out": os.path.abspath(args.out),
            "paired_only": paired_only,
            "counts_in_zip": {
                "exco": len(exco),
                "relationship": len(rel),
                "paired_bases": len(bases_paired),
                "exco_only_bases": len(bases_ex_only),
                "relationship_only_bases": len(bases_rel_only),
            },
            "extracted": dict(extracted_counts),
            "manifest_csv": manifest_csv,
        }
        stats_json = os.path.join(args.out, "stats.json")
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # Also write lists of mismatches for debugging
        if bases_ex_only:
            with open(os.path.join(args.out, "exco_without_relationship.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(bases_ex_only) + "\n")
        if bases_rel_only:
            with open(os.path.join(args.out, "relationship_without_exco.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(bases_rel_only) + "\n")

        print(f"[DONE] Extracted exco={extracted_counts['exco']} relationship={extracted_counts['relationship']}")
        print(f"[DONE] Wrote manifest: {manifest_csv}")
        print(f"[DONE] Wrote stats: {stats_json}")


if __name__ == "__main__":
    main()
