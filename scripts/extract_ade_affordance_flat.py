#!/usr/bin/env python3
"""
Flatten-extract ADE-Affordance *_exco.json and *_relationship.txt into:
  out/training/
  out/validation/
  out/testing/

The ADE-Affordance zip usually has nested scene folders. This script removes them
and keeps only the files we care about, optionally requiring exco+relationship pairs.

Usage:
  python extract_ade_affordance_flat.py \
    --zip /mnt/data/ADE-Affordance.zip \
    --out ./ADE-Affordance-flat \
    --paired_only true \
    --on_conflict error

Options:
  --paired_only true|false   Extract only paired samples (default: true)
  --on_conflict error|skip|overwrite
                            What to do if a flattened filename already exists
                            (default: error)
"""

import argparse
import csv
import json
import os
import posixpath
import zipfile
from collections import defaultdict
from typing import Dict, Tuple, Optional


SUFFIX_EXCO = "_exco.json"
SUFFIX_REL = "_relationship.txt"


def detect_split_from_path(zip_path: str) -> Optional[str]:
    """
    Detect split based on ADE-Affordance folder names.
    Expected patterns somewhere in the path:
      /training/ , /validation/ , /testing/
    Returns one of: training, validation, testing, or None.
    """
    parts = zip_path.strip("/").split("/")
    for p in parts:
        if p in ("training", "validation", "testing"):
            return p
    return None


def safe_member_name(member: str) -> str:
    """
    Normalize and protect against zip-slip.
    """
    norm = posixpath.normpath(member)
    if norm.startswith("../") or norm.startswith("/") or "/../" in norm:
        raise ValueError(f"Unsafe zip path detected: {member}")
    return norm


def base_key(member: str) -> Optional[str]:
    """
    Return base key (path without the suffix), preserving full path (minus suffix).
    Used only for pairing within the same original location.
    """
    if member.endswith(SUFFIX_EXCO):
        return member[: -len(SUFFIX_EXCO)]
    if member.endswith(SUFFIX_REL):
        return member[: -len(SUFFIX_REL)]
    return None


def flattened_filename(member: str) -> str:
    """
    Flattened output filename = original basename only.
    Example:
      .../alcove/ADE_train_00001220_exco.json -> ADE_train_00001220_exco.json
    """
    return os.path.basename(member)


def write_member_to_flat(
    zf: zipfile.ZipFile,
    member: str,
    out_dir: str,
    on_conflict: str,
) -> str:
    """
    Extract one member to out_dir with flattened filename.
    Returns output file path.
    """
    fname = flattened_filename(member)
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        if on_conflict == "skip":
            return out_path
        if on_conflict == "overwrite":
            pass
        else:
            raise FileExistsError(
                f"Conflict: {out_path} already exists. "
                f"Use --on_conflict skip|overwrite or clean the output dir."
            )

    os.makedirs(out_dir, exist_ok=True)
    with zf.open(member) as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ADE-Affordance.zip")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument(
        "--paired_only",
        default="true",
        help="If true, extract only samples with both exco and relationship (default: true).",
    )
    ap.add_argument(
        "--on_conflict",
        default="error",
        choices=["error", "skip", "overwrite"],
        help="What to do if a flattened filename already exists (default: error).",
    )
    args = ap.parse_args()

    paired_only = str(args.paired_only).lower() in ("1", "true", "yes", "y")
    on_conflict = args.on_conflict

    out_root = args.out
    out_split_dirs = {
        "training": os.path.join(out_root, "training"),
        "validation": os.path.join(out_root, "validation"),
        "testing": os.path.join(out_root, "testing"),
    }
    os.makedirs(out_root, exist_ok=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        members = [safe_member_name(m) for m in members]

        exco_members = [m for m in members if m.endswith(SUFFIX_EXCO)]
        rel_members = [m for m in members if m.endswith(SUFFIX_REL)]

        # Pairing keys: full original path without suffix
        exco_map: Dict[str, str] = {}
        rel_map: Dict[str, str] = {}

        for m in exco_members:
            k = base_key(m)
            if k is not None:
                exco_map[k] = m
        for m in rel_members:
            k = base_key(m)
            if k is not None:
                rel_map[k] = m

        exco_keys = set(exco_map.keys())
        rel_keys = set(rel_map.keys())
        paired_keys = sorted(exco_keys & rel_keys)
        exco_only = sorted(exco_keys - rel_keys)
        rel_only = sorted(rel_keys - exco_keys)

        print(f"[INFO] Found exco: {len(exco_members)}")
        print(f"[INFO] Found relationship: {len(rel_members)}")
        print(f"[INFO] Paired bases: {len(paired_keys)}")
        print(f"[INFO] Exco-only: {len(exco_only)}")
        print(f"[INFO] Rel-only: {len(rel_only)}")
        print(f"[INFO] paired_only={paired_only}, on_conflict={on_conflict}")

        keys_to_extract = paired_keys if paired_only else sorted(exco_keys | rel_keys)

        stats = {
            "zip": os.path.abspath(args.zip),
            "out": os.path.abspath(out_root),
            "paired_only": paired_only,
            "on_conflict": on_conflict,
            "counts_in_zip": {
                "exco": len(exco_members),
                "relationship": len(rel_members),
                "paired_bases": len(paired_keys),
                "exco_only_bases": len(exco_only),
                "relationship_only_bases": len(rel_only),
            },
            "extracted": {
                "training": {"exco": 0, "relationship": 0, "pairs": 0},
                "validation": {"exco": 0, "relationship": 0, "pairs": 0},
                "testing": {"exco": 0, "relationship": 0, "pairs": 0},
                "unknown_split": 0,
            },
        }

        manifest_rows = []
        seen_flat_names = defaultdict(int)  # track collisions per split

        for k in keys_to_extract:
            exco_m = exco_map.get(k)
            rel_m = rel_map.get(k)

            # Choose a reference member to detect split + basename
            ref = exco_m or rel_m
            split = detect_split_from_path(ref)
            if split not in ("training", "validation", "testing"):
                stats["extracted"]["unknown_split"] += 1
                # If split is unknown, skip (or could default to training)
                continue

            out_dir = out_split_dirs[split]

            exco_out = ""
            rel_out = ""
            has_exco = exco_m is not None
            has_rel = rel_m is not None

            # Extract (flatten)
            if has_exco:
                flat = flattened_filename(exco_m)
                seen_flat_names[(split, flat)] += 1
                exco_out = write_member_to_flat(zf, exco_m, out_dir, on_conflict)
                stats["extracted"][split]["exco"] += 1

            if has_rel:
                flat = flattened_filename(rel_m)
                seen_flat_names[(split, flat)] += 1
                rel_out = write_member_to_flat(zf, rel_m, out_dir, on_conflict)
                stats["extracted"][split]["relationship"] += 1

            if has_exco and has_rel:
                stats["extracted"][split]["pairs"] += 1

            manifest_rows.append(
                {
                    "base_key": k,
                    "split": split,
                    "exco_out": exco_out,
                    "relationship_out": rel_out,
                    "has_exco": int(has_exco),
                    "has_relationship": int(has_rel),
                    "src_exco_member": exco_m or "",
                    "src_relationship_member": rel_m or "",
                }
            )

        # Write manifest + stats
        manifest_csv = os.path.join(out_root, "manifest.csv")
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "base_key",
                    "split",
                    "exco_out",
                    "relationship_out",
                    "has_exco",
                    "has_relationship",
                    "src_exco_member",
                    "src_relationship_member",
                ],
            )
            w.writeheader()
            w.writerows(manifest_rows)

        stats["manifest_csv"] = manifest_csv

        # Write collision report (if any)
        collisions = [
            {"split": s, "filename": fn, "count": c}
            for (s, fn), c in seen_flat_names.items()
            if c > 1
        ]
        stats["collisions_detected"] = len(collisions)
        collision_csv = os.path.join(out_root, "collisions.csv")
        if collisions:
            with open(collision_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["split", "filename", "count"])
                w.writeheader()
                w.writerows(collisions)
            stats["collisions_csv"] = collision_csv

        stats_json = os.path.join(out_root, "stats.json")
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # Also write mismatch lists (by base_key)
        if exco_only:
            with open(os.path.join(out_root, "exco_without_relationship.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(exco_only) + "\n")
        if rel_only:
            with open(os.path.join(out_root, "relationship_without_exco.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(rel_only) + "\n")

        print("[DONE] Flatten extraction complete.")
        print(f"[DONE] training: {stats['extracted']['training']}")
        print(f"[DONE] validation: {stats['extracted']['validation']}")
        print(f"[DONE] testing: {stats['extracted']['testing']}")
        print(f"[DONE] Manifest: {manifest_csv}")
        print(f"[DONE] Stats: {stats_json}")
        if collisions:
            print(f"[WARN] Filename collisions detected: {len(collisions)}. See {collision_csv}")


if __name__ == "__main__":
    main()
