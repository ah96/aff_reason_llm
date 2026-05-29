import os
import json
import argparse
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import numpy as np

from ade_parsing import (
    load_objectinfo150,
    decode_ade_annotation,
    extract_instances_from_ade_png,
    compute_touching_adjacency,
    load_relationship_file,
    load_exco_json,
)

from llm_clients import load_llm_configs, make_client
from metrics_relationship import compute_macc_metrics
from metrics_caption import compute_caption_metrics


# -----------------------------
# LLM prompting
# -----------------------------
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
{
  "relationship_id": <int 0..6>,
  "explanation": <string>,
  "consequence": <string>
}

Rules:
- Always output relationship_id.
- If relationship_id is 0 or 1, explanation and consequence must be empty strings.
- If relationship_id is 2..6, explanation and consequence must be ONE short sentence each.
"""


# -----------------------------
# Utilities
# -----------------------------
def _norm(p: str) -> str:
    return os.path.normpath(p)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_refs(x: Any) -> List[str]:
    """Normalize GT explanation/consequence to list[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        return [v.strip() for v in x if isinstance(v, str) and v.strip()]
    return []


# -----------------------------
# Dataset indexing (FLAT)
# -----------------------------
def build_aff_index_flat(ade_aff_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Build index from FLAT ADE-Affordance directory.

    Returns:
      image_id -> {"rel": path_to_relationship, "exco": path_to_exco_or_None}
    """
    root = ade_aff_dir
    if not os.path.isdir(root):
        raise RuntimeError(f"Affordance split not found: {root}")

    rel_map = {}
    exco_map = {}

    for fn in os.listdir(root):
        p = os.path.join(root, fn)
        if not os.path.isfile(p):
            continue
        if fn.endswith("_relationship.txt"):
            key = fn.replace("_relationship.txt", "")
            rel_map[key] = p
        elif fn.endswith("_exco.json"):
            key = fn.replace("_exco.json", "")
            exco_map[key] = p

    out = {}
    for k, rel_path in rel_map.items():
        out[k] = {
            "rel": rel_path,
            "exco": exco_map.get(k, None),
        }
    return out


def build_ann_name_list(ade_ann_dir: str) -> List[str]:
    root = ade_ann_dir
    if not os.path.isdir(root):
        raise RuntimeError(f"ADE annotation split not found: {root}")

    idx = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            idx.append(fn[:-4])
    return idx


# -----------------------------
# Prompt construction
# -----------------------------
def build_user_prompt(action: str, target: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> str:
    neigh_txt = "\n".join(
        f"- class={n['class_name']} bbox={n['bbox']} area={n['area']}"
        for n in neighbors
    ) or "- none"

    return f"""Action: {action}

Target object:
- class={target['class_name']}
- bbox={target['bbox']}
- area={target['area']}

Adjacent objects:
{neigh_txt}

Predict relationship and explanation if needed.
"""


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade_ann_dir", required=True,
                    help="Path to ADE20K annotation PNGs directory.")
    ap.add_argument("--objectinfo150", required=True,
                    help="Path to ADE20K objectInfo150.txt.")
    ap.add_argument("--ade_aff_dir", required=True,
                    help="Path to flat ADE-Affordance directory (*_relationship.txt, *_exco.json).")
    ap.add_argument("--llms", required=True,
                    help="Path to llms.json config file.")
    ap.add_argument("--actions", default="sit,run,grasp",
                    help="Comma-separated actions to evaluate. Must match ADE-Affordance annotation columns.")
    ap.add_argument("--limit_images", type=int, default=None,
                    help="Cap number of images (useful for quick sanity checks).")
    ap.add_argument("--cache_dir", default="cache_llm",
                    help="Directory to cache LLM responses (hash-keyed JSON files).")
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    id2name = load_objectinfo150(args.objectinfo150)
    ann_name_list = build_ann_name_list(args.ade_ann_dir)
    ann_name_list.sort()
    aff_index = build_aff_index_flat(args.ade_aff_dir)
    aff_index = dict(sorted(aff_index.items()))

    image_names = sorted(set(ann_name_list) & set(aff_index))
    if args.limit_images:
        image_names = image_names[: args.limit_images]

    llm_cfgs = load_llm_configs(args.llms)
    clients = [(cfg.name, make_client(cfg)) for cfg in llm_cfgs]

    # -----------------------------
    # Accumulators
    # -----------------------------
    results = {}
    for name, _ in clients:
        results[name] = {
            "gt_rel": [],
            "pred_rel": [],
            "expl_metrics": [],
            "cons_metrics": [],
            "n_rel": 0,
            "n_text": 0,
        }

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    for image_name in tqdm(image_names, desc="Images"):
        ann = decode_ade_annotation(args.ade_ann_dir, image_name)
        instances = extract_instances_from_ade_png(ann, id2name)
        adjacency = compute_touching_adjacency(instances)

        rel_map = load_relationship_file(aff_index[image_name]["rel"], actions=actions)
        exco_map = (
            load_exco_json(aff_index[image_name]["exco"])
            if aff_index[image_name]["exco"]
            else {}
        )

        for iid, inst in instances.items():
            if iid not in rel_map:
                continue

            neighbors = []
            for nid in adjacency.get(iid, []):
                if nid in instances:
                    n = instances[nid]
                    neighbors.append({
                        "class_name": n.class_name,
                        "bbox": list(n.bbox),
                        "area": int(n.area),
                    })

            target = {
                "class_name": inst.class_name,
                "bbox": list(inst.bbox),
                "area": int(inst.area),
            }

            for action in actions:
                if action not in rel_map[iid]:
                    continue

                gt_rel = int(rel_map[iid][action])
                prompt = build_user_prompt(action, target, neighbors)

                for llm_name, client in clients:
                    cache_path = os.path.join(
                        args.cache_dir, llm_name, image_name, f"{action}_{iid}.json"
                    )

                    pred = read_json(cache_path)
                    if pred is None:
                        try:
                            pred = client.chat_json(SYSTEM_PROMPT, prompt)
                        except Exception:
                            pred = {"relationship_id": -1, "explanation": "", "consequence": ""}
                        write_json(cache_path, pred)

                    pred_rel = int(pred.get("relationship_id", -1))

                    results[llm_name]["gt_rel"].append(gt_rel)
                    results[llm_name]["pred_rel"].append(pred_rel)
                    results[llm_name]["n_rel"] += 1

                    if 2 <= gt_rel <= 6 and action in exco_map and iid in exco_map[action]:
                        gt = exco_map[action][iid]
                        expl_refs = to_refs(gt.get("explanation"))
                        cons_refs = to_refs(gt.get("consequence"))

                        if expl_refs:
                            results[llm_name]["expl_metrics"].append(
                                compute_caption_metrics(pred.get("explanation", ""), expl_refs)
                            )
                        if cons_refs:
                            results[llm_name]["cons_metrics"].append(
                                compute_caption_metrics(pred.get("consequence", ""), cons_refs)
                            )
                        results[llm_name]["n_text"] += 1

    # -----------------------------
    # Reporting
    # -----------------------------
    print("\n=== Experiment A Results ===")
    for llm, r in results.items():
        rel_metrics = compute_macc_metrics(r["gt_rel"], r["pred_rel"])

        def avg(ms, k):
            return sum(m[k] for m in ms if k in m) / max(1, len(ms))

        print(f"\nLLM: {llm}")
        print(f"  Relationship samples: {r['n_rel']}")
        print(f"  Text samples (exceptions): {r['n_text']}")
        print(f"  mAcc:   {rel_metrics['mAcc']:.4f}")
        print(f"  mAcc-E: {rel_metrics['mAcc-E']:.4f}")

        if r["expl_metrics"]:
            print("  Explanation:")
            for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
                print(f"    {k}: {avg(r['expl_metrics'], k):.4f}")

        if r["cons_metrics"]:
            print("  Consequence:")
            for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
                print(f"    {k}: {avg(r['cons_metrics'], k):.4f}")


if __name__ == "__main__":
    main()
