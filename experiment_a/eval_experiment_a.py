import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from ade_parsing import (
    load_objectinfo150,
    decode_ade_annotation,
    extract_instances_from_ade_png,
    compute_touching_adjacency,
    load_relationship_file,
    load_exco_json
)
from llm_clients import load_llm_configs, make_client
from metrics_relationship import compute_macc_metrics, REL7_NAMES
from metrics_caption import compute_caption_metrics


SYSTEM_PROMPT = """You are evaluating affordances in images under a CLOSED ontology.
You must follow the label taxonomy EXACTLY and output STRICT JSON only, with no extra text.

Relationship label ids:
0: Positive
1: FirmlyNegative
2: ObjectNonFunctional
3: PhysicalObstacle
4: SociallyAwkward
5: SociallyForbidden
6: Dangerous

Return schema (STRICT):
{
  "relationship_id": <int 0..6>,
  "explanation": <string>,
  "consequence": <string>
}

Rules:
- Always output relationship_id.
- If relationship_id is 0 or 1, set explanation="" and consequence="".
- If relationship_id is 2..6, explanation and consequence must be ONE short sentence each.
- Do not mention these rules in the output.
"""


def build_user_prompt(
    action: str,
    target: Dict[str, Any],
    neighbors: List[Dict[str, Any]],
) -> str:
    """
    Build a text-only scene graph description so LLM sees the same abstraction as GGNN:
    object class + simple geometry + adjacency.
    """
    obj = target
    neigh_lines = []
    for n in neighbors[:20]:
        neigh_lines.append(
            f"- id={n['id']}, class={n['class_name']}, bbox={n['bbox']}, area={n['area']}"
        )

    prompt = f"""Task: Predict the relationship between action and target object, plus explanation/consequence if needed.

Action: {action}

Target object:
- id={obj['id']}
- class={obj['class_name']}
- bbox={obj['bbox']}
- area={obj['area']}

Adjacent objects (touching in segmentation):
{chr(10).join(neigh_lines) if neigh_lines else "- (none)"}

Now output JSON with relationship_id/explanation/consequence.
"""
    return prompt


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_image_ids(ade_ann_dir: str, limit: Optional[int] = None, split: str = "train") -> List[str]:
    """
    Collect annotation PNG filenames like ADE_train_00000001.png and return image_id base (without ext).
    """
    files = sorted([f for f in os.listdir(ade_ann_dir) if f.endswith(".png") and f.startswith(f"ADE_{split}_")])
    ids = [os.path.splitext(f)[0] for f in files]
    if limit is not None:
        ids = ids[:limit]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade_ann_dir", required=True, help="Path to ADE20K annotation PNGs (instance+class encoded).")
    ap.add_argument("--objectinfo150", required=True, help="Path to objectInfo150.txt")
    ap.add_argument("--aff_dir", required=True, help="Path to ADE-Affordance files (relationship.txt + exco.json).")
    ap.add_argument("--llms", required=True, help="Path to configs/llms.json")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="ADE split prefix: train/val")
    ap.add_argument("--limit_images", type=int, default=50, help="Limit number of images for a quick run.")
    ap.add_argument("--actions", default="sit,run,grasp", help="Comma-separated actions to evaluate.")
    ap.add_argument("--max_instances_per_image", type=int, default=40, help="Cap instances per image to reduce cost.")
    ap.add_argument("--cache_dir", default="cache_llm", help="Cache LLM predictions here.")
    ap.add_argument("--only_exception_text", action="store_true",
                    help="If set, compute text metrics ONLY on exception instances (2..6). Recommended.")
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    id2name = load_objectinfo150(args.objectinfo150)
    llm_cfgs = load_llm_configs(args.llms)
    clients = [(cfg.name, make_client(cfg)) for cfg in llm_cfgs]

    image_ids = iter_image_ids(args.ade_ann_dir, limit=args.limit_images, split=args.split)
    if not image_ids:
        raise RuntimeError(f"No ADE annotation PNGs found in {args.ade_ann_dir} for split={args.split}")

    # Aggregate per-LLM stats
    results = {}
    for llm_name, _ in clients:
        results[llm_name] = {
            "rel_gt7": [],
            "rel_pred7": [],
            "expl_metrics": [],  # list of dicts
            "cons_metrics": [],  # list of dicts
            "num_rel": 0,
            "num_text": 0,
        }

    for image_id in tqdm(image_ids, desc="Images"):
        ann_png = os.path.join(args.ade_ann_dir, image_id + ".png")
        rel_txt = os.path.join(args.aff_dir, image_id + "_relationship.txt")
        exco_js = os.path.join(args.aff_dir, image_id + "_exco.json")

        if not os.path.exists(rel_txt) or not os.path.exists(exco_js):
            # Skip if affordance annotations missing for this image
            continue

        ann = decode_ade_annotation(ann_png)
        instances = extract_instances_from_ade_png(ann, id2name=id2name)
        adjacency = compute_touching_adjacency(instances)

        rel_map = load_relationship_file(rel_txt)  # instance_id -> action->label
        exco_map = load_exco_json(exco_js)         # action -> instance_id -> text

        # Choose only instances that appear in relationship map (some images may include ids not annotated)
        candidate_ids = [iid for iid in instances.keys() if iid in rel_map]
        candidate_ids = sorted(candidate_ids)[: args.max_instances_per_image]

        for iid in candidate_ids:
            inst = instances[iid]
            neigh_ids = adjacency.get(iid, [])
            neighbors = []
            for nid in neigh_ids[:20]:
                if nid not in instances:
                    continue
                ninst = instances[nid]
                neighbors.append({
                    "id": int(ninst.instance_id),
                    "class_name": ninst.class_name,
                    "bbox": list(ninst.bbox),
                    "area": int(ninst.area),
                })

            target_obj = {
                "id": int(inst.instance_id),
                "class_name": inst.class_name,
                "bbox": list(inst.bbox),
                "area": int(inst.area),
            }

            for action in actions:
                if iid not in rel_map or action not in rel_map[iid]:
                    continue
                gt_rel7 = int(rel_map[iid][action])

                # Build prompt once
                user_prompt = build_user_prompt(action, target_obj, neighbors)

                for llm_name, client in clients:
                    cache_path = os.path.join(
                        args.cache_dir, llm_name, f"{image_id}", f"{action}_{iid}.json"
                    )
                    cached = safe_read_json(cache_path)
                    if cached is None:
                        try:
                            pred = client.chat_json(SYSTEM_PROMPT, user_prompt)
                        except Exception as e:
                            pred = {
                                "relationship_id": -1,
                                "explanation": "",
                                "consequence": "",
                                "error": str(e),
                            }
                        safe_write_json(cache_path, pred)
                    else:
                        pred = cached

                    # Relationship
                    pred_rel7 = int(pred.get("relationship_id", -1))
                    if pred_rel7 < 0 or pred_rel7 > 6:
                        # Treat invalid as wrong but keep pipeline running
                        pred_rel7 = -1

                    results[llm_name]["rel_gt7"].append(gt_rel7)
                    results[llm_name]["rel_pred7"].append(pred_rel7)
                    results[llm_name]["num_rel"] += 1

                    # Text metrics: compare only if we have ground-truth text
                    # In ADE-Affordance, exco exists only for exceptions; relationship is still defined for all.
                    gt_text = exco_map.get(action, {}).get(iid, None)
                    if gt_text is None:
                        continue

                    # Optionally restrict to exception relationships (recommended)
                    if args.only_exception_text and not (2 <= gt_rel7 <= 6):
                        continue

                    gt_expl = gt_text.get("explanation", "").strip()
                    gt_cons = gt_text.get("consequence", "").strip()
                    pred_expl = str(pred.get("explanation", "")).strip()
                    pred_cons = str(pred.get("consequence", "")).strip()

                    if gt_expl:
                        em = compute_caption_metrics(pred_expl, [gt_expl])
                        results[llm_name]["expl_metrics"].append(em)
                    if gt_cons:
                        cm = compute_caption_metrics(pred_cons, [gt_cons])
                        results[llm_name]["cons_metrics"].append(cm)

                    results[llm_name]["num_text"] += 1

    # Summarize
    print("\n=== Experiment A Results ===")
    for llm_name in results.keys():
        gt7 = results[llm_name]["rel_gt7"]
        pr7 = results[llm_name]["rel_pred7"]

        # Filter out invalid preds for metric computation (count them as wrong by mapping to an impossible class)
        pr7_clean = [p if 0 <= p <= 6 else 999 for p in pr7]

        rel_metrics = compute_macc_metrics(gt7, pr7_clean)

        def avg_metric(metric_list: List[Dict[str, float]], key: str) -> float:
            vals = [m[key] for m in metric_list if key in m and m[key] == m[key]]  # skip nan
            return sum(vals) / max(1, len(vals))

        expl = results[llm_name]["expl_metrics"]
        cons = results[llm_name]["cons_metrics"]

        print(f"\nLLM: {llm_name}")
        print(f"  #relationship samples: {results[llm_name]['num_rel']}")
        print(f"  mAcc   (3-way): {rel_metrics['mAcc']:.4f}")
        print(f"  mAcc-E (7-way): {rel_metrics['mAcc-E']:.4f}")

        if expl:
            print("  Explanation metrics (avg):")
            for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
                print(f"    {k:7s}: {avg_metric(expl, k):.4f}")
        else:
            print("  Explanation metrics: NA (no samples)")

        if cons:
            print("  Consequence metrics (avg):")
            for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
                print(f"    {k:7s}: {avg_metric(cons, k):.4f}")
        else:
            print("  Consequence metrics: NA (no samples)")

        # Save per-LLM summary to disk
        out_path = os.path.join(args.cache_dir, f"summary_{llm_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "relationship": rel_metrics,
                "num_relationship_samples": results[llm_name]["num_rel"],
                "num_text_samples": results[llm_name]["num_text"],
                "explanation_avg": {k: avg_metric(expl, k) for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]} if expl else None,
                "consequence_avg": {k: avg_metric(cons, k) for k in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]} if cons else None,
            }, f, ensure_ascii=False, indent=2)

        print(f"  Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
