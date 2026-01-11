import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2


@dataclass
class InstanceInfo:
    instance_id: int
    class_id: int
    class_name: str
    mask: np.ndarray  # bool HxW
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    area: int


def load_objectinfo150(path: str) -> Dict[int, str]:
    """
    Parses ADE20K objectInfo150.txt.
    Returns mapping: class_id (1..150) -> class_name.
    """
    id2name: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Common ADE format has fields separated by whitespace; class name often in 5th column,
            # but formats vary slightly across mirrors.
            # We'll robustly find the first token that looks like an integer id, then keep a reasonable name token.
            parts = line.split()
            # Heuristic: first column is class_id
            try:
                cid = int(parts[0])
            except Exception:
                continue

            # Heuristic: class name is often at parts[4] or parts[1], but we choose the last non-numeric token group
            # after removing obvious numeric fields.
            name_candidates = []
            for p in parts[1:]:
                if p.replace(".", "", 1).isdigit():
                    suggests_numeric = True
                else:
                    suggests_numeric = False
                if not suggests_numeric:
                    name_candidates.append(p)

            if not name_candidates:
                continue

            # ADE names sometimes contain commas or are multi-token; keep the first candidate
            # but strip trailing commas.
            cname = name_candidates[0].strip(",")
            id2name[cid] = cname

    return id2name


def decode_ade_annotation(png_path: str) -> np.ndarray:
    """
    Loads an ADE20K annotation PNG as uint8/uint16/uint32 array.
    We keep raw values and decode instance/class from channels robustly.
    """
    ann = np.array(Image.open(png_path))
    return ann


def extract_instances_from_ade_png(
    ann: np.ndarray,
    id2name: Dict[int, str],
    ignore_class_ids: Optional[List[int]] = None
) -> Dict[int, InstanceInfo]:
    """
    ADE20K annotations are commonly encoded as RGB where:
      R: class_id
      G,B: instance_id (or similar)
    But different dumps exist. We'll implement a robust decoder:
      - class_id = ann[:,:,0]
      - instance_id = ann[:,:,1]*256 + ann[:,:,2]
    This matches the common ADE20K instance encoding.

    Returns dict mapping instance_id -> InstanceInfo
    """
    ignore_class_ids = ignore_class_ids or []

    if ann.ndim == 2:
        raise ValueError("Expected RGB annotation PNG. Got single-channel.")

    if ann.shape[2] < 3:
        raise ValueError(f"Expected >=3 channels. Got shape={ann.shape}.")

    class_map = ann[:, :, 0].astype(np.int32)
    instance_map = (ann[:, :, 1].astype(np.int32) << 8) + ann[:, :, 2].astype(np.int32)

    instances: Dict[int, InstanceInfo] = {}
    unique_ids = np.unique(instance_map)

    H, W = instance_map.shape

    for iid in unique_ids:
        if iid == 0:
            continue  # background
        mask = instance_map == iid
        if not mask.any():
            continue
        # class id: take mode of class_map on mask
        class_ids = class_map[mask]
        if class_ids.size == 0:
            continue
        cid = int(np.bincount(class_ids).argmax())
        if cid in ignore_class_ids:
            continue
        cname = id2name.get(cid, f"class_{cid}")

        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        bbox = (x1, y1, x2, y2)
        area = int(mask.sum())

        instances[iid] = InstanceInfo(
            instance_id=int(iid),
            class_id=cid,
            class_name=cname,
            mask=mask,
            bbox=bbox,
            area=area
        )

    return instances


def compute_touching_adjacency(instances: Dict[int, InstanceInfo]) -> Dict[int, List[int]]:
    """
    Two instances are adjacent if their masks touch (share boundary adjacency).
    We'll compute boundaries via dilation and check overlaps.
    Complexity O(n^2) but fine for typical ADE images with moderate instance count.
    """
    ids = list(instances.keys())
    adjacency: Dict[int, List[int]] = {iid: [] for iid in ids}

    # Precompute dilated masks for boundary-touch detection
    kernel = np.ones((3, 3), np.uint8)
    dilated = {}
    for iid in ids:
        m = instances[iid].mask.astype(np.uint8)
        dilated[iid] = cv2.dilate(m, kernel, iterations=1).astype(bool)

    for i, id_a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            id_b = ids[j]
            # Touch if dilated(A) intersects B OR dilated(B) intersects A
            if (dilated[id_a] & instances[id_b].mask).any() or (dilated[id_b] & instances[id_a].mask).any():
                adjacency[id_a].append(id_b)
                adjacency[id_b].append(id_a)

    return adjacency


def load_relationship_file(path: str) -> Dict[int, Dict[str, int]]:
    """
    Parses *_relationship.txt.
    Returns: instance_id -> { 'sit':label, 'run':label, 'grasp':label }
    File format lines like: "209 # 2 # 6 # 0"
    """
    rel: Dict[int, Dict[str, int]] = {}
    actions = ["sit", "run", "grasp"]
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("#")]
            if len(parts) < 4:
                continue
            try:
                iid = int(parts[0])
                vals = [int(parts[1]), int(parts[2]), int(parts[3])]
            except Exception:
                continue
            rel[iid] = {actions[k]: vals[k] for k in range(3)}
    return rel


def load_exco_json(path: str) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Parses *_exco.json.
    Returns: action -> instance_id -> {'explanation':..., 'consequence':...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, Dict[int, Dict[str, str]]] = {}
    for action, by_inst in data.items():
        out[action] = {}
        for iid_str, v in by_inst.items():
            try:
                iid = int(iid_str)
            except Exception:
                continue
            out[action][iid] = {
                "explanation": v.get("explanation", ""),
                "consequence": v.get("consequence", "")
            }
    return out
