#!/usr/bin/env python3
"""
OOAL Saliency Adapter for Experiment B.

This adapter wraps the OOAL affordance saliency model (models.ooal.Net) and exposes a
simple callable interface to obtain saliency maps per action/affordance for a single RGB image.

It mirrors your `test.py` logic:
- load Seen/Unseen affordance class lists
- instantiate Net(args, 768, 512)
- forward with `gt_aff` to condition the prediction
- normalize the output map (normalize_map)
- resize saliency back to original image resolution

Expected usage in Experiment B (conceptually):
    adapter = OOALSaliencyAdapter(seen_ckpt=..., unseen_ckpt=..., device="cuda")
    saliency = adapter.predict_saliency(image_rgb, actions=["sit", "grasp", ...])
    # saliency[action] is HxW float32 in [0,1] (normalized)

Notes:
- This assumes your OOAL repo structure is available (models/, data/, utils/).
- It does NOT require the TestData loader; we preprocess directly.
- If your training used a specific normalization (mean/std), adjust below.

Author: generated for your Experiment B pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# OOAL model
from ooal.models.ooal import Net as OOALNet

# Affordance name lists (as in test.py)
from ooal.data.agd20k_ego import SEEN_AFF, UNSEEN_AFF

# Same normalization utility you used in test.py
from ooal.utils.util import normalize_map


# -----------------------------
# Helper: preprocessing
# -----------------------------
def _preprocess_rgb_to_tensor(
    rgb: np.ndarray,
    crop_size: int = 224,
    resize_size: int = 256,
) -> torch.Tensor:
    """
    Preprocess image similarly to common pipelines:
    - resize to resize_size (square)
    - center crop to crop_size
    - convert to float tensor in [0,1]
    - normalize with ImageNet mean/std (change if your repo uses something else)

    Returns: (1,3,crop_size,crop_size)
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 image. Got shape={rgb.shape}")

    pil = Image.fromarray(rgb)

    # Resize to square (simple, stable)
    pil = pil.resize((resize_size, resize_size), resample=Image.BILINEAR)

    # Center crop
    left = (resize_size - crop_size) // 2
    top = (resize_size - crop_size) // 2
    pil = pil.crop((left, top, left + crop_size, top + crop_size))

    arr = np.asarray(pil).astype(np.float32) / 255.0  # HWC, [0,1]
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    # ImageNet normalization (edit if needed)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    ten = torch.from_numpy(arr).float().unsqueeze(0)  # (1,3,H,W)
    return ten


# -----------------------------
# Adapter
# -----------------------------
@dataclass
class _ModelBundle:
    model: torch.nn.Module
    class_names: List[str]


class OOALSaliencyAdapter:
    """
    Provides affordance saliency maps for a given image and list of affordance/action names.

    Two-model setup:
      - Seen model handles affordances in SEEN_AFF
      - Unseen model handles affordances in UNSEEN_AFF

    If an action name isn't found in either list, we raise a clear error
    (better than silently returning zeros).
    """

    def __init__(
        self,
        seen_ckpt: str,
        unseen_ckpt: str,
        device: str = "cuda",
        crop_size: int = 224,
        resize_size: int = 256,
        backbone_dim: int = 768,
        hidden_dim: int = 512,
    ):
        self.device = torch.device(device)
        self.crop_size = crop_size
        self.resize_size = resize_size

        self.seen = self._load_model_bundle(
            ckpt_path=seen_ckpt,
            class_names=list(SEEN_AFF),
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            divide="Seen",
        )
        self.unseen = self._load_model_bundle(
            ckpt_path=unseen_ckpt,
            class_names=list(UNSEEN_AFF),
            backbone_dim=backbone_dim,
            hidden_dim=hidden_dim,
            divide="Unseen",
        )

        # Build quick lookup: affordance -> (bundle, index)
        self._aff_to_bundle: Dict[str, Tuple[_ModelBundle, int]] = {}
        for i, a in enumerate(self.seen.class_names):
            self._aff_to_bundle[a] = (self.seen, i)
        for i, a in enumerate(self.unseen.class_names):
            self._aff_to_bundle[a] = (self.unseen, i)

    def _load_model_bundle(
        self,
        ckpt_path: str,
        class_names: List[str],
        backbone_dim: int,
        hidden_dim: int,
        divide: str,
    ) -> _ModelBundle:
        """
        Instantiate OOALNet similarly to test.py: model(args, 768, 512).
        We provide a minimal args-like object with required fields.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Minimal args object expected by OOALNet
        class _Args:
            pass

        args = _Args()
        args.crop_size = self.crop_size
        args.resize_size = self.resize_size
        args.divide = divide
        args.class_names = class_names

        net = OOALNet(args, backbone_dim, hidden_dim).to(self.device)
        net.eval()

        # Load state dict in the same style as test.py
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            # sometimes checkpoints are saved directly as state_dict
            state = ckpt

        net.load_state_dict(state, strict=False)
        return _ModelBundle(model=net, class_names=class_names)

    def predict_saliency(
        self,
        rgb: np.ndarray,
        actions: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Compute saliency maps for requested actions/affordances.

        Args:
            rgb: HxWx3 uint8 RGB image
            actions: list of affordance names (strings) matching SEEN_AFF/UNSEEN_AFF entries

        Returns:
            dict action -> HxW float32 saliency in [0,1] (normalized)
        """
        H, W = rgb.shape[:2]

        # Preprocess once
        img_t = _preprocess_rgb_to_tensor(rgb, self.crop_size, self.resize_size).to(self.device)

        out: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for a in actions:
                if a not in self._aff_to_bundle:
                    raise ValueError(
                        f"Action/affordance '{a}' not found in SEEN_AFF or UNSEEN_AFF.\n"
                        f"Tip: ensure your Experiment B action vocabulary matches the OOAL affordance names."
                    )

                bundle, idx = self._aff_to_bundle[a]

                # Build gt_aff conditioning vector (1, num_aff)
                gt_aff = torch.zeros((1, len(bundle.class_names)), dtype=torch.float32, device=self.device)
                gt_aff[0, idx] = 1.0

                # Forward pass (same signature as test.py: model(image, gt_aff=gt_aff))
                pred = bundle.model(img_t, gt_aff=gt_aff)

                # pred is typically (1, 1, H, W) or (1, H, W); normalize like test.py
                pred_np = np.array(pred.squeeze().detach().cpu())

                # Normalize to [0,1] at crop resolution
                pred_np = normalize_map(pred_np, self.crop_size).astype(np.float32)

                # Resize back to original image resolution for mask overlap scoring
                pred_full = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_LINEAR)

                out[a] = pred_full

        return out


# -----------------------------
# Optional CLI for quick testing
# -----------------------------
def _cli():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seen_ckpt", required=True)
    ap.add_argument("--unseen_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--actions", nargs="+", required=True)
    ap.add_argument("--out_dir", default="./saliency_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    adapter = OOALSaliencyAdapter(
        seen_ckpt=args.seen_ckpt,
        unseen_ckpt=args.unseen_ckpt,
        device=args.device,
    )
    sal = adapter.predict_saliency(rgb, args.actions)

    base = os.path.splitext(os.path.basename(args.image))[0]
    for a, m in sal.items():
        # save as grayscale heatmap
        m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, f"{base}_{a}.png"), m8)

    print("Wrote saliency maps to:", args.out_dir)


if __name__ == "__main__":
    _cli()


# ---------------------------------------------------------------------------
# Experiment-B runner compatibility layer
# ---------------------------------------------------------------------------
# The Experiment B runner (experiment_b_run.py) dynamically imports this file
# and expects TWO module-level callables:
#   - _load_model_bundle(ckpt_path: str, device: str)
#   - predict_saliency(model_bundle, pil_image: PIL.Image.Image, action: str) -> np.ndarray
#
# The original adapter above is class-based. To avoid forcing changes in the
# runner, we expose thin wrappers that reuse the same OOAL Net + preprocessing
# logic.


def _infer_divide_and_classes(ckpt_path: str):
    """Infer whether a checkpoint is for Seen or Unseen affordances."""
    p = str(ckpt_path).lower()
    if "unseen" in p:
        return "Unseen", list(UNSEEN_AFF)
    # default
    return "Seen", list(SEEN_AFF)


def _load_model_bundle(ckpt_path: str, device: str = "cuda") -> dict:
    """Load an OOAL model bundle for the runner (seen or unseen).

    Returns a plain dict so it stays JSON/pickle friendly and simple:
      {"model": torch.nn.Module, "class_names": List[str], "device": torch.device, ...}
    """
    divide, class_names = _infer_divide_and_classes(ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dev = torch.device(device)

    # Minimal args object expected by OOALNet
    class _Args:
        pass

    args = _Args()
    args.crop_size = 224
    args.resize_size = 256
    args.divide = divide
    args.class_names = class_names

    net = OOALNet(args, 768, 512).to(dev)
    net.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=False)

    return {
        "model": net,
        "class_names": class_names,
        "divide": divide,
        "device": dev,
        "crop_size": 224,
        "resize_size": 256,
        "_name_to_idx": {a: i for i, a in enumerate(class_names)},
    }


def predict_saliency(model_bundle: dict, pil_image: Image.Image, action: str) -> np.ndarray:
    """Return an HxW saliency map for the given action.

    Important behavior for the Experiment B runner:
    - If the action is NOT supported by this bundle (e.g., an unseen affordance
      queried on the seen model), we return a (1,1) dummy array. The runner
      detects the shape mismatch and falls back to the other bundle.
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError("pil_image must be a PIL.Image.Image")

    rgb = np.asarray(pil_image.convert("RGB"))
    H, W = rgb.shape[:2]

    idx = model_bundle.get("_name_to_idx", {}).get(action, None)
    if idx is None:
        # Trigger runner fallback (shape mismatch vs. masks)
        return np.zeros((1, 1), dtype=np.float32)

    dev = model_bundle["device"]
    crop_size = int(model_bundle.get("crop_size", 224))
    resize_size = int(model_bundle.get("resize_size", 256))
    net = model_bundle["model"]
    class_names = model_bundle["class_names"]

    img_t = _preprocess_rgb_to_tensor(rgb, crop_size=crop_size, resize_size=resize_size).to(dev)

    gt_aff = torch.zeros((1, len(class_names)), dtype=torch.float32, device=dev)
    gt_aff[0, int(idx)] = 1.0

    with torch.no_grad():
        pred = net(img_t, gt_aff=gt_aff)

    pred_np = np.array(pred.squeeze().detach().cpu())
    pred_np = normalize_map(pred_np, crop_size).astype(np.float32)
    pred_full = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_LINEAR)
    return pred_full

