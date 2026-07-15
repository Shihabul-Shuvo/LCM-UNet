"""Evaluation metrics (methodology section 8): DSC, mIoU, Sensitivity,
Specificity, Accuracy (primary/supporting), HD95, ASSD (boundary,
supporting-only). Binarisation threshold is fixed at 0.5 for every dataset
-- no per-dataset convention override (Fairness rule); if one is ever
needed it must be documented here, not silently applied per-dataset.

HD95/ASSD are implemented directly with scipy (already a dependency)
rather than adding medpy, using the same standard surface-distance
definitions medpy.metric.binary.hd95/assd use: pool the two one-directional
surface-distance sets and take the 95th percentile for HD95; average the
two one-directional MEANS for ASSD (not a pooled mean -- that would weight
by border-pixel count instead of by direction).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

THRESHOLD = 0.5
EPS = 1e-6


# ---- per-image metrics (numpy, binary masks) --------------------------------


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = EPS) -> float:
    intersection = np.sum(pred_bin & gt_bin)
    return float((2.0 * intersection + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = EPS) -> float:
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    return float((intersection + eps) / (union + eps))


def _confusion_counts(pred_bin: np.ndarray, gt_bin: np.ndarray):
    tp = int(np.sum(pred_bin & gt_bin))
    tn = int(np.sum((~pred_bin) & (~gt_bin)))
    fp = int(np.sum(pred_bin & (~gt_bin)))
    fn = int(np.sum((~pred_bin) & gt_bin))
    return tp, tn, fp, fn


def sensitivity_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = EPS) -> float:
    tp, _tn, _fp, fn = _confusion_counts(pred_bin, gt_bin)
    return float((tp + eps) / (tp + fn + eps))


def specificity_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = EPS) -> float:
    _tp, tn, fp, _fn = _confusion_counts(pred_bin, gt_bin)
    return float((tn + eps) / (tn + fp + eps))


def accuracy_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    tp, tn, fp, fn = _confusion_counts(pred_bin, gt_bin)
    total = tp + tn + fp + fn
    return float((tp + tn) / total) if total > 0 else float("nan")


def _surface_points_distances(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing):
    """One-directional-both-ways surface distances, matching medpy's
    __surface_distances convention: border = mask XOR eroded-mask."""
    pred_border = pred_bin ^ binary_erosion(pred_bin, border_value=1)
    gt_border = gt_bin ^ binary_erosion(gt_bin, border_value=1)

    dt_from_gt_border = distance_transform_edt(~gt_border, sampling=spacing)
    dt_from_pred_border = distance_transform_edt(~pred_border, sampling=spacing)

    pred_to_gt = dt_from_gt_border[pred_border]
    gt_to_pred = dt_from_pred_border[gt_border]
    return pred_to_gt, gt_to_pred


def hd95_and_assd(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0, 1.0)):
    """Returns (hd95, assd). NaN for either metric when one mask is empty and
    the other isn't (undefined: no boundary to measure against) -- this is a
    documented edge case, never silently coerced to 0 or a fabricated value.
    Both-empty (no lesion, correctly predicted none) returns (0.0, 0.0):
    trivially perfect boundary agreement.
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    if not pred_bin.any() and not gt_bin.any():
        return 0.0, 0.0
    if not pred_bin.any() or not gt_bin.any():
        return float("nan"), float("nan")

    pred_to_gt, gt_to_pred = _surface_points_distances(pred_bin, gt_bin, spacing)
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        # degenerate: a mask with foreground but no detectable border pixel
        # (e.g. a single-pixel blob at the extreme of binary_erosion behaviour)
        return float("nan"), float("nan")

    hd95 = float(np.percentile(np.concatenate([pred_to_gt, gt_to_pred]), 95))
    assd = float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)
    return hd95, assd


def compute_all_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0, 1.0), boundary: bool = True) -> Dict[str, float]:
    out = {
        "dsc": dice_score(pred_bin, gt_bin),
        "miou": iou_score(pred_bin, gt_bin),
        "sensitivity": sensitivity_score(pred_bin, gt_bin),
        "specificity": specificity_score(pred_bin, gt_bin),
        "accuracy": accuracy_score(pred_bin, gt_bin),
    }
    if boundary:
        hd95, assd = hd95_and_assd(pred_bin, gt_bin, spacing=spacing)
        out["hd95"], out["assd"] = hd95, assd
    return out


# ---- dataset-level evaluation -------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = THRESHOLD,
    boundary: bool = True,
) -> Dict[str, object]:
    """Run model over loader; return aggregate metrics + per-image arrays.

    boundary=False skips HD95/ASSD (distance-transform-based, the slowest
    metrics) -- used for cheap per-epoch validation; the final test-set
    evaluation should use boundary=True (the default).

    Returns a dict with scalar aggregate metrics (nanmean over images -- see
    hd95_and_assd for when a per-image value is NaN), 'ids': List[str], and
    'per_image': Dict[str, np.ndarray] (one array per metric, aligned to ids).
    """
    model.eval()
    ids: List[str] = []
    per_image: Dict[str, List[float]] = {
        "dsc": [], "miou": [], "sensitivity": [], "specificity": [], "accuracy": [],
    }
    if boundary:
        per_image["hd95"] = []
        per_image["assd"] = []

    for images, masks, batch_ids in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gts = masks.numpy()

        for i, item_id in enumerate(batch_ids):
            pred_bin = probs[i, 0] >= threshold
            gt_bin = gts[i, 0] >= 0.5
            m = compute_all_metrics(pred_bin, gt_bin, boundary=boundary)
            for k, v in m.items():
                per_image[k].append(v)
            ids.append(item_id)

    per_image_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in per_image.items()}
    with warnings.catch_warnings():
        # All-NaN slices happen legitimately (e.g. every image in a tiny
        # sanity batch has an empty prediction or empty ground truth) -- the
        # resulting NaN aggregate is correct, so silence numpy's advisory
        # RuntimeWarning rather than let it look like an error in logs.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        aggregate = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in per_image_arrays.items()}
    aggregate["ids"] = ids
    aggregate["per_image"] = per_image_arrays
    aggregate["n_images"] = len(ids)
    return aggregate


def save_per_image_dice(config_id: str, seed: int, ids: List[str], dsc: np.ndarray, results_dir) -> Path:
    """Save per-image test Dice (methodology section 8: paired Wilcoxon needs
    the same test images' per-image Dice for two models)."""
    out_dir = Path(results_dir) / "perimage"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config_id}_{seed}.npy"
    # store ids alongside the array so the file is self-describing; np.save
    # of a structured array keeps this a single artifact per methodology's
    # "per-image Dice arrays" requirement.
    dtype = np.dtype([("id", f"U{max((len(i) for i in ids), default=1)}"), ("dsc", "f8")])
    structured = np.array(list(zip(ids, dsc.tolist())), dtype=dtype)
    np.save(path, structured)
    return path


def load_per_image_dice(config_id: str, seed: int, results_dir) -> Dict[str, np.ndarray]:
    path = Path(results_dir) / "perimage" / f"{config_id}_{seed}.npy"
    structured = np.load(path)
    return {"ids": structured["id"], "dsc": structured["dsc"]}
