"""Train-only augmentation (methodology section 7, documented in
data/PREPROCESSING.md -- keep the two in sync).

Operates on already-preprocessed arrays: image (H, W, 3) float32 in [0, 1],
mask (H, W) float32 in {0, 1}. Deliberately excludes anything that can erase
a small lesion: no aggressive/random cropping, no cutout/erasing, no heavy
elastic deformation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

ROTATION_DEGREES = 30.0
SCALE_RANGE = (0.9, 1.1)
PHOTOMETRIC_FACTOR_RANGE = (0.85, 1.15)
AUG_PROB = 0.5


def random_hflip(image, mask, rng: np.random.Generator):
    if rng.random() < AUG_PROB:
        image = image[:, ::-1, :]
        mask = mask[:, ::-1]
    return image, mask


def random_vflip(image, mask, rng: np.random.Generator):
    if rng.random() < AUG_PROB:
        image = image[::-1, :, :]
        mask = mask[::-1, :]
    return image, mask


def random_rotate(image, mask, rng: np.random.Generator, degrees: float = ROTATION_DEGREES):
    if rng.random() < AUG_PROB:
        angle = rng.uniform(-degrees, degrees)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode="reflect")
        mask = ndimage.rotate(mask, angle, axes=(0, 1), reshape=False, order=0, mode="constant", cval=0.0)
    return image, mask


def random_scale(image, mask, rng: np.random.Generator, scale_range: Tuple[float, float] = SCALE_RANGE):
    """Mild zoom in/out, then center-crop/pad back to the original size.

    Deliberately NOT a random crop: the crop/pad is always centered, so it
    cannot shift a lesion out of frame the way a random-position crop could.
    """
    if rng.random() < AUG_PROB:
        h, w = image.shape[:2]
        scale = rng.uniform(*scale_range)
        image = ndimage.zoom(image, (scale, scale, 1), order=1, mode="reflect")
        mask = ndimage.zoom(mask, (scale, scale), order=0, mode="constant", cval=0.0)
        image = _center_crop_or_pad(image, h, w, pad_value=0.0)
        mask = _center_crop_or_pad(mask, h, w, pad_value=0.0)
    return image, mask


def random_photometric(image, rng: np.random.Generator, factor_range: Tuple[float, float] = PHOTOMETRIC_FACTOR_RANGE):
    """Mild brightness/contrast jitter. Image only -- never applied to the mask."""
    if rng.random() < AUG_PROB:
        brightness = rng.uniform(*factor_range)
        image = np.clip(image * brightness, 0.0, 1.0)
    if rng.random() < AUG_PROB:
        contrast = rng.uniform(*factor_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0.0, 1.0)
    return image


def _center_crop_or_pad(arr: np.ndarray, target_h: int, target_w: int, pad_value: float) -> np.ndarray:
    h, w = arr.shape[:2]

    # crop if larger
    if h > target_h:
        top = (h - target_h) // 2
        arr = arr[top : top + target_h]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        arr = arr[:, left : left + target_w]
        w = target_w

    # pad if smaller
    if h < target_h or w < target_w:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if arr.ndim == 3:
            pad_width.append((0, 0))
        arr = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)

    return arr


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Apply the full train-only augmentation pipeline to one (image, mask) pair."""
    image, mask = random_hflip(image, mask, rng)
    image, mask = random_vflip(image, mask, rng)
    image, mask = random_rotate(image, mask, rng)
    image, mask = random_scale(image, mask, rng)
    image = random_photometric(image, rng)
    mask = (mask > 0.5).astype(np.float32)  # rotation/scale interpolation can only ever be 0/nearest, but re-assert
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)
