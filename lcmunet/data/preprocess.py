"""Fixed, deterministic preprocessing (methodology section 7): resize 256x256,
intensity-normalise, binarise masks to {0,1}. One scheme, used identically
for baseline/GLGF/LC-SS2D and for every dataset (Fairness rule). The scheme
chosen here is documented in data/PREPROCESSING.md -- that text is meant to
go directly into the paper, so keep the two in sync if this changes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

INPUT_SIZE = 256
MASK_BINARY_THRESHOLD = 127  # out of 0-255; see data/PREPROCESSING.md


def load_image(path) -> np.ndarray:
    """Load an image as RGB uint8, any source format (jpg/png/tif/bmp)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_mask(path) -> np.ndarray:
    """Load a mask as single-channel uint8 (grayscale)."""
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def preprocess_image(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize (bilinear) to size x size and scale to [0,1] float32.

    Per-image min-max via /255 (not a corpus-level mean/std): simple,
    deterministic, and never depends on train/val/test statistics, so there
    is no risk of val/test information leaking into normalisation.
    """
    img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (H, W, 3), range [0, 1]


def preprocess_mask(mask: np.ndarray, size: int = INPUT_SIZE, threshold: int = MASK_BINARY_THRESHOLD) -> np.ndarray:
    """Resize (nearest) to size x size and binarise to {0,1} float32.

    Nearest-neighbour (not bilinear) resize keeps mask edges hard instead of
    blurring them into intermediate grey values, which matters for small
    lesions/polyps -- consistent with the "no augmentation that erases small
    lesions" rule extended to preprocessing.
    """
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img = img.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    return (arr > threshold).astype(np.float32)  # (H, W), {0, 1}


def load_and_preprocess_pair(image_path, mask_path, size: int = INPUT_SIZE):
    """Convenience: load + preprocess an (image, mask) file pair in one call."""
    image = preprocess_image(load_image(image_path), size=size)
    mask = preprocess_mask(load_mask(mask_path), size=size)
    return image, mask
