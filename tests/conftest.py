"""Shared fixtures for the data-pipeline tests: synthetic, self-contained
raw datasets (no network, no dependency on real downloaded data) written
under tmp_path, mirroring the exact directory layouts lcmunet/data/raw_layout.py
expects.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lcmunet.paths import get_paths


def _make_image_mask_pair(size=40, seed=0):
    rng = np.random.default_rng(seed)
    image = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    # a synthetic circular "lesion" so binarisation/normalisation do real work
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    radius = size // 4
    mask[(yy - center) ** 2 + (xx - center) ** 2 <= radius**2] = 255
    return Image.fromarray(image, mode="RGB"), Image.fromarray(mask, mode="L")


@pytest.fixture
def paths(tmp_path):
    return get_paths(root=tmp_path / "drive_root")


@pytest.fixture
def make_kvasir_raw(paths):
    def _make(n=20):
        root = paths.data_raw / "Kvasir-SEG"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, mask = _make_image_mask_pair(seed=i)
            img.save(root / "images" / f"img{i:04d}.jpg")
            mask.save(root / "masks" / f"img{i:04d}.jpg")
        return paths

    return _make


@pytest.fixture
def make_cvc_raw(paths):
    def _make(n=612):
        root = paths.data_raw / "CVC-ClinicDB"
        (root / "Original").mkdir(parents=True, exist_ok=True)
        (root / "Ground Truth").mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            img, mask = _make_image_mask_pair(seed=i)
            img.save(root / "Original" / f"{i}.tif")
            mask.save(root / "Ground Truth" / f"{i}.tif")
        return paths

    return _make


@pytest.fixture
def make_isic_raw(paths):
    def _make(version="isic2017", n=None):
        from lcmunet.data import raw_layout as rl

        tag = "ISIC2017" if version == "isic2017" else "ISIC2018"
        n = n if n is not None else rl.ISIC_IMAGE_COUNTS[version]
        root = paths.data_raw / tag
        images_dir = root / f"{tag}_Task1-2_Training_Input"
        masks_dir = root / f"{tag}_Task1_Training_GroundTruth"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            image_id = f"ISIC_{i:07d}"
            img, mask = _make_image_mask_pair(seed=i)
            img.save(images_dir / f"{image_id}.jpg")
            mask.save(masks_dir / f"{image_id}_segmentation.png")
        return paths

    return _make
