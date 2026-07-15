"""Raw-file discovery: where each dataset's images/masks live in data_raw/,
and how an image id maps to its image+mask file paths.

Deliberately dataset-specific (not a generic plugin registry) -- there are
exactly four datasets (methodology section 7) and no more will be added, so
a registry abstraction here would be speculative machinery.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class RawPair:
    id: str
    image_path: Path
    mask_path: Path


class RawDataMissingError(RuntimeError):
    """Raised when a dataset's raw files aren't present under data_raw/.

    `instructions` is the exact, printable message telling the user what to
    download and where to put it -- fail loud with precise instructions,
    never guess or proceed without the real data.
    """

    def __init__(self, dataset: str, instructions: str):
        self.dataset = dataset
        self.instructions = instructions
        super().__init__(f"Raw data missing for {dataset}.\n{instructions}")


def _index_by_stem(d: Path, exts: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not d.is_dir():
        return out
    for ext in exts:
        for p in list(d.glob(f"*{ext}")) + list(d.glob(f"*{ext.upper()}")):
            out.setdefault(p.stem, p)
    return out


def _pair_by_stem(images_dir: Path, masks_dir: Path, exts: Sequence[str]) -> List[RawPair]:
    images = _index_by_stem(images_dir, exts)
    masks = _index_by_stem(masks_dir, exts)

    common_ids = sorted(set(images) & set(masks))
    if not common_ids and images and masks:
        # Bridge zero-padding differences between mirrors (e.g. "01" vs "1"),
        # but only when every stem on both sides is purely numeric -- this is
        # a narrow, explicit fallback, not a silent guess.
        if all(s.isdigit() for s in images) and all(s.isdigit() for s in masks):
            images_by_int = {int(s): p for s, p in images.items()}
            masks_by_int = {int(s): p for s, p in masks.items()}
            common_int_ids = sorted(set(images_by_int) & set(masks_by_int))
            return [
                RawPair(id=str(i), image_path=images_by_int[i], mask_path=masks_by_int[i])
                for i in common_int_ids
            ]
    return [RawPair(id=i, image_path=images[i], mask_path=masks[i]) for i in common_ids]


# ---- Kvasir-SEG -------------------------------------------------------------

KVASIR_IMAGE_COUNT = 1000


def kvasir_root(data_raw_dir: Path) -> Path:
    return Path(data_raw_dir) / "Kvasir-SEG"


def list_kvasir_pairs(data_raw_dir: Path) -> List[RawPair]:
    root = kvasir_root(data_raw_dir)
    images_dir, masks_dir = root / "images", root / "masks"
    pairs = _pair_by_stem(images_dir, masks_dir, exts=(".jpg",)) if images_dir.is_dir() else []
    if not pairs:
        raise RawDataMissingError(
            "kvasir_seg",
            "Kvasir-SEG should auto-download (see download.py). If this failed "
            "(e.g. no network in this environment), download it yourself from "
            "https://datasets.simula.no/downloads/kvasir-seg.zip and extract "
            f"it so that {root} contains 'images/' and 'masks/' subfolders "
            "(1000 matching .jpg files each).",
        )
    return pairs


# ---- CVC-ClinicDB -----------------------------------------------------------

CVC_IMAGE_COUNT = 612
CVC_SEQUENCE_COUNT = 29

_CVC_IMAGE_DIR_NAMES = ("original", "images")
_CVC_MASK_DIR_NAMES = ("ground truth", "groundtruth", "ground_truth", "masks")


def cvc_root(data_raw_dir: Path) -> Path:
    return Path(data_raw_dir) / "CVC-ClinicDB"


def _find_subdir(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    if not root.is_dir():
        return None
    lowered = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _cvc_instructions(root: Path) -> str:
    return (
        "CVC-ClinicDB is not auto-downloadable (registration required). "
        "Download it from https://polyp.grand-challenge.org/CVCClinicDB/ "
        "(or a mirror such as "
        "https://www.kaggle.com/datasets/balraj98/cvcclinicdb) and place it "
        f"so that {root} contains an 'Original' (or 'images') folder with "
        "612 image files and a 'Ground Truth' (or 'masks') folder with 612 "
        "corresponding mask files, named/numbered consistently (e.g. "
        "1.tif..612.tif or 1.png..612.png). A single .zip placed directly in "
        f"{root.parent} with 'cvc' in its filename will be auto-extracted "
        "(download.py)."
    )


def list_cvc_pairs(data_raw_dir: Path) -> List[RawPair]:
    root = cvc_root(data_raw_dir)
    images_dir = _find_subdir(root, _CVC_IMAGE_DIR_NAMES)
    masks_dir = _find_subdir(root, _CVC_MASK_DIR_NAMES)
    pairs = (
        _pair_by_stem(images_dir, masks_dir, exts=(".tif", ".tiff", ".png", ".bmp", ".jpg"))
        if images_dir is not None and masks_dir is not None
        else []
    )
    if not pairs:
        raise RawDataMissingError("cvc_clinicdb", _cvc_instructions(root))
    return pairs


# ---- ISIC2017 / ISIC2018 -----------------------------------------------------

ISIC_IMAGE_COUNTS = {"isic2017": 2000, "isic2018": 2594}
_ISIC_TAGS = {"isic2017": "ISIC2017", "isic2018": "ISIC2018"}


def isic_root(data_raw_dir: Path, version: str) -> Path:
    return Path(data_raw_dir) / _ISIC_TAGS[version]


def _isic_instructions(version: str, root: Path) -> str:
    tag = _ISIC_TAGS[version]
    n = ISIC_IMAGE_COUNTS[version]
    return (
        f"{tag} is not auto-downloadable (registration required). Register "
        "at https://challenge.isic-archive.com/ and download the "
        f"{tag} Task 1-2 Training Input images and Task 1 Training Ground "
        f"Truth masks. Extract so that {root} contains "
        f"'{tag}_Task1-2_Training_Input/' ({n} .jpg images) and "
        f"'{tag}_Task1_Training_GroundTruth/' ({n} *_segmentation.png "
        f"masks). A single .zip placed directly in {root.parent} with "
        f"'{tag.lower()}' in its filename will be auto-extracted (download.py)."
    )


def list_isic_pairs(data_raw_dir: Path, version: str) -> List[RawPair]:
    if version not in ISIC_IMAGE_COUNTS:
        raise ValueError(f"unknown ISIC version: {version!r}")
    tag = _ISIC_TAGS[version]
    root = isic_root(data_raw_dir, version)
    images_dir = root / f"{tag}_Task1-2_Training_Input"
    masks_dir = root / f"{tag}_Task1_Training_GroundTruth"

    pairs: List[RawPair] = []
    if images_dir.is_dir() and masks_dir.is_dir():
        for image_path in sorted(images_dir.glob("*.jpg")):
            mask_path = masks_dir / f"{image_path.stem}_segmentation.png"
            if mask_path.exists():
                pairs.append(RawPair(id=image_path.stem, image_path=image_path, mask_path=mask_path))
    if not pairs:
        raise RawDataMissingError(version, _isic_instructions(version, root))
    return pairs


# ---- dispatch ----------------------------------------------------------

DATASET_NAMES = ("kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018")


def list_pairs(dataset: str, data_raw_dir: Path) -> List[RawPair]:
    if dataset == "kvasir_seg":
        return list_kvasir_pairs(data_raw_dir)
    if dataset == "cvc_clinicdb":
        return list_cvc_pairs(data_raw_dir)
    if dataset in ISIC_IMAGE_COUNTS:
        return list_isic_pairs(data_raw_dir, dataset)
    raise ValueError(f"unknown dataset: {dataset!r} (expected one of {DATASET_NAMES})")
