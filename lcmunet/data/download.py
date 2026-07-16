"""Dataset acquisition (methodology section 7) -- EXTRACTION ONLY. The user
downloads all four datasets as .zip files by hand and places them under
DRIVE_ROOT/data_raw/<CanonicalName>/ (any filename); this module never
downloads anything itself (no wget, no Kaggle API, no S3 URLs). Where to
get each archive from (for the placement instructions printed when none is
found):

  1. Kvasir-SEG: https://datasets.simula.no/downloads/kvasir-seg.zip (or
     search datasets.simula.no/kvasir-seg if that link has moved).
     Structure after extraction: images/ and masks/, matching filenames.
  2. CVC-ClinicDB: Kaggle dataset "balraj98/cvcclinicdb" (or the official
     release at https://polyp.grand-challenge.org/CVCClinicDB/). Structure
     after extraction: "Original/" and "Ground Truth/" folders,
     1.png..612.png.
  3. ISIC2018: Kaggle dataset
     "tschandl/isic2018-challenge-task1-data-segmentation". Training
     portion ONLY (2594 pairs) is used; Validation/Test are detected and
     ignored automatically (no usable masks for our protocol).
  4. ISIC2017: Kaggle dataset "johnchfr/isic-2017". Training(2000)+
     Validation(150)+Test(600)=2750 are combined into one pool, then
     Training is subsampled to 1250 (fixed seed) so the final kept pool
     matches UltraLight's 1250/150/600 protocol (2000 total, methodology
     section 7) -- exactly which images were kept is written to
     data_raw/ISIC2017/isic2017_source_manifest.json.

Every ensure_* function is idempotent: it first checks whether the
canonical extracted layout raw_layout.py expects already has the right
pair count (fast path, no filesystem scan beyond that) before looking for
a placed .zip.

The ISIC2018/ISIC2017 Kaggle mirrors' internal folder layout is not under
this project's control and can vary, so both use best-effort directory
detection (_detect_bucket_dirs) instead of a hardcoded path -- and FAIL
LOUD with a full directory listing if detection can't confidently settle
on exactly one images dir and one masks dir, never silently guessing
wrong. Whatever naming/suffix convention is actually found is printed
("detect and log the transformed naming"), and every file is normalised
(format-converted if needed) into the canonical layout raw_layout.py
expects, so no other module has to know about mirror-specific quirks.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from lcmunet.data import cvc_sequence
from lcmunet.data import raw_layout as rl

# Documentation-only hints for the "no zip found" instructions -- NOT used to
# download anything (see module docstring).
KVASIR_URL_HINT = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CVC_KAGGLE_SLUG_HINT = "balraj98/cvcclinicdb"
ISIC2018_KAGGLE_SLUG_HINT = "tschandl/isic2018-challenge-task1-data-segmentation"
ISIC2017_KAGGLE_SLUG_HINT = "johnchfr/isic-2017"

ISIC2017_TRAIN_KEEP = 1250  # UltraLight's protocol: 1250/150/600 (methodology section 7)

_MASK_HINTS = ("groundtruth", "ground_truth", "ground-truth", "mask", "segmentation", "label", "annotation")


# ---- generic helpers ---------------------------------------------------------


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _find_any_zip(root: Path) -> Optional[Path]:
    """The user manually places exactly one archive anywhere under `root`
    (the dataset's own data_raw/<Name>/ folder); any filename is accepted.
    If more than one is found, the first (alphabetically) is used and a
    warning is printed -- never silently picks one without saying so."""
    if not root.is_dir():
        return None
    matches = sorted(root.rglob("*.zip"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: found {len(matches)} .zip files under {root}; using {matches[0].name} (first alphabetically). Remove the others if this is wrong.")
    return matches[0]


_CVC_FORMAT_PREFERENCE = ("png", "tif", "tiff")


def _select_cvc_format(root: Path) -> None:
    """The official CVC-ClinicDB release (and the balraj98 Kaggle mirror)
    ship BOTH a 'PNG/' and a 'TIF/' top-level folder, each containing its own
    complete 'Original'/'Ground Truth' pair for the same 612 frames -- pick
    one (PNG preferred: no TIFF codec dependency) and flatten it into root;
    delete the other format folder so it can never spuriously match. No-op
    if root already directly contains Original/Ground Truth (archives that
    don't have this PNG/TIF split)."""
    if not root.is_dir():
        return
    lowered = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    if any(name in lowered for name in rl._CVC_IMAGE_DIR_NAMES + rl._CVC_MASK_DIR_NAMES):
        return  # already flat -- nothing to select between
    chosen_fmt = next((fmt for fmt in _CVC_FORMAT_PREFERENCE if fmt in lowered), None)
    if chosen_fmt is None:
        return  # not this layout; list_cvc_pairs will raise its own fail-loud error
    chosen = lowered[chosen_fmt]
    other_fmts = sorted(fmt for fmt in _CVC_FORMAT_PREFERENCE if fmt in lowered and fmt != chosen_fmt)
    print(f"CVC-ClinicDB archive ships multiple formats {sorted(lowered)}; using {chosen.name}/ (flattening into {root}, discarding {other_fmts}).")
    for item in chosen.iterdir():
        shutil.move(str(item), str(root / item.name))
    chosen.rmdir()
    for fmt in other_fmts:
        shutil.rmtree(lowered[fmt])


def _relocate_nested_extraction(root: Path, expected_subdir_groups: Sequence[Sequence[str]], max_depth: int = 5) -> None:
    """If `root` doesn't directly contain any of the expected subdirectories
    (case-insensitive), descend through a chain of pure single-subdirectory
    wrapper folders -- each level with exactly one nested directory and
    nothing else meaningful to disambiguate -- until a level directly
    contains the expected subdirectories, then flatten that level's contents
    up into root and discard the now-empty wrapper chain. Handles archives
    wrapped in one extra folder (e.g. 'cvcclinicdb-master/Original/...') AND
    double-wrapped ones (e.g. the debeshjha1/kvasirseg Kaggle mirror nests
    'Kvasir-SEG/Kvasir-SEG/images/...'). Stops (without guessing) the moment
    a level has zero or multiple directory children and still doesn't match
    -- list_*_pairs's own fail-loud error takes over from there."""

    def has_any(d: Path) -> bool:
        if not d.is_dir():
            return False
        lowered = {p.name.lower() for p in d.iterdir() if p.is_dir()}
        return any(any(name in lowered for name in group) for group in expected_subdir_groups)

    def only_subdir(d: Path) -> Optional[Path]:
        subdirs = [c for c in d.iterdir() if c.is_dir()]
        return subdirs[0] if len(subdirs) == 1 else None

    chain: list[Path] = []
    current = root
    for _ in range(max_depth):
        if has_any(current):
            if not chain:
                return  # already flat at root -- nothing to do
            print(f"Extraction wrapped in {len(chain)} extra folder(s) ({'/'.join(p.name for p in chain)}); flattening into {root}.")
            for item in current.iterdir():
                shutil.move(str(item), str(root / item.name))
            shutil.rmtree(chain[0])
            return
        nxt = only_subdir(current)
        if nxt is None:
            return  # dead end (nothing nested) or ambiguous (multiple candidates) -- don't guess
        chain.append(nxt)
        current = nxt


def _iter_leaf_dirs_with_many_files(root: Path, exts: Sequence[str], min_files: int):
    dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for d in dirs:
        n = sum(1 for ext in exts for _ in d.glob(f"*{ext}")) + sum(1 for ext in exts for _ in d.glob(f"*{ext.upper()}"))
        if n >= min_files:
            yield d, n


def _detect_bucket_dirs(root: Path, include_hints: Sequence[str], exclude_hints: Sequence[str], min_files: int = 20) -> Tuple[Path, Path, str]:
    """Best-effort detection of one images dir + one masks dir under `root`
    whose path components mention `include_hints` (case-insensitive) and
    none of `exclude_hints`; falls back to "every non-excluded candidate" if
    nothing matches include_hints (some mirrors ship a single bucket with no
    keyword anywhere in the path). Only ever used against a Kaggle mirror's
    structure, which this project cannot inspect ahead of time -- see module
    docstring. Raises with a full directory listing if it can't settle on
    exactly one of each; never silently guesses.
    """
    candidates = list(_iter_leaf_dirs_with_many_files(root, (".jpg", ".jpeg", ".png"), min_files=min_files))
    if not candidates:
        raise RuntimeError(f"No directory under {root} contains >={min_files} image files -- unexpected archive layout.")

    def matches(d: Path, hints: Sequence[str]) -> bool:
        parts = [p.lower() for p in d.relative_to(root).parts] or [d.name.lower()]
        return any(any(h in part for h in hints) for part in parts)

    non_excluded = [(d, n) for d, n in candidates if not exclude_hints or not matches(d, exclude_hints)]
    labelled = [(d, n) for d, n in non_excluded if matches(d, include_hints)]
    pool = labelled if labelled else non_excluded

    mask_pool = [(d, n) for d, n in pool if matches(d, _MASK_HINTS)]
    image_pool = [(d, n) for d, n in pool if not matches(d, _MASK_HINTS)]

    if len(image_pool) != 1 or len(mask_pool) != 1:
        listing = "\n".join(f"  {d} ({n} files)" for d, n in candidates)
        raise RuntimeError(
            f"Could not confidently detect exactly one images dir and one masks "
            f"dir under {root} for hints={include_hints!r} (excluding "
            f"{exclude_hints!r}); found {len(image_pool)} image candidate(s) and "
            f"{len(mask_pool)} mask candidate(s):\n{listing}\n"
            "Inspect the archive by hand and adjust lcmunet/data/download.py's "
            "_detect_bucket_dirs if the mirror's layout has changed."
        )
    return image_pool[0][0], mask_pool[0][0], f"{len(candidates)} candidate dirs scanned, hints={include_hints!r}"


def _find_files(root: Path, exts: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not root.is_dir():
        return out
    for ext in exts:
        for p in sorted(root.glob(f"*{ext}")) + sorted(root.glob(f"*{ext.upper()}")):
            out.setdefault(p.stem, p)
    return out


def _pair_isic_style(images: Dict[str, Path], masks: Dict[str, Path]) -> Tuple[Dict[str, Tuple[Path, Path]], str]:
    """Pairs image stem `X` with mask stem `X<suffix>`, trying the canonical
    ISIC "_segmentation" suffix first, then a couple of common variants
    (including the capitalised "_Segmentation" some Kaggle mirrors use),
    then exact-stem (no suffix). Returns whichever convention matched the
    most pairs, plus that suffix string (so callers can log it)."""
    best: Dict[str, Tuple[Path, Path]] = {}
    best_suffix = ""
    for suffix in ("_segmentation", "_Segmentation", "_mask", ""):
        candidate = {stem: (img, masks[f"{stem}{suffix}"]) for stem, img in images.items() if f"{stem}{suffix}" in masks}
        if len(candidate) > len(best):
            best, best_suffix = candidate, suffix
    return best, best_suffix


def _copy_as_format(src: Path, dst: Path, fmt: str) -> None:
    """Copies src to dst, converting to `fmt` ('JPEG' or 'PNG') only if the
    source extension doesn't already match -- so raw_layout's fixed-extension
    glob (*.jpg images, *_segmentation.png masks) works regardless of what
    format a given mirror actually shipped."""
    wants_ext = ".jpg" if fmt == "JPEG" else ".png"
    if src.suffix.lower() == wants_ext:
        shutil.copy(src, dst)
        return
    from PIL import Image as _PILImage

    with _PILImage.open(src) as img:
        if fmt == "JPEG":
            img = img.convert("RGB")
        img.save(dst, format=fmt)


# ---- Kvasir-SEG ---------------------------------------------------------------


def ensure_kvasir(data_raw_dir: str | Path) -> int:
    """Idempotent: only extracts if not already present with the right count."""
    data_raw_dir = Path(data_raw_dir)
    root = rl.kvasir_root(data_raw_dir)

    try:
        pairs = rl.list_kvasir_pairs(data_raw_dir)
        if len(pairs) == rl.KVASIR_IMAGE_COUNT:
            print(f"Kvasir-SEG already present and verified: {len(pairs)} pairs. Skipping extraction.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    zip_path = _find_any_zip(root)
    if zip_path is None:
        raise rl.RawDataMissingError(
            "kvasir_seg",
            f"No Kvasir-SEG .zip found under {root}. Download it from "
            f"{KVASIR_URL_HINT} and place the .zip file anywhere under {root} "
            "(any filename) -- it will be auto-extracted. Expected structure "
            "after extraction: 'images/' and 'masks/' subfolders with matching "
            f"filenames ({rl.KVASIR_IMAGE_COUNT} pairs).",
        )

    print(f"Found {zip_path}; extracting -> {root} ...")
    _extract_zip(zip_path, root)
    _relocate_nested_extraction(root, [("images",), ("masks",)])  # the official zip nests one "Kvasir-SEG/" wrapper folder

    pairs = rl.list_kvasir_pairs(data_raw_dir)
    if len(pairs) != rl.KVASIR_IMAGE_COUNT:
        raise RuntimeError(
            f"Kvasir-SEG extracted to {len(pairs)} image/mask pairs under {root} "
            f"(from {zip_path.name}), expected {rl.KVASIR_IMAGE_COUNT}. Verify "
            "it's the complete, official archive."
        )
    print(f"Kvasir-SEG ready: {len(pairs)} pairs at {root}")
    return len(pairs)


# ---- CVC-ClinicDB ---------------------------------------------------------------


def ensure_cvc(data_raw_dir: str | Path) -> int:
    data_raw_dir = Path(data_raw_dir)
    root = rl.cvc_root(data_raw_dir)

    try:
        pairs = rl.list_cvc_pairs(data_raw_dir)
        if len(pairs) == rl.CVC_IMAGE_COUNT:
            print(f"CVC-ClinicDB already present and verified: {len(pairs)} pairs. Skipping extraction.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    zip_path = _find_any_zip(root)
    if zip_path is None:
        raise rl.RawDataMissingError(
            "cvc_clinicdb",
            f"No CVC-ClinicDB .zip found under {root}. Download the Kaggle "
            f"dataset '{CVC_KAGGLE_SLUG_HINT}' (or the official release from "
            "https://polyp.grand-challenge.org/CVCClinicDB/) and place the .zip "
            f"file anywhere under {root} (any filename) -- it will be "
            "auto-extracted. Expected structure after extraction: an "
            "'Original/' folder and a 'Ground Truth/' folder, "
            f"1.png..{rl.CVC_IMAGE_COUNT}.png.",
        )

    print(f"Found {zip_path}; extracting -> {root} ...")
    _extract_zip(zip_path, root)
    _relocate_nested_extraction(root, [rl._CVC_IMAGE_DIR_NAMES, rl._CVC_MASK_DIR_NAMES, _CVC_FORMAT_PREFERENCE])
    _select_cvc_format(root)
    pairs = rl.list_cvc_pairs(data_raw_dir)

    if len(pairs) != rl.CVC_IMAGE_COUNT:
        raise RuntimeError(
            f"CVC-ClinicDB has {len(pairs)} matched image/mask pairs under "
            f"{root} (from {zip_path.name}), expected {rl.CVC_IMAGE_COUNT}. "
            "Verify it's the complete, official 612-frame release."
        )
    print(f"CVC-ClinicDB ready: {len(pairs)} pairs.")
    return len(pairs)


# ---- ISIC2018 (Training portion only) --------------------------------------


def ensure_isic2018(data_raw_dir: str | Path) -> int:
    data_raw_dir = Path(data_raw_dir)
    version = "isic2018"
    root = rl.isic_root(data_raw_dir, version)

    try:
        pairs = rl.list_isic_pairs(data_raw_dir, version)
        if len(pairs) == rl.ISIC_IMAGE_COUNTS[version]:
            print(f"ISIC2018 already present and verified: {len(pairs)} pairs. Skipping extraction.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    zip_path = _find_any_zip(root)
    if zip_path is None:
        raise rl.RawDataMissingError(
            version,
            f"No ISIC2018 .zip found under {root}. Download the Kaggle dataset "
            f"'{ISIC2018_KAGGLE_SLUG_HINT}' and place the .zip file anywhere "
            f"under {root} (any filename) -- it will be auto-extracted. Only "
            f"the Training portion ({rl.ISIC_IMAGE_COUNTS[version]} images + "
            f"{rl.ISIC_IMAGE_COUNTS[version]} masks) is used; Validation/Test "
            "are detected and ignored automatically.",
        )

    staging = Path(tempfile.mkdtemp(prefix="isic2018_staging_"))
    try:
        print(f"Found {zip_path}; extracting -> {staging} ...")
        _extract_zip(zip_path, staging)

        images_dir, masks_dir, note = _detect_bucket_dirs(staging, include_hints=("train", "training"), exclude_hints=("valid", "test"), min_files=50)
        print(f"Detected ISIC2018 Training layout: images={images_dir} masks={masks_dir} ({note})")

        images = _find_files(images_dir, (".jpg", ".jpeg", ".png"))
        masks = _find_files(masks_dir, (".png", ".jpg", ".jpeg"))
        matched, suffix_used = _pair_isic_style(images, masks)
        print(f"Detected ISIC2018 mask naming: stem + {suffix_used!r} ({len(matched)} pairs found)")

        expected = rl.ISIC_IMAGE_COUNTS[version]
        if len(matched) != expected:
            raise RuntimeError(
                f"ISIC2018 zip {zip_path.name} produced {len(matched)} Training "
                f"image/mask pairs, expected exactly {expected}. Inspect "
                f"{staging} by hand -- the archive's layout may not match "
                "what's expected."
            )

        dest_images = root / "ISIC2018_Task1-2_Training_Input"
        dest_masks = root / "ISIC2018_Task1_Training_GroundTruth"
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_masks.mkdir(parents=True, exist_ok=True)
        for stem, (img_path, mask_path) in matched.items():
            _copy_as_format(img_path, dest_images / f"{stem}.jpg", "JPEG")
            _copy_as_format(mask_path, dest_masks / f"{stem}_segmentation.png", "PNG")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    pairs = rl.list_isic_pairs(data_raw_dir, version)
    if len(pairs) != rl.ISIC_IMAGE_COUNTS[version]:
        raise RuntimeError(
            f"ISIC2018 has {len(pairs)} matched image/mask pairs under {root} "
            f"after normalisation, expected {rl.ISIC_IMAGE_COUNTS[version]}."
        )
    print(f"ISIC2018 ready: {len(pairs)} pairs (Training portion only) at {root}")
    return len(pairs)


# ---- ISIC2017 (combined pool, subsampled to 1250/150/600) -------------------


def ensure_isic2017(data_raw_dir: str | Path) -> int:
    data_raw_dir = Path(data_raw_dir)
    version = "isic2017"
    root = rl.isic_root(data_raw_dir, version)

    try:
        pairs = rl.list_isic_pairs(data_raw_dir, version)
        if len(pairs) == rl.ISIC_IMAGE_COUNTS[version]:
            print(f"ISIC2017 already present and verified: {len(pairs)} pairs. Skipping extraction.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    zip_path = _find_any_zip(root)
    if zip_path is None:
        raise rl.RawDataMissingError(
            version,
            f"No ISIC2017 .zip found under {root}. Download the Kaggle dataset "
            f"'{ISIC2017_KAGGLE_SLUG_HINT}' and place the .zip file anywhere "
            f"under {root} (any filename) -- it will be auto-extracted. "
            "Training/Validation/Test are combined into one pool, then "
            f"Training is subsampled to {ISIC2017_TRAIN_KEEP} (fixed seed) so "
            f"the final kept pool matches {rl.ISIC_IMAGE_COUNTS[version]} total "
            "(UltraLight's 1250/150/600 protocol).",
        )

    staging = Path(tempfile.mkdtemp(prefix="isic2017_staging_"))
    try:
        print(f"Found {zip_path}; extracting -> {staging} ...")
        extracted = staging / "extracted"
        _extract_zip(zip_path, extracted)

        buckets: Dict[str, Dict[str, Dict[str, Path]]] = {}
        for bucket, include_hints, exclude_hints in (
            ("train", ("train", "training"), ("valid", "test")),
            ("val", ("valid", "validation"), ("train", "test")),
            ("test", ("test",), ("train", "valid")),
        ):
            images_dir, masks_dir, note = _detect_bucket_dirs(extracted, include_hints, exclude_hints, min_files=20)
            print(f"ISIC2017: detected {bucket} images={images_dir} masks={masks_dir} ({note})")
            buckets[bucket] = {
                "images": _find_files(images_dir, (".jpg", ".jpeg", ".png")),
                "masks": _find_files(masks_dir, (".png", ".jpg", ".jpeg")),
            }

        paired: Dict[str, Dict[str, Tuple[Path, Path]]] = {}
        for bucket in ("train", "val", "test"):
            matched, suffix_used = _pair_isic_style(buckets[bucket]["images"], buckets[bucket]["masks"])
            paired[bucket] = matched
            print(f"ISIC2017 bucket={bucket}: {len(matched)} matched pairs (mask suffix {suffix_used!r})")

        train_ids_sorted = sorted(paired["train"].keys())
        if len(train_ids_sorted) < ISIC2017_TRAIN_KEEP:
            raise RuntimeError(f"ISIC2017 Training bucket only has {len(train_ids_sorted)} pairs, need at least {ISIC2017_TRAIN_KEEP} to subsample from.")
        kept_train_ids = sorted(random.Random(cvc_sequence.SPLIT_SEED).sample(train_ids_sorted, ISIC2017_TRAIN_KEEP))

        kept_by_bucket = {
            "train": kept_train_ids,
            "val": sorted(paired["val"].keys()),
            "test": sorted(paired["test"].keys()),
        }
        total_kept = sum(len(v) for v in kept_by_bucket.values())
        if total_kept != rl.ISIC_IMAGE_COUNTS[version]:
            raise RuntimeError(
                f"ISIC2017 selection produced {total_kept} total pairs "
                f"({', '.join(f'{k}={len(v)}' for k, v in kept_by_bucket.items())}), "
                f"expected exactly {rl.ISIC_IMAGE_COUNTS[version]}."
            )

        dest_images = root / "ISIC2017_Task1-2_Training_Input"
        dest_masks = root / "ISIC2017_Task1_Training_GroundTruth"
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_masks.mkdir(parents=True, exist_ok=True)
        for bucket, ids in kept_by_bucket.items():
            for stem in ids:
                img_path, mask_path = paired[bucket][stem]
                _copy_as_format(img_path, dest_images / f"{stem}.jpg", "JPEG")
                _copy_as_format(mask_path, dest_masks / f"{stem}_segmentation.png", "PNG")

        manifest = {
            "source_zip": zip_path.name,
            "train_pool_size": len(train_ids_sorted),
            "train_kept": kept_by_bucket["train"],
            "val_kept": kept_by_bucket["val"],
            "test_kept": kept_by_bucket["test"],
            "subsample_seed": cvc_sequence.SPLIT_SEED,
        }
        manifest_path = root / "isic2017_source_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Wrote {manifest_path} documenting exactly which ISIC2017 images were kept.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    pairs = rl.list_isic_pairs(data_raw_dir, version)
    if len(pairs) != rl.ISIC_IMAGE_COUNTS[version]:
        raise RuntimeError(
            f"ISIC2017 has {len(pairs)} matched image/mask pairs under {root} "
            f"after selection, expected {rl.ISIC_IMAGE_COUNTS[version]}."
        )
    print(f"ISIC2017 ready: {len(pairs)} pairs at {root}")
    return len(pairs)


# ---- dispatch ----------------------------------------------------------


def ensure_dataset_ready(dataset: str, data_raw_dir: str | Path) -> int:
    """Dispatch to the right ensure_* function. Raises RawDataMissingError
    or RuntimeError if the dataset isn't genuinely ready -- callers must not
    catch these to "proceed anyway" (lcmunet.data.prepare_all.
    prepare_all_datasets catches per-dataset so one missing/bad archive
    doesn't block the other three)."""
    if dataset == "kvasir_seg":
        return ensure_kvasir(data_raw_dir)
    if dataset == "cvc_clinicdb":
        return ensure_cvc(data_raw_dir)
    if dataset == "isic2017":
        return ensure_isic2017(data_raw_dir)
    if dataset == "isic2018":
        return ensure_isic2018(data_raw_dir)
    raise ValueError(f"unknown dataset: {dataset!r} (expected one of {rl.DATASET_NAMES})")


if __name__ == "__main__":
    import argparse

    from lcmunet.paths import get_paths

    parser = argparse.ArgumentParser(description="Extract raw datasets already placed under data_raw/.")
    parser.add_argument("datasets", nargs="*", default=list(rl.DATASET_NAMES), choices=list(rl.DATASET_NAMES))
    args = parser.parse_args()

    paths = get_paths()
    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            ensure_dataset_ready(name, paths.data_raw)
        except rl.RawDataMissingError:
            print(f"[STOPPED] {name}: raw data not available; see instructions above.")
