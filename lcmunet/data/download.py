"""Dataset acquisition (methodology section 7). LOCKED sources -- do not
substitute without updating this docstring and the calling prompt:

  1. Kvasir-SEG: direct wget, no auth (KVASIR_URL below). Verified live by
     HEAD request when this module was written; ensure_kvasir() still fails
     loud with the exact instructions if the archive ever moves.
  2. CVC-ClinicDB: Kaggle dataset "balraj98/cvcclinicdb" (CVC_KAGGLE_SLUG).
  3. ISIC2018: Kaggle dataset
     "tschandl/isic2018-challenge-task1-data-segmentation"
     (ISIC2018_KAGGLE_SLUG). Training portion ONLY (2594 pairs); Validation/
     Test portions are ignored (no usable masks for our protocol).
  4. ISIC2017: PRIMARY is the official ISIC S3 bucket, direct wget, no auth
     (ISIC2017_S3_PARTS below -- verified live by HEAD request when this
     module was written). FALLBACK, only if the S3 primary fails on any of
     the 6 parts, is the Kaggle dataset "johnchfr/isic-2017"
     (ISIC2017_KAGGLE_FALLBACK_SLUG) -- structure unverified against the
     official split, so ensure_isic2017() prints a WARNING whenever it is
     actually used. Either way the official Training(2000)+Validation(150)+
     Test(600)=2750 pool is combined, then Training is subsampled down to
     1250 (fixed seed) so the final kept pool matches UltraLight's
     1250/150/600 protocol (2000 total, methodology section 7) -- exactly
     which images were kept is written to
     data_raw/ISIC2017/isic2017_source_manifest.json, and which source
     succeeded is recorded in results/env.json under "isic2017_source".

CVC-ClinicDB, ISIC2018, and the ISIC2017 Kaggle fallback all need a Kaggle
API token (lcmunet.data.kaggle_auth); Kvasir-SEG and the ISIC2017 S3 primary
path do not and must proceed regardless of whether Kaggle auth is set up.

Every ensure_* function is idempotent: it first checks whether the
canonical extracted layout raw_layout.py expects already has the right
pair count (fast path, no network at all), then checks for an
already-downloaded archive before hitting the network again.

Kaggle mirrors are not under this project's control and their internal
folder layout cannot be verified without live Kaggle credentials at the
time this module was written, so ISIC2018 and the ISIC2017 Kaggle fallback
use best-effort directory detection (_detect_bucket_dirs) instead of a
hardcoded path, and FAIL LOUD with a full directory listing if detection
can't confidently settle on exactly one images dir and one masks dir --
never silently guesses wrong. Whatever naming/suffix convention is actually
found is printed (methodology prompt: "detect and log the transformed
naming"), and every file is normalised (format-converted if needed) into
the canonical layout raw_layout.py expects, so no other module has to know
about mirror-specific quirks.
"""

from __future__ import annotations

import json
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from lcmunet.data import cvc_sequence
from lcmunet.data import raw_layout as rl
from lcmunet.data.kaggle_auth import ensure_kaggle_auth

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_ZIP_NAME = "kvasir-seg.zip"

CVC_KAGGLE_SLUG = "balraj98/cvcclinicdb"

ISIC2018_KAGGLE_SLUG = "tschandl/isic2018-challenge-task1-data-segmentation"

ISIC2017_S3_BASE = "https://isic-challenge-data.s3.amazonaws.com/2017"
# (filename, kind, bucket) -- kind in {"images","masks"}, bucket in {"train","val","test"}.
ISIC2017_S3_PARTS: Tuple[Tuple[str, str, str], ...] = (
    ("ISIC-2017_Training_Data.zip", "images", "train"),
    ("ISIC-2017_Training_Part1_GroundTruth.zip", "masks", "train"),
    ("ISIC-2017_Validation_Data.zip", "images", "val"),
    ("ISIC-2017_Validation_Part1_GroundTruth.zip", "masks", "val"),
    ("ISIC-2017_Test_v2_Data.zip", "images", "test"),
    ("ISIC-2017_Test_v2_Part1_GroundTruth.zip", "masks", "test"),
)
ISIC2017_BUCKET_COUNTS = {"train": 2000, "val": 150, "test": 600}
ISIC2017_TRAIN_KEEP = 1250  # UltraLight's protocol: 1250/150/600 (methodology section 7)
ISIC2017_KAGGLE_FALLBACK_SLUG = "johnchfr/isic-2017"

DOWNLOAD_TIMEOUT_SECONDS = 1800  # generous: the ISIC2017 S3 Training_Data.zip alone is ~6GB
_MASK_HINTS = ("groundtruth", "ground_truth", "ground-truth", "mask", "segmentation", "label", "annotation")


# ---- generic helpers ---------------------------------------------------------


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _find_archive(data_raw_dir: Path, name_substrings: Sequence[str]) -> Optional[Path]:
    """Manual-override escape hatch: if the user already placed a matching
    .zip directly under data_raw/, use it instead of hitting the network --
    same behaviour for every Kaggle-backed dataset, not just CVC."""
    data_raw_dir = Path(data_raw_dir)
    if not data_raw_dir.is_dir():
        return None
    for p in sorted(data_raw_dir.rglob("*.zip")):
        lname = p.name.lower()
        if any(s in lname for s in name_substrings):
            return p
    return None


def _urlretrieve_with_timeout(url: str, dest: Path, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        urllib.request.urlretrieve(url, dest)
    finally:
        socket.setdefaulttimeout(old_timeout)


def _kaggle_cli_download(slug: str, dest_dir: Path, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> Path:
    """Downloads a Kaggle dataset zip (no --unzip -- we control extraction
    ourselves, same as every other archive here) via
    `python -m kaggle datasets download`. Assumes
    lcmunet.data.kaggle_auth.ensure_kaggle_auth() already succeeded."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--force"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`kaggle datasets download -d {slug}` failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    expected = dest_dir / (slug.split("/")[-1] + ".zip")
    if expected.is_file():
        return expected
    candidates = sorted(dest_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"`kaggle datasets download -d {slug}` reported success but no .zip file appeared in {dest_dir}")
    return candidates[0]


def _relocate_nested_extraction(root: Path, expected_subdir_groups: Sequence[Sequence[str]]) -> None:
    """If `root` doesn't directly contain any of the expected subdirectories
    (case-insensitive) but exactly one nested child does, move that child's
    contents up into root -- handles archives that wrap everything in an
    extra top-level folder (e.g. 'cvcclinicdb-master/Original/...')."""

    def has_any(d: Path) -> bool:
        if not d.is_dir():
            return False
        lowered = {p.name.lower() for p in d.iterdir() if p.is_dir()}
        return any(any(name in lowered for name in group) for group in expected_subdir_groups)

    if has_any(root):
        return
    nested_hits = [c for c in root.iterdir() if c.is_dir() and has_any(c)]
    if len(nested_hits) == 1:
        wrapper = nested_hits[0]
        print(f"Extraction wrapped in an extra folder ({wrapper.name}); flattening into {root}.")
        for item in wrapper.iterdir():
            shutil.move(str(item), str(root / item.name))
        wrapper.rmdir()


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
    keyword anywhere in the path). Only ever used against a live Kaggle
    mirror's structure, which this project cannot inspect ahead of time --
    see module docstring. Raises with a full directory listing if it can't
    settle on exactly one of each; never silently guesses.
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
    ISIC "_segmentation" suffix first, then a couple of common variants,
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


def ensure_kvasir(data_raw_dir: str | Path, force_download: bool = False) -> int:
    """Idempotent: only downloads+extracts if not already present with the right count."""
    data_raw_dir = Path(data_raw_dir)
    root = rl.kvasir_root(data_raw_dir)

    if not force_download:
        try:
            pairs = rl.list_kvasir_pairs(data_raw_dir)
        except rl.RawDataMissingError:
            pairs = []
        if len(pairs) == rl.KVASIR_IMAGE_COUNT:
            print(f"Kvasir-SEG already present and verified: {len(pairs)} pairs. Skipping download.")
            return len(pairs)
        if pairs:
            print(f"Kvasir-SEG present but count mismatch ({len(pairs)} != {rl.KVASIR_IMAGE_COUNT}); re-downloading.")

    zip_path = data_raw_dir / KVASIR_ZIP_NAME
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kvasir-SEG from {KVASIR_URL} ...")
    _urlretrieve_with_timeout(KVASIR_URL, zip_path)
    print(f"Downloaded {zip_path} ({zip_path.stat().st_size} bytes). Extracting...")
    _extract_zip(zip_path, data_raw_dir)

    pairs = rl.list_kvasir_pairs(data_raw_dir)
    if len(pairs) != rl.KVASIR_IMAGE_COUNT:
        raise RuntimeError(
            f"Kvasir-SEG download/extract produced {len(pairs)} image/mask pairs, "
            f"expected {rl.KVASIR_IMAGE_COUNT}. The archive at {KVASIR_URL} may have "
            f"changed -- inspect {root} by hand."
        )
    print(f"Kvasir-SEG ready: {len(pairs)} pairs at {root}")
    return len(pairs)


# ---- CVC-ClinicDB ---------------------------------------------------------------


def ensure_cvc(data_raw_dir: str | Path, paths) -> int:
    data_raw_dir = Path(data_raw_dir)
    try:
        pairs = rl.list_cvc_pairs(data_raw_dir)
        if len(pairs) == rl.CVC_IMAGE_COUNT:
            print(f"CVC-ClinicDB already present and verified: {len(pairs)} pairs. Skipping download.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    root = rl.cvc_root(data_raw_dir)
    archive = _find_archive(data_raw_dir, ("cvc",))
    if archive is not None:
        print(f"Found cached/placed archive {archive}; skipping re-download.")
    else:
        default_zip = root / "cvcclinicdb.zip"
        if default_zip.is_file():
            print(f"Found cached archive {default_zip}; skipping re-download.")
            archive = default_zip
        else:
            ensure_kaggle_auth(paths)
            print(f"Downloading CVC-ClinicDB (Kaggle: {CVC_KAGGLE_SLUG}) ...")
            archive = _kaggle_cli_download(CVC_KAGGLE_SLUG, root)

    print(f"Extracting {archive} -> {root} ...")
    _extract_zip(archive, root)
    _relocate_nested_extraction(root, [rl._CVC_IMAGE_DIR_NAMES, rl._CVC_MASK_DIR_NAMES])
    pairs = rl.list_cvc_pairs(data_raw_dir)

    if len(pairs) != rl.CVC_IMAGE_COUNT:
        raise RuntimeError(
            f"CVC-ClinicDB has {len(pairs)} matched image/mask pairs under {root}, "
            f"expected {rl.CVC_IMAGE_COUNT}. Verify the Kaggle dataset "
            f"'{CVC_KAGGLE_SLUG}' still matches the official 612-frame release."
        )
    print(f"CVC-ClinicDB ready: {len(pairs)} pairs.")
    return len(pairs)


# ---- ISIC2018 (Training portion only) --------------------------------------


def ensure_isic2018(data_raw_dir: str | Path, paths) -> int:
    data_raw_dir = Path(data_raw_dir)
    version = "isic2018"
    try:
        pairs = rl.list_isic_pairs(data_raw_dir, version)
        if len(pairs) == rl.ISIC_IMAGE_COUNTS[version]:
            print(f"ISIC2018 already present and verified: {len(pairs)} pairs. Skipping download.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    root = rl.isic_root(data_raw_dir, version)
    archive = _find_archive(data_raw_dir, ("isic2018", "isic-2018"))
    if archive is not None:
        print(f"Found cached/placed archive {archive}; skipping re-download.")
    else:
        default_zip = root / "isic2018-challenge-task1-data-segmentation.zip"
        if default_zip.is_file():
            print(f"Found cached archive {default_zip}; skipping re-download.")
            archive = default_zip
        else:
            ensure_kaggle_auth(paths)
            print(f"Downloading ISIC2018 (Kaggle: {ISIC2018_KAGGLE_SLUG}) ...")
            archive = _kaggle_cli_download(ISIC2018_KAGGLE_SLUG, root)

    staging = Path(tempfile.mkdtemp(prefix="isic2018_staging_"))
    try:
        print(f"Extracting {archive} -> {staging} ...")
        _extract_zip(archive, staging)

        images_dir, masks_dir, note = _detect_bucket_dirs(staging, include_hints=("train", "training"), exclude_hints=("valid", "test"), min_files=50)
        print(f"Detected ISIC2018 Training layout: images={images_dir} masks={masks_dir} ({note})")

        images = _find_files(images_dir, (".jpg", ".jpeg", ".png"))
        masks = _find_files(masks_dir, (".png", ".jpg", ".jpeg"))
        matched, suffix_used = _pair_isic_style(images, masks)
        print(f"Detected ISIC2018 mask naming: stem + {suffix_used!r} ({len(matched)} pairs found)")

        expected = rl.ISIC_IMAGE_COUNTS[version]
        if len(matched) != expected:
            raise RuntimeError(
                f"ISIC2018 Kaggle download produced {len(matched)} Training image/mask "
                f"pairs, expected exactly {expected}. Inspect {staging} by hand -- the "
                "Kaggle mirror's layout may have changed."
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
            f"ISIC2018 has {len(pairs)} matched image/mask pairs under {root} after "
            f"normalisation, expected {rl.ISIC_IMAGE_COUNTS[version]}."
        )
    print(f"ISIC2018 ready: {len(pairs)} pairs (Training portion only) at {root}")
    return len(pairs)


# ---- ISIC2017 (S3 primary, Kaggle fallback, subsampled to 1250/150/600) -----


def _download_isic2017_s3(archives_dir: Path) -> Optional[Dict[str, Path]]:
    """Downloads all 6 official S3 parts (or reuses cached files). Returns
    {filename: zip_path} on full success, or None (cleaning up any partial
    file from the failing part) the moment any single part fails -- the
    caller falls through to the Kaggle mirror for ALL of ISIC2017, not a
    per-file mix of sources."""
    archives_dir.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}
    for fname, _kind, _bucket in ISIC2017_S3_PARTS:
        dest = archives_dir / fname
        if dest.is_file():
            downloaded[fname] = dest
            continue
        url = f"{ISIC2017_S3_BASE}/{fname}"
        try:
            print(f"Downloading {url} ...")
            _urlretrieve_with_timeout(url, dest)
            downloaded[fname] = dest
        except Exception as exc:  # noqa: BLE001 -- any failure here means "fall through to Kaggle", not crash
            print(f"ISIC2017 S3 primary download FAILED on {fname}: {exc!r}")
            if dest.is_file():
                dest.unlink()  # remove partial file so it isn't mistaken for a valid cached archive next run
            return None
    return downloaded


def ensure_isic2017(data_raw_dir: str | Path, paths) -> int:
    data_raw_dir = Path(data_raw_dir)
    version = "isic2017"
    try:
        pairs = rl.list_isic_pairs(data_raw_dir, version)
        if len(pairs) == rl.ISIC_IMAGE_COUNTS[version]:
            print(f"ISIC2017 already present and verified: {len(pairs)} pairs. Skipping download.")
            return len(pairs)
    except rl.RawDataMissingError:
        pass

    root = rl.isic_root(data_raw_dir, version)
    archives_dir = root / "s3_archives"

    zips = _download_isic2017_s3(archives_dir)
    source = "s3"
    kaggle_zip: Optional[Path] = None
    if zips is None:
        print(
            "WARNING: ISIC2017 S3 primary failed; using Kaggle fallback dataset "
            f"'{ISIC2017_KAGGLE_FALLBACK_SLUG}'. Structure/counts must be verified "
            "by hand against the official 2000/150/600 (Training/Validation/Test) "
            "split before trusting results."
        )
        source = "kaggle_fallback"
        archive = _find_archive(data_raw_dir, ("isic-2017", "isic2017"))
        if archive is not None:
            print(f"Found cached/placed archive {archive}; skipping re-download.")
            kaggle_zip = archive
        else:
            default_zip = root / "isic-2017.zip"
            if default_zip.is_file():
                print(f"Found cached archive {default_zip}; skipping re-download.")
                kaggle_zip = default_zip
            else:
                ensure_kaggle_auth(paths)
                kaggle_zip = _kaggle_cli_download(ISIC2017_KAGGLE_FALLBACK_SLUG, root)

    staging = Path(tempfile.mkdtemp(prefix="isic2017_staging_"))
    try:
        buckets: Dict[str, Dict[str, Dict[str, Path]]] = {"train": {}, "val": {}, "test": {}}
        if source == "s3":
            for fname, kind, bucket in ISIC2017_S3_PARTS:
                extracted_dir = staging / bucket / kind
                extracted_dir.mkdir(parents=True, exist_ok=True)
                _extract_zip(zips[fname], extracted_dir)
                exts = (".jpg", ".jpeg") if kind == "images" else (".png",)
                found_root = extracted_dir
                # official S3 zips nest one wrapper folder (e.g. "ISIC-2017_Training_Data/")
                nested = [p for p in extracted_dir.iterdir() if p.is_dir()]
                if not list(extracted_dir.glob(f"*{exts[0]}")) and len(nested) == 1:
                    found_root = nested[0]
                files = _find_files(found_root, exts)
                buckets[bucket][kind] = files
                n_expected = ISIC2017_BUCKET_COUNTS[bucket]
                if len(files) != n_expected:
                    raise RuntimeError(
                        f"ISIC2017 S3 {fname} extracted to {len(files)} {kind} files, "
                        f"expected {n_expected}. The archive may have changed -- "
                        f"inspect {extracted_dir} by hand."
                    )
        else:
            extracted = staging / "kaggle_extract"
            _extract_zip(kaggle_zip, extracted)
            for bucket, include_hints, exclude_hints in (
                ("train", ("train", "training"), ("valid", "test")),
                ("val", ("valid", "validation"), ("train", "test")),
                ("test", ("test",), ("train", "valid")),
            ):
                images_dir, masks_dir, note = _detect_bucket_dirs(extracted, include_hints, exclude_hints, min_files=20)
                print(f"ISIC2017 Kaggle fallback: detected {bucket} images={images_dir} masks={masks_dir} ({note})")
                buckets[bucket]["images"] = _find_files(images_dir, (".jpg", ".jpeg", ".png"))
                buckets[bucket]["masks"] = _find_files(masks_dir, (".png", ".jpg", ".jpeg"))

        paired: Dict[str, Dict[str, Tuple[Path, Path]]] = {}
        for bucket in ("train", "val", "test"):
            matched, suffix_used = _pair_isic_style(buckets[bucket]["images"], buckets[bucket]["masks"])
            paired[bucket] = matched
            print(f"ISIC2017 bucket={bucket}: {len(matched)} matched pairs (mask suffix {suffix_used!r})")
            if source == "s3" and len(matched) != ISIC2017_BUCKET_COUNTS[bucket]:
                raise RuntimeError(f"ISIC2017 S3 bucket={bucket} matched {len(matched)} image/mask pairs, expected {ISIC2017_BUCKET_COUNTS[bucket]}.")

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
            "source": source,
            "train_pool_size": len(train_ids_sorted),
            "train_kept": kept_by_bucket["train"],
            "val_kept": kept_by_bucket["val"],
            "test_kept": kept_by_bucket["test"],
            "subsample_seed": cvc_sequence.SPLIT_SEED,
        }
        manifest_path = root / "isic2017_source_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Wrote {manifest_path} documenting exactly which ISIC2017 images were kept (source={source}).")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    from lcmunet.env_report import update_env_json

    update_env_json(paths.results, {"isic2017_source": source})

    pairs = rl.list_isic_pairs(data_raw_dir, version)
    if len(pairs) != rl.ISIC_IMAGE_COUNTS[version]:
        raise RuntimeError(f"ISIC2017 has {len(pairs)} matched image/mask pairs under {root} after selection, expected {rl.ISIC_IMAGE_COUNTS[version]}.")
    print(f"ISIC2017 ready: {len(pairs)} pairs (source={source}) at {root}")
    return len(pairs)


# ---- dispatch ----------------------------------------------------------


def ensure_dataset_ready(dataset: str, paths) -> int:
    """Dispatch to the right ensure_* function. Raises RawDataMissingError,
    lcmunet.data.kaggle_auth.KaggleAuthMissingError, or RuntimeError if the
    dataset isn't genuinely ready -- callers must not catch these to
    "proceed anyway" (lcmunet.data.prepare_all.prepare_all_datasets catches
    per-dataset so one failure doesn't block the other three)."""
    if dataset == "kvasir_seg":
        return ensure_kvasir(paths.data_raw)
    if dataset == "cvc_clinicdb":
        return ensure_cvc(paths.data_raw, paths)
    if dataset == "isic2017":
        return ensure_isic2017(paths.data_raw, paths)
    if dataset == "isic2018":
        return ensure_isic2018(paths.data_raw, paths)
    raise ValueError(f"unknown dataset: {dataset!r} (expected one of {rl.DATASET_NAMES})")


if __name__ == "__main__":
    import argparse

    from lcmunet.paths import get_paths

    parser = argparse.ArgumentParser(description="Ensure raw datasets are present under data_raw/.")
    parser.add_argument("datasets", nargs="*", default=list(rl.DATASET_NAMES), choices=list(rl.DATASET_NAMES))
    args = parser.parse_args()

    paths = get_paths()
    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            ensure_dataset_ready(name, paths)
        except rl.RawDataMissingError:
            print(f"[STOPPED] {name}: raw data not available; see instructions above.")
