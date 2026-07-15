"""Dataset acquisition (methodology section 7).

Kvasir-SEG auto-downloads (public, no registration). CVC-ClinicDB and
ISIC2017/2018 require registration, so the user places the raw archive (or
already-extracted folder) in data_raw/ themselves; this module detects it,
auto-extracts a placed .zip, verifies the expected count, and -- if the data
truly isn't there -- fails loudly with the exact instructions for what to
download and where to put it (never proceeds on a partial/guessed dataset).
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

from lcmunet.data import raw_layout as rl

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_ZIP_NAME = "kvasir-seg.zip"


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _find_archive(data_raw_dir: Path, name_substrings: Sequence[str]) -> Optional[Path]:
    data_raw_dir = Path(data_raw_dir)
    if not data_raw_dir.is_dir():
        return None
    for p in sorted(data_raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".zip":
            lname = p.name.lower()
            if any(s in lname for s in name_substrings):
                return p
    return None


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
            print(
                f"Kvasir-SEG present but count mismatch "
                f"({len(pairs)} != {rl.KVASIR_IMAGE_COUNT}); re-downloading."
            )

    zip_path = data_raw_dir / KVASIR_ZIP_NAME
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kvasir-SEG from {KVASIR_URL} ...")
    urllib.request.urlretrieve(KVASIR_URL, zip_path)
    print(f"Downloaded {zip_path} ({zip_path.stat().st_size} bytes). Extracting...")
    _extract_zip(zip_path, data_raw_dir)

    pairs = rl.list_kvasir_pairs(data_raw_dir)
    if len(pairs) != rl.KVASIR_IMAGE_COUNT:
        raise RuntimeError(
            f"Kvasir-SEG download/extract produced {len(pairs)} image/mask "
            f"pairs, expected {rl.KVASIR_IMAGE_COUNT}. The archive at "
            f"{KVASIR_URL} may have changed -- inspect {root} by hand."
        )
    print(f"Kvasir-SEG ready: {len(pairs)} pairs at {root}")
    return len(pairs)


def ensure_cvc(data_raw_dir: str | Path) -> int:
    data_raw_dir = Path(data_raw_dir)
    try:
        pairs = rl.list_cvc_pairs(data_raw_dir)
    except rl.RawDataMissingError as exc:
        archive = _find_archive(data_raw_dir, ("cvc",))
        if archive is None:
            print(exc.instructions)
            raise
        dest = rl.cvc_root(data_raw_dir)
        print(f"Found archive {archive}; extracting to {dest} ...")
        _extract_zip(archive, dest)
        pairs = rl.list_cvc_pairs(data_raw_dir)  # re-raises with instructions if still not found

    if len(pairs) != rl.CVC_IMAGE_COUNT:
        raise RuntimeError(
            f"CVC-ClinicDB has {len(pairs)} matched image/mask pairs under "
            f"{rl.cvc_root(data_raw_dir)}, expected {rl.CVC_IMAGE_COUNT}. "
            "Verify the placed archive is the complete, official release."
        )
    print(f"CVC-ClinicDB ready: {len(pairs)} pairs.")
    return len(pairs)


def ensure_isic(data_raw_dir: str | Path, version: str) -> int:
    data_raw_dir = Path(data_raw_dir)
    try:
        pairs = rl.list_isic_pairs(data_raw_dir, version)
    except rl.RawDataMissingError as exc:
        archive = _find_archive(data_raw_dir, (version,))
        if archive is None:
            print(exc.instructions)
            raise
        dest = rl.isic_root(data_raw_dir, version)
        print(f"Found archive {archive}; extracting to {dest} ...")
        _extract_zip(archive, dest)
        pairs = rl.list_isic_pairs(data_raw_dir, version)

    expected = rl.ISIC_IMAGE_COUNTS[version]
    if len(pairs) != expected:
        raise RuntimeError(
            f"{version} has {len(pairs)} matched image/mask pairs under "
            f"{rl.isic_root(data_raw_dir, version)}, expected {expected}. "
            "Verify the placed archive is the complete, official Task 1-2 "
            "training release."
        )
    print(f"{version} ready: {len(pairs)} pairs.")
    return len(pairs)


def ensure_dataset_ready(dataset: str, data_raw_dir: str | Path) -> int:
    """Dispatch to the right ensure_* function. Raises RawDataMissingError
    (with .instructions) or RuntimeError (count mismatch) if the dataset
    isn't genuinely ready -- callers must not catch these to "proceed anyway"."""
    if dataset == "kvasir_seg":
        return ensure_kvasir(data_raw_dir)
    if dataset == "cvc_clinicdb":
        return ensure_cvc(data_raw_dir)
    if dataset in rl.ISIC_IMAGE_COUNTS:
        return ensure_isic(data_raw_dir, dataset)
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
            ensure_dataset_ready(name, paths.data_raw)
        except rl.RawDataMissingError:
            print(f"[STOPPED] {name}: raw data not available; see instructions above.")
