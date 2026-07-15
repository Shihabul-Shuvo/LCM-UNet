"""Build and cache split id-lists under DRIVE_ROOT/splits/ (methodology
section 7). Each dataset is split exactly once with a fixed seed, reused by
every training seed (Fairness rule) -- this is a separate concern from
RunConfig.seed, which controls model/training stochasticity only.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from lcmunet.data import cvc_sequence, raw_layout as rl

# Fixed, project-wide split seed -- intentionally decoupled from
# RunConfig.seed (which varies 1..5 across headline-row repeats). Re-used
# from cvc_sequence so there is exactly one constant, not two that could
# drift apart.
SPLIT_SEED = cvc_sequence.SPLIT_SEED

ISIC_COUNTS = {
    "isic2017": (1250, 150, 600),
    "isic2018": (1815, 259, 520),
}


def _split_path(splits_dir: str | Path, dataset: str) -> Path:
    return Path(splits_dir) / f"{dataset}.json"


def _write_split(splits_dir: str | Path, dataset: str, payload: dict) -> Path:
    path = _split_path(splits_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load_split_file(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No split file at {path}. Run lcmunet.data.splits.build_all_splits() first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(splits_dir: str | Path, dataset: str) -> dict:
    return load_split_file(_split_path(splits_dir, dataset))


def _fixed_seed_shuffle_split(
    ids: List[str], seed: int, fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[List[str], List[str], List[str]]:
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1.0")
    ids_sorted = sorted(ids)  # deterministic starting order, independent of filesystem/OS
    shuffled = ids_sorted[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * fracs[0])
    n_val = round(n * fracs[1])
    train, val, test = shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]
    assert len(train) + len(val) + len(test) == n
    return train, val, test


# ---- Kvasir-SEG -------------------------------------------------------------


def build_kvasir_split(paths, seed: int = SPLIT_SEED) -> dict:
    pairs = rl.list_kvasir_pairs(paths.data_raw)
    ids = [p.id for p in pairs]
    train, val, test = _fixed_seed_shuffle_split(ids, seed, fracs=(0.8, 0.1, 0.1))
    payload = {
        "dataset": "kvasir_seg",
        "seed": seed,
        "n_total": len(ids),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train": train,
        "val": val,
        "test": test,
        "notes": "80/10/10, fixed split seed, shuffled from a sorted id list (methodology section 7).",
    }
    path = _write_split(paths.splits, "kvasir_seg", payload)
    print(f"kvasir_seg split written to {path}: {payload['counts']}")
    return payload


# ---- CVC-ClinicDB -----------------------------------------------------------


def build_cvc_split(paths, seed: int = SPLIT_SEED) -> dict:
    pairs = rl.list_cvc_pairs(paths.data_raw)
    frame_ids = [p.id for p in pairs]

    seq_result = cvc_sequence.build_cvc_sequence_split(frame_ids, data_raw_dir=paths.data_raw, seed=seed)
    seq_map_path = cvc_sequence.save_cvc_sequence_map(seq_result, paths.splits)

    by_partition: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for frame_id, partition in seq_result["partition_of_frame"].items():
        by_partition[partition].append(frame_id)
    for k in by_partition:
        by_partition[k].sort()

    payload = {
        "dataset": "cvc_clinicdb",
        "seed": seed,
        "n_total": len(frame_ids),
        "counts": {k: len(v) for k, v in by_partition.items()},
        "train": by_partition["train"],
        "val": by_partition["val"],
        "test": by_partition["test"],
        "notes": (
            "SEQUENCE-LEVEL split (29 sequences), not a random frame split -- "
            "see splits/cvc_sequence_map.json and lcmunet/data/cvc_sequence.py. "
            f"Mapping source: {seq_result['mapping_source']}"
        ),
        "paper_sentence": cvc_sequence.PAPER_SENTENCE,
    }
    path = _write_split(paths.splits, "cvc_clinicdb", payload)

    print(f"cvc_clinicdb split written to {path}: {payload['counts']}")
    print(f"cvc_sequence_map written to {seq_map_path}")
    print(f"Mapping source: {seq_result['mapping_source']}")
    print("Verify the CVC 29-sequence mapping against the original documentation before submission.")
    print(f"Paper sentence: {cvc_sequence.PAPER_SENTENCE}")
    return payload


# ---- ISIC2017 / ISIC2018 -----------------------------------------------------


def _load_ultralight_split_override(data_raw_dir: str | Path, version: str) -> dict | None:
    path = Path(data_raw_dir) / "ultralight_splits" / f"{version}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_isic_split(paths, version: str, seed: int = SPLIT_SEED) -> dict:
    if version not in ISIC_COUNTS:
        raise ValueError(f"unknown ISIC version: {version!r}")
    pairs = rl.list_isic_pairs(paths.data_raw, version)
    all_ids = {p.id for p in pairs}
    n_train, n_val, n_test = ISIC_COUNTS[version]

    override = _load_ultralight_split_override(paths.data_raw, version)
    if override is not None:
        train, val, test = override["train"], override["val"], override["test"]
        given = set(train) | set(val) | set(test)
        if given != all_ids or len(train) + len(val) + len(test) != len(all_ids):
            raise ValueError(
                f"data_raw/ultralight_splits/{version}.json does not exactly "
                f"partition the {len(all_ids)} available ids (missing="
                f"{len(all_ids - given)}, extra={len(given - all_ids)}, "
                f"overlap-or-dup={len(train)+len(val)+len(test) != len(all_ids)}). "
                "Fix the file or remove it to use the deterministic fallback split."
            )
        notes = "Loaded verbatim from data_raw/ultralight_splits/ (exact UltraLight split)."
        source = "ultralight_splits override"
    else:
        if len(all_ids) != n_train + n_val + n_test:
            raise ValueError(
                f"{version}: found {len(all_ids)} raw pairs, expected exactly "
                f"{n_train + n_val + n_test} ({n_train}/{n_val}/{n_test}) to match "
                "UltraLight's counts. Check the raw archive is the complete "
                "official training release."
            )
        train, val, test = _sorted_slice_split(sorted(all_ids), (n_train, n_val, n_test))
        notes = (
            "Deterministic alphabetical-sort contiguous slice, counts matched to "
            "UltraLight VM-UNet's own split "
            f"({n_train}/{n_val}/{n_test}; see third_party/UltraLight-VM-UNet/"
            "dataprepare/Prepare_"
            + ("ISIC2017" if version == "isic2017" else "ISIC2018")
            + ".py). DEVIATION: UltraLight's original split order comes from "
            "unsorted glob.glob() and is not reproducible from public "
            "documentation, so exact per-sample membership may differ from "
            "the original paper even though the counts match. Supply "
            "data_raw/ultralight_splits/"
            + version
            + ".json (train/val/test id lists) for an exact match."
        )
        source = "deterministic sorted-slice fallback"

    payload = {
        "dataset": version,
        "seed": seed,
        "n_total": len(all_ids),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "split_source": source,
        "notes": notes,
    }
    path = _write_split(paths.splits, version, payload)
    print(f"{version} split written to {path}: {payload['counts']} (source: {source})")
    return payload


def _sorted_slice_split(ids_sorted: List[str], counts: Tuple[int, int, int]) -> Tuple[List[str], List[str], List[str]]:
    n_train, n_val, n_test = counts
    if len(ids_sorted) != n_train + n_val + n_test:
        raise ValueError(f"expected {n_train + n_val + n_test} ids, got {len(ids_sorted)}")
    return ids_sorted[:n_train], ids_sorted[n_train : n_train + n_val], ids_sorted[n_train + n_val :]


# ---- cross-dataset (methodology section 7, required) -------------------------

CROSS_DATASET_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("kvasir_seg", "cvc_clinicdb"),
    ("cvc_clinicdb", "kvasir_seg"),
)


# ---- orchestration -------------------------------------------------------


def build_all_splits(paths, seed: int = SPLIT_SEED) -> Dict[str, dict]:
    """Build every split this project needs. Each dataset that isn't ready in
    data_raw/ fails loudly for that dataset (raw_layout.RawDataMissingError)
    but does not block the others -- report is printed at the end either way.
    """
    results: Dict[str, dict] = {}
    errors: Dict[str, str] = {}

    builders = {
        "kvasir_seg": lambda: build_kvasir_split(paths, seed=seed),
        "cvc_clinicdb": lambda: build_cvc_split(paths, seed=seed),
        "isic2017": lambda: build_isic_split(paths, "isic2017", seed=seed),
        "isic2018": lambda: build_isic_split(paths, "isic2018", seed=seed),
    }
    for name, builder in builders.items():
        print(f"\n=== building split: {name} ===")
        try:
            results[name] = builder()
        except rl.RawDataMissingError as exc:
            print(exc.instructions)
            errors[name] = str(exc)

    print("\n=== summary ===")
    for name in builders:
        if name in results:
            print(f"  {name}: OK {results[name]['counts']}")
        else:
            print(f"  {name}: SKIPPED (raw data not ready)")

    return results


if __name__ == "__main__":
    from lcmunet.paths import get_paths

    build_all_splits(get_paths())
