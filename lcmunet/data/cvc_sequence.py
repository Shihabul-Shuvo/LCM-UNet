"""CVC-ClinicDB sequence-level split (methodology section 7 -- REQUIRED
leakage guard).

CVC-ClinicDB's 612 frames come from 29 colonoscopy sequences; consecutive
frames within a sequence are near-identical. A random frame-level split
leaks near-duplicates across train/val/test and inflates reported Dice by
1-3%. Sequences must be assigned to partitions as whole units.

Resolution order (never fall back to a random frame split silently):
  (a) a user-supplied frame->sequence mapping file in data_raw/ (e.g. the
      original ISBI-2015 / Bernal et al. 2015 release's own correspondence
      file) -- used verbatim if present.
  (b) the FRAME_TO_SEQUENCE_TABLE below, built from the published
      frame-number/sequence-number correspondence.
  (c) if neither is available/usable: raise. Do not proceed with a leaky
      split.

*** VERIFY BEFORE SUBMISSION ***
FRAME_TO_SEQUENCE_TABLE below is transcribed from a third-party dataset
documentation page (Dataset Ninja, https://datasetninja.com/cvc-612), cross-
checked for internal consistency (29 contiguous, non-overlapping ranges
summing to exactly 612 frames) but NOT independently confirmed against the
original CVC/Hospital Clinic Barcelona release documentation or the Bernal
et al. 2015 paper (Computerized Medical Imaging and Graphics, vol. 43,
pp. 99-111). The same source page also states two different sequence-related
counts in different places ("25 different video studies" vs. "29 different
sequences"), which is a real inconsistency worth resolving before trusting
this table for publication. Path (a) -- an explicit user-supplied mapping --
always overrides this table; use it if you can obtain the original
correspondence file.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Editable, explicit table -- NOT a magic constant buried in code. Each row is
# (first_frame, last_frame, sequence_id), inclusive, 1-indexed, inserted in
# ascending frame order. Edit this table directly if you obtain a corrected
# mapping; frame_to_sequence() below will pick up the change automatically.
FRAME_TO_SEQUENCE_TABLE: List[Tuple[int, int, int]] = [
    (1, 25, 1),
    (26, 50, 2),
    (51, 67, 3),
    (68, 78, 4),
    (79, 103, 5),
    (104, 126, 6),
    (127, 151, 7),
    (152, 177, 8),
    (178, 199, 9),
    (200, 205, 10),
    (206, 227, 11),
    (228, 252, 12),
    (253, 277, 13),
    (278, 297, 14),
    (298, 317, 15),
    (318, 342, 16),
    (343, 363, 17),
    (364, 383, 18),
    (384, 408, 19),
    (409, 428, 20),
    (429, 447, 21),
    (448, 466, 22),
    (467, 478, 23),
    (479, 503, 24),
    (504, 528, 25),
    (529, 546, 26),
    (547, 571, 27),
    (572, 591, 28),
    (592, 612, 29),
]

CVC_SEQUENCE_COUNT = 29
CVC_FRAME_COUNT = 612

# Fixed, independent of any RunConfig.seed (Fairness rule: one split, reused
# by every training seed).
SPLIT_SEED = 42

PAPER_SENTENCE = (
    "CVC-ClinicDB was partitioned at the sequence level (29 sequences); no "
    "frames from the same sequence appear in more than one split."
)


def _validate_table(table: List[Tuple[int, int, int]]) -> None:
    seq_ids = [row[2] for row in table]
    if sorted(seq_ids) != list(range(1, CVC_SEQUENCE_COUNT + 1)):
        raise ValueError(
            f"FRAME_TO_SEQUENCE_TABLE must cover sequences 1..{CVC_SEQUENCE_COUNT} "
            f"exactly once each; got {sorted(seq_ids)}"
        )
    covered = set()
    for start, end, _seq in table:
        if start > end:
            raise ValueError(f"invalid range in FRAME_TO_SEQUENCE_TABLE: ({start}, {end})")
        rng = set(range(start, end + 1))
        if covered & rng:
            raise ValueError(f"FRAME_TO_SEQUENCE_TABLE has overlapping frame ranges near {start}-{end}")
        covered |= rng
    if covered != set(range(1, CVC_FRAME_COUNT + 1)):
        missing = sorted(set(range(1, CVC_FRAME_COUNT + 1)) - covered)
        extra = sorted(covered - set(range(1, CVC_FRAME_COUNT + 1)))
        raise ValueError(
            f"FRAME_TO_SEQUENCE_TABLE does not exactly cover frames 1..{CVC_FRAME_COUNT}. "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
            f"extra={extra[:10]}{'...' if len(extra) > 10 else ''}"
        )


_validate_table(FRAME_TO_SEQUENCE_TABLE)  # fail loud at import time if the table is ever edited incorrectly


def frame_to_sequence(frame_id: int) -> int:
    """Look up the sequence id (1..29) for a 1-indexed frame number, via the table above."""
    for start, end, seq in FRAME_TO_SEQUENCE_TABLE:
        if start <= frame_id <= end:
            return seq
    raise ValueError(f"frame_id {frame_id} is outside the known range 1..{CVC_FRAME_COUNT}")


def _frame_id_to_int(frame_id: str) -> int:
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    if not digits:
        raise ValueError(f"could not parse a frame number out of id {frame_id!r}")
    return int(digits)


def load_user_supplied_mapping(data_raw_dir: str | Path) -> Optional[Dict[str, int]]:
    """Path (a): data_raw/cvc_sequence_map.csv with columns frame_id,sequence_id (any header names containing those words), if present."""
    path = Path(data_raw_dir) / "cvc_sequence_map.csv"
    if not path.is_file():
        return None

    mapping: Dict[str, int] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or not a valid CSV")
        frame_col = next((c for c in reader.fieldnames if "frame" in c.lower()), None)
        seq_col = next((c for c in reader.fieldnames if "seq" in c.lower()), None)
        if frame_col is None or seq_col is None:
            raise ValueError(
                f"{path} must have a frame-id column and a sequence-id column "
                f"(found columns: {reader.fieldnames})"
            )
        for row in reader:
            mapping[str(row[frame_col]).strip()] = int(row[seq_col])
    return mapping


def resolve_sequence_map(frame_ids: List[str], data_raw_dir: str | Path) -> Tuple[Dict[str, int], str]:
    """Resolve frame_id -> sequence_id for the given ids. Returns (mapping, source_description).

    Priority: (a) user-supplied file, verbatim -> (b) FRAME_TO_SEQUENCE_TABLE
    -> (c) raise (never fabricate a random split).
    """
    user_mapping = load_user_supplied_mapping(data_raw_dir)
    if user_mapping is not None:
        missing = [f for f in frame_ids if f not in user_mapping]
        if missing:
            raise ValueError(
                f"data_raw/cvc_sequence_map.csv is missing {len(missing)} of the "
                f"{len(frame_ids)} present frame ids (e.g. {missing[:5]}). Fix the "
                "file so it covers every frame, or remove it to fall back to the "
                "built-in FRAME_TO_SEQUENCE_TABLE."
            )
        return {f: user_mapping[f] for f in frame_ids}, "user-supplied data_raw/cvc_sequence_map.csv"

    try:
        mapping = {f: frame_to_sequence(_frame_id_to_int(f)) for f in frame_ids}
    except ValueError as exc:
        raise ValueError(
            "No user-supplied CVC frame->sequence mapping found "
            "(data_raw/cvc_sequence_map.csv), and the built-in "
            "FRAME_TO_SEQUENCE_TABLE could not resolve every present frame id "
            f"({exc}). Refusing to fall back to a random frame-level split -- "
            "that would leak near-duplicate frames across train/val/test and "
            "inflate Dice. Supply data_raw/cvc_sequence_map.csv (frame_id,"
            "sequence_id columns) with the correct mapping."
        ) from exc
    return mapping, "built-in FRAME_TO_SEQUENCE_TABLE (see cvc_sequence.py header -- verify before submission)"


def assign_sequences_to_partitions(
    seq_frame_counts: Dict[int, int],
    seed: int = SPLIT_SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Dict[int, str]:
    """Deterministically assign WHOLE sequences to train/val/test, targeting
    ~80/10/10 BY FRAME COUNT (methodology section 7). Never splits a sequence.

    Greedy balanced assignment: shuffle sequence ids with a fixed seed, then
    repeatedly assign the next sequence to whichever partition is currently
    furthest below its frame-count target. Exact 80/10/10 is not always
    achievable with only 29 unevenly-sized sequences (methodology says
    "~80/10/10"); this converges close to it (empirically within ~1-2pp at
    seed=42) while guaranteeing whole-sequence, non-overlapping partitions.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    total = sum(seq_frame_counts.values())
    targets = {"train": total * train_frac, "val": total * val_frac, "test": total * test_frac}
    counts = {"train": 0, "val": 0, "test": 0}

    seq_ids = list(seq_frame_counts.keys())
    random.Random(seed).shuffle(seq_ids)

    assignment: Dict[int, str] = {}
    for sid in seq_ids:
        deficits = {p: targets[p] - counts[p] for p in ("train", "val", "test")}
        chosen = max(deficits, key=lambda p: (deficits[p], p))
        assignment[sid] = chosen
        counts[chosen] += seq_frame_counts[sid]

    return assignment


def assert_no_sequence_leakage(sequence_of_frame: Dict[str, int], partition_of_frame: Dict[str, str]) -> None:
    """The mandatory leakage guard: no sequence id may appear under more than one partition."""
    partitions_by_seq: Dict[int, set] = {}
    for frame_id, seq_id in sequence_of_frame.items():
        partitions_by_seq.setdefault(seq_id, set()).add(partition_of_frame[frame_id])

    offenders = {seq: parts for seq, parts in partitions_by_seq.items() if len(parts) > 1}
    if offenders:
        raise AssertionError(
            f"CVC sequence-level split leakage: {len(offenders)} sequence(s) span "
            f"more than one partition: {offenders}"
        )


def build_cvc_sequence_split(
    frame_ids: List[str],
    data_raw_dir: str | Path,
    seed: int = SPLIT_SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict:
    """Build the full CVC sequence-level split. Returns the dict written to
    splits/cvc_sequence_map.json: frame->sequence, sequence->partition,
    frame->partition, provenance, and the paper sentence. Asserts no leakage
    before returning.
    """
    sequence_of_frame, source = resolve_sequence_map(frame_ids, data_raw_dir)

    seq_frame_counts: Dict[int, int] = {}
    for seq_id in sequence_of_frame.values():
        seq_frame_counts[seq_id] = seq_frame_counts.get(seq_id, 0) + 1

    partition_of_seq = assign_sequences_to_partitions(
        seq_frame_counts, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    partition_of_frame = {f: partition_of_seq[seq_id] for f, seq_id in sequence_of_frame.items()}

    assert_no_sequence_leakage(sequence_of_frame, partition_of_frame)

    frame_counts_by_partition = {"train": 0, "val": 0, "test": 0}
    for p in partition_of_frame.values():
        frame_counts_by_partition[p] += 1

    result = {
        "mapping_source": source,
        "n_frames": len(frame_ids),
        "n_sequences": len(seq_frame_counts),
        "split_seed": seed,
        "target_fracs": {"train": train_frac, "val": val_frac, "test": test_frac},
        "frame_counts_by_partition": frame_counts_by_partition,
        "sequence_of_frame": sequence_of_frame,
        "partition_of_sequence": {str(k): v for k, v in partition_of_seq.items()},
        "partition_of_frame": partition_of_frame,
        "paper_sentence": PAPER_SENTENCE,
    }
    return result


def save_cvc_sequence_map(result: dict, splits_dir: str | Path) -> Path:
    path = Path(splits_dir) / "cvc_sequence_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return path
