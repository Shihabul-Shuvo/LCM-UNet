import csv

import pytest

from lcmunet.data import cvc_sequence as cs


def test_table_covers_all_612_frames_exactly_once():
    # _validate_table already runs at import time; re-running here documents
    # the invariant explicitly and fails loudly if the table is ever edited badly.
    cs._validate_table(cs.FRAME_TO_SEQUENCE_TABLE)


def test_frame_to_sequence_boundaries():
    assert cs.frame_to_sequence(1) == 1
    assert cs.frame_to_sequence(25) == 1
    assert cs.frame_to_sequence(26) == 2
    assert cs.frame_to_sequence(612) == 29
    assert cs.frame_to_sequence(592) == 29


def test_frame_to_sequence_out_of_range_raises():
    with pytest.raises(ValueError):
        cs.frame_to_sequence(0)
    with pytest.raises(ValueError):
        cs.frame_to_sequence(613)


def test_resolve_sequence_map_uses_builtin_table_by_default(tmp_path):
    frame_ids = [str(i) for i in range(1, 613)]
    mapping, source = cs.resolve_sequence_map(frame_ids, tmp_path)
    assert "FRAME_TO_SEQUENCE_TABLE" in source
    assert mapping["1"] == 1
    assert mapping["612"] == 29


def test_resolve_sequence_map_prefers_user_supplied_csv(tmp_path):
    csv_path = tmp_path / "cvc_sequence_map.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "sequence_id"])
        for i in range(1, 613):
            w.writerow([str(i), (i % 29) + 1])  # deliberately different from the built-in table

    frame_ids = [str(i) for i in range(1, 613)]
    mapping, source = cs.resolve_sequence_map(frame_ids, tmp_path)
    assert "user-supplied" in source
    assert mapping["1"] == 2  # (1 % 29) + 1, not the built-in table's 1


def test_resolve_sequence_map_incomplete_csv_raises(tmp_path):
    csv_path = tmp_path / "cvc_sequence_map.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "sequence_id"])
        w.writerow(["1", "1"])  # only one frame, but we'll ask for more below

    with pytest.raises(ValueError, match="missing"):
        cs.resolve_sequence_map([str(i) for i in range(1, 613)], tmp_path)


def test_resolve_sequence_map_never_falls_back_to_random_split(tmp_path):
    """Frame ids outside the known table AND no user CSV -> must raise, not guess."""
    with pytest.raises(ValueError):
        cs.resolve_sequence_map(["99999"], tmp_path)


def test_assign_sequences_to_partitions_is_deterministic():
    counts = {i: 20 for i in range(1, 30)}
    a = cs.assign_sequences_to_partitions(counts, seed=42)
    b = cs.assign_sequences_to_partitions(counts, seed=42)
    assert a == b


def test_assign_sequences_to_partitions_approximately_80_10_10():
    sizes = [25, 25, 17, 11, 25, 23, 25, 26, 22, 6, 22, 25, 25, 20, 20, 25, 21, 20, 25, 20, 19, 19, 12, 25, 25, 18, 25, 20, 21]
    counts = {i + 1: s for i, s in enumerate(sizes)}
    total = sum(sizes)
    assignment = cs.assign_sequences_to_partitions(counts, seed=cs.SPLIT_SEED)
    by_partition = {"train": 0, "val": 0, "test": 0}
    for seq_id, part in assignment.items():
        by_partition[part] += counts[seq_id]

    assert set(assignment.values()) <= {"train", "val", "test"}
    assert sum(by_partition.values()) == total
    # whole-sequence assignment can't hit exact fractions; require "close"
    assert 0.70 <= by_partition["train"] / total <= 0.90
    assert 0.05 <= by_partition["val"] / total <= 0.20
    assert 0.05 <= by_partition["test"] / total <= 0.20


def test_assert_no_sequence_leakage_passes_on_clean_split():
    sequence_of_frame = {"1": 1, "2": 1, "3": 2}
    partition_of_frame = {"1": "train", "2": "train", "3": "val"}
    cs.assert_no_sequence_leakage(sequence_of_frame, partition_of_frame)  # must not raise


def test_assert_no_sequence_leakage_catches_split_sequence():
    sequence_of_frame = {"1": 1, "2": 1}
    partition_of_frame = {"1": "train", "2": "test"}  # same sequence, two partitions
    with pytest.raises(AssertionError):
        cs.assert_no_sequence_leakage(sequence_of_frame, partition_of_frame)


def test_build_cvc_sequence_split_end_to_end_no_leakage(tmp_path):
    frame_ids = [str(i) for i in range(1, 613)]
    result = cs.build_cvc_sequence_split(frame_ids, data_raw_dir=tmp_path)

    assert result["n_frames"] == 612
    assert result["n_sequences"] == 29
    assert sum(result["frame_counts_by_partition"].values()) == 612
    cs.assert_no_sequence_leakage(result["sequence_of_frame"], result["partition_of_frame"])
    assert result["paper_sentence"] == cs.PAPER_SENTENCE


def test_save_cvc_sequence_map_writes_json(tmp_path):
    frame_ids = [str(i) for i in range(1, 613)]
    result = cs.build_cvc_sequence_split(frame_ids, data_raw_dir=tmp_path)
    path = cs.save_cvc_sequence_map(result, tmp_path / "splits")
    assert path.is_file()
    assert path.name == "cvc_sequence_map.json"
