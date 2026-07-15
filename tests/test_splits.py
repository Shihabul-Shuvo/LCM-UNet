import json

import pytest

from lcmunet.data import raw_layout as rl
from lcmunet.data import splits as sp


def test_kvasir_split_counts_and_no_overlap(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    payload = sp.build_kvasir_split(paths)
    assert payload["counts"] == {"train": 80, "val": 10, "test": 10}
    train, val, test = set(payload["train"]), set(payload["val"]), set(payload["test"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(range_ids := {f"img{i:04d}" for i in range(100)})


def test_kvasir_split_is_deterministic_across_seed_but_not_across_calls_state(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    a = sp.build_kvasir_split(paths, seed=42)
    b = sp.build_kvasir_split(paths, seed=42)
    assert a["train"] == b["train"] and a["val"] == b["val"] and a["test"] == b["test"]


def test_kvasir_split_different_seed_gives_different_partition(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    a = sp.build_kvasir_split(paths, seed=1)
    b = sp.build_kvasir_split(paths, seed=2)
    assert a["train"] != b["train"]


def test_kvasir_split_seed_is_independent_of_training_seed_concept():
    # SPLIT_SEED is a module constant, not derived from any RunConfig -- this
    # test documents that decoupling exists as a fixed, importable constant.
    assert isinstance(sp.SPLIT_SEED, int)


def test_load_split_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sp.load_split_file(tmp_path / "nope.json")


def test_cvc_split_end_to_end_no_leakage_and_paper_sentence(make_cvc_raw):
    paths = make_cvc_raw(n=612)
    payload = sp.build_cvc_split(paths)

    assert sum(payload["counts"].values()) == 612
    assert payload["paper_sentence"] == (
        "CVC-ClinicDB was partitioned at the sequence level (29 sequences); no "
        "frames from the same sequence appear in more than one split."
    )
    seq_map_path = paths.splits / "cvc_sequence_map.json"
    assert seq_map_path.is_file()
    with open(seq_map_path) as f:
        seq_map = json.load(f)

    # re-verify the mandatory leakage guard directly from the saved artifact
    from lcmunet.data.cvc_sequence import assert_no_sequence_leakage

    assert_no_sequence_leakage(seq_map["sequence_of_frame"], seq_map["partition_of_frame"])


def test_isic_split_matches_ultralight_counts_when_no_override(make_isic_raw):
    paths = make_isic_raw(version="isic2017")  # full 2000
    payload = sp.build_isic_split(paths, "isic2017")
    assert payload["counts"] == {"train": 1250, "val": 150, "test": 600}
    assert payload["split_source"] == "deterministic sorted-slice fallback"
    assert "DEVIATION" in payload["notes"]


def test_isic_split_wrong_count_raises(make_isic_raw):
    paths = make_isic_raw(version="isic2017", n=1999)  # one short
    with pytest.raises(ValueError, match="expected exactly"):
        sp.build_isic_split(paths, "isic2017")


def test_isic2018_split_counts(make_isic_raw):
    paths = make_isic_raw(version="isic2018")  # full 2594
    payload = sp.build_isic_split(paths, "isic2018")
    assert payload["counts"] == {"train": 1815, "val": 259, "test": 520}


def test_isic_ultralight_override_used_verbatim(make_isic_raw):
    paths = make_isic_raw(version="isic2017")
    pairs = rl.list_isic_pairs(paths.data_raw, "isic2017")
    all_ids = sorted(p.id for p in pairs)
    override = {"train": all_ids[:1250], "val": all_ids[1250:1400], "test": all_ids[1400:]}

    override_dir = paths.data_raw / "ultralight_splits"
    override_dir.mkdir(parents=True, exist_ok=True)
    with open(override_dir / "isic2017.json", "w") as f:
        json.dump(override, f)

    payload = sp.build_isic_split(paths, "isic2017")
    assert payload["split_source"] == "ultralight_splits override"
    assert payload["train"] == sorted(override["train"])


def test_isic_ultralight_override_bad_partition_raises(make_isic_raw):
    paths = make_isic_raw(version="isic2017")
    override_dir = paths.data_raw / "ultralight_splits"
    override_dir.mkdir(parents=True, exist_ok=True)
    with open(override_dir / "isic2017.json", "w") as f:
        json.dump({"train": ["ISIC_0000000"], "val": [], "test": []}, f)  # way too few ids

    with pytest.raises(ValueError, match="does not exactly partition"):
        sp.build_isic_split(paths, "isic2017")


def test_build_all_splits_skips_missing_datasets_gracefully(make_kvasir_raw):
    paths = make_kvasir_raw(n=20)  # only kvasir present; CVC/ISIC raw data absent
    results = sp.build_all_splits(paths)
    assert "kvasir_seg" in results
    assert "cvc_clinicdb" not in results
    assert "isic2017" not in results
    assert "isic2018" not in results


def test_cross_dataset_pairs_are_the_required_two():
    assert sp.CROSS_DATASET_PAIRS == (
        ("kvasir_seg", "cvc_clinicdb"),
        ("cvc_clinicdb", "kvasir_seg"),
    )
