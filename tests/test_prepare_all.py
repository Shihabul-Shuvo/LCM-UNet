"""lcmunet/data/prepare_all.py -- orchestration only (ensure_dataset_ready
and the split builders are exercised directly in test_download.py /
test_splits.py-equivalent real-data tests); here we monkeypatch both layers
to verify the per-dataset isolation and summary-table contract."""

from __future__ import annotations

import pytest

from lcmunet.data import prepare_all as pa
from lcmunet.data import raw_layout as rl
from lcmunet.data.kaggle_auth import KaggleAuthMissingError


def test_prepare_all_datasets_all_pass(paths, monkeypatch):
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, p: 10)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 8, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert set(report) == set(rl.DATASET_NAMES)
    assert all(row["status"] == "PASS" for row in report.values())
    assert all(row["n_pairs"] == 10 for row in report.values())


def test_prepare_all_datasets_one_failure_does_not_block_others(paths, monkeypatch):
    def flaky_ensure(name, p):
        if name == "cvc_clinicdb":
            raise KaggleAuthMissingError()
        return 5

    monkeypatch.setattr(pa, "ensure_dataset_ready", flaky_ensure)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 3, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["cvc_clinicdb"]["status"] == "FAIL"
    assert "kaggle.json" in report["cvc_clinicdb"]["error"]
    for name in rl.DATASET_NAMES:
        if name != "cvc_clinicdb":
            assert report[name]["status"] == "PASS"


def test_prepare_all_datasets_raw_data_missing_is_caught(paths, monkeypatch):
    def missing_ensure(name, p):
        if name == "isic2017":
            raise rl.RawDataMissingError("isic2017", "download it from ...")
        return 1

    monkeypatch.setattr(pa, "ensure_dataset_ready", missing_ensure)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 0, "val": 0, "test": 0}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["isic2017"]["status"] == "FAIL"
    assert report["kvasir_seg"]["status"] == "PASS"


def test_prepare_all_datasets_generic_exception_is_caught(paths, monkeypatch):
    def boom_ensure(name, p):
        if name == "isic2018":
            raise RuntimeError("archive layout changed")
        return 1

    monkeypatch.setattr(pa, "ensure_dataset_ready", boom_ensure)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 0, "val": 0, "test": 0}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["isic2018"]["status"] == "FAIL"
    assert "archive layout changed" in report["isic2018"]["error"]


def test_prepare_all_datasets_split_builder_failure_is_also_caught(paths, monkeypatch):
    """A split-builder failure (e.g. the CVC sequence-leakage assertion)
    must be caught exactly like a download failure -- it is not special-cased."""

    def flaky_split(p):
        raise AssertionError("CVC sequence-level split leakage: 1 sequence(s) span more than one partition")

    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, p: 5)
    builders = {name: (lambda p: {"counts": {"train": 1, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES}
    builders["cvc_clinicdb"] = flaky_split
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", builders)

    report = pa.prepare_all_datasets(paths)

    assert report["cvc_clinicdb"]["status"] == "FAIL"
    assert "leakage" in report["cvc_clinicdb"]["error"]
    assert report["kvasir_seg"]["status"] == "PASS"


def test_print_summary_table_does_not_crash(paths, monkeypatch, capsys):
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, p: 5)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 3, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    pa.prepare_all_datasets(paths)
    out = capsys.readouterr().out
    assert "DATA PREPARATION SUMMARY" in out
    assert "4/4 datasets ready." in out


def test_prepare_all_datasets_defaults_to_get_paths(monkeypatch, tmp_path):
    from lcmunet import paths as paths_module

    fake_paths = paths_module.get_paths(root=tmp_path / "drive_root")
    monkeypatch.setattr(paths_module, "get_paths", lambda: fake_paths)

    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, p: 1)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 0, "val": 0, "test": 1}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets()  # paths=None -> should call get_paths()
    assert all(row["status"] == "PASS" for row in report.values())
