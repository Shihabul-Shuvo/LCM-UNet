"""lcmunet/data/prepare_all.py -- orchestration + the data/<name>/prepared.json
marker cache. ensure_dataset_ready and the split builders are exercised
directly in test_download.py; here we monkeypatch both layers to verify
per-dataset isolation, the marker fast-path, and the summary-table contract.
"""

from __future__ import annotations

import json

from lcmunet import config as config_module
from lcmunet.data import prepare_all as pa
from lcmunet.data import raw_layout as rl


def _patch_all_pass(monkeypatch, n_pairs=10, counts=None):
    counts = counts or {"train": 8, "val": 1, "test": 1}
    # These generic orchestration tests predate ACTIVE_DATASETS and are about
    # PASS/FAIL isolation across "every dataset", independent of the scope
    # feature -- keep them exercising all 4 (the scope-specific behaviour has
    # its own dedicated tests further down).
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", list(rl.DATASET_NAMES))
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: n_pairs)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": counts}) for name in rl.DATASET_NAMES})


def test_prepare_all_datasets_all_pass(paths, monkeypatch):
    _patch_all_pass(monkeypatch)

    report = pa.prepare_all_datasets(paths)

    assert set(report) == set(rl.DATASET_NAMES)
    assert all(row["status"] == "PASS" for row in report.values())
    assert all(row["n_pairs"] == 10 for row in report.values())
    assert all(row["cached"] is False for row in report.values())


def test_prepare_all_datasets_one_failure_does_not_block_others(paths, monkeypatch):
    def flaky_ensure(name, data_raw_dir):
        if name == "cvc_clinicdb":
            raise rl.RawDataMissingError("cvc_clinicdb", "No CVC-ClinicDB .zip found under ...")
        return 5

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", list(rl.DATASET_NAMES))
    monkeypatch.setattr(pa, "ensure_dataset_ready", flaky_ensure)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 3, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["cvc_clinicdb"]["status"] == "FAIL"
    assert "No CVC-ClinicDB .zip found" in report["cvc_clinicdb"]["error"]
    for name in rl.DATASET_NAMES:
        if name != "cvc_clinicdb":
            assert report[name]["status"] == "PASS"


def test_prepare_all_datasets_generic_exception_is_caught(paths, monkeypatch):
    def boom_ensure(name, data_raw_dir):
        if name == "isic2018":
            raise RuntimeError("archive layout changed")
        return 1

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", list(rl.DATASET_NAMES))
    monkeypatch.setattr(pa, "ensure_dataset_ready", boom_ensure)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 0, "val": 0, "test": 0}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["isic2018"]["status"] == "FAIL"
    assert "archive layout changed" in report["isic2018"]["error"]


def test_prepare_all_datasets_split_builder_failure_is_also_caught(paths, monkeypatch):
    """A split-builder failure (e.g. the CVC sequence-leakage assertion)
    must be caught exactly like an extraction failure -- it is not special-cased."""

    def flaky_split(p):
        raise AssertionError("CVC sequence-level split leakage: 1 sequence(s) span more than one partition")

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", list(rl.DATASET_NAMES))
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: 5)
    builders = {name: (lambda p: {"counts": {"train": 1, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES}
    builders["cvc_clinicdb"] = flaky_split
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", builders)

    report = pa.prepare_all_datasets(paths)

    assert report["cvc_clinicdb"]["status"] == "FAIL"
    assert "leakage" in report["cvc_clinicdb"]["error"]
    assert report["kvasir_seg"]["status"] == "PASS"


def test_print_summary_table_does_not_crash(paths, monkeypatch, capsys):
    _patch_all_pass(monkeypatch)

    pa.prepare_all_datasets(paths)
    out = capsys.readouterr().out
    assert "DATA PREPARATION SUMMARY" in out
    assert "4/4 in-scope datasets ready." in out


def test_prepare_all_datasets_defaults_to_get_paths(monkeypatch, tmp_path):
    from lcmunet import paths as paths_module

    fake_paths = paths_module.get_paths(root=tmp_path / "drive_root")
    monkeypatch.setattr(paths_module, "get_paths", lambda: fake_paths)

    _patch_all_pass(monkeypatch, counts={"train": 0, "val": 0, "test": 1})

    report = pa.prepare_all_datasets()  # paths=None -> should call get_paths()
    assert all(row["status"] == "PASS" for row in report.values())


# ---- data/<name>/prepared.json marker cache ---------------------------------


def test_marker_write_and_load_roundtrip(paths):
    path = pa._write_marker(paths, "kvasir_seg", 1000, {"train": 800, "val": 100, "test": 100})
    assert path == paths.data / "kvasir_seg" / "prepared.json"

    cached = pa._load_marker(paths, "kvasir_seg")
    assert cached == {"n_pairs": 1000, "counts": {"train": 800, "val": 100, "test": 100}}


def test_marker_absent_returns_none(paths):
    assert pa._load_marker(paths, "kvasir_seg") is None


def test_marker_corrupt_file_treated_as_absent(paths):
    marker_path = paths.data / "kvasir_seg" / "prepared.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("{not valid json", encoding="utf-8")

    assert pa._load_marker(paths, "kvasir_seg") is None


def test_prepare_one_dataset_skips_instantly_when_marker_present(paths, monkeypatch):
    pa._write_marker(paths, "kvasir_seg", 1000, {"train": 800, "val": 100, "test": 100})

    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: (_ for _ in ()).throw(AssertionError("should not extract when cached")))
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {"kvasir_seg": lambda p: (_ for _ in ()).throw(AssertionError("should not re-split when cached"))})

    result = pa._prepare_one_dataset("kvasir_seg", paths)
    assert result == {"status": "PASS", "n_pairs": 1000, "counts": {"train": 800, "val": 100, "test": 100}, "cached": True}


def test_prepare_all_datasets_writes_marker_after_first_success(paths, monkeypatch):
    _patch_all_pass(monkeypatch, n_pairs=612, counts={"train": 490, "val": 61, "test": 61})

    pa.prepare_all_datasets(paths)

    marker = json.loads((paths.data / "cvc_clinicdb" / "prepared.json").read_text(encoding="utf-8"))
    assert marker == {"n_pairs": 612, "counts": {"train": 490, "val": 61, "test": 61}}


# ---- ACTIVE_DATASETS scope -----------------------------------------------


def test_out_of_scope_datasets_are_skipped_not_failed(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: (_ for _ in ()).throw(AssertionError(f"should never attempt {name}: not in ACTIVE_DATASETS")) if name in ("isic2017", "isic2018") else 10)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 8, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    report = pa.prepare_all_datasets(paths)

    assert report["kvasir_seg"]["status"] == "PASS"
    assert report["cvc_clinicdb"]["status"] == "PASS"
    assert report["isic2017"] == {"status": "SKIPPED", "error": "not in ACTIVE_DATASETS"}
    assert report["isic2018"] == {"status": "SKIPPED", "error": "not in ACTIVE_DATASETS"}


def test_out_of_scope_datasets_do_not_require_data_raw_folder(paths, monkeypatch):
    """No data_raw/ISIC2017 or data_raw/ISIC2018 directory exists at all in
    this fixture -- SKIPPED must not care."""
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: 10 if name in ("kvasir_seg", "cvc_clinicdb") else (_ for _ in ()).throw(AssertionError("unreachable")))
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 8, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    assert not (paths.data_raw / "ISIC2017").exists()
    assert not (paths.data_raw / "ISIC2018").exists()

    report = pa.prepare_all_datasets(paths)
    assert report["isic2017"]["status"] == "SKIPPED"
    assert report["isic2018"]["status"] == "SKIPPED"
    # still true afterwards -- nothing was created for the skipped datasets
    assert not (paths.data_raw / "ISIC2017").exists()
    assert not (paths.data_raw / "ISIC2018").exists()


def test_print_summary_table_labels_skipped_datasets(paths, monkeypatch, capsys):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: 10)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 8, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    pa.prepare_all_datasets(paths)
    out = capsys.readouterr().out

    assert "isic2017" in out and "SKIPPED" in out
    assert "2/2 in-scope datasets ready; 2 SKIPPED (not in ACTIVE_DATASETS)." in out


def test_adding_a_dataset_back_to_active_datasets_prepares_it_next_call(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(pa, "ensure_dataset_ready", lambda name, data_raw_dir: 10)
    monkeypatch.setattr(pa, "_SPLIT_BUILDERS", {name: (lambda p: {"counts": {"train": 8, "val": 1, "test": 1}}) for name in rl.DATASET_NAMES})

    report1 = pa.prepare_all_datasets(paths)
    assert report1["isic2017"]["status"] == "SKIPPED"

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])
    report2 = pa.prepare_all_datasets(paths)
    assert report2["isic2017"]["status"] == "PASS"
    assert report2["kvasir_seg"]["cached"] is True  # already-prepared datasets are untouched, still fast
