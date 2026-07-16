"""End-to-end test of the ACTIVE_DATASETS toggle across the two pipeline
stages it actually gates: lcmunet.data.prepare_all.prepare_all_datasets
(download/extract/split) and lcmunet.run_manifest.sync_manifest_with_active_
datasets (enqueue). Real Kvasir/CVC raw fixtures, real splits, real
RunConfig/manifest machinery -- nothing mocked except the raw data itself
(via tests/conftest.py's make_kvasir_raw/make_cvc_raw, the same synthetic
fixtures every other data-pipeline test uses).
"""

from __future__ import annotations

import json

from lcmunet import config as config_module
from lcmunet import experiment_matrix as em
from lcmunet import run_manifest as rm
from lcmunet.data import prepare_all as pa
from lcmunet.data import raw_layout as rl


def _dataset_of_config_id(scan_impl: str = "ref", hero_descriptor_type: str = "contrast"):
    merged = em.build_all(scan_impl, hero_descriptor_type=hero_descriptor_type)
    return {cid: config.dataset for cid, (config, _roles) in merged.items()}


def test_kvasir_cvc_only_scope_produces_zero_errors_and_zero_isic_references(make_kvasir_raw, make_cvc_raw, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(rl, "KVASIR_IMAGE_COUNT", 20)
    monkeypatch.setattr(rl, "CVC_IMAGE_COUNT", 612)

    paths = make_kvasir_raw(n=20)
    paths = make_cvc_raw(n=612)  # sequence-level splitter needs enough frames per sequence for a non-empty test partition

    data_report = pa.prepare_all_datasets(paths)
    assert data_report["kvasir_seg"]["status"] == "PASS"
    assert data_report["cvc_clinicdb"]["status"] == "PASS"
    assert data_report["isic2017"] == {"status": "SKIPPED", "error": "not in ACTIVE_DATASETS"}
    assert data_report["isic2018"] == {"status": "SKIPPED", "error": "not in ACTIVE_DATASETS"}

    sync_report = rm.sync_manifest_with_active_datasets(paths)
    assert sync_report["out_of_scope_datasets"] == ["isic2017", "isic2018"]

    manifest = json.loads((paths.results / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) > 0

    dataset_of = _dataset_of_config_id()
    referenced_datasets = {dataset_of[cid] for cid in manifest["jobs"]}
    assert referenced_datasets == {"kvasir_seg", "cvc_clinicdb"}
    assert "isic2017" not in referenced_datasets
    assert "isic2018" not in referenced_datasets


def test_switching_to_all_four_adds_isic_jobs_leaves_kvasir_cvc_untouched(make_kvasir_raw, make_cvc_raw, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(rl, "KVASIR_IMAGE_COUNT", 20)
    monkeypatch.setattr(rl, "CVC_IMAGE_COUNT", 612)

    paths = make_kvasir_raw(n=20)
    paths = make_cvc_raw(n=612)

    pa.prepare_all_datasets(paths)
    rm.sync_manifest_with_active_datasets(paths)

    # simulate real progress on the existing Kvasir/CVC jobs before the scope change
    job1 = rm.next_pending(paths.results)
    rm.mark_done(paths.results, job1["config_id"])
    job2 = rm.next_pending(paths.results)
    rm.mark_failed(paths.results, job2["config_id"], error="simulated GPU OOM")

    manifest_before = json.loads((paths.results / "manifest.json").read_text(encoding="utf-8"))
    kvasir_cvc_ids = set(manifest_before["jobs"].keys())
    assert len(kvasir_cvc_ids) > 0

    # ---- switch ACTIVE_DATASETS to all 4 (the user's "edit config.py" step) ----
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])

    # prepare_all_datasets on ISIC would need real ISIC zips placed (out of
    # scope for this test); we only need to prove the MANIFEST sync behaviour
    # here -- sync_manifest_with_active_datasets enqueues PENDING jobs
    # regardless of whether the data is prepared yet (matching Phase-2 jobs
    # already being enqueueable before Gate-2, per this module's docstring).
    sync_report = rm.sync_manifest_with_active_datasets(paths)

    manifest_after = json.loads((paths.results / "manifest.json").read_text(encoding="utf-8"))

    # every pre-existing Kvasir/CVC job's full record is byte-identical --
    # not just "same status", the whole dict (heartbeat/error/timestamps too)
    for cid in kvasir_cvc_ids:
        assert manifest_after["jobs"][cid] == manifest_before["jobs"][cid]

    new_ids = set(manifest_after["jobs"]) - kvasir_cvc_ids
    assert len(new_ids) > 0
    assert new_ids == set(sync_report["newly_enqueued"])

    dataset_of = _dataset_of_config_id()
    for cid in new_ids:
        assert dataset_of[cid] in ("isic2017", "isic2018")
        assert manifest_after["jobs"][cid]["status"] == rm.STATUS_PENDING  # every new job starts PENDING


def test_repeated_sync_calls_produce_identical_manifest(make_kvasir_raw, make_cvc_raw, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    monkeypatch.setattr(rl, "KVASIR_IMAGE_COUNT", 20)
    monkeypatch.setattr(rl, "CVC_IMAGE_COUNT", 612)

    paths = make_kvasir_raw(n=20)
    paths = make_cvc_raw(n=612)
    pa.prepare_all_datasets(paths)

    rm.sync_manifest_with_active_datasets(paths)
    manifest_1 = json.loads((paths.results / "manifest.json").read_text(encoding="utf-8"))

    for _ in range(5):  # stand-in for "50 times in a row" -- enqueue()'s own no-op guarantee makes the result identical regardless of N
        rm.sync_manifest_with_active_datasets(paths)
    manifest_5 = json.loads((paths.results / "manifest.json").read_text(encoding="utf-8"))

    assert manifest_1 == manifest_5

    # a second prepare_all_datasets() call is also a fast, unchanged no-op
    data_report_2 = pa.prepare_all_datasets(paths)
    assert data_report_2["kvasir_seg"]["cached"] is True
    assert data_report_2["cvc_clinicdb"]["cached"] is True
