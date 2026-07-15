import csv

import pytest
import torch

from lcmunet import efficiency_report as er


def test_build_all_models_ultralight_baseline_fails_others_succeed(monkeypatch):
    import lcmunet.engine as engine_mod

    monkeypatch.setattr(engine_mod, "SCAN_IMPL", "ref")  # this machine has no mamba-ssm CUDA build
    models, errors = er.build_all_models(scan_impl="ref")

    assert "ultralight_baseline" in errors
    assert set(models.keys()) == {"glgf", "lc_ss2d", "unet", "malunet", "egeunet"}


def test_verify_lc_ss2d_lighter_than_glgf_real_measurement():
    """Not a synthetic fixture -- builds the REAL project models and reports
    the REAL measured params delta, exactly like the production code path."""
    from lcmunet.engine import build_model

    scan_impl = "ref"
    glgf_row = {"model_name": "glgf", "params_M": _params_of(build_model(er._build_config("glgf", scan_impl)))}
    hero_row = {"model_name": "lc_ss2d", "params_M": _params_of(build_model(er._build_config("lc_ss2d", scan_impl)))}

    result = er.verify_lc_ss2d_lighter_than_glgf([glgf_row, hero_row])
    assert result is not None
    assert result["lc_ss2d_params_M"] == hero_row["params_M"]
    assert result["glgf_params_M"] == glgf_row["params_M"]
    assert result["delta_M"] == pytest.approx(hero_row["params_M"] - glgf_row["params_M"])
    assert isinstance(result["claim_confirmed"], bool)


def _params_of(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def test_verify_lc_ss2d_lighter_than_glgf_returns_none_when_row_missing():
    assert er.verify_lc_ss2d_lighter_than_glgf([{"model_name": "glgf", "params_M": 1.0}]) is None
    assert er.verify_lc_ss2d_lighter_than_glgf([]) is None


def test_render_report_warns_when_no_cuda_device():
    rows = [{"model_name": "unet", "params_M": 31.0, "gflops_module_thop": 50.0, "gflops_scan_supplementary": 0.0,
             "gflops_total": 50.0, "fps_b1": 10.0, "fps_b8": 60.0, "peak_mem_MB_b8": None}]
    md = er.render_efficiency_report_md(rows, {}, "ref", "test-source", torch.device("cpu"), None, "2.0.0")
    assert "NO CUDA device" in md
    assert "unet" in md


def test_render_report_lists_errors_and_claim_check():
    rows = [{"model_name": "glgf", "params_M": 0.10, "gflops_module_thop": 1.0, "gflops_scan_supplementary": 0.1,
             "gflops_total": 1.1, "fps_b1": 5.0, "fps_b8": 30.0, "peak_mem_MB_b8": None},
            {"model_name": "lc_ss2d", "params_M": 0.08, "gflops_module_thop": 0.9, "gflops_scan_supplementary": 0.1,
             "gflops_total": 1.0, "fps_b1": 5.0, "fps_b8": 30.0, "peak_mem_MB_b8": None}]
    errors = {"ultralight_baseline": "RuntimeError: needs cuda"}
    md = er.render_efficiency_report_md(rows, errors, "ref", "test-source", torch.device("cpu"), None, "2.0.0")

    assert "NOT measured" in md
    assert "ultralight_baseline" in md
    assert "CONFIRMED" in md  # 0.08 < 0.10 -> lc_ss2d is lighter


def test_generate_efficiency_report_end_to_end_cpu(paths):
    result = er.generate_efficiency_report(
        paths, input_size=256, fps_batch_sizes=(1, 8), memory_batch_size=8, n_warmup=1, n_measure=2,
    )

    assert result["efficiency_csv"].is_file()
    assert result["efficiency_report_md"].is_file()
    assert "ultralight_baseline" in result["errors"]
    assert len(result["rows"]) == 5  # every model except ultralight_baseline

    with open(result["efficiency_csv"], newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 5
    model_names = {row["model_name"] for row in csv_rows}
    assert model_names == {"glgf", "lc_ss2d", "unet", "malunet", "egeunet"}
    for row in csv_rows:
        assert row["scan_impl"] == csv_rows[0]["scan_impl"]  # SAME scan_impl for every measured row

    md = result["efficiency_report_md"].read_text(encoding="utf-8")
    assert "Claim check" in md
    assert "NOT measured" in md and "ultralight_baseline" in md
