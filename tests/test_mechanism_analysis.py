"""Mechanism analysis (methodology section 11). Real hero-side captures use
a real, CPU-runnable LCMUNet (no mamba-ssm needed -- lcmunet/lc_vss.py
reimplements the Mamba forward directly). The vendored baseline's mamba_ssm
hook is untestable end-to-end on this CPU dev machine (mamba_ssm is not
installed -- GLOBAL RULES rule 5); its hook-target/use_fast_path LOGIC is
verified with a fake mamba_ssm-shaped module tree (same technique as
tests/test_efficiency.py's regression test for the identical bug class).
"""

from __future__ import annotations

import sys
import textwrap
import types

import numpy as np
import pandas as pd
import pytest
import torch

import lcmunet.delta_diff as dd
import lcmunet.mechanism_analysis as ma
from lcmunet.lcm_unet import LCMUNet


# ---- pure math: un-flatten / reduction helpers -----------------------------


def test_dts_calls_to_2d_map_applies_softplus_and_averages_channels_and_groups():
    torch.manual_seed(0)
    h, w = 2, 3
    l = h * w
    b, d_inner = 1, 4
    # two identical PVM-group calls, all-zero dts -> softplus(0) = ln(2) everywhere
    calls = [torch.zeros(b, d_inner, l), torch.zeros(b, d_inner, l)]

    result = ma._dts_calls_to_2d_map(calls, h, w)

    assert result.shape == (b, h, w)
    assert np.allclose(result, np.log(2.0), atol=1e-6)


def test_dts_calls_to_2d_map_unflattens_row_major_correctly():
    h, w = 2, 2
    l = h * w
    b, d_inner = 1, 1
    # dts values 0,1,2,3 along L -- after softplus, un-flattening must preserve
    # row-major order: token t = row*w + col
    dts = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])  # (1, 1, 4)
    result = ma._dts_calls_to_2d_map([dts], h, w)

    expected = torch.nn.functional.softplus(torch.tensor([[0.0, 1.0], [2.0, 3.0]])).numpy()
    assert np.allclose(result[0], expected)


def test_dts_calls_to_2d_map_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="un-flatten shape mismatch"):
        ma._dts_calls_to_2d_map([torch.zeros(1, 2, 5)], 2, 3)  # 5 != 6


def test_modulation_calls_to_2d_map_averages_last_axis_and_unflattens():
    h, w = 2, 2
    l = h * w
    mod = torch.tensor([[[1.0, 3.0], [2.0, 4.0], [0.0, 0.0], [10.0, -10.0]]])  # (1, 4, 2)
    result = ma._modulation_calls_to_2d_map([mod], h, w)

    expected_flat = mod.mean(dim=2).reshape(1, h, w).numpy()
    assert result.shape == (1, h, w)
    assert np.allclose(result, expected_flat)


# ---- region masks -----------------------------------------------------------


def test_region_masks_partition_every_pixel_exactly_once():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[3:7, 3:7] = 1.0  # a filled 4x4 square

    regions = ma._region_masks_at_resolution(mask, 10, 10)
    boundary, interior, background = regions["boundary"], regions["interior"], regions["background"]

    stacked = np.stack([boundary, interior, background])
    assert (stacked.sum(axis=0) == 1).all()  # every pixel in EXACTLY one region
    assert interior.any()  # a 4x4 filled square has a nonempty eroded interior
    assert boundary.any()
    assert background.any()


def test_region_masks_downsamples_nearest_to_stage_resolution():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    regions = ma._region_masks_at_resolution(mask, 4, 4)
    for r in regions.values():
        assert r.shape == (4, 4)


def test_region_masks_all_background_when_mask_empty():
    mask = np.zeros((8, 8), dtype=np.float32)
    regions = ma._region_masks_at_resolution(mask, 8, 8)
    assert regions["background"].all()
    assert not regions["boundary"].any()
    assert not regions["interior"].any()


# ---- _baseline_stage_modules: structural, no mamba_ssm needed --------------


def test_baseline_stage_modules_extracts_the_right_positional_slots():
    class _Fake:
        pass

    model = _Fake()
    model.encoder4 = [ "e4" ]
    model.encoder5 = [ "e5" ]
    model.decoder1 = [ "d5" ]
    model.decoder2 = [ "d4" ]

    stages = ma._baseline_stage_modules(model)
    assert stages == {"E4": "e4", "E5": "e5", "D5": "d5", "D4": "d4"}


# ---- baseline mamba_ssm hook: fake module tree (mirrors test_efficiency.py) --


def _install_fake_mamba_simple(monkeypatch):
    """See tests/test_efficiency.py's _install_fake_mamba_simple -- identical
    technique: a real exec()'d module (not a plain object) so `selective_scan_fn`
    is a GLOBAL name bound in the fake module's own namespace, exactly
    reproducing mamba_ssm's `from ...selective_scan_interface import
    selective_scan_fn` static-binding structure that makes the hook-target
    choice matter in the first place."""
    fake_interface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")

    def _stub_selective_scan_fn(u, dts, A, B, C, D=None, **kw):
        return u

    fake_interface.selective_scan_fn = _stub_selective_scan_fn

    fake_simple = types.ModuleType("mamba_ssm.modules.mamba_simple")
    src = textwrap.dedent(
        """
        import torch
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        class Mamba(torch.nn.Module):
            def __init__(self, d_model, use_fast_path=True):
                super().__init__()
                self.use_fast_path = use_fast_path
                self.d_inner = d_model * 2

            def forward(self, hidden_states):
                b, l, _d = hidden_states.shape
                if self.use_fast_path:
                    return hidden_states  # simulates the fused path -- never calls selective_scan_fn
                u = torch.zeros(b, self.d_inner, l)
                dt = torch.arange(b * self.d_inner * l, dtype=torch.float32).reshape(b, self.d_inner, l)
                A = torch.zeros(self.d_inner, 4)
                B = torch.zeros(b, 4, l)
                C = torch.zeros(b, 4, l)
                selective_scan_fn(u, dt, A, B, C)
                return hidden_states
        """
    )
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.selective_scan_interface", fake_interface)
    exec(compile(src, "<fake_mamba_simple>", "exec"), fake_simple.__dict__)
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules.mamba_simple", fake_simple)
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules", types.ModuleType("mamba_ssm.modules"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", types.ModuleType("mamba_ssm.ops"))
    return fake_simple


class _FakeVendoredPVMLayer(torch.nn.Module):
    def __init__(self, fake_simple, d_model=8):
        super().__init__()
        self.mamba = fake_simple.Mamba(d_model=d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w).transpose(-1, -2)
        self.mamba(x_flat)
        return x


def test_capture_baseline_dts_hooks_correct_module_and_forces_slow_path(monkeypatch):
    fake_simple = _install_fake_mamba_simple(monkeypatch)
    stage = _FakeVendoredPVMLayer(fake_simple)
    assert stage.mamba.use_fast_path is True

    calls = ma._capture_baseline_dts(stage, torch.randn(2, 8, 4, 4))

    assert len(calls) == 1
    assert calls[0].shape == (2, 16, 16)  # (B, d_inner=2*8, L=4*4)
    assert stage.mamba.use_fast_path is True  # restored


def test_capture_baseline_dts_raises_without_use_fast_path_attribute(monkeypatch):
    fake_simple = _install_fake_mamba_simple(monkeypatch)
    stage = _FakeVendoredPVMLayer(fake_simple)
    del stage.mamba.use_fast_path

    with pytest.raises(AttributeError, match="use_fast_path"):
        ma._capture_baseline_dts(stage, torch.randn(1, 8, 4, 4))


# ---- hero-side capture: real LCMUNet, no mocks -----------------------------


def test_hero_stage_capture_finds_all_four_stages_with_4_groups_each():
    torch.manual_seed(0)
    model = LCMUNet(placement="hero")
    x = torch.randn(2, 3, 64, 64)

    capture = ma._hero_stage_capture(model, x)

    assert set(capture.keys()) == set(ma.STAGES)
    for stage, info in capture.items():
        assert len(info["dts"]) == 4  # 4 PVM groups
        h, w = info["input_hw"]
        assert info["dts"][0].shape[2] == h * w


def test_hero_stage_capture_raises_when_no_active_lc_vss_stage():
    model = LCMUNet(placement="P0")
    with pytest.raises(ValueError, match="no active LC-VSS alpha"):
        ma._hero_stage_capture(model, torch.randn(1, 3, 64, 64))


def test_hero_modulation_maps_shapes_and_finiteness():
    torch.manual_seed(0)
    model = LCMUNet(placement="hero", alpha_init=0.05)
    x = torch.randn(2, 3, 64, 64)

    maps = ma.hero_modulation_maps(model, x)

    assert set(maps.keys()) == set(ma.STAGES)
    for stage, m in maps.items():
        assert m.shape[0] == 2
        assert np.isfinite(m).all()


# ---- region_wise_modulation_stats: real model + synthetic masks ------------


def _circular_mask_batch(batch=2, size=64, radius=None):
    masks = torch.zeros(batch, 1, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    center = size // 2
    radius = radius if radius is not None else size // 8  # small relative to size -- see caller's note on coarse-stage margins
    circle = ((yy - center) ** 2 + (xx - center) ** 2 <= radius**2).float()
    for b in range(batch):
        masks[b, 0] = circle
    return masks


def test_region_wise_modulation_stats_full_report_structure():
    torch.manual_seed(0)
    model = LCMUNet(placement="hero", alpha_init=0.05)
    images = torch.randn(2, 3, 64, 64)
    masks = _circular_mask_batch(batch=2, size=64)

    report = ma.region_wise_modulation_stats(model, images, masks)

    assert set(report.keys()) == set(ma.STAGES)
    for stage, s in report.items():
        for region in ("boundary", "interior", "background"):
            assert set(s[region].keys()) == {"mean", "std", "n"}
        assert isinstance(s["boundary_ne_background"], bool)

    # Only E4 (8x8 tokens for a 64x64 input -- the finest of the 4 stages;
    # E5=4x4, D5=D4=2x2, confirmed by direct capture) has enough resolution
    # margin for a small centered circular lesion to reliably leave all 3
    # regions non-empty. At D5/D4's 2x2 grid, one 3x3 dilation of a single
    # foreground pixel fills the WHOLE grid -- background legitimately
    # empties there, handled via NaN (see the dedicated test below), not
    # something a real 64x64 CPU-test-only resolution can avoid.
    s = report["E4"]
    assert s["boundary"]["n"] > 0
    assert s["background"]["n"] > 0
    assert not np.isnan(s["kruskal_p"])


def test_region_wise_modulation_stats_handles_empty_regions_gracefully():
    """At coarse token resolutions (E5/D5, 4x4 tokens for a 64x64 input) a
    single 3x3 dilation/erosion can fill or empty the entire grid for a
    centered lesion -- _region_stat/the Kruskal-Wallis call must report NaN
    rather than crashing (e.g. on an empty-array .mean() or scipy.kruskal
    with an empty sample)."""
    torch.manual_seed(0)
    model = LCMUNet(placement="hero", alpha_init=0.05)
    images = torch.randn(2, 3, 64, 64)
    masks = _circular_mask_batch(batch=2, size=64)

    report = ma.region_wise_modulation_stats(model, images, masks)  # must not raise
    for stage in ("E5", "D5", "D4"):
        s = report[stage]
        for region in ("boundary", "interior", "background"):
            if s[region]["n"] == 0:
                assert np.isnan(s[region]["mean"]) and np.isnan(s[region]["std"])
        if s["boundary"]["n"] == 0 or s["background"]["n"] == 0:
            assert np.isnan(s["kruskal_p"])
            assert s["boundary_ne_background"] is False


def test_pooled_boundary_vs_background_summarises_worst_stage():
    region_stats = {
        "E4": {"kruskal_p": 0.5, "boundary_ne_background": False},
        "E5": {"kruskal_p": 0.001, "boundary_ne_background": True},
        "D5": {"kruskal_p": 0.02, "boundary_ne_background": True},
        "D4": {"kruskal_p": 0.9, "boundary_ne_background": False},
    }
    pooled = ma.pooled_boundary_vs_background(region_stats)
    assert pooled["conservative_worst_stage"] == "D4"
    assert pooled["all_stages_significant"] is False


# ---- delta_difference_maps: assembly/fail-loud logic, capture seams mocked --


def _fake_capture(stage_shapes, n_calls_per_stage):
    def _capture(model, images):
        out = {}
        for stage in ma.STAGES:
            h, w = stage_shapes[stage]
            l = h * w
            out[stage] = {
                "input_hw": (h, w),
                "dts": [torch.zeros(1, 4, l) for _ in range(n_calls_per_stage)],
            }
        return out
    return _capture


def test_delta_difference_maps_computes_diff_from_mocked_captures(monkeypatch):
    shapes = {"E4": (4, 4), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    monkeypatch.setattr(ma, "_hero_stage_capture", _fake_capture(shapes, 4))
    monkeypatch.setattr(ma, "_baseline_stage_capture", _fake_capture(shapes, 4))

    maps = ma.delta_difference_maps(torch.nn.Module(), torch.nn.Module(), torch.randn(1, 3, 8, 8))

    assert set(maps.keys()) == set(ma.STAGES)
    for stage, m in maps.items():
        assert np.allclose(m["diff"], 0.0)  # identical hero/baseline dts -> exactly zero diff


def test_delta_difference_maps_raises_on_resolution_mismatch(monkeypatch):
    hero_shapes = {"E4": (4, 4), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    baseline_shapes = {"E4": (2, 2), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    monkeypatch.setattr(ma, "_hero_stage_capture", _fake_capture(hero_shapes, 4))
    monkeypatch.setattr(ma, "_baseline_stage_capture", _fake_capture(baseline_shapes, 4))

    with pytest.raises(ValueError, match="resolution mismatch"):
        ma.delta_difference_maps(torch.nn.Module(), torch.nn.Module(), torch.randn(1, 3, 8, 8))


def test_delta_difference_maps_raises_on_zero_scan_calls(monkeypatch):
    shapes = {"E4": (4, 4), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    monkeypatch.setattr(ma, "_hero_stage_capture", _fake_capture(shapes, 0))
    monkeypatch.setattr(ma, "_baseline_stage_capture", _fake_capture(shapes, 4))

    with pytest.raises(RuntimeError, match="captured zero scan calls"):
        ma.delta_difference_maps(torch.nn.Module(), torch.nn.Module(), torch.randn(1, 3, 8, 8))


# ---- figure-saving smoke tests ----------------------------------------------


def test_save_delta_difference_figure_writes_a_file(tmp_path):
    shapes = {"E4": (4, 4), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    maps = {
        stage: {"hero": np.random.rand(1, h, w), "baseline": np.random.rand(1, h, w), "diff": np.random.randn(1, h, w)}
        for stage, (h, w) in shapes.items()
    }
    out = tmp_path / "delta.png"
    result = ma.save_delta_difference_figure(maps, out)
    assert result == out
    assert out.is_file() and out.stat().st_size > 0


def test_save_region_wise_figure_writes_a_file(tmp_path):
    region_stats = {
        stage: {
            "boundary": {"mean": 0.1, "std": 0.02, "n": 50},
            "interior": {"mean": 0.05, "std": 0.01, "n": 100},
            "background": {"mean": 0.01, "std": 0.005, "n": 200},
            "kruskal_h": 5.0, "kruskal_p": 0.01, "boundary_ne_background": True,
        }
        for stage in ma.STAGES
    }
    out = tmp_path / "region.png"
    result = ma.save_region_wise_figure(region_stats, out)
    assert result == out
    assert out.is_file() and out.stat().st_size > 0


def test_save_alpha_figure_writes_a_file(tmp_path):
    df = pd.DataFrame({"epoch": [10, 20, 30], "E4": [0.01, 0.02, 0.03], "E5": [0.01, 0.015, 0.02]})
    out = tmp_path / "alpha.png"
    result = ma.save_alpha_figure(df, out)
    assert result == out
    assert out.is_file() and out.stat().st_size > 0


# ---- load_alpha_log ----------------------------------------------------------


def test_load_alpha_log_raises_clearly_when_missing(paths):
    from lcmunet.config import RunConfig

    config = RunConfig(run_name="x", model_name="lc_ss2d", dataset="kvasir_seg", seed=0, split_file="splits/kvasir_seg.json")
    with pytest.raises(RuntimeError, match="No alpha log"):
        ma.load_alpha_log(paths, config)


def test_load_alpha_log_reads_a_real_written_log(paths):
    from lcmunet.config import RunConfig
    from lcmunet.engine import _alpha_csv_path, _log_alphas

    config = RunConfig(run_name="x", model_name="lc_ss2d", dataset="kvasir_seg", seed=0, split_file="splits/kvasir_seg.json")
    path = _alpha_csv_path(paths, config)
    _log_alphas(path, 10, {"E4": 0.01, "E5": 0.02, "D5": 0.03, "D4": 0.04})
    _log_alphas(path, 20, {"E4": 0.02, "E5": 0.03, "D5": 0.04, "D4": 0.05})

    df = ma.load_alpha_log(paths, config)
    assert list(df["epoch"]) == [10, 20]
    assert df["E4"].iloc[-1] == 0.02


# ---- render_mechanism_report_md: smoke test --------------------------------


def test_render_mechanism_report_md_contains_disclaimer_and_sections():
    from lcmunet.config import RunConfig

    hero_config = RunConfig(run_name="hero", model_name="lc_ss2d", dataset="kvasir_seg", seed=42, split_file="splits/kvasir_seg.json")
    baseline_config = RunConfig(run_name="baseline", model_name="ultralight_baseline", dataset="kvasir_seg", seed=42, split_file="splits/kvasir_seg.json")

    shapes = {"E4": (4, 4), "E5": (2, 2), "D5": (2, 2), "D4": (4, 4)}
    delta_maps = {stage: {"diff": np.random.randn(1, h, w)} for stage, (h, w) in shapes.items()}
    region_stats = {
        stage: {
            "boundary": {"mean": 0.1, "std": 0.02, "n": 50},
            "interior": {"mean": 0.05, "std": 0.01, "n": 100},
            "background": {"mean": 0.01, "std": 0.005, "n": 200},
            "kruskal_h": 5.0, "kruskal_p": 0.01, "boundary_ne_background": True,
        }
        for stage in ma.STAGES
    }
    alpha_df = pd.DataFrame({"epoch": [10, 20], "E4": [0.01, 0.02], "E5": [0.01, 0.02], "D5": [0.01, 0.02], "D4": [0.01, 0.02]})
    figure_paths = {"delta_difference_figure": "figures/delta.png", "region_wise_figure": "figures/region.png",
                     "region_wise_csv": "results/region.csv", "alpha_figure": "figures/alpha.png"}

    md = ma.render_mechanism_report_md(hero_config, baseline_config, ["img1", "img2"], delta_maps, region_stats, alpha_df, figure_paths)

    assert ma.DISCLAIMER in md
    assert "## 1. Delta-difference map" in md
    assert "## 2. Region-wise modulation statistics" in md
    assert "## 3. Per-stage learned alpha" in md
    for stage in ma.STAGES:
        assert stage in md


# ---- end-to-end: real (cheap) training for hero + a CPU-runnable stand-in --
# ---- for the GPU-gated ultralight_baseline (same rationale as              --
# ---- test_gate2_report.py's end-to-end test)                               --


def test_generate_mechanism_report_end_to_end(make_kvasir_raw, monkeypatch):
    """Real run_one() training for the hero (descriptor_type='contrast') and
    a CPU-runnable stand-in for 'baseline' (lc_ss2d, descriptor_type='none'
    -- mathematically identical to stock PVMLayer per lcmunet/lc_vss.py's
    own docstring). _baseline_stage_capture is monkeypatched to route
    through the identical real capture mechanics _hero_stage_capture uses
    (dd._capture_dts hooking lcmunet.lc_vss.selective_scan), since the
    stand-in is an LC_PVMLayer-based model, not the vendored mamba_ssm one
    -- the REAL vendored-model hook is separately, directly unit-tested
    above with a fake mamba_ssm module tree.
    """
    from lcmunet.config import RunConfig
    from lcmunet.data.splits import build_kvasir_split
    from lcmunet.engine import ALPHA_LOG_EVERY as _unused  # noqa: F401
    from lcmunet import engine as engine_mod
    from lcmunet.engine import run_one

    monkeypatch.setattr(engine_mod, "ALPHA_LOG_EVERY", 1)  # log alpha every epoch so a short run still produces a log

    def _stand_in_baseline_capture(model, images):
        stage_modules = ma._baseline_stage_modules(model)  # unconditional E4/E5/D5/D4 grab
        stage_inputs = dd._capture_stage_inputs(model, images, stage_modules)
        return {
            stage: {
                "input_hw": (int(stage_inputs[stage].shape[2]), int(stage_inputs[stage].shape[3])),
                "dts": dd._capture_dts(stage_modules[stage], stage_inputs[stage]),
            }
            for stage in ma.STAGES
        }

    monkeypatch.setattr(ma, "_baseline_stage_capture", _stand_in_baseline_capture)

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)

    hero_config = RunConfig(
        run_name="mech_hero", model_name="lc_ss2d", dataset="kvasir_seg", seed=0,
        split_file="splits/kvasir_seg.json", epochs=3, batch_size=4, input_size=64,
    )
    baseline_config = RunConfig(
        run_name="mech_baseline_standin", model_name="lc_ss2d", dataset="kvasir_seg", seed=0,
        split_file="splits/kvasir_seg.json", epochs=3, batch_size=4, input_size=64,
        model_cfg={"descriptor_type": "none"},
    )

    run_one(hero_config, paths=paths, num_workers=0)
    run_one(baseline_config, paths=paths, num_workers=0)

    result = ma.generate_mechanism_report(paths, hero_config=hero_config, baseline_config=baseline_config, n_images=2)

    for key in ("delta_difference_figure", "region_wise_figure", "region_wise_csv", "alpha_figure", "mechanism_report_md"):
        assert result[key].is_file(), f"{key} missing"

    content = result["mechanism_report_md"].read_text(encoding="utf-8")
    assert ma.DISCLAIMER in content
    for stage in ma.STAGES:
        assert stage in content
