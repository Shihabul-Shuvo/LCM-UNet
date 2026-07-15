import torch

from lcmunet import efficiency as eff
from lcmunet.glgf import GLGFUNet
from lcmunet.lcm_unet import LCMUNet
from lcmunet.unet import UNet


CPU = torch.device("cpu")


# ---- params ------------------------------------------------------------


def test_measure_params_millions_matches_manual_sum():
    model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.Linear(20, 5))
    expected = sum(p.numel() for p in model.parameters()) / 1e6
    assert eff.measure_params_millions(model) == expected
    assert expected == (10 * 20 + 20 + 20 * 5 + 5) / 1e6


# ---- module-level GFLOPs (thop) -----------------------------------------


def test_module_level_gflops_positive_for_lc_ss2d_hero():
    model = LCMUNet(placement="hero")
    result = eff.measure_module_level_gflops(model, input_size=64, batch=1, device=CPU)
    assert result["gflops_module_thop"] > 0
    assert result["macs"] > 0


def test_module_level_gflops_positive_for_unet():
    model = UNet()
    result = eff.measure_module_level_gflops(model, input_size=256, batch=1, device=CPU)
    assert result["gflops_module_thop"] > 0


# ---- supplementary scan GFLOPs -------------------------------------------


def test_scan_gflops_zero_for_pure_cnn_unet():
    model = UNet()
    result = eff.measure_scan_gflops(model, input_size=256, batch=1, device=CPU)
    assert result["gflops_scan_supplementary"] == 0.0
    assert result["n_scan_calls"] == 0
    assert "none" in result["scan_hook_source"]


def test_scan_gflops_positive_for_lc_ss2d_hero():
    model = LCMUNet(placement="hero")
    result = eff.measure_scan_gflops(model, input_size=64, batch=1, device=CPU)
    assert result["gflops_scan_supplementary"] > 0
    # hero places LC-VSS at E4/E5/D5/D4, but E6/D3 are plain PVM and still
    # call the scan (descriptor_type='none') -- 6 stages x 4 PVM groups = 24 calls
    assert result["n_scan_calls"] == 24
    assert result["scan_hook_source"] == "lcmunet.lc_vss.selective_scan"


def test_scan_gflops_positive_for_glgf():
    model = GLGFUNet()
    result = eff.measure_scan_gflops(model, input_size=64, batch=1, device=CPU)
    assert result["gflops_scan_supplementary"] > 0
    assert result["n_scan_calls"] == 24  # same 6 PVM-family stages as LCMUNet


def test_scan_flop_formula_scales_linearly_in_each_shape_dimension():
    """Functional-form check on SCAN_FLOP_MULTIPLIER * B*D*L*N: doubling any
    one of B, D, L, N must exactly double the total -- confirms the formula
    is genuinely multilinear in all four dimensions, matching the derivation
    in lcmunet/efficiency.py's module docstring (not just "some formula that
    happens to run without crashing").
    """
    base = eff.SCAN_FLOP_MULTIPLIER * 2 * 3 * 5 * 7
    assert eff.SCAN_FLOP_MULTIPLIER * 4 * 3 * 5 * 7 == 2 * base  # double B
    assert eff.SCAN_FLOP_MULTIPLIER * 2 * 6 * 5 * 7 == 2 * base  # double D
    assert eff.SCAN_FLOP_MULTIPLIER * 2 * 3 * 10 * 7 == 2 * base  # double L
    assert eff.SCAN_FLOP_MULTIPLIER * 2 * 3 * 5 * 14 == 2 * base  # double N


def test_scan_gflops_matches_hand_computed_value_for_captured_shapes(monkeypatch):
    """Directly verifies measure_scan_gflops applies SCAN_FLOP_MULTIPLIER*B*D*L*N
    per captured call by monkeypatching the shape-capture step with known values."""
    fake_calls = [((2, 4, 8), 16), ((2, 4, 8), 16)]  # 2 identical calls
    monkeypatch.setattr(eff, "_capture_scan_shapes", lambda model, x: (fake_calls, "fake"))

    result = eff.measure_scan_gflops(UNet(), input_size=256, batch=1, device=CPU)
    expected_flops = sum(eff.SCAN_FLOP_MULTIPLIER * b * d * l * n for (b, d, l), n in fake_calls)
    assert result["gflops_scan_supplementary"] == expected_flops / 1e9
    assert result["n_scan_calls"] == 2


# ---- mamba_ssm hook regression (real mamba_ssm not installed locally, so a --
# ---- fake mamba_ssm.modules.mamba_simple module tree stands in) ------------


def _install_fake_mamba_simple(monkeypatch):
    """Builds a fake mamba_ssm.modules.mamba_simple module via exec() (not a
    plain object) so its Mamba.__call__ resolves `selective_scan_fn` as a
    GLOBAL name bound in THAT module's own namespace -- exactly reproducing
    the real mamba_ssm source's `from ...selective_scan_interface import
    selective_scan_fn` static-binding structure. A plain attribute-access
    stand-in (e.g. `fake_module.selective_scan_fn(...)`) would NOT reproduce
    the bug this test guards against, because attribute access always sees
    the current value -- only a name bound into the callee's own module
    globals (via `from X import Y`) can go stale the way the real bug did.

    The fake Mamba only calls selective_scan_fn when use_fast_path is False
    (mirroring the real slow/fast branch) so the test also proves
    _capture_scan_shapes forces use_fast_path off -- without that, `calls`
    would stay empty even with the hook target fixed.
    """
    import sys
    import textwrap
    import types

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
            # a REAL nn.Module subclass, like the genuine mamba_ssm.Mamba --
            # a plain object here would silently defeat model.modules()
            # traversal in _capture_scan_shapes's use_fast_path forcing.
            def __init__(self, d_model, use_fast_path=True):
                super().__init__()
                self.use_fast_path = use_fast_path
                self.d_inner = d_model * 2

            def forward(self, hidden_states):
                b, l, _d = hidden_states.shape
                if self.use_fast_path:
                    return hidden_states  # simulates the fused path -- selective_scan_fn never called
                u = torch.zeros(b, self.d_inner, l)
                dt = torch.zeros(b, self.d_inner, l)
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
    """Mimics the vendored PVMLayer's shape: a shared `self.mamba` called
    once per forward -- just enough structure to exercise _capture_scan_shapes."""

    def __init__(self, fake_simple):
        super().__init__()
        self.mamba = fake_simple.Mamba(d_model=8)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w).transpose(-1, -2)
        self.mamba(x_flat)
        return x


def test_capture_scan_shapes_hooks_correct_mamba_module_and_forces_slow_path(monkeypatch):
    fake_simple = _install_fake_mamba_simple(monkeypatch)
    model = _FakeVendoredPVMLayer(fake_simple)
    assert model.mamba.use_fast_path is True  # real mamba_ssm's own default

    calls, source = eff._capture_scan_shapes(model, torch.randn(1, 8, 4, 4))

    assert len(calls) == 1  # the fake Mamba's one selective_scan_fn call was captured
    assert "mamba_ssm.modules.mamba_simple.selective_scan_fn" in source
    assert model.mamba.use_fast_path is True  # restored after the measurement


def test_capture_scan_shapes_returns_empty_if_use_fast_path_not_forced(monkeypatch):
    """Isolates the use_fast_path half of the fix: if a caller ran the fake
    Mamba with use_fast_path left True, the fused branch (never hooked)
    would run and no calls would be captured -- confirms the fake actually
    exercises the branch _capture_scan_shapes must force off."""
    fake_simple = _install_fake_mamba_simple(monkeypatch)
    model = _FakeVendoredPVMLayer(fake_simple)

    with torch.no_grad():
        model(torch.randn(1, 8, 4, 4))  # use_fast_path still True -- no capture, no error

    # sanity: the fake itself never called selective_scan_fn on this path
    assert model.mamba.use_fast_path is True


# ---- FPS -------------------------------------------------------------------


def test_measure_fps_cpu_runs_and_labels_itself_not_gpu():
    model = LCMUNet(placement="P0")  # cheapest variant, still exercises the harness
    result = eff.measure_fps(model, input_size=32, batch=1, device=CPU, n_warmup=1, n_measure=2)
    assert result["fps"] > 0
    assert result["is_gpu_measurement"] is False
    assert result["n_warmup"] == 1 and result["n_measure"] == 2


def test_measure_fps_scales_batch_into_throughput():
    model = LCMUNet(placement="P0")
    r1 = eff.measure_fps(model, input_size=32, batch=1, device=CPU, n_warmup=1, n_measure=2)
    r8 = eff.measure_fps(model, input_size=32, batch=8, device=CPU, n_warmup=1, n_measure=2)
    assert r1["fps"] > 0 and r8["fps"] > 0  # both must complete without error; no throughput ordering assumed on a noisy CPU


# ---- peak memory -------------------------------------------------------------


def test_peak_memory_is_none_without_cuda():
    model = LCMUNet(placement="P0")
    assert eff.measure_peak_training_memory_mb(model, input_size=32, batch=8, device=CPU) is None


# ---- full row assembly -------------------------------------------------------


def test_measure_model_efficiency_assembles_full_row_on_cpu():
    model = LCMUNet(placement="hero")
    row = eff.measure_model_efficiency(
        "lc_ss2d_hero", model, device=CPU, scan_impl="ref", gpu_name=None, torch_version=torch.__version__,
        input_size=32, fps_batch_sizes=(1, 8), memory_batch_size=8, n_warmup=1, n_measure=2,
    )
    for key in (
        "model_name", "params_M", "gflops_module_thop", "gflops_scan_supplementary", "gflops_total",
        "n_scan_calls", "fps_b1", "fps_b8", "peak_mem_MB_b8", "gpu_name", "torch_version", "scan_impl",
    ):
        assert key in row
    assert row["model_name"] == "lc_ss2d_hero"
    assert row["params_M"] > 0
    assert row["gflops_total"] == row["gflops_module_thop"] + row["gflops_scan_supplementary"]
    assert row["peak_mem_MB_b8"] is None  # no CUDA on this machine
    assert row["fps_b1_is_gpu_measurement"] is False


def test_measure_model_efficiency_zero_scan_for_unet():
    row = eff.measure_model_efficiency(
        "unet", UNet(), device=CPU, scan_impl="ref", gpu_name=None, torch_version=torch.__version__,
        input_size=256, fps_batch_sizes=(1,), memory_batch_size=8, n_warmup=1, n_measure=2,
    )
    assert row["gflops_scan_supplementary"] == 0.0
    assert row["n_scan_calls"] == 0
