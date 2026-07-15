"""Full-pipeline smoke test using purely synthetic in-memory data.

No dataset, no Drive mount, no download: 20 fake images/masks are generated
directly with torch.rand/torch.randint and fed through the real project
code (lcmunet.lcm_unet.LCMUNet, lcmunet.losses.CombinedLoss,
lcmunet.engine's checkpoint machinery, ...). This proves the pipeline's
plumbing -- forward shapes, the LC-VSS gradient path, the token-order/
delta-differs claims, the training loop, and checkpoint save/resume -- is
wired correctly BEFORE spending any real compute/time on a real dataset
run. It is not a substitute for the real Gate-1/Gate-2/Phase-2 runs.

CHECK 1 (baseline forward) needs mamba-ssm's CUDA build (GLOBAL RULES rule
5) to construct the REAL vendored UltraLight VM-UNet, exactly like
lcmunet.engine.build_model('ultralight_baseline') does. If that build is
not available (e.g. a CPU-only runtime), CHECK 1 is SKIPPED -- not faked,
not silently passed -- since that model has no CPU fallback at all (see
lcmunet/backbone.py).

CHECKS 4/5 (token order unchanged / dts differs) do not need the real
vendored baseline: they reuse lcmunet.delta_diff's established "alpha=0
proxy" (methodology section 11.1) -- at alpha=0 the SAME LC-SS2D weights
reduce exactly to the stock-PVM computation, so comparing alpha=0 vs
alpha=real on the same captured stage input IS the baseline-vs-LC-SS2D
comparison, without a second model or mamba-ssm.

Usage:
    python scripts/dummy_run.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/dummy_run.py` works from any cwd

import lcmunet.lc_vss as lc_vss_module  # noqa: E402
from lcmunet.config import DEFAULT_MODEL_CFG  # noqa: E402
from lcmunet.delta_diff import _capture_stage_inputs, _stage_modules, delta_difference_report  # noqa: E402
from lcmunet.engine import _load_checkpoint, _save_checkpoint  # noqa: E402
from lcmunet.lcm_unet import LCMUNet  # noqa: E402
from lcmunet.losses import CombinedLoss  # noqa: E402
from lcmunet.scan import SCAN_IMPL  # noqa: E402
from lcmunet.seed import set_seed  # noqa: E402

N_SAMPLES = 20
IMG_SIZE = 256
BATCH_SIZE = 4
CHECK_BATCH = 4  # samples used per single-batch check (1-3)
SEED = 0
TRAIN_EPOCHS = 3
TRAIN_LR = 3e-3


class CheckSkipped(Exception):
    """Raised by a check to report SKIP (not PASS, not FAIL) -- e.g. a real
    GPU-only dependency (mamba-ssm) is unavailable on this runtime."""


# ---- dummy data -------------------------------------------------------------


def make_dummy_data(n: int = N_SAMPLES, size: int = IMG_SIZE, seed: int = SEED) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, size, size, generator=g)
    masks = torch.randint(0, 2, (n, 1, size, size), generator=g).float()
    return images, masks


def make_dummy_loader(images: torch.Tensor, masks: torch.Tensor, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(TensorDataset(images, masks), batch_size=batch_size, shuffle=True)


def _hero_model() -> LCMUNet:
    """The hero LC-SS2D config (methodology sections 3-5) -- identical
    construction to lcmunet.engine.build_model(model_name='lc_ss2d') with
    the default config."""
    mc = DEFAULT_MODEL_CFG
    return LCMUNet(
        descriptor_type=mc["descriptor_type"],
        kernel_size=mc["kernel_size"],
        inject_target=mc["inject_target"],
        placement=mc["placement"],
        use_e6=mc["use_e6"],
        alpha_init=mc["alpha_init"],
        wdelta_std=mc["wdelta_std"],
    )


# ---- CHECK 1: baseline forward pass -----------------------------------------


def check_baseline_forward(images: torch.Tensor) -> str:
    if SCAN_IMPL != "cuda":
        raise CheckSkipped(
            f"SCAN_IMPL={SCAN_IMPL!r} (mamba-ssm CUDA build not importable / no CUDA "
            "device on this runtime) -- the real vendored UltraLight VM-UNet has NO "
            "CPU fallback (its PVMLayer calls mamba_ssm.Mamba directly; see "
            "lcmunet/backbone.py and lcmunet/engine.py build_model()). Expected on a "
            "CPU-only runtime -- re-run on a Colab GPU runtime with mamba-ssm "
            "installed (notebooks/01_env.ipynb) to exercise this check."
        )
    from lcmunet.adapters import LogitsAdapter
    from lcmunet.backbone import load_ultralight_vmunet

    device = torch.device("cuda")
    model = LogitsAdapter(load_ultralight_vmunet()).to(device)
    model.eval()
    x = images[:CHECK_BATCH].to(device)
    with torch.no_grad():
        out = model(x)

    expected_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    if tuple(out.shape) != expected_shape:
        raise AssertionError(f"output shape {tuple(out.shape)} != expected {expected_shape}")
    if not torch.isfinite(out).all():
        raise AssertionError("NaN/Inf in baseline (UltraLight VM-UNet) output")
    return f"real vendored UltraLight VM-UNet forward OK on {device}, output shape {tuple(out.shape)}, all finite"


# ---- CHECK 2: LC-SS2D forward pass ------------------------------------------


def check_hero_forward(hero_model: LCMUNet, images: torch.Tensor) -> str:
    hero_model.eval()
    x = images[:CHECK_BATCH]
    with torch.no_grad():
        out = hero_model(x)

    expected_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    if tuple(out.shape) != expected_shape:
        raise AssertionError(f"output shape {tuple(out.shape)} != expected {expected_shape}")
    if not torch.isfinite(out).all():
        raise AssertionError("NaN/Inf in LC-SS2D output")
    return f"LC-SS2D (hero) forward OK, output shape {tuple(out.shape)}, all finite"


# ---- CHECK 3: gradient flow (mechanism alive) -------------------------------


def check_gradient_flow(hero_model: LCMUNet, images: torch.Tensor, masks: torch.Tensor) -> str:
    hero_model.train()
    hero_model.zero_grad(set_to_none=True)
    x = images[:CHECK_BATCH]
    y = masks[:CHECK_BATCH]

    logits = hero_model(x)
    loss = CombinedLoss()(logits, y)
    loss.backward()

    stage_modules = _stage_modules(hero_model)
    if not stage_modules:
        raise RuntimeError("hero model has no active LC-VSS stage (alpha is None everywhere)")

    checked: List[str] = []
    for name, mod in stage_modules.items():
        if mod.alpha.grad is None:
            raise AssertionError(f"{name}: alpha.grad is None -- no gradient reached alpha")
        if float(mod.alpha.grad.abs().item()) == 0.0:
            raise AssertionError(f"{name}: alpha.grad is exactly zero")

        for i, w_delta in enumerate(mod.w_deltas):
            grad = w_delta.weight.grad
            if grad is None:
                raise AssertionError(f"{name} group {i}: W_delta.weight.grad is None")
            if not torch.isfinite(grad).all():
                raise AssertionError(f"{name} group {i}: W_delta.weight.grad has non-finite values")
            if torch.count_nonzero(grad).item() == 0:
                raise AssertionError(f"{name} group {i}: W_delta.weight.grad is all-zero")
            checked.append(f"{name}.w_deltas[{i}]")

    return f"gradient flowed to alpha and W_delta in all {len(stage_modules)} active stages ({len(checked)} W_delta tensors verified non-zero/finite)"


# ---- CHECKS 4/5: token order unchanged / dts differs ------------------------
# Both reuse lcmunet.delta_diff's "alpha=0 proxy": at alpha=0 the SAME
# trained weights reduce exactly to the stock-PVM computation (see
# lcmunet/delta_diff.py module docstring), so this IS the baseline-vs-LC-SS2D
# comparison without a second model.


def _capture_u(stage_module, stage_input: torch.Tensor) -> List[torch.Tensor]:
    """Mirrors lcmunet.delta_diff._capture_dts's hook pattern, but records
    `u` (the token sequence going into selective_scan) instead of `dts`."""
    real_scan = lc_vss_module.selective_scan
    calls: List[torch.Tensor] = []

    def spy(u, dts, *args, **kwargs):
        calls.append(u.detach().clone())
        return real_scan(u, dts, *args, **kwargs)

    lc_vss_module.selective_scan = spy
    try:
        with torch.no_grad():
            stage_module(stage_input)
    finally:
        lc_vss_module.selective_scan = real_scan
    return calls


def check_token_order_unchanged(hero_model: LCMUNet, images: torch.Tensor) -> str:
    hero_model.eval()
    stage_modules = _stage_modules(hero_model)
    if not stage_modules:
        raise RuntimeError("hero model has no active LC-VSS stage -- cannot check token order")

    x = images[:CHECK_BATCH]
    stage_inputs = _capture_stage_inputs(hero_model, x, stage_modules)

    checked: List[str] = []
    for name, mod in stage_modules.items():
        xin = stage_inputs[name]
        u_hero = _capture_u(mod, xin)

        original_alpha = mod.alpha.data.clone()
        mod.alpha.data.zero_()
        try:
            u_baseline = _capture_u(mod, xin)
        finally:
            mod.alpha.data.copy_(original_alpha)

        if len(u_hero) != len(u_baseline):
            raise RuntimeError(f"{name}: scan call count mismatch ({len(u_hero)} vs {len(u_baseline)})")
        for i, (h, b) in enumerate(zip(u_hero, u_baseline)):
            if not torch.equal(h, b):
                raise AssertionError(
                    f"{name} group {i}: u (scan input) differs between alpha=0 "
                    "(baseline-equivalent) and alpha=real (LC-SS2D) -- token order is NOT unchanged"
                )
        checked.append(name)

    return f"u (token sequence into selective_scan) is bit-identical between baseline-equivalent (alpha=0) and LC-SS2D in all {len(checked)} active stages: {checked}"


def check_dts_differs(hero_model: LCMUNet, images: torch.Tensor) -> str:
    x = images[:CHECK_BATCH]
    report = delta_difference_report(hero_model, x)
    if not report:
        raise RuntimeError("delta_difference_report returned no stages")

    non_constant = {name: stats["non_constant"] for name, stats in report.items()}
    if not all(non_constant.values()):
        raise AssertionError(f"dts did NOT differ (constant/zero) in some stage(s): {non_constant}")

    means = {name: round(stats["mean_abs_diff"], 6) for name, stats in report.items()}
    return f"dts(LC-SS2D) != dts(baseline-equivalent, alpha=0) in all {len(report)} stages (mean|diff|: {means})"


# ---- CHECK 6: loss computes -------------------------------------------------


def check_loss_computes() -> str:
    torch.manual_seed(SEED)
    logits = torch.randn(BATCH_SIZE, 1, 64, 64)
    targets = torch.randint(0, 2, (BATCH_SIZE, 1, 64, 64)).float()
    loss = CombinedLoss()(logits, targets)
    if not torch.isfinite(loss):
        raise AssertionError(f"CombinedLoss produced a non-finite value: {loss.item()}")
    return f"CombinedLoss(BCE+Dice) = {loss.item():.4f} (finite)"


# ---- CHECKS 7/8: mini-train + checkpoint save/resume ------------------------
# Share one training run (a checkpoint captured after epoch 2 is meaningless
# on its own): check 7 asserts on the loss/alpha trajectory, check 8 asserts
# on the epoch-2 checkpoint's save/reload round-trip.


def _run_mini_train_and_checkpoint(images: torch.Tensor, masks: torch.Tensor) -> Dict[str, Any]:
    set_seed(SEED)  # must precede model construction -- see lcmunet/engine.py run_one
    model = _hero_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    criterion = CombinedLoss()
    loader = make_dummy_loader(images, masks)

    alpha_before = dict(model.alpha_by_stage())
    epoch_losses: List[float] = []
    ckpt_state: Dict[str, Any] = None  # type: ignore[assignment]

    for epoch in range(TRAIN_EPOCHS):  # 0-indexed; epoch==1 is "after epoch 2"
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        epoch_losses.append(total / max(n, 1))

        if epoch == 1:
            ckpt_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

    return {
        "epoch_losses": epoch_losses,
        "alpha_before": alpha_before,
        "alpha_after": dict(model.alpha_by_stage()),
        "ckpt_state": ckpt_state,
    }


def check_mini_train(train_result: Dict[str, Any]) -> str:
    losses = train_result["epoch_losses"]
    if not losses[-1] < losses[0]:
        raise AssertionError(f"loss did not decrease: epoch1={losses[0]:.4f} -> epoch{len(losses)}={losses[-1]:.4f} ({losses})")

    alpha_before, alpha_after = train_result["alpha_before"], train_result["alpha_after"]
    if not any(alpha_before[k] != alpha_after[k] for k in alpha_before):
        raise AssertionError(f"alpha did not change during training: before={alpha_before} after={alpha_after}")

    return f"loss decreased {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs; alpha changed ({alpha_before} -> {alpha_after})"


def check_checkpoint_save_resume(train_result: Dict[str, Any]) -> str:
    ckpt_state = train_result["ckpt_state"]
    if ckpt_state is None:
        raise RuntimeError("no checkpoint was captured after epoch 2 (mini-train check must run first)")

    with tempfile.TemporaryDirectory(prefix="lcmunet_dummy_run_") as tmp:
        ckpt_dir = Path(tmp) / "ckpt"
        _save_checkpoint(ckpt_dir, ckpt_state, is_best=False)
        reloaded = _load_checkpoint(ckpt_dir)

    if reloaded is None:
        raise RuntimeError("checkpoint failed to reload from disk")
    if reloaded["epoch"] != 1:
        raise AssertionError(f"epoch counter mismatch: expected 1 (0-indexed 'epoch 2'), got {reloaded['epoch']}")

    for key, orig_tensor in ckpt_state["model"].items():
        reloaded_tensor = reloaded["model"][key]
        if not torch.equal(orig_tensor, reloaded_tensor):
            raise AssertionError(f"weight mismatch after reload: {key}")

    return f"checkpoint saved after epoch 2 and reloaded from disk: epoch={reloaded['epoch']}, {len(ckpt_state['model'])} tensors bit-identical"


# ---- runner ------------------------------------------------------------------


def run_check(fn: Callable[[], str]) -> Tuple[str, str]:
    try:
        return "PASS", fn()
    except CheckSkipped as exc:
        return "SKIP", str(exc)
    except Exception:
        return "FAIL", traceback.format_exc()


def main() -> int:
    print("=" * 78)
    print("LCM-UNet dummy pipeline smoke test (scripts/dummy_run.py)")
    print(f"SCAN_IMPL={SCAN_IMPL!r}  cuda_available={torch.cuda.is_available()}")
    print("=" * 78)

    images, masks = make_dummy_data()
    print(
        f"\nGenerated {images.shape[0]} dummy images {tuple(images.shape[1:])} and "
        f"masks {tuple(masks.shape[1:])} in memory (no file I/O, no Drive, no download)."
    )

    hero_holder: Dict[str, LCMUNet] = {}

    def get_hero_model() -> LCMUNet:
        if "model" not in hero_holder:
            set_seed(SEED)
            hero_holder["model"] = _hero_model()
        return hero_holder["model"]

    train_holder: Dict[str, Any] = {}

    def get_train_result() -> Dict[str, Any]:
        if "result" not in train_holder:
            if "error" in train_holder:
                raise RuntimeError(f"mini-train prerequisite failed: {train_holder['error']}") from None
            try:
                train_holder["result"] = _run_mini_train_and_checkpoint(images, masks)
            except Exception as exc:
                train_holder["error"] = repr(exc)
                raise
        return train_holder["result"]

    checks: List[Tuple[str, Callable[[], str]]] = [
        ("1. Baseline (UltraLight VM-UNet) forward pass", lambda: check_baseline_forward(images)),
        ("2. LC-SS2D (hero) forward pass", lambda: check_hero_forward(get_hero_model(), images)),
        ("3. Gradient flow through LC-VSS mechanism", lambda: check_gradient_flow(get_hero_model(), images, masks)),
        ("4. Token order unchanged (u bit-identical)", lambda: check_token_order_unchanged(get_hero_model(), images)),
        ("5. dts differs (mechanism modifies delta)", lambda: check_dts_differs(get_hero_model(), images)),
        ("6. Loss computes (BCE + Dice)", check_loss_computes),
        ("7. 3-epoch mini-train (loss decreases, alpha changes)", lambda: check_mini_train(get_train_result())),
        ("8. Checkpoint save + resume", lambda: check_checkpoint_save_resume(get_train_result())),
    ]

    results: List[Tuple[str, str, str]] = []
    for name, fn in checks:
        print(f"\n--- {name} ---")
        status, detail = run_check(fn)
        results.append((name, status, detail))
        print(f"{status}: {detail}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print("\n" + "=" * 78)
    summary = f"SUMMARY: {passed}/{len(checks)} checks passed"
    if skipped:
        summary += f", {skipped} skipped"
    if failed:
        summary += f", {failed} FAILED"
    print(summary)
    print("=" * 78)
    marker = {"PASS": "[PASS]", "SKIP": "[SKIP]", "FAIL": "[FAIL]"}
    for name, status, _detail in results:
        print(f"{marker[status]} {name}")

    if failed:
        print("\nFailed check details:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"\n--- {name} ---\n{detail}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
