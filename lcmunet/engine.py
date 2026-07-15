"""Resumable training/eval engine (methodology sections 6 and 8).

Colab sessions die without warning (GLOBAL RULES). Every run checkpoints
every epoch and resumes seamlessly (model/optimizer/scheduler/scaler/RNG
state all restored) from wherever it was killed -- run_one() is safe to call
repeatedly on the same config until it finishes.

MODEL AVAILABILITY (methodology section 9):
  - "ultralight_baseline": the vendored, unmodified UltraLight VM-UNet
    (Gate 1). Requires mamba-ssm's CUDA build (GLOBAL RULES rule 5) -- its
    forward() cannot run at all without it (no CPU fallback exists inside
    mamba_ssm.Mamba; see third_party/UltraLight-VM-UNet/VENDORED.md).
    build_model() checks lcmunet.scan.SCAN_IMPL == "cuda" before attempting
    to build it and raises a clear error otherwise, rather than a confusing
    import failure deep inside mamba_ssm.
  - "unet", "malunet", "egeunet": comparators (section 9), all CPU-runnable.
  - "lc_ss2d": LCM-UNet with LC-VSS (methodology sections 3-5), CPU-runnable
    (lcmunet/lc_vss.py reimplements Mamba's non-fused forward directly, no
    mamba-ssm dependency). All ablations are config toggles on
    RunConfig.model_cfg (descriptor_type/kernel_size/inject_target/
    placement/use_e6/alpha_init/wdelta_std), not separate model_names.
  - "glgf": GLGF late-fusion baseline/ablation variant (methodology sections
    1, 3.5, 10.1 Ablation A; CONTEXT.md) -- the project's prior design
    iteration, kept only to prove LC-SS2D beats late feature fusion.
    CPU-runnable (lcmunet/glgf.py reuses lcmunet.lc_vss.LC_PVMLayer with
    descriptor_type='none' for its Mamba branch, same as lc_ss2d). No config
    toggles -- it is a single fixed row, not itself ablated.
  - "engine_test_tiny": test scaffolding only (lcmunet.testing_models.
    EngineSanityNet), never a candidate architecture.

Every vendored model (ultralight_baseline/malunet/egeunet) hard-codes a
final sigmoid in forward() and is wrapped in lcmunet.adapters.LogitsAdapter
so all models present a uniform logits interface to CombinedLoss/evaluate().
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from lcmunet.config import RunConfig
from lcmunet.data.loaders import build_dataloaders
from lcmunet.logging_utils import get_run_logger
from lcmunet.losses import CombinedLoss
from lcmunet.metrics import evaluate, save_per_image_dice
from lcmunet.results_store import upsert_result
from lcmunet.scan import SCAN_IMPL
from lcmunet.seed import set_seed

ALPHA_LOG_EVERY = 10
ALPHA_COLLAPSE_EPOCH = 100
ALPHA_COLLAPSE_THRESHOLD = 0.02


# ---- model construction (see module docstring: real models don't exist yet) --


def build_model(config: RunConfig) -> torch.nn.Module:
    if config.model_name == "engine_test_tiny":
        from lcmunet.testing_models import EngineSanityNet

        return EngineSanityNet()

    if config.model_name == "ultralight_baseline":
        if SCAN_IMPL != "cuda":
            raise RuntimeError(
                "model_name='ultralight_baseline' requires lcmunet.scan.SCAN_IMPL "
                f"== 'cuda', got {SCAN_IMPL!r}. The vendored UltraLight_VM_UNet's "
                "PVMLayer uses mamba_ssm.Mamba directly, which has NO CPU/ref "
                "fallback (both its fused and non-fused internal paths require "
                "the compiled selective_scan_cuda extension) -- so this baseline "
                "cannot run at all without a working mamba-ssm CUDA build. This "
                "is a GLOBAL RULES rule-5 GPU gate: run notebooks/01_env.ipynb "
                "in Colab and confirm SCAN_IMPL=='cuda' before Gate 1."
            )
        from lcmunet.adapters import LogitsAdapter
        from lcmunet.backbone import load_ultralight_vmunet

        return LogitsAdapter(load_ultralight_vmunet())

    if config.model_name == "unet":
        from lcmunet.unet import UNet

        return UNet()

    if config.model_name == "malunet":
        from lcmunet.adapters import LogitsAdapter
        from lcmunet.comparators import load_malunet

        return LogitsAdapter(load_malunet())

    if config.model_name == "egeunet":
        from lcmunet.adapters import LogitsAdapter
        from lcmunet.comparators import load_egeunet

        return LogitsAdapter(load_egeunet())

    if config.model_name == "lc_ss2d":
        from lcmunet.lcm_unet import LCMUNet

        mc = config.model_cfg
        return LCMUNet(
            descriptor_type=mc["descriptor_type"],
            kernel_size=mc["kernel_size"],
            inject_target=mc["inject_target"],
            placement=mc["placement"],
            use_e6=mc["use_e6"],
            alpha_init=mc["alpha_init"],
            wdelta_std=mc["wdelta_std"],
        )

    if config.model_name == "glgf":
        from lcmunet.glgf import GLGFUNet

        return GLGFUNet()

    raise ValueError(f"unknown model_name: {config.model_name!r}")


# ---- RNG state (python/numpy/torch CPU+CUDA) --------------------------------


def _get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---- checkpointing -----------------------------------------------------


def checkpoint_dir(paths, config: RunConfig) -> Path:
    return Path(paths.checkpoints) / f"{config.config_id}_{config.seed}"


def _save_checkpoint(ckpt_dir: Path, state: dict, is_best: bool) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / "last.pt.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_dir / "last.pt")  # atomic-ish: never leaves a half-written last.pt
    if is_best:
        tmp_best = ckpt_dir / "best.pt.tmp"
        torch.save(state, tmp_best)
        tmp_best.replace(ckpt_dir / "best.pt")


def _load_checkpoint(ckpt_dir: Path) -> Optional[dict]:
    path = ckpt_dir / "last.pt"
    if not path.is_file():
        return None
    # weights_only=False: our checkpoints intentionally carry non-tensor
    # state (python/numpy RNG state, plain ints/floats) alongside tensors,
    # and are always files this same codebase wrote (never a third-party
    # checkpoint), so the arbitrary-unpickling risk weights_only=True guards
    # against does not apply here.
    return torch.load(path, map_location="cpu", weights_only=False)


# ---- alpha logging (methodology section 6) -----------------------------


def collect_alphas(model: torch.nn.Module) -> Dict[str, float]:
    """Find every learnable scalar named `alpha` (LC-VSS blocks, methodology
    section 3.3), keyed by stage name. Prefers the model's own
    alpha_by_stage() when available (e.g. lcmunet.lcm_unet.LCMUNet ->
    {"E4":..., "E5":..., "D5":..., "D4":...}); otherwise falls back to
    generic named_modules() introspection (module path names) so this still
    works for any model exposing an `.alpha` nn.Parameter, not just LCMUNet.
    Empty for any model that has none (e.g. a comparator baseline).
    """
    alpha_by_stage = getattr(model, "alpha_by_stage", None)
    if callable(alpha_by_stage):
        return alpha_by_stage()

    alphas = {}
    for name, module in model.named_modules():
        alpha = getattr(module, "alpha", None)
        if isinstance(alpha, torch.nn.Parameter):
            alphas[name or "root"] = float(alpha.detach().cpu().item())
    return alphas


def _alpha_csv_path(paths, config: RunConfig) -> Path:
    return Path(paths.results) / "alpha" / f"{config.config_id}_{config.seed}.csv"


def _log_alphas(path: Path, epoch_1indexed: int, alphas: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(alphas.keys())
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["epoch"] + keys)
        writer.writerow([epoch_1indexed] + [alphas[k] for k in keys])


# ---- one epoch of training -----------------------------------------------


def _train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp_enabled, grad_accum_steps) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pending_grad = False

    for step, (images, masks, _ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks) / grad_accum_steps

        scaler.scale(loss).backward()
        pending_grad = True
        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_grad = False

    if pending_grad:  # flush a partial accumulation window at epoch end
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1)


# ---- full run, resumable -------------------------------------------------


def run_one(
    config_or_job: Union[RunConfig, Dict[str, Any]],
    paths=None,
    stop_after_epoch: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Train (or resume) one run to completion; writes checkpoints every
    epoch and a final results row + per-image test Dice.

    Accepts either a RunConfig directly, or a run_manifest job dict
    ({"config_yaml_path": ...}) -- so this function can be passed straight
    to lcmunet.run_manifest.run_queue as runner_fn.

    stop_after_epoch (0-indexed): TEST-ONLY hook that returns right after
    that epoch's checkpoint is written, simulating a killed Colab session,
    so resumability can be verified by calling run_one() again on the same
    config without needing a real process kill.

    num_workers: forwarded to build_dataloaders (defaults to loaders.py's
    own default, NUM_WORKERS=2, if None). IMPORTANT: multi-worker DataLoader
    shuffling is only reproducible up to OS-level process-scheduling
    nondeterminism -- a universal PyTorch limitation, not specific to this
    engine. "Deterministic after a kill" here means resuming restores
    model/optimizer/scheduler/scaler/RNG state exactly and continues from
    there; bit-exact end-to-end reproduction of an uninterrupted run is only
    guaranteed with num_workers=0 (which is what the resume test below
    uses to verify the restore mechanism itself).
    """
    if paths is None:
        from lcmunet.paths import get_paths

        paths = get_paths()

    config = config_or_job if isinstance(config_or_job, RunConfig) else RunConfig.load_yaml(config_or_job["config_yaml_path"])

    logger = get_run_logger(config.config_id, paths.logs)
    ckpt_dir = checkpoint_dir(paths, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.amp) and device.type == "cuda"

    ckpt = _load_checkpoint(ckpt_dir)
    if ckpt is None:
        # Must precede model construction: nn.Module weight init consumes
        # the global torch RNG, so seeding after building the model would
        # leave initial weights un-seeded (and every cold start would get a
        # different random init instead of a reproducible one).
        set_seed(config.seed)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    criterion = CombinedLoss()

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dsc = ckpt["best_val_dsc"]
        logger.info(f"Resumed {config.config_id} seed={config.seed} from epoch {ckpt['epoch']} -> continuing at epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        best_val_dsc = -1.0
        logger.info(f"Starting {config.config_id} seed={config.seed} from scratch ({config.epochs} epochs)")

    dataloader_kwargs = {} if num_workers is None else {"num_workers": num_workers}
    train_loader, val_loader, test_loader = build_dataloaders(config, paths, **dataloader_kwargs)

    reached_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs):
        train_loader.dataset.set_epoch(epoch)  # augmentation RNG is a pure function of (seed, epoch, item) -- see SegmentationDataset
        train_loss = _train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, amp_enabled, config.grad_accum_steps)
        val_metrics = evaluate(model, val_loader, device, boundary=False)  # cheap: skip HD95/ASSD every epoch
        scheduler.step()

        is_best = val_metrics["dsc"] > best_val_dsc
        if is_best:
            best_val_dsc = val_metrics["dsc"]

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_dsc": best_val_dsc,
            "rng_state": _get_rng_state(),
        }
        _save_checkpoint(ckpt_dir, state, is_best=is_best)
        reached_epoch = epoch

        epoch_1indexed = epoch + 1
        alphas = collect_alphas(model)
        if alphas and epoch_1indexed % ALPHA_LOG_EVERY == 0:
            _log_alphas(_alpha_csv_path(paths, config), epoch_1indexed, alphas)
        if alphas and epoch_1indexed == ALPHA_COLLAPSE_EPOCH and all(v < ALPHA_COLLAPSE_THRESHOLD for v in alphas.values()):
            logger.warning(
                f"Potential mechanism collapse: all LC-VSS alpha < {ALPHA_COLLAPSE_THRESHOLD} "
                f"at epoch {ALPHA_COLLAPSE_EPOCH} ({alphas}). Continuing training (not stopping)."
            )

        logger.info(
            f"epoch {epoch_1indexed}/{config.epochs} train_loss={train_loss:.4f} "
            f"val_dsc={val_metrics['dsc']:.4f} best_val_dsc={best_val_dsc:.4f}"
        )

        if stop_after_epoch is not None and epoch == stop_after_epoch:
            logger.info(f"stop_after_epoch={stop_after_epoch} reached (test hook) -- returning without final test eval.")
            return {
                "config_id": config.config_id,
                "seed": config.seed,
                "completed": False,
                "reached_epoch": reached_epoch,
                "best_val_dsc": best_val_dsc,
            }

    test_metrics = evaluate(model, test_loader, device, boundary=True)
    per_image_path = save_per_image_dice(config.config_id, config.seed, test_metrics["ids"], test_metrics["per_image"]["dsc"], paths.results)

    row = {
        "config_id": config.config_id,
        "model_name": config.model_name,
        "dataset": config.dataset,
        "seed": config.seed,
        "split_file": config.split_file,
        "dsc": test_metrics["dsc"],
        "miou": test_metrics["miou"],
        "sensitivity": test_metrics["sensitivity"],
        "specificity": test_metrics["specificity"],
        "accuracy": test_metrics["accuracy"],
        "hd95": test_metrics["hd95"],
        "assd": test_metrics["assd"],
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scan_impl": SCAN_IMPL,
        "notes": "engine.run_one final test evaluation",
    }
    upsert_result(paths.results, row)
    logger.info(f"Run complete: test_dsc={test_metrics['dsc']:.4f} per-image Dice saved to {per_image_path}")

    return {
        "config_id": config.config_id,
        "seed": config.seed,
        "completed": True,
        "reached_epoch": reached_epoch,
        "best_val_dsc": best_val_dsc,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("per_image",)},
    }
