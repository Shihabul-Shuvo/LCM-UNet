import uuid

import numpy as np
import pandas as pd
import torch

from lcmunet.config import RunConfig
from lcmunet.data.splits import build_kvasir_split
from lcmunet.metrics import load_per_image_dice
import lcmunet.engine as engine_mod
from lcmunet.engine import checkpoint_dir, run_one


def _tiny_config(**overrides):
    # run_name is randomised so every call gets a distinct config_id, even
    # across different test functions with otherwise-identical settings --
    # logging.getLogger() caches loggers by name (== config_id) for the
    # whole pytest process, so a colliding config_id would make a later
    # test's get_run_logger() call silently return an earlier test's
    # (already torn-down) log file handle instead of a fresh one.
    defaults = dict(
        run_name=f"engine_test_{uuid.uuid4().hex[:8]}",
        model_name="engine_test_tiny",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
        epochs=2,
        batch_size=4,
        input_size=32,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_run_one_end_to_end_writes_all_artifacts(make_kvasir_raw):
    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=2)

    result = run_one(config, paths=paths, num_workers=0)

    assert result["completed"] is True
    assert result["reached_epoch"] == 1  # 0-indexed, 2 epochs -> last index 1

    ckpt_dir = checkpoint_dir(paths, config)
    assert (ckpt_dir / "last.pt").is_file()
    assert (ckpt_dir / "best.pt").is_file()

    results_csv = paths.results / "results.csv"
    assert results_csv.is_file()
    df = pd.read_csv(results_csv)
    row = df[(df["config_id"] == config.config_id) & (df["seed"] == config.seed)]
    assert len(row) == 1
    assert row.iloc[0]["model_name"] == "engine_test_tiny"
    assert row.iloc[0]["dataset"] == "kvasir_seg"

    per_image = load_per_image_dice(config.config_id, config.seed, paths.results)
    assert len(per_image["ids"]) > 0
    assert len(per_image["ids"]) == len(per_image["dsc"])


def test_run_one_is_idempotent_once_complete(make_kvasir_raw):
    """Calling run_one again after completion must not error or duplicate the results row."""
    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=2)

    run_one(config, paths=paths, num_workers=0)
    run_one(config, paths=paths, num_workers=0)  # should just re-run training loop range() as empty and re-evaluate

    df = pd.read_csv(paths.results / "results.csv")
    row = df[(df["config_id"] == config.config_id) & (df["seed"] == config.seed)]
    assert len(row) == 1  # upsert_result must not duplicate


def test_stop_after_epoch_hook_returns_without_final_eval(make_kvasir_raw):
    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=5)

    result = run_one(config, paths=paths, stop_after_epoch=1, num_workers=0)

    assert result["completed"] is False
    assert result["reached_epoch"] == 1
    assert not (paths.results / "results.csv").exists()  # no final eval yet


def test_resume_reproduces_uninterrupted_run_bit_exact(make_kvasir_raw, tmp_path):
    """The core resumability requirement: killing after epoch 2 (0-indexed)
    and resuming must reproduce exactly what an uninterrupted 5-epoch run
    would have produced -- model weights and final test metrics both.
    num_workers=0 is required for this to be verifiable: multi-worker
    DataLoader shuffling has OS-scheduling nondeterminism independent of
    this engine (see engine.run_one's docstring).
    """
    from lcmunet.paths import get_paths

    def make_data(root_dir):
        p = get_paths(root=root_dir)
        # reuse the exact same synthetic generation as conftest.make_kvasir_raw,
        # but on two SEPARATE roots so run A and run B don't share state.
        from PIL import Image

        root = p.data_raw / "Kvasir-SEG"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(20):
            img = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[10:30, 10:30] = 255
            Image.fromarray(img).save(root / "images" / f"img{i:04d}.jpg")
            Image.fromarray(mask).save(root / "masks" / f"img{i:04d}.jpg")
        build_kvasir_split(p)
        return p

    paths_a = make_data(tmp_path / "a")
    config_a = _tiny_config(epochs=5)
    result_a = run_one(config_a, paths=paths_a, num_workers=0)
    ckpt_a = torch.load(checkpoint_dir(paths_a, config_a) / "last.pt", weights_only=False)

    paths_b = make_data(tmp_path / "b")
    config_b = _tiny_config(epochs=5)
    partial = run_one(config_b, paths=paths_b, stop_after_epoch=2, num_workers=0)
    assert partial["completed"] is False
    assert partial["reached_epoch"] == 2

    result_b = run_one(config_b, paths=paths_b, num_workers=0)  # resume, no stop hook -> runs to completion
    ckpt_b = torch.load(checkpoint_dir(paths_b, config_b) / "last.pt", weights_only=False)

    assert result_b["reached_epoch"] == result_a["reached_epoch"] == 4

    for key in ckpt_a["model"]:
        assert torch.equal(ckpt_a["model"][key], ckpt_b["model"][key]), f"mismatch in {key}"

    assert result_a["test_metrics"]["dsc"] == result_b["test_metrics"]["dsc"]


def test_alpha_csv_written_every_10_epochs(make_kvasir_raw, monkeypatch):
    monkeypatch.setattr(engine_mod, "ALPHA_LOG_EVERY", 2)
    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=2)

    run_one(config, paths=paths, num_workers=0)

    alpha_csv = paths.results / "alpha" / f"{config.config_id}_{config.seed}.csv"
    assert alpha_csv.is_file()
    df = pd.read_csv(alpha_csv)
    assert list(df["epoch"]) == [2]
    assert "alpha_block" in df.columns


def test_no_alpha_csv_when_model_has_no_alpha_block(make_kvasir_raw, monkeypatch):
    monkeypatch.setattr(engine_mod, "ALPHA_LOG_EVERY", 1)

    def build_no_alpha_model(config):
        from lcmunet.testing_models import EngineSanityNet

        return EngineSanityNet(with_alpha_block=False)

    monkeypatch.setattr(engine_mod, "build_model", build_no_alpha_model)

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=2)
    run_one(config, paths=paths, num_workers=0)

    alpha_csv = paths.results / "alpha" / f"{config.config_id}_{config.seed}.csv"
    assert not alpha_csv.exists()


def test_mechanism_collapse_warning_logged(make_kvasir_raw, monkeypatch):
    monkeypatch.setattr(engine_mod, "ALPHA_COLLAPSE_EPOCH", 2)
    monkeypatch.setattr(engine_mod, "collect_alphas", lambda model: {"fake_stage": 0.001})

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)
    config = _tiny_config(epochs=2)
    run_one(config, paths=paths, num_workers=0)

    log_path = paths.logs / f"{config.config_id}.log"
    content = log_path.read_text(encoding="utf-8")
    assert "Potential mechanism collapse" in content


def test_build_model_ultralight_baseline_requires_scan_impl_cuda(monkeypatch):
    import pytest

    import lcmunet.engine as engine_mod

    monkeypatch.setattr(engine_mod, "SCAN_IMPL", "ref")
    config = _tiny_config(model_name="ultralight_baseline")
    with pytest.raises(RuntimeError, match="SCAN_IMPL"):
        engine_mod.build_model(config)


def test_build_model_unet_malunet_egeunet_are_real_working_models():
    from lcmunet.engine import build_model

    for name in ("unet", "malunet", "egeunet"):
        config = _tiny_config(model_name=name)
        model = build_model(config)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == (1, 1, 64, 64)
        # logits, not probabilities -- LogitsAdapter must have unwrapped the
        # vendored models' internal sigmoid (or, for unet, never applied one)
        assert (y < 0).any() or (y > 1).any(), f"{name}: output looks sigmoid-bounded, LogitsAdapter may be missing"


def test_collect_alphas_finds_named_alpha_parameters():
    from lcmunet.engine import collect_alphas
    from lcmunet.testing_models import EngineSanityNet

    model = EngineSanityNet(with_alpha_block=True)
    alphas = collect_alphas(model)
    assert "alpha_block" in alphas
    assert isinstance(alphas["alpha_block"], float)

    model_no_alpha = EngineSanityNet(with_alpha_block=False)
    assert collect_alphas(model_no_alpha) == {}
