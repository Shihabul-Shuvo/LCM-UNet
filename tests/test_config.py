import dataclasses

import pytest

from lcmunet.config import DEFAULT_MODEL_CFG, RunConfig


def _make_config(**overrides) -> RunConfig:
    defaults = dict(
        run_name="hero_kvasir_seed0",
        model_name="lcm_unet",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seed0.json",
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_defaults_are_filled_in():
    cfg = _make_config()
    assert cfg.model_cfg == DEFAULT_MODEL_CFG
    assert cfg.scan_impl == "ref"


def test_invalid_scan_impl_raises():
    with pytest.raises(ValueError):
        _make_config(scan_impl="numpy")


def test_yaml_round_trip(tmp_path):
    cfg = _make_config(model_cfg={"kernel_size": 1})  # Ablation C control
    path = cfg.save_yaml(tmp_path / "run.yaml")
    loaded = RunConfig.load_yaml(path)

    assert loaded == cfg
    assert loaded.config_id == cfg.config_id
    # open key merged with defaults, override preserved
    assert loaded.model_cfg["kernel_size"] == 1
    assert loaded.model_cfg["descriptor_type"] == DEFAULT_MODEL_CFG["descriptor_type"]


def test_config_id_is_stable_regardless_of_dict_key_order():
    cfg_a = _make_config(
        model_cfg={"kernel_size": 1, "descriptor_type": "plain"}
    )
    cfg_b = _make_config(
        model_cfg={"descriptor_type": "plain", "kernel_size": 1}
    )
    assert cfg_a.config_id == cfg_b.config_id


def test_config_id_changes_with_content():
    cfg_a = _make_config(seed=0)
    cfg_b = _make_config(seed=1)
    assert cfg_a.config_id != cfg_b.config_id


def test_run_config_is_frozen():
    cfg = _make_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 5
