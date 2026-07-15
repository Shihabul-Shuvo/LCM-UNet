import json

import pytest

from lcmunet import experiment_matrix as em
from lcmunet.config import DEFAULT_MODEL_CFG


def test_build_all_has_no_duplicate_config_ids_by_construction():
    merged = em.build_all(scan_impl="ref")
    config_ids = list(merged.keys())
    assert len(config_ids) == len(set(config_ids))


def test_build_all_dedupes_hero_baseline_glgf_kvasir_seed42_across_phases():
    merged = em.build_all(scan_impl="ref")

    total_rows = sum(len(roles) for _config, roles in merged.values())
    n_shared = sum(1 for _config, roles in merged.values() if len(roles) > 1)

    # Ablation A's baseline/glgf/hero rows on Kvasir at PHASE1_SEED are the
    # same experiments as Phase-2 headline's seed=42 Kvasir rows -- exactly
    # 3 configs should carry 2 roles each (one phase1 role, one phase2 role).
    assert n_shared == 3
    assert total_rows - len(merged) == 3  # each shared config saves exactly 1 duplicate row

    for config, roles in merged.values():
        if len(roles) > 1:
            assert any(r.startswith("phase1") for r in roles)
            assert any(r.startswith("phase2") for r in roles)
            assert config.dataset == "kvasir_seg"
            assert config.seed == em.PHASE1_SEED


def test_phase1_kvasir_has_exactly_ten_rows():
    rows = em.build_phase1_kvasir(scan_impl="ref")
    assert len(rows) == 10
    roles = [r for r, _c in rows]
    assert len(roles) == len(set(roles))  # every phase1 role is unique
    for _role, config in rows:
        assert config.dataset == "kvasir_seg"
        assert config.seed == em.PHASE1_SEED
        assert config.epochs == em.EPOCHS


def test_phase1_hero_row_matches_default_model_cfg():
    rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    hero = rows["phase1_ablation_A_lc_ss2d_hero"]
    assert hero.model_cfg == DEFAULT_MODEL_CFG


def test_phase1_ablation_c_only_kernel_size_differs_from_hero():
    rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    hero = rows["phase1_ablation_A_lc_ss2d_hero"]
    control = rows["phase1_ablation_C_k1_control"]
    diffs = {k for k in hero.model_cfg if hero.model_cfg[k] != control.model_cfg[k]}
    assert diffs == {"kernel_size"}
    assert control.model_cfg["kernel_size"] == 1
    assert hero.model_cfg["kernel_size"] == 3


def test_phase1_ablation_d_only_inject_target_differs_from_hero():
    rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    hero = rows["phase1_ablation_A_lc_ss2d_hero"]
    variant = rows["phase1_ablation_D_inject_input"]
    diffs = {k for k in hero.model_cfg if hero.model_cfg[k] != variant.model_cfg[k]}
    assert diffs == {"inject_target"}
    assert variant.model_cfg["inject_target"] == "input"


def test_phase1_desc_only_descriptor_type_differs_from_hero():
    rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    hero = rows["phase1_ablation_A_lc_ss2d_hero"]
    variant = rows["phase1_desc_plain"]
    diffs = {k for k in hero.model_cfg if hero.model_cfg[k] != variant.model_cfg[k]}
    assert diffs == {"descriptor_type"}
    assert variant.model_cfg["descriptor_type"] == "plain"


@pytest.mark.parametrize(
    "role,expected_placement",
    [
        ("phase1_placement_P0", "P0"),
        ("phase1_placement_E4E5", "E4E5"),
        ("phase1_placement_plusE6", "+E6"),
        ("phase1_placement_E4D4", "E4D4"),
    ],
)
def test_phase1_placement_rows_only_placement_differs_from_hero(role, expected_placement):
    rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    hero = rows["phase1_ablation_A_lc_ss2d_hero"]
    variant = rows[role]
    diffs = {k for k in hero.model_cfg if hero.model_cfg[k] != variant.model_cfg[k]}
    assert diffs == {"placement"}
    assert variant.model_cfg["placement"] == expected_placement


def test_phase2_headline_covers_kvasir_and_cvc_with_five_seeds_each():
    rows = em.build_phase2_headline(scan_impl="ref")
    assert len(rows) == 2 * 5 * 3  # 2 datasets x 5 seeds x {baseline, glgf, hero}
    for _role, config in rows:
        assert config.dataset in ("kvasir_seg", "cvc_clinicdb")
        assert config.seed in em.HEADLINE_SEEDS
        assert config.model_name in ("ultralight_baseline", "glgf", "lc_ss2d")


def test_phase2_isic_covers_both_versions_hero_and_baseline_only_three_seeds():
    rows = em.build_phase2_isic(scan_impl="ref")
    assert len(rows) == 2 * 3 * 2  # 2 datasets x 3 seeds x {baseline, hero}
    model_names = {config.model_name for _role, config in rows}
    assert model_names == {"ultralight_baseline", "lc_ss2d"}  # no GLGF on ISIC per spec
    for _role, config in rows:
        assert config.seed in em.OTHER_SEEDS
        assert config.dataset in ("isic2017", "isic2018")


def test_phase2_comparators_covers_unet_malunet_egeunet_kvasir_and_cvc():
    rows = em.build_phase2_comparators(scan_impl="ref")
    assert len(rows) == 2 * 3 * 3  # 2 datasets x 3 seeds x {unet, malunet, egeunet}
    model_names = {config.model_name for _role, config in rows}
    assert model_names == {"unet", "malunet", "egeunet"}
    cvc_roles = [role for role, _c in rows if "cvc_clinicdb" in role]
    assert all("optional_if_time" in role for role in cvc_roles)
    kvasir_roles = [role for role, _c in rows if "kvasir_seg" in role]
    assert not any("optional_if_time" in role for role in kvasir_roles)


def test_run_name_is_identical_regardless_of_which_builder_produced_it():
    # The Kvasir hero row at seed=42 is built independently by both
    # build_phase1_kvasir and build_phase2_headline -- they must agree on
    # every field (including run_name) or the dedup in build_all breaks.
    phase1_rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    phase2_rows = dict(em.build_phase2_headline(scan_impl="ref"))

    hero_p1 = phase1_rows["phase1_ablation_A_lc_ss2d_hero"]
    hero_p2 = phase2_rows["phase2_headline_kvasir_seg_lc_ss2d_hero_seed42"]
    assert hero_p1.to_dict() == hero_p2.to_dict()
    assert hero_p1.config_id == hero_p2.config_id


def test_scan_impl_propagates_to_every_config():
    merged = em.build_all(scan_impl="cuda")
    assert all(config.scan_impl == "cuda" for config, _roles in merged.values())


def test_resolve_scan_impl_falls_back_without_env_json(paths):
    scan_impl, source = em.resolve_scan_impl(paths)
    from lcmunet.scan import SCAN_IMPL

    assert scan_impl == SCAN_IMPL
    assert "live fallback" in source


def test_resolve_scan_impl_prefers_env_json(paths):
    env_json_path = paths.results / "env.json"
    env_json_path.write_text(json.dumps({"scan_impl": "cuda", "timestamp": "t"}), encoding="utf-8")

    scan_impl, source = em.resolve_scan_impl(paths)
    assert scan_impl == "cuda"
    assert "env.json" in source


def test_phase_of():
    assert em.phase_of("phase1_ablation_A_baseline_pvm") == "phase1"
    assert em.phase_of("phase2_headline_kvasir_seg_glgf_seed42") == "phase2"
