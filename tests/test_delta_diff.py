import torch

from lcmunet.delta_diff import delta_difference_report
from lcmunet.lcm_unet import LCMUNet


def test_delta_difference_report_finds_all_four_hero_stages():
    torch.manual_seed(0)
    model = LCMUNet(placement="hero")  # descriptor=contrast default -> E4/E5/D5/D4 all inject
    x = torch.randn(2, 3, 64, 64)

    report = delta_difference_report(model, x)

    assert set(report.keys()) == {"E4", "E5", "D5", "D4"}
    for stage, stats in report.items():
        assert stats["mean_abs_diff"] >= 0
        assert stats["std_diff"] >= 0
        assert stats["max_abs_diff"] >= stats["mean_abs_diff"]


def test_delta_difference_is_non_constant_for_a_real_trained_looking_model():
    """With a random (but non-trivial, non-collapsed) alpha/W_delta, the
    per-token modulation should vary across a real random image -- this is
    exactly the 'quick sanity' section 13 asks for."""
    torch.manual_seed(1)
    model = LCMUNet(placement="hero", alpha_init=0.05)  # away from the near-zero init so the effect isn't numerically negligible
    x = torch.randn(2, 3, 64, 64)

    report = delta_difference_report(model, x)
    assert all(stats["non_constant"] for stats in report.values())


def test_alpha_restored_after_delta_difference_report():
    torch.manual_seed(0)
    model = LCMUNet(placement="hero")
    before = {name: mod.alpha.detach().clone() for name, mod in [("E4", model.encoder4[0]), ("E5", model.encoder5[0]), ("D5", model.decoder1[0]), ("D4", model.decoder2[0])]}

    delta_difference_report(model, torch.randn(1, 3, 64, 64))

    after = {"E4": model.encoder4[0].alpha, "E5": model.encoder5[0].alpha, "D5": model.decoder1[0].alpha, "D4": model.decoder2[0].alpha}
    for name in before:
        assert torch.equal(before[name], after[name].detach())


def test_zero_alpha_model_gives_exactly_zero_diff_everywhere():
    """Sanity check on the method itself: if alpha is already 0 for every
    stage, 'ours' and 'baseline' are mathematically identical, so the diff
    must be exactly zero (not just small) -- confirms the toggling mechanism
    is correct, not just approximately so."""
    torch.manual_seed(0)
    model = LCMUNet(placement="hero", alpha_init=0.0)
    x = torch.randn(2, 3, 64, 64)

    report = delta_difference_report(model, x)
    for stage, stats in report.items():
        assert stats["mean_abs_diff"] == 0.0
        assert stats["max_abs_diff"] == 0.0
        assert stats["non_constant"] is False


def test_raises_when_model_has_no_active_lc_vss_stage():
    import pytest

    model = LCMUNet(placement="P0")  # no stage injects
    with pytest.raises(ValueError, match="no LC-VSS stage"):
        delta_difference_report(model, torch.randn(1, 3, 64, 64))
