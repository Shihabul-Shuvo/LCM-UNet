from lcmunet.results_store import COLUMNS, load_results, upsert_result


def _row(config_id="abc123", seed=0, dsc=0.80, **overrides):
    row = dict(
        config_id=config_id,
        model_name="lcm_unet",
        dataset="kvasir_seg",
        seed=seed,
        split_file="splits/kvasir_seed0.json",
        dsc=dsc,
        miou=0.70,
        scan_impl="ref",
        gpu_name="cpu-test",
    )
    row.update(overrides)
    return row


def test_load_results_empty_has_correct_columns(tmp_path):
    df = load_results(tmp_path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 0


def test_upsert_creates_row(tmp_path):
    df = upsert_result(tmp_path, _row())
    assert len(df) == 1
    assert df.iloc[0]["config_id"] == "abc123"
    assert df.iloc[0]["dsc"] == 0.80


def test_upsert_missing_required_key_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        upsert_result(tmp_path, {"model_name": "lcm_unet"})


def test_upsert_different_seed_appends(tmp_path):
    upsert_result(tmp_path, _row(seed=0))
    df = upsert_result(tmp_path, _row(seed=1))
    assert len(df) == 2


def test_upsert_same_config_and_seed_replaces_not_duplicates(tmp_path):
    upsert_result(tmp_path, _row(seed=0, dsc=0.80))
    df = upsert_result(tmp_path, _row(seed=0, dsc=0.91))

    assert len(df) == 1
    assert df.iloc[0]["dsc"] == 0.91


def test_upsert_persists_across_loads(tmp_path):
    upsert_result(tmp_path, _row(config_id="c1", seed=0))
    upsert_result(tmp_path, _row(config_id="c2", seed=0))
    upsert_result(tmp_path, _row(config_id="c1", seed=1))

    reloaded = load_results(tmp_path)
    assert len(reloaded) == 3
    # idempotent re-run of c1/seed0 still yields 3 rows total
    upsert_result(tmp_path, _row(config_id="c1", seed=0, dsc=0.5))
    reloaded_again = load_results(tmp_path)
    assert len(reloaded_again) == 3


def test_timestamp_auto_filled(tmp_path):
    df = upsert_result(tmp_path, _row())
    assert df.iloc[0]["timestamp"] not in (None, "")
