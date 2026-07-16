import json

from lcmunet import env_report


def test_update_env_json_creates_file_when_absent(tmp_path):
    results_dir = tmp_path / "results"
    path = env_report.update_env_json(results_dir, {"custom_note": "hello"})

    assert path == results_dir / "env.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"custom_note": "hello"}


def test_update_env_json_merges_without_clobbering_existing_keys(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "env.json").write_text(json.dumps({"scan_impl": "cuda", "gpu_name": "Tesla T4"}), encoding="utf-8")

    env_report.update_env_json(results_dir, {"custom_note": "hello"})

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["scan_impl"] == "cuda"
    assert data["gpu_name"] == "Tesla T4"
    assert data["custom_note"] == "hello"


def test_update_env_json_overwrites_same_key(tmp_path):
    results_dir = tmp_path / "results"
    env_report.update_env_json(results_dir, {"custom_note": "first"})
    env_report.update_env_json(results_dir, {"custom_note": "second"})

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["custom_note"] == "second"


def test_write_env_json_preserves_keys_written_by_something_else_first(tmp_path):
    """Regardless of run order, an extra key written via update_env_json by
    some other module must survive a later write_env_json() refresh
    (write_env_json only owns collect_env_info()'s own fixed key set)."""
    results_dir = tmp_path / "results"
    env_report.update_env_json(results_dir, {"custom_note": "hello"})

    env_report.write_env_json(results_dir, repo_root=".")

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["custom_note"] == "hello"
    assert "scan_impl" in data  # collect_env_info's own keys are still (freshly) present
