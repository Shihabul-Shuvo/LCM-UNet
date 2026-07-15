import json
from pathlib import Path

from lcmunet import env_report


def test_update_env_json_creates_file_when_absent(tmp_path):
    results_dir = tmp_path / "results"
    path = env_report.update_env_json(results_dir, {"isic2017_source": "s3"})

    assert path == results_dir / "env.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"isic2017_source": "s3"}


def test_update_env_json_merges_without_clobbering_existing_keys(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "env.json").write_text(json.dumps({"scan_impl": "cuda", "gpu_name": "Tesla T4"}), encoding="utf-8")

    env_report.update_env_json(results_dir, {"isic2017_source": "kaggle_fallback"})

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["scan_impl"] == "cuda"
    assert data["gpu_name"] == "Tesla T4"
    assert data["isic2017_source"] == "kaggle_fallback"


def test_update_env_json_overwrites_same_key(tmp_path):
    results_dir = tmp_path / "results"
    env_report.update_env_json(results_dir, {"isic2017_source": "s3"})
    env_report.update_env_json(results_dir, {"isic2017_source": "kaggle_fallback"})

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["isic2017_source"] == "kaggle_fallback"


def test_write_env_json_preserves_isic2017_source_written_first(tmp_path):
    """Regardless of run order (prepare_all_datasets before or after
    01_env.ipynb's write_env_json), isic2017_source must survive."""
    results_dir = tmp_path / "results"
    env_report.update_env_json(results_dir, {"isic2017_source": "s3"})

    env_report.write_env_json(results_dir, repo_root=".")

    data = json.loads((results_dir / "env.json").read_text(encoding="utf-8"))
    assert data["isic2017_source"] == "s3"
    assert "scan_impl" in data  # collect_env_info's own keys are still (freshly) present
