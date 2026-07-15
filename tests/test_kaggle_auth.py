from pathlib import Path

import pytest

from lcmunet.data import kaggle_auth as ka


def test_missing_kaggle_json_raises_with_exact_message(paths):
    with pytest.raises(ka.KaggleAuthMissingError) as exc_info:
        ka.ensure_kaggle_auth(paths)
    assert str(exc_info.value) == ka.MISSING_MESSAGE
    assert "DRIVE_ROOT/secrets/kaggle.json" in str(exc_info.value)
    assert "kaggle.com -> Settings -> Create New Token" in str(exc_info.value)


def test_ensure_kaggle_auth_copies_to_home_kaggle_dir(paths, monkeypatch, tmp_path):
    paths.secrets.mkdir(parents=True, exist_ok=True)
    src = paths.secrets / "kaggle.json"
    src.write_text('{"username": "u", "key": "k"}', encoding="utf-8")

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(ka, "_kaggle_installed", lambda: True)  # skip real pip install

    dst = ka.ensure_kaggle_auth(paths)

    assert dst == fake_home / ".kaggle" / "kaggle.json"
    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_ensure_kaggle_auth_pip_installs_kaggle_if_missing(paths, monkeypatch, tmp_path):
    paths.secrets.mkdir(parents=True, exist_ok=True)
    (paths.secrets / "kaggle.json").write_text("{}", encoding="utf-8")

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(ka, "_kaggle_installed", lambda: False)

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)

    monkeypatch.setattr(ka.subprocess, "run", fake_run)

    ka.ensure_kaggle_auth(paths)

    assert len(calls) == 1
    assert "kaggle" in calls[0]
    assert "pip" in calls[0] or "-m" in calls[0]


def test_try_ensure_kaggle_auth_does_not_raise_when_missing(paths, capsys):
    result = ka.try_ensure_kaggle_auth(paths)
    assert result is None
    out = capsys.readouterr().out
    assert "Kvasir-SEG" in out


def test_try_ensure_kaggle_auth_returns_path_when_present(paths, monkeypatch, tmp_path):
    paths.secrets.mkdir(parents=True, exist_ok=True)
    (paths.secrets / "kaggle.json").write_text("{}", encoding="utf-8")

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(ka, "_kaggle_installed", lambda: True)

    result = ka.try_ensure_kaggle_auth(paths)
    assert result == fake_home / ".kaggle" / "kaggle.json"
