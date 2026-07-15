"""lcmunet/data/download.py -- pure-logic helpers tested directly (no
network), plus fast-path idempotency and full synthetic-zip flow tests for
each ensure_* function. Every network/Kaggle-CLI call is monkeypatched to a
local synthetic .zip built with lcmunet.data.download._extract_zip's own
counterpart (zipfile.ZipFile.writestr) -- these tests never touch a real
network or require kaggle credentials.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from lcmunet.data import download as dl
from lcmunet.data import kaggle_auth as ka
from lcmunet.data import raw_layout as rl


def _make_zip(zip_path: Path, files: dict) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, content in files.items():
            zf.writestr(arcname, content)
    return zip_path


# ---- pure-logic helpers ------------------------------------------------------


def test_pair_isic_style_prefers_segmentation_suffix():
    images = {"ISIC_0000001": Path("a.jpg"), "ISIC_0000002": Path("b.jpg")}
    masks = {"ISIC_0000001_segmentation": Path("a_seg.png"), "ISIC_0000002_segmentation": Path("b_seg.png")}
    matched, suffix = dl._pair_isic_style(images, masks)
    assert suffix == "_segmentation"
    assert set(matched) == {"ISIC_0000001", "ISIC_0000002"}


def test_pair_isic_style_falls_back_to_exact_stem():
    images = {"x1": Path("x1.jpg")}
    masks = {"x1": Path("x1.png")}  # no suffix at all
    matched, suffix = dl._pair_isic_style(images, masks)
    assert suffix == ""
    assert matched["x1"] == (Path("x1.jpg"), Path("x1.png"))


def test_pair_isic_style_picks_whichever_suffix_matches_more():
    images = {f"id{i}": Path(f"id{i}.jpg") for i in range(5)}
    masks = {f"id{i}_segmentation": Path(f"id{i}_segmentation.png") for i in range(4)}
    masks["id4"] = Path("id4.png")  # only one exact-stem match
    matched, suffix = dl._pair_isic_style(images, masks)
    assert suffix == "_segmentation"
    assert len(matched) == 4


def test_find_archive_matches_any_substring(tmp_path):
    (tmp_path / "some_other.zip").write_bytes(b"x")
    target = tmp_path / "cvcclinicdb-release.zip"
    target.write_bytes(b"x")
    found = dl._find_archive(tmp_path, ("cvc",))
    assert found == target


def test_find_archive_returns_none_when_no_match(tmp_path):
    (tmp_path / "unrelated.zip").write_bytes(b"x")
    assert dl._find_archive(tmp_path, ("cvc",)) is None


def test_find_files_indexes_by_stem(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    files = dl._find_files(tmp_path, (".jpg", ".png"))
    assert set(files) == {"a", "b"}


def test_relocate_nested_extraction_flattens_single_wrapper(tmp_path):
    root = tmp_path / "root"
    wrapper = root / "cvcclinicdb-release"
    (wrapper / "Original").mkdir(parents=True)
    (wrapper / "Ground Truth").mkdir(parents=True)
    (wrapper / "Original" / "1.png").write_bytes(b"x")

    dl._relocate_nested_extraction(root, [rl._CVC_IMAGE_DIR_NAMES, rl._CVC_MASK_DIR_NAMES])

    assert (root / "Original" / "1.png").is_file()
    assert not wrapper.exists()


def test_relocate_nested_extraction_no_op_when_already_flat(tmp_path):
    root = tmp_path / "root"
    (root / "Original").mkdir(parents=True)
    (root / "Ground Truth").mkdir(parents=True)
    (root / "Original" / "1.png").write_bytes(b"x")

    dl._relocate_nested_extraction(root, [rl._CVC_IMAGE_DIR_NAMES, rl._CVC_MASK_DIR_NAMES])

    assert (root / "Original" / "1.png").is_file()  # untouched, still there


def test_detect_bucket_dirs_distinguishes_images_and_masks(tmp_path):
    root = tmp_path / "extract"
    images_dir = root / "ISIC2018_Task1-2_Training_Input"
    masks_dir = root / "ISIC2018_Task1_Training_GroundTruth"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for i in range(25):
        (images_dir / f"ISIC_{i:07d}.jpg").write_bytes(b"x")
        (masks_dir / f"ISIC_{i:07d}_segmentation.png").write_bytes(b"x")

    found_images, found_masks, note = dl._detect_bucket_dirs(root, include_hints=("train", "training"), exclude_hints=("valid", "test"), min_files=20)
    assert found_images == images_dir
    assert found_masks == masks_dir


def test_detect_bucket_dirs_excludes_validation(tmp_path):
    root = tmp_path / "extract"
    train_images = root / "Training" / "images"
    train_masks = root / "Training" / "masks"
    val_images = root / "Validation" / "images"
    train_images.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    val_images.mkdir(parents=True)
    for i in range(25):
        (train_images / f"id{i}.jpg").write_bytes(b"x")
        (train_masks / f"id{i}_segmentation.png").write_bytes(b"x")
        (val_images / f"id{i}.jpg").write_bytes(b"x")

    found_images, found_masks, note = dl._detect_bucket_dirs(root, include_hints=("train", "training"), exclude_hints=("valid", "test"), min_files=20)
    assert found_images == train_images
    assert found_masks == train_masks


def test_detect_bucket_dirs_raises_when_ambiguous(tmp_path):
    root = tmp_path / "extract"
    a = root / "images_a"
    b = root / "images_b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    for i in range(25):
        (a / f"id{i}.jpg").write_bytes(b"x")
        (b / f"id{i}.jpg").write_bytes(b"x")

    with pytest.raises(RuntimeError, match="Could not confidently detect"):
        dl._detect_bucket_dirs(root, include_hints=(), exclude_hints=(), min_files=20)


def test_copy_as_format_copies_when_extension_already_matches(tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"raw-bytes")
    dst = tmp_path / "out.jpg"
    dl._copy_as_format(src, dst, "JPEG")
    assert dst.read_bytes() == b"raw-bytes"


def test_copy_as_format_converts_when_extension_differs(tmp_path):
    from PIL import Image
    import numpy as np

    src = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(src)
    dst = tmp_path / "out.jpg"
    dl._copy_as_format(src, dst, "JPEG")
    assert dst.is_file()
    with Image.open(dst) as img:
        assert img.format == "JPEG"


# ---- fast-path idempotency (already-ready data -- no network at all) --------


def test_ensure_kvasir_fast_path_no_network(make_kvasir_raw, monkeypatch):
    monkeypatch.setattr(rl, "KVASIR_IMAGE_COUNT", 5)
    p = make_kvasir_raw(n=5)
    monkeypatch.setattr(dl, "_urlretrieve_with_timeout", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download")))

    assert dl.ensure_kvasir(p.data_raw) == 5


def test_ensure_cvc_fast_path_no_network(make_cvc_raw, monkeypatch):
    p = make_cvc_raw(n=612)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download")))
    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not need kaggle auth")))

    assert dl.ensure_cvc(p.data_raw, p) == 612


def test_ensure_isic2018_fast_path_no_network(make_isic_raw, monkeypatch):
    monkeypatch.setitem(rl.ISIC_IMAGE_COUNTS, "isic2018", 5)
    p = make_isic_raw(version="isic2018", n=5)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download")))
    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not need kaggle auth")))

    assert dl.ensure_isic2018(p.data_raw, p) == 5


def test_ensure_isic2017_fast_path_no_network(make_isic_raw, monkeypatch):
    monkeypatch.setitem(rl.ISIC_IMAGE_COUNTS, "isic2017", 5)
    p = make_isic_raw(version="isic2017", n=5)
    monkeypatch.setattr(dl, "_download_isic2017_s3", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not attempt S3")))
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download")))

    assert dl.ensure_isic2017(p.data_raw, p) == 5


# ---- KaggleAuthMissingError propagation (no token, data genuinely missing) ---


def test_ensure_cvc_raises_kaggle_auth_missing_without_token(paths):
    with pytest.raises(ka.KaggleAuthMissingError):
        dl.ensure_cvc(paths.data_raw, paths)


def test_ensure_isic2018_raises_kaggle_auth_missing_without_token(paths):
    with pytest.raises(ka.KaggleAuthMissingError):
        dl.ensure_isic2018(paths.data_raw, paths)


def test_ensure_isic2017_falls_back_to_kaggle_and_raises_auth_missing_when_s3_fails(paths, monkeypatch):
    monkeypatch.setattr(dl, "_urlretrieve_with_timeout", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated network failure")))

    with pytest.raises(ka.KaggleAuthMissingError):
        dl.ensure_isic2017(paths.data_raw, paths)


# ---- full synthetic-zip flows -------------------------------------------------


def test_ensure_isic2018_full_kaggle_flow(paths, monkeypatch, tmp_path):
    n = 50
    monkeypatch.setitem(rl.ISIC_IMAGE_COUNTS, "isic2018", n)

    files = {}
    for i in range(n):
        stem = f"ISIC_{i:07d}"
        files[f"ISIC2018_Task1-2_Training_Input/{stem}.jpg"] = b"img"
        files[f"ISIC2018_Task1_Training_GroundTruth/{stem}_segmentation.png"] = b"mask"
    for i in range(n):  # a same-sized Validation portion that must be ignored
        files[f"ISIC2018_Task1-2_Validation_Input/ISIC_9{i:06d}.jpg"] = b"img"

    zip_path = _make_zip(tmp_path / "isic2018.zip", files)
    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda p: None)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda slug, dest_dir: zip_path)

    n_pairs = dl.ensure_isic2018(paths.data_raw, paths)
    assert n_pairs == n
    assert len(rl.list_isic_pairs(paths.data_raw, "isic2018")) == n


def test_ensure_cvc_full_kaggle_flow_direct_structure(paths, monkeypatch, tmp_path):
    n = 6
    monkeypatch.setattr(rl, "CVC_IMAGE_COUNT", n)

    files = {}
    for i in range(1, n + 1):
        files[f"Original/{i}.png"] = b"img"
        files[f"Ground Truth/{i}.png"] = b"mask"
    zip_path = _make_zip(tmp_path / "cvc.zip", files)

    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda p: None)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda slug, dest_dir: zip_path)

    n_pairs = dl.ensure_cvc(paths.data_raw, paths)
    assert n_pairs == n


def test_ensure_cvc_handles_extra_wrapper_folder(paths, monkeypatch, tmp_path):
    n = 6
    monkeypatch.setattr(rl, "CVC_IMAGE_COUNT", n)

    files = {}
    for i in range(1, n + 1):
        files[f"cvcclinicdb-release/Original/{i}.png"] = b"img"
        files[f"cvcclinicdb-release/Ground Truth/{i}.png"] = b"mask"
    zip_path = _make_zip(tmp_path / "cvc_nested.zip", files)

    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda p: None)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda slug, dest_dir: zip_path)

    n_pairs = dl.ensure_cvc(paths.data_raw, paths)
    assert n_pairs == n


def test_ensure_isic2017_full_s3_flow(paths, monkeypatch, tmp_path):
    bucket_counts = {"train": 20, "val": 5, "test": 5}
    keep_train = 8
    monkeypatch.setattr(dl, "ISIC2017_BUCKET_COUNTS", bucket_counts)
    monkeypatch.setattr(dl, "ISIC2017_TRAIN_KEEP", keep_train)
    total_kept = keep_train + bucket_counts["val"] + bucket_counts["test"]
    monkeypatch.setitem(rl.ISIC_IMAGE_COUNTS, "isic2017", total_kept)

    bucket_ranges = {"train": range(0, bucket_counts["train"]), "val": range(1000, 1000 + bucket_counts["val"]), "test": range(2000, 2000 + bucket_counts["test"])}
    zips = {}
    for fname, kind, bucket in dl.ISIC2017_S3_PARTS:
        wrapper = fname[:-4]  # official S3 zips nest one top-level folder named like the zip stem
        files = {}
        for i in bucket_ranges[bucket]:
            stem = f"ISIC_{i:07d}"
            if kind == "images":
                files[f"{wrapper}/{stem}.jpg"] = b"img"
            else:
                files[f"{wrapper}/{stem}_segmentation.png"] = b"mask"
        zips[fname] = _make_zip(tmp_path / fname, files)

    def fake_urlretrieve(url, dest, timeout=None):
        shutil.copy(zips[url.rsplit("/", 1)[-1]], dest)

    monkeypatch.setattr(dl, "_urlretrieve_with_timeout", fake_urlretrieve)

    n_pairs = dl.ensure_isic2017(paths.data_raw, paths)
    assert n_pairs == total_kept
    assert len(rl.list_isic_pairs(paths.data_raw, "isic2017")) == total_kept

    manifest = json.loads((rl.isic_root(paths.data_raw, "isic2017") / "isic2017_source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "s3"
    assert len(manifest["train_kept"]) == keep_train
    assert len(manifest["val_kept"]) == bucket_counts["val"]
    assert len(manifest["test_kept"]) == bucket_counts["test"]

    env_data = json.loads((paths.results / "env.json").read_text(encoding="utf-8"))
    assert env_data["isic2017_source"] == "s3"


def test_ensure_isic2017_kaggle_fallback_flow(paths, monkeypatch, tmp_path):
    train_n, val_n, test_n, keep_train = 25, 20, 20, 10
    monkeypatch.setattr(dl, "ISIC2017_TRAIN_KEEP", keep_train)
    total_kept = keep_train + val_n + test_n
    monkeypatch.setitem(rl.ISIC_IMAGE_COUNTS, "isic2017", total_kept)

    monkeypatch.setattr(dl, "_urlretrieve_with_timeout", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated network failure")))
    monkeypatch.setattr(dl, "ensure_kaggle_auth", lambda p: None)

    bucket_ranges = {"train": range(0, train_n), "val": range(1000, 1000 + val_n), "test": range(2000, 2000 + test_n)}
    bucket_dirnames = {"train": "Training", "val": "Validation", "test": "Test"}
    files = {}
    for bucket, ids in bucket_ranges.items():
        dirname = bucket_dirnames[bucket]
        for i in ids:
            stem = f"ISIC_{i:07d}"
            files[f"{dirname}/images/{stem}.jpg"] = b"img"
            files[f"{dirname}/masks/{stem}_segmentation.png"] = b"mask"
    zip_path = _make_zip(tmp_path / "isic-2017.zip", files)
    monkeypatch.setattr(dl, "_kaggle_cli_download", lambda slug, dest_dir: zip_path)

    n_pairs = dl.ensure_isic2017(paths.data_raw, paths)
    assert n_pairs == total_kept

    manifest = json.loads((rl.isic_root(paths.data_raw, "isic2017") / "isic2017_source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "kaggle_fallback"

    env_data = json.loads((paths.results / "env.json").read_text(encoding="utf-8"))
    assert env_data["isic2017_source"] == "kaggle_fallback"


# ---- dispatch -----------------------------------------------------------


def test_ensure_dataset_ready_dispatches_kvasir(make_kvasir_raw, monkeypatch):
    monkeypatch.setattr(rl, "KVASIR_IMAGE_COUNT", 3)
    p = make_kvasir_raw(n=3)
    assert dl.ensure_dataset_ready("kvasir_seg", p) == 3


def test_ensure_dataset_ready_unknown_dataset_raises(paths):
    with pytest.raises(ValueError):
        dl.ensure_dataset_ready("not_a_real_dataset", paths)
