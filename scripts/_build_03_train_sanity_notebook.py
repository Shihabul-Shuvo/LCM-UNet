"""One-off generator for notebooks/03_train_sanity.ipynb.
Run with: python scripts/_build_03_train_sanity_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 03: Training/Eval Engine Sanity + Resume Check

Run All top to bottom in Colab. This notebook does NOT train a real model --
no baseline (reproduced UltraLight PVM), GLGF, or LC-SS2D/LCM-UNet exists in
lcmunet/ yet (see lcmunet/testing_models.py's docstring: the real backbone
needs mamba-ssm, a GPU-only build, and LC-VSS implementation is pending an
architecture decision flagged in a prior session).

What this DOES prove, against real Kvasir-SEG data and the real engine.py:
1. A run checkpoints every epoch (model/optimizer/scheduler/scaler/RNG state).
2. Killing a run mid-training and calling run_one() again resumes seamlessly
   -- continuing from the saved epoch, not restarting -- and reproduces
   exactly what an uninterrupted run would have produced.
3. Alpha logging and the per-image Dice / results-row artifacts all get
   written correctly.

**Action needed from you:** paste back the printed verification block at the
end -- specifically that `weights match exactly: True` and the artifact
checklist all show True.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Mount Drive + pull this repo fresh.
import os
import subprocess

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = "https://github.com/Shihabul-Shuvo/LCM-UNet.git"
REPO_DIR = "/content/LCM-UNet"
DRIVE_ROOT = "/content/drive/MyDrive/LCM-UNet/"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", "main"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", "main"], check=True)

os.chdir(REPO_DIR)
os.environ["DRIVE_ROOT"] = DRIVE_ROOT
print("cwd:", os.getcwd())
"""
))

cells.append(nbf.v4.new_code_cell(
"""%pip install -q -r requirements.txt
"""
))

cells.append(nbf.v4.new_code_cell(
'''# Ensure Kvasir-SEG is present (auto-downloads if missing) and build its split.
from lcmunet.paths import get_paths
from lcmunet.data.download import ensure_kvasir
from lcmunet.data.splits import build_kvasir_split

paths = get_paths()
ensure_kvasir(paths.data_raw)
build_kvasir_split(paths)
print("GPU:", __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "none (CPU)")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Carve out a TINY subset of the real split (fast sanity run, real images).
import json

from lcmunet.data.splits import load_split

full_split = load_split(paths.splits, "kvasir_seg")
tiny_split = {
    **full_split,
    "train": full_split["train"][:16],
    "val": full_split["val"][:4],
    "test": full_split["test"][:4],
    "counts": {"train": 16, "val": 4, "test": 4},
    "notes": "TINY subset of the real kvasir_seg split, for engine sanity-run demonstration only.",
}
tiny_path = paths.splits / "kvasir_seg_tiny.json"
with open(tiny_path, "w") as f:
    json.dump(tiny_split, f)
print("wrote", tiny_path)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Run A: uninterrupted 5 epochs (ground truth for comparison).
from lcmunet.config import RunConfig
from lcmunet.engine import run_one, checkpoint_dir

config_a = RunConfig(
    run_name="sanity_uninterrupted",
    model_name="engine_test_tiny",
    dataset="kvasir_seg",
    seed=0,
    split_file="splits/kvasir_seg_tiny.json",
    epochs=5,
    batch_size=4,
    input_size=64,
)
result_a = run_one(config_a, paths=paths, num_workers=0)
print("Run A (uninterrupted):", result_a)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Run B: kill after epoch 2 (0-indexed), then resume -- must reproduce Run A exactly.
config_b = RunConfig(
    run_name="sanity_killed_then_resumed",
    model_name="engine_test_tiny",
    dataset="kvasir_seg",
    seed=0,
    split_file="splits/kvasir_seg_tiny.json",
    epochs=5,
    batch_size=4,
    input_size=64,
)

partial = run_one(config_b, paths=paths, stop_after_epoch=2, num_workers=0)
print("Simulated kill after epoch 2:", partial)
assert partial["completed"] is False and partial["reached_epoch"] == 2

result_b = run_one(config_b, paths=paths, num_workers=0)  # fresh call, must detect + resume the checkpoint
print("Resumed to completion:", result_b)
assert result_b["completed"] is True and result_b["reached_epoch"] == 4
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Verify: resumed run reproduces the uninterrupted run bit-exactly.
import torch

ckpt_a = torch.load(checkpoint_dir(paths, config_a) / "last.pt", weights_only=False)
ckpt_b = torch.load(checkpoint_dir(paths, config_b) / "last.pt", weights_only=False)

mismatches = [k for k in ckpt_a["model"] if not torch.equal(ckpt_a["model"][k], ckpt_b["model"][k])]
print("weights match exactly:", len(mismatches) == 0, "-- mismatched keys:", mismatches)
print("test dsc A:", result_a["test_metrics"]["dsc"], " test dsc B:", result_b["test_metrics"]["dsc"])
print("test dsc matches exactly:", result_a["test_metrics"]["dsc"] == result_b["test_metrics"]["dsc"])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Verify all required artifacts exist for config_b\'s run.
import pandas as pd

ckpt_dir = checkpoint_dir(paths, config_b)
print("checkpoint files:", sorted(p.name for p in ckpt_dir.iterdir()))

df = pd.read_csv(paths.results / "results.csv")
row = df[(df["config_id"] == config_b.config_id) & (df["seed"] == config_b.seed)]
print("results row present:", len(row) == 1)

per_image_path = paths.results / "perimage" / f"{config_b.config_id}_{config_b.seed}.npy"
print("per-image Dice file present:", per_image_path.is_file())

alpha_csv = paths.results / "alpha" / f"{config_b.config_id}_{config_b.seed}.csv"
print("alpha csv present:", alpha_csv.is_file())
if alpha_csv.is_file():
    print(pd.read_csv(alpha_csv))
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## Paste this back

- `weights match exactly: True` (and an empty mismatched-keys list)
- `test dsc matches exactly: True`
- The artifact checklist (checkpoint files / results row / per-image Dice / alpha csv) all `True`

This confirms the resumable engine is safe to build real models on top of
once the pending model decision lands.
"""
))

nb["cells"] = cells
with open("notebooks/03_train_sanity.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/03_train_sanity.ipynb")
