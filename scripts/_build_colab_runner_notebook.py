"""One-off generator for notebooks/colab_runner.ipynb. Not part of the
package; kept here so the notebook can be regenerated deterministically if
edited (same convention as every scripts/_build_XX_notebook.py).
Run with: python scripts/_build_colab_runner_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet — Colab Runner (the only notebook you open in Colab)

This notebook is the **bridge**. It never contains hand-copied model/training
code — every run it pulls the latest code fresh from GitHub, so there is
never a divergence between what's on GitHub and what runs here.

Run All top to bottom each session. Zero manual steps beyond the one-time
dataset placement below:
- Drive mounts, the repo is pulled, and dependencies install.
- ALL FOUR datasets (Kvasir-SEG, CVC-ClinicDB, ISIC2017, ISIC2018) are
  extracted and split automatically from whatever .zip you've placed
  (`lcmunet.data.prepare_all.prepare_all_datasets`) -- idempotent, so a
  dataset already prepared in Drive is skipped instantly on every later
  session.
- `run_all_pending()` resumes the training job queue from Google Drive with
  zero manual state (see `lcmunet/run_manifest.py`).

If the session dies mid-run, just reopen this notebook and Run All again.

**One-time setup:** download each dataset's .zip yourself (see
`lcmunet/data/download.py`'s module docstring for exactly where to get each
one) and place it anywhere under `DRIVE_ROOT/data_raw/<Name>/` (any
filename) -- `Kvasir-SEG/`, `CVC-ClinicDB/`, `ISIC2017/`, `ISIC2018/`. This
notebook never downloads anything itself; it only extracts + splits what's
already there. Any dataset with no .zip placed yet is reported FAILED with
the exact download link/filename to use, without blocking the others.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (a) Mount Google Drive — this is the persistence layer for everything
# (checkpoints/splits/configs/logs/results/figures/raw data).
from google.colab import drive
drive.mount('/content/drive')
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (b) git clone/pull THIS repo into /content/ — no code is ever hand-copied.
# Also sets DRIVE_ROOT and resolves `paths` immediately so every later cell
# can rely on both being set.
import os
import subprocess

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
print("Working directory:", os.getcwd())
print("DRIVE_ROOT:", os.environ["DRIVE_ROOT"])

from lcmunet.paths import get_paths
paths = get_paths()
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (c) pip install requirements (infra-layer deps only; torch is Colab's own).
%pip install -q -r requirements.txt
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (d) print GPU name and free VRAM — GPU gates are confirmed by YOU pasting
# this output back, never claimed by the agent.
import torch

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"GPU: {name}")
    print(f"Free VRAM:  {free_b / 1024**3:.2f} GB")
    print(f"Total VRAM: {total_b / 1024**3:.2f} GB")
else:
    print("WARNING: no CUDA GPU detected. Runtime > Change runtime type > GPU, then Run All again.")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (e) LAST setup cell: extract + split ALL FOUR datasets (methodology
# section 7) from whatever .zip you've placed under data_raw/, one call,
# fully idempotent. This is the entire "prepare" pipeline — no other
# notebook, no other cell, is needed, and nothing is downloaded here (see
# lcmunet/data/download.py). A per-dataset PASS/FAIL summary table prints
# at the end; any single dataset with no .zip placed yet (or a bad archive)
# does not block the others. Re-running this in a new session is instant
# for already-prepared datasets (see lcmunet/data/prepare_all.py).
from lcmunet.data.prepare_all import prepare_all_datasets

data_report = prepare_all_datasets(paths)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (f) single call that drives the job queue (results/manifest.json on Drive).
# Training/runner logic does not exist yet (infra-layer only, per methodology
# Week 1); run_all_pending() will raise NotImplementedError until it does.
from lcmunet.run_manifest import run_queue


def run_all_pending(max_minutes: float = 300.0) -> None:
    \"\"\"Process PENDING jobs in results/manifest.json until empty or time runs out.

    Safe to call from a brand-new Colab session after a disconnect: stale
    RUNNING jobs (killed by the previous session) are reclaimed automatically.
    \"\"\"

    def runner_fn(job):
        raise NotImplementedError(
            "No training entrypoint yet — infra layer only (Week 1). "
            f"Job requested: {job['config_id']} -> {job['config_yaml_path']}"
        )

    run_queue(paths.results, runner_fn, max_minutes=max_minutes)


print("Bridge ready. Call run_all_pending() once the training entrypoint exists.")
"""
))

nb["cells"] = cells
with open("notebooks/colab_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/colab_runner.ipynb")
