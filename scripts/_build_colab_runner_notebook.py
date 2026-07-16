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
"""**To change dataset scope:** edit `ACTIVE_DATASETS` in `lcmunet/config.py`, commit, push, then just re-run this notebook (Run All). Nothing else needs to be run by hand.

# LCM-UNet — Colab Runner (the only notebook you open in Colab)

This notebook is the **bridge**. It never contains hand-copied model/training
code — every run it pulls the latest code fresh from GitHub, so there is
never a divergence between what's on GitHub and what runs here.

Run All top to bottom each session. Zero manual steps beyond the one-time
dataset placement below:
- Drive mounts, the repo is pulled, and dependencies install.
- Every dataset in `lcmunet.config.ACTIVE_DATASETS` is extracted and split
  automatically from whatever .zip you've placed
  (`lcmunet.data.prepare_all.prepare_all_datasets`) -- idempotent, so a
  dataset already prepared in Drive is skipped instantly on every later
  session. Datasets NOT in `ACTIVE_DATASETS` are cleanly SKIPPED, not
  treated as an error.
- Every in-scope experiment (Phase-1 ablations, Phase-2 headline/ISIC/
  comparator rows) is automatically enqueued as PENDING
  (`lcmunet.run_manifest.sync_manifest_with_active_datasets`) -- also
  idempotent: re-running this never duplicates, resets, or touches a job
  that's already PENDING/RUNNING/DONE/FAILED.
- A short summary prints job counts (PENDING/RUNNING/DONE/FAILED) broken
  down by dataset, plus which datasets are currently skipped.
- Actually **running** the job queue (`run_all_pending()`, hours of GPU
  time) is a separate, explicit cell below -- discovering and enqueuing
  work is automatic, starting training is a conscious action you take.

If the session dies mid-run, just reopen this notebook and Run All again.

**One-time setup:** download each ACTIVE dataset's .zip yourself (see
`lcmunet/data/download.py`'s module docstring for exactly where to get each
one) and place it anywhere under `DRIVE_ROOT/data_raw/<Name>/` (any
filename) -- `Kvasir-SEG/`, `CVC-ClinicDB/`, `ISIC2017/`, `ISIC2018/`. This
notebook never downloads anything itself; it only extracts + splits what's
already there, for whichever datasets are currently in scope. Any in-scope
dataset with no .zip placed yet is reported FAILED with the exact download
link/filename to use, without blocking the others.
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
"""# GPU name and free VRAM — GPU gates are confirmed by YOU pasting this
# output back, never claimed by the agent.
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
"""# (d) extract + split every dataset in lcmunet.config.ACTIVE_DATASETS from
# whatever .zip you've placed under data_raw/, one call, fully idempotent.
# Nothing is downloaded here (see lcmunet/data/download.py). Datasets NOT
# in ACTIVE_DATASETS are reported SKIPPED (not an error, no data_raw/
# folder required); any per-dataset failure among the ACTIVE ones does not
# block the others. Re-running this in a new session is instant for
# already-prepared datasets (see lcmunet/data/prepare_all.py).
from lcmunet.data.prepare_all import prepare_all_datasets

data_report = prepare_all_datasets(paths)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (e) automatically enqueue every in-scope experiment (Phase-1 ablations,
# Phase-2 headline/ISIC/comparator rows) as PENDING in results/manifest.json
# -- fully idempotent (never duplicates/resets/touches an existing job; see
# lcmunet/run_manifest.py's sync_manifest_with_active_datasets docstring).
# Datasets not in ACTIVE_DATASETS simply have no jobs enqueued for them.
from lcmunet.run_manifest import sync_manifest_with_active_datasets

sync_report = sync_manifest_with_active_datasets(paths)
print(f"scan_impl = {sync_report['scan_impl']!r} (source: {sync_report['scan_impl_source']})")
print(f"hero_descriptor_type = {sync_report['hero_descriptor_type']!r}")
print(f"ACTIVE_DATASETS = {sync_report['active_datasets']}")
print(f"in-scope jobs: {len(sync_report['in_scope'])} ({len(sync_report['newly_enqueued'])} newly enqueued this run)")
if sync_report["out_of_scope_datasets"]:
    print(f"out-of-scope datasets (no jobs enqueued): {sync_report['out_of_scope_datasets']} ({sync_report['n_out_of_scope_configs']} config(s) skipped)")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (f) short summary: PENDING/RUNNING/DONE/FAILED job counts broken down by
# dataset, plus which datasets are currently skipped -- so a glance at this
# cell tells you exactly what's queued and what isn't, every session.
from lcmunet.data import raw_layout as rl
from lcmunet.config import ACTIVE_DATASETS
from lcmunet.run_manifest import manifest_status_counts_by_dataset

counts_by_dataset = manifest_status_counts_by_dataset(paths)
print("=== Job queue summary (results/manifest.json) ===")
print(f"{'dataset':<14} {'PENDING':>8} {'RUNNING':>8} {'DONE':>8} {'FAILED':>8}")
for dataset in rl.DATASET_NAMES:
    if dataset not in ACTIVE_DATASETS:
        print(f"{dataset:<14} {'SKIPPED (not in ACTIVE_DATASETS)':>34}")
        continue
    c = counts_by_dataset.get(dataset, {})
    print(f"{dataset:<14} {c.get('PENDING', 0):>8} {c.get('RUNNING', 0):>8} {c.get('DONE', 0):>8} {c.get('FAILED', 0):>8}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# (g) single call that drives the job queue (results/manifest.json on Drive).
# Discovering/enqueuing work above is automatic; actually running hours of
# training is a separate, deliberate action -- call run_all_pending()
# yourself in a new cell/session when you're ready, it is never invoked
# automatically by Run All.
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


print("Bridge ready. Call run_all_pending() yourself, in a new cell, when you're ready to spend GPU-hours.")
"""
))

nb["cells"] = cells
with open("notebooks/colab_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/colab_runner.ipynb")
