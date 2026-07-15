"""One-off generator for notebooks/07_phase2.ipynb.
Run with: python scripts/_build_07_phase2_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 07: Phase-2 (multi-seed headline + ISIC + cross-dataset) + statistics

Run All top to bottom in Colab. **This notebook refuses to run Phase-2 jobs
unless Gate-2's decision is PROCEED** -- `lcmunet.gate2_report.
require_gate2_proceed` re-derives the Gate-2 decision FRESH from current
`results.csv` state every time this is run (never trusting a possibly-stale
`gate2_report.md` file) and raises a clear error otherwise. If you have not
yet run `notebooks/06_phase1_gate2.ipynb` to a PROCEED decision, this
notebook will stop at the gate cell below -- that is correct behaviour, not
a bug, per this prompt's own instruction: "Do NOT auto-proceed to Phase-2;
require the user's confirmation."

**Resumable.** Phase-2 is 60 config rows across up to 5 seeds each -- many
multi-hour runs on Colab free tier. `run_queue()` is filtered to ONLY
Phase-2 config_ids and checkpoints every epoch; if this session dies
partway through, reopen and Run All again.

**GPU needed for `ultralight_baseline` rows only** (mamba-ssm's CUDA
build). `glgf`, `lc_ss2d`, `unet`, `malunet`, `egeunet` are all
CPU-runnable, though obviously much faster on a GPU.

**Desc ablation winner.** Gate-2's `desc_winner` is read from the FRESH
decision (not the file) and used as the fixed hero descriptor for every
Phase-2 hero row (methodology section 10.1: "winner used in all later
results") -- `scripts/enqueue_all.py --hero-descriptor <winner>` is called
automatically below.
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
'''# Best-effort mamba-ssm (+ causal-conv1d) install. Safe to re-run; a no-op
# if already installed. Needed only for the ultralight_baseline rows.
import subprocess

result = subprocess.run(["pip", "install", "-q", "causal-conv1d>=1.1.0", "mamba-ssm"],
                         capture_output=True, text=True, timeout=1800)
print("mamba-ssm install return code:", result.returncode)
print((result.stdout + result.stderr)[-2000:])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Run the full local pytest suite for the modules this notebook exercises.
import subprocess

result = subprocess.run(
    ["python", "-m", "pytest", "-q",
     "tests/test_experiment_matrix.py", "tests/test_run_manifest.py",
     "tests/test_gate2_report.py", "tests/test_stats.py",
     "tests/test_cross_dataset_eval.py", "tests/test_phase2_report.py",
     "tests/test_enqueue_all.py"],
    capture_output=True, text=True,
)
print(result.stdout[-6000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed -- see output above"
print("pytest: all green.")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# env.json + ensure Kvasir-SEG, CVC-ClinicDB, ISIC2017, ISIC2018 are present
# and their splits are built (Phase-2 needs all four datasets; Phase-1 only
# needed Kvasir). Idempotent -- fast no-op if already ready in Drive (see
# lcmunet/data/prepare_all.py, also called automatically every session by
# notebooks/colab_runner.ipynb).
from lcmunet.paths import get_paths
from lcmunet.env_report import write_env_json
from lcmunet.data.prepare_all import prepare_all_datasets

paths = get_paths()
write_env_json(paths.results, repo_root=".")

data_report = prepare_all_datasets(paths)
assert all(row["status"] == "PASS" for row in data_report.values()), (
    "Phase-2 needs all four datasets ready -- see the FAILED rows above for "
    "exactly what to fix, then re-run this cell."
)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
print("SCAN_IMPL:", __import__("lcmunet.scan", fromlist=["SCAN_IMPL"]).SCAN_IMPL)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# HARD GATE -- re-derives the Gate-2 decision FRESH from current
# results.csv state and RAISES unless it is PROCEED. Do not skip, do not
# catch-and-continue past this cell: a PIVOT here means Phase-1's ablation
# rows did not support proceeding to Phase-2 (see results/gate2_report.md
# for the exact reason and named next step).
from lcmunet.gate2_report import require_gate2_proceed

gate2_rules = require_gate2_proceed(paths)
desc_winner = gate2_rules["desc_winner"]
print("Gate-2: PROCEED confirmed.")
print("Desc ablation winner (fixed hero descriptor for all Phase-2 hero rows):", desc_winner)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Populate/refresh the manifest with the Desc winner baked into Phase-2\\'s
# hero rows (idempotent -- config_id-keyed, never duplicates; Phase-1\\'s
# rows are completely unaffected by --hero-descriptor).
import subprocess

result = subprocess.run(["python", "scripts/enqueue_all.py", "--hero-descriptor", desc_winner],
                         capture_output=True, text=True)
print(result.stdout)
print(result.stderr[-2000:])
assert result.returncode == 0, "enqueue_all.py failed -- see output above"
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Drive run_queue over ONLY the Phase-2 config_ids (60 rows: headline
# Kvasir+CVC 5 seeds, ISIC 3 seeds, comparators 3 seeds) -- Phase-1 jobs
# (already DONE) are left untouched by this filtered run.
from lcmunet.run_manifest import run_queue
from lcmunet.engine import run_one
from lcmunet.experiment_matrix import phase2_config_ids, resolve_scan_impl

scan_impl, _source = resolve_scan_impl(paths)
phase2_ids = phase2_config_ids(scan_impl, hero_descriptor_type=desc_winner)
print(f"Phase-2 config_ids to run: {len(phase2_ids)}")

def runner_fn(job):
    run_one(job, paths=paths)

run_queue(paths.results, runner_fn, max_minutes=600, job_filter=lambda job: job["config_id"] in phase2_ids)
print("run_queue returned (either Phase-2 is fully DONE, or this session\\'s time budget ran out --")
print("if the latter, just Run All again; PENDING/reclaimed jobs resume automatically).")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Check Phase-2 status before evaluating cross-dataset / generating the report.
import json

manifest = json.load(open(paths.results / "manifest.json"))
phase2_jobs = {cid: job for cid, job in manifest["jobs"].items() if cid in phase2_ids}
by_status = {}
for job in phase2_jobs.values():
    by_status.setdefault(job["status"], 0)
    by_status[job["status"]] += 1
print("Phase-2 job status counts:", by_status)
for cid, job in phase2_jobs.items():
    if job["status"] != "DONE":
        print(" -", job["status"], cid, job.get("error"))
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Cross-dataset generalisation (methodology section 7 "required"; section
# 14 week 5): evaluates the ALREADY-TRAINED Kvasir/CVC headline checkpoints
# (hero + baseline) on the OTHER dataset\\'s test split. Not a training job --
# reuses lcmunet.data.loaders.build_cross_dataset_loader. Resumable/
# idempotent (upserts one row at a time into results/cross_dataset_results.csv).
from lcmunet.cross_dataset_eval import run_cross_dataset_suite

cross_df = run_cross_dataset_suite(paths, hero_descriptor_type=desc_winner)
print(cross_df)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Generate results/phase2_summary.csv + results/stats_report.md (methodology
# section 8: mean+/-std, 95% CI, paired Wilcoxon, effect sizes). Re-checks
# Gate-2 PROCEED itself (defence in depth) and raises a clear error naming
# exactly which Phase-2 row(s) are still missing if it isn\\'t fully DONE yet.
from lcmunet.phase2_report import generate_phase2_summary

result_paths = generate_phase2_summary(paths)
print(result_paths["stats_report_md"].read_text(encoding="utf-8"))
print()
print(">>> PASTE BACK: the full stats_report.md content printed above <<<")
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- pytest output (must be all green).
- `desc_winner` and confirmation Gate-2 said PROCEED.
- The Phase-2 job status counts (all should be DONE).
- `cross_df` (cross-dataset results).
- The full `results/stats_report.md` content printed above.
"""
))

nb["cells"] = cells
with open("notebooks/07_phase2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/07_phase2.ipynb")
