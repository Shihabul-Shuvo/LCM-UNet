"""One-off generator for notebooks/06_phase1_gate2.ipynb.
Run with: python scripts/_build_06_phase1_gate2_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 06: Phase-1 (Kvasir-SEG, 1 seed) + Gate-2 decision

Run All top to bottom in Colab. **Resumable**: run_queue() is filtered to
ONLY the 10 Phase-1 config_ids (methodology section 10.1/10.2, 13) -- if
this session dies partway through, reopen and Run All again; it picks up
exactly where it left off (RUNNING jobs from a dead session are reclaimed
after 30 min, PENDING jobs are untouched either way).

**GPU needed for one row only.** 9 of the 10 Phase-1 rows (`lc_ss2d` hero +
4 ablation variants, `glgf`) are CPU-runnable. The remaining row
(`ultralight_baseline`, "Baseline PVM (reproduced)") needs mamba-ssm's CUDA
build (GLOBAL RULES rule 5) -- confirm `SCAN_IMPL == "cuda"` below before
this notebook can produce a complete Gate-2 report; `lcmunet/gate2_report.py`
will raise a clear error naming exactly which row(s) are still missing if
you try to generate the report early.

**This notebook does NOT auto-enqueue or run Phase-2.** Gate-2's decision
(PROCEED/PIVOT) is for you and your supervisor to review; Phase-2 is many
multi-hour runs and should only be enqueued after that review.
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
'''# Best-effort mamba-ssm (+ causal-conv1d) install, in case this is a fresh
# runtime that never ran notebooks/01_env.ipynb. Safe to re-run; a no-op if
# already installed. Needed only for the ultralight_baseline row.
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
     "tests/test_delta_diff.py", "tests/test_gate2_report.py", "tests/test_glgf.py"],
    capture_output=True, text=True,
)
print(result.stdout[-4000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed -- see output above"
print("pytest: all green.")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# env.json (records SCAN_IMPL for provenance -- section 5.5 fairness rule)
# + ensure Kvasir-SEG is present and its split is built.
from lcmunet.paths import get_paths
from lcmunet.env_report import write_env_json
from lcmunet.data.download import ensure_kvasir
from lcmunet.data.splits import build_kvasir_split

paths = get_paths()
env_info = write_env_json(paths.results, repo_root=".")
ensure_kvasir(paths.data_raw)
split = build_kvasir_split(paths)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
print("SCAN_IMPL:", __import__("lcmunet.scan", fromlist=["SCAN_IMPL"]).SCAN_IMPL)
print("kvasir split counts:", split["counts"])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Populate/refresh the full manifest (idempotent -- config_id-keyed, never
# duplicates). Uses the real scan_impl just recorded in results/env.json.
import subprocess

result = subprocess.run(["python", "scripts/enqueue_all.py"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr[-2000:])
assert result.returncode == 0, "enqueue_all.py failed -- see output above"
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Drive run_queue over ONLY the Phase-1 config_ids (methodology section
# 10.1/10.2, 13) -- Phase-2 jobs (also PENDING in the manifest by now) are
# left completely untouched by this filtered run.
from lcmunet.run_manifest import run_queue
from lcmunet.engine import run_one
from lcmunet.gate2_report import phase1_config_ids

phase1_ids = phase1_config_ids(paths)
print(f"Phase-1 config_ids to run: {len(phase1_ids)}")

def runner_fn(job):
    run_one(job, paths=paths)

run_queue(paths.results, runner_fn, max_minutes=600, job_filter=lambda job: job["config_id"] in phase1_ids)
print("run_queue returned (either Phase-1 is fully DONE, or this session\\'s time budget ran out --")
print("if the latter, just Run All again; PENDING/reclaimed jobs resume automatically).")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Check Phase-1 status before generating the report.
import json

manifest = json.load(open(paths.results / "manifest.json"))
phase1_jobs = {cid: job for cid, job in manifest["jobs"].items() if cid in phase1_ids}
by_status = {}
for job in phase1_jobs.values():
    by_status.setdefault(job["status"], 0)
    by_status[job["status"]] += 1
print("Phase-1 job status counts:", by_status)
for cid, job in phase1_jobs.items():
    if job["status"] != "DONE":
        print(" -", job["status"], cid, job.get("error"))
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Generate results/gate2_report.md (methodology section 13 Phase-1 decision
# rules). Raises a clear error naming exactly which row(s) are still
# missing if Phase-1 isn\\'t fully DONE yet -- rerun the run_queue cell above
# until this succeeds.
from lcmunet.gate2_report import generate_gate2_report

report_path = generate_gate2_report(paths)
print(report_path.read_text(encoding="utf-8"))
print()
print(">>> PASTE BACK: the full gate2_report.md content printed above <<<")
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- pytest output (must be all green).
- `SCAN_IMPL` value and the Phase-1 job status counts.
- The full `results/gate2_report.md` content printed above.

## What happens next

This notebook stops here. Read the DECISION (PROCEED or PIVOT) and the
per-rule table in `gate2_report.md`, discuss with your supervisor, and only
THEN decide whether to run Phase-2 (`scripts/enqueue_all.py` already
enqueued its 60 rows as PENDING -- driving `run_queue` without a Phase-1
`job_filter` picks them up). Nothing in this repository auto-proceeds to
Phase-2 on your behalf.
"""
))

nb["cells"] = cells
with open("notebooks/06_phase1_gate2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/06_phase1_gate2.ipynb")
