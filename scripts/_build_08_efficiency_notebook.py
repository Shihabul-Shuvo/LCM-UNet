"""One-off generator for notebooks/08_efficiency.ipynb.
Run with: python scripts/_build_08_efficiency_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 08: Efficiency (Params, GFLOPs, FPS, peak GPU memory)

Run All top to bottom in Colab. **This is a GPU gate for 2 reasons:**
1. `ultralight_baseline` needs mamba-ssm's CUDA build (GLOBAL RULES rule 5)
   -- cannot be measured at all without it.
2. FPS and peak training GPU memory are, by definition, GPU facts -- a CPU
   number is not a valid substitute for any of the 6 models, even the 5
   that happen to be CPU-runnable for training.

**Params and GFLOPs were already measured locally (CPU) for the 5
CPU-runnable models** (glgf, lc_ss2d, unet, malunet, egeunet) -- these are
structural counts, not timing measurements, so they do not depend on device
or scan_impl and do NOT need to be re-measured here; this notebook
re-measures them anyway for a single self-contained `efficiency.csv` with
every column filled from ONE run. `ultralight_baseline`'s Params/GFLOPs can
ONLY come from this notebook (mamba-ssm required just to build it).

**Same scan_impl for every model (section 5.5 fairness rule).** This
notebook asserts `SCAN_IMPL == "cuda"` before measuring anything --
otherwise `ultralight_baseline` cannot be built at all, and a table missing
one of the compared models is not "every compared model under identical
conditions."
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
# runtime that never ran notebooks/01_env.ipynb. Safe to re-run.
import subprocess

result = subprocess.run(["pip", "install", "-q", "causal-conv1d>=1.1.0", "mamba-ssm"],
                         capture_output=True, text=True, timeout=1800)
print("mamba-ssm install return code:", result.returncode)
print((result.stdout + result.stderr)[-2000:])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Run the local pytest suite for the efficiency modules.
import subprocess

result = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_efficiency.py", "tests/test_efficiency_report.py"],
                         capture_output=True, text=True)
print(result.stdout[-4000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed -- see output above"
print("pytest: all green.")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Confirm SCAN_IMPL == "cuda" -- required for ultralight_baseline to build
# at all, and for a valid "same scan_impl across every model" comparison.
from lcmunet.paths import get_paths
from lcmunet.env_report import write_env_json
from lcmunet.scan import SCAN_IMPL
from lcmunet import scan as scan_module

paths = get_paths()
write_env_json(paths.results, repo_root=".")

print("SCAN_IMPL =", SCAN_IMPL)
if SCAN_IMPL != "cuda":
    print("mamba-ssm import error:", repr(scan_module._CUDA_IMPORT_ERROR))
    raise RuntimeError(
        "SCAN_IMPL != 'cuda' -- this efficiency run cannot proceed: "
        "ultralight_baseline cannot be built at all without it, and every "
        "OTHER model would then be measured under a different scan_impl "
        "than baseline, which section 5.5 says is an invalid comparison. "
        "Fix the mamba-ssm build (see notebooks/01_env.ipynb) first."
    )
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Measure Params / GFLOPs (thop + supplementary scan-GFLOPs) / FPS (batch=1,
# batch=8, 20 warm-up + 100 measured passes) / peak training GPU memory
# (batch=8) for EVERY compared model, all under the SAME scan_impl just
# confirmed above. Writes results/efficiency.csv + results/efficiency_report.md.
from lcmunet.efficiency_report import generate_efficiency_report

result = generate_efficiency_report(
    paths, input_size=256, fps_batch_sizes=(1, 8), memory_batch_size=8,
    n_warmup=20, n_measure=100,
)
print(f"Measured {len(result[\\'rows\\'])}/6 models.")
if result["errors"]:
    print("NOT measured (see efficiency_report.md for why):", list(result["errors"]))
print()
print(result["efficiency_report_md"].read_text(encoding="utf-8"))
print()
print(">>> PASTE BACK: the full efficiency_report.md content printed above <<<")
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- pytest output (must be all green).
- `SCAN_IMPL` value (must be `"cuda"`) and GPU name / torch version.
- How many of 6 models were measured (should be 6/6 -- if not, paste the
  error(s) too).
- The full `results/efficiency_report.md` content printed above, including
  the LC-SS2D-vs-GLGF params claim check (CONFIRMED or FALSE).
"""
))

nb["cells"] = cells
with open("notebooks/08_efficiency.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/08_efficiency.ipynb")
