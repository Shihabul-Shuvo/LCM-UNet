"""One-off generator for notebooks/05_lc_ss2d_sanity.ipynb.
Run with: python scripts/_build_05_lc_ss2d_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 05: LC-SS2D Step-0 Audit + Sanity Training

Run All top to bottom in Colab. Unlike ultralight_baseline, **lc_ss2d does
NOT need mamba-ssm or a GPU** -- lcmunet/lc_vss.py reimplements the Mamba
forward pass directly (no mamba_ssm import at all), using
lcmunet.scan.selective_scan (whatever SCAN_IMPL locked to). Everything in
this notebook already passed on the agent's CPU-only dev machine; this run
is to confirm it also holds on your Colab GPU runtime and to get a longer,
real-hardware sanity signal on alpha behaviour.

**Read this first: three documentation-vs-code discrepancies were found and
resolved while building this (not silently -- each is flagged in code
comments and needs your/your supervisor's confirmation):**

1. The vendored PVMLayer implements 4-way CHANNEL-GROUP parallelism (one
   shared Mamba applied to 4 channel slices), not VMamba-style 4-directional
   spatial cross-scanning. Methodology section 5.3's own operational
   language ("slice M into num_groups parts along the channel dimension")
   matches this; section 3.1's "4 scan directions" is read as the
   mechanism's general framing. See lcmunet/lc_vss.py's module docstring.
2. Methodology's stage table lists "D3-D1: Conv blocks" but the real
   vendored code has decoder3 (D3) as a PVMLayer. Implemented to match the
   real code (D3 = plain/inherited PVM, never LC-VSS) rather than the
   table. See lcmunet/lcm_unet.py's module docstring.
3. The stage table's channel numbers (E4=32, E5=48, D5=48, D4=32) match
   each stage's OUTPUT width; LC-VSS operates on each block's actual INPUT
   width (E4=24, E5=32, D5=64, D4=48), which is what section 3.1's
   "Xn = LayerNorm(X)" and the "PVM operator...inherited, unchanged"
   constraint (section 4) require. See lcmunet/audit.py's item 4 report.

If any of these three readings is wrong, everything downstream (ablation
rows, efficiency numbers) inherits the error -- please confirm or correct
before Phase-1 ablations (section 13/14 Week 3).
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
'''# Run the full local pytest suite for lc_vss/engine/audit first.
import subprocess

result = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_lc_vss.py", "tests/test_engine.py"],
                         capture_output=True, text=True)
print(result.stdout[-4000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed -- see output above"
print("pytest: all green.")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Step-0 audit (methodology section 5.1 / section 16), items 1-6, against a
# fresh hero-placement LCMUNet. Raises AssertionError on the first failure
# (fail loud) -- if this cell errors, STOP, do not proceed to training.
from lcmunet.audit import run_step0_audit

report = run_step0_audit(verbose=True)
assert report["all_6_items_passed"]
print()
print(">>> PASTE BACK: the full JSON report printed above <<<")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Ensure Kvasir-SEG is present and build its split.
from lcmunet.paths import get_paths
from lcmunet.data.download import ensure_kvasir
from lcmunet.data.splits import build_kvasir_split

paths = get_paths()
ensure_kvasir(paths.data_raw)
build_kvasir_split(paths)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
print("SCAN_IMPL:", __import__("lcmunet.scan", fromlist=["SCAN_IMPL"]).SCAN_IMPL)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# A few real sanity epochs on the full Kvasir split, model_name="lc_ss2d",
# alpha logged every 10 epochs (default cadence) -- lower it here for a
# short run so you actually see it fire within this sanity check.
import lcmunet.engine as engine_mod
from lcmunet.config import RunConfig
from lcmunet.engine import run_one

engine_mod.ALPHA_LOG_EVERY = 5

config = RunConfig(
    run_name="lc_ss2d_colab_sanity",
    model_name="lc_ss2d",
    dataset="kvasir_seg",
    seed=42,
    split_file="splits/kvasir_seg.json",
    epochs=20,
    batch_size=8,
    input_size=256,
)
result = run_one(config, paths=paths)
print(result)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Show the per-stage alpha trajectory.
import pandas as pd

alpha_csv = paths.results / "alpha" / f"{config.config_id}_{config.seed}.csv"
print(pd.read_csv(alpha_csv))
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- pytest output (must be all green).
- The full Step-0 audit JSON report.
- The 20-epoch sanity run's `result` dict (loss/Dice trend, completed=True).
- The per-stage alpha CSV contents.
- Confirmation (or correction) on the three discrepancies listed at the top.
"""
))

nb["cells"] = cells
with open("notebooks/05_lc_ss2d_sanity.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/05_lc_ss2d_sanity.ipynb")
