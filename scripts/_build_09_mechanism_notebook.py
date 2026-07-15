"""One-off generator for notebooks/09_mechanism.ipynb.
Run with: python scripts/_build_09_mechanism_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 09: Mechanism analysis and visualisation (methodology section 11)

Run All top to bottom in Colab. **GPU gate**: the baseline side of this
notebook is the REAL trained `ultralight_baseline` checkpoint (the vendored
UltraLight_VM_UNet, using `mamba_ssm.Mamba` directly) -- it needs mamba-ssm's
CUDA build (GLOBAL RULES rule 5) both to have been TRAINED (Phase-1) and to
be re-loaded here. The hero (`lc_ss2d`) side is CPU-runnable, but this
notebook still requires the real trained Phase-1 checkpoints for BOTH, so it
needs Phase-1 to be DONE either way.

**Uses Phase-1's fixed Ablation-A reference rows** (`lcmunet.gate2_report.
REQUIRED_ROLES["hero"]` / `["baseline_pvm"]`) by default -- that hero row's
descriptor_type is ALWAYS `'contrast'` by construction regardless of the
Gate-2 Desc-ablation winner used later for Phase-2, so this notebook does
NOT require Gate-2 to have run, let alone PROCEED -- only that Phase-1
training finished.

Produces exactly the 3 methodology section 11 deliverables, no more:
1. `figures/delta_difference_map.png` -- Delta(ours) - Delta(baseline).
2. `figures/region_wise_modulation.png` + `results/region_wise_modulation_stats.csv`.
3. `figures/per_stage_alpha.png` (from the training-time alpha log).

...and `results/mechanism_report.md` summarising all three, with the
methodology section 11 disclaimer.
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
# already installed. REQUIRED for the baseline side of this notebook.
import subprocess

result = subprocess.run(["pip", "install", "-q", "causal-conv1d>=1.1.0", "mamba-ssm"],
                         capture_output=True, text=True, timeout=1800)
print("mamba-ssm install return code:", result.returncode)
print((result.stdout + result.stderr)[-2000:])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Run the local pytest suite for the modules this notebook exercises.
import subprocess

result = subprocess.run(
    ["python", "-m", "pytest", "-q", "tests/test_mechanism_analysis.py", "tests/test_delta_diff.py", "tests/test_efficiency.py"],
    capture_output=True, text=True,
)
print(result.stdout[-4000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed -- see output above"
print("pytest: all green.")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Confirm SCAN_IMPL == "cuda" -- required to even BUILD ultralight_baseline
# (the vendored PVMLayer uses mamba_ssm.Mamba directly, no CPU fallback).
from lcmunet.paths import get_paths
from lcmunet.env_report import write_env_json
from lcmunet.scan import SCAN_IMPL
from lcmunet import scan as scan_module
from lcmunet.data.download import ensure_kvasir
from lcmunet.data.splits import build_kvasir_split

paths = get_paths()
write_env_json(paths.results, repo_root=".")
ensure_kvasir(paths.data_raw)
build_kvasir_split(paths)

print("SCAN_IMPL =", SCAN_IMPL)
if SCAN_IMPL != "cuda":
    print("mamba-ssm import error:", repr(scan_module._CUDA_IMPORT_ERROR))
    raise RuntimeError(
        "SCAN_IMPL != 'cuda' -- the baseline (ultralight_baseline) checkpoint "
        "cannot be built at all without it. Fix the mamba-ssm build "
        "(see notebooks/01_env.ipynb) first."
    )
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Confirm the Phase-1 hero + baseline checkpoints exist before generating
# the report -- a clear, named error otherwise (rather than a confusing
# failure deep inside model-loading).
from lcmunet import experiment_matrix as em
from lcmunet.engine import checkpoint_dir
from lcmunet.gate2_report import REQUIRED_ROLES

scan_impl, _source = em.resolve_scan_impl(paths)
phase1_rows = dict(em.build_phase1_kvasir(scan_impl))
hero_config = phase1_rows[REQUIRED_ROLES["hero"]]
baseline_config = phase1_rows[REQUIRED_ROLES["baseline_pvm"]]

for label, config in (("hero", hero_config), ("baseline", baseline_config)):
    ckpt_path = checkpoint_dir(paths, config) / "best.pt"
    print(label, "->", ckpt_path, "EXISTS" if ckpt_path.is_file() else "MISSING -- run notebooks/06_phase1_gate2.ipynb first")
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Generate all 3 methodology section 11 deliverables + results/mechanism_report.md.
from lcmunet.mechanism_analysis import generate_mechanism_report

result = generate_mechanism_report(paths, n_images=8)
for key, path in result.items():
    print(f"{key}: {path}")
print()
print(result["mechanism_report_md"].read_text(encoding="utf-8"))
print()
print(">>> PASTE BACK: the full mechanism_report.md content printed above, plus confirm the 3 figure files exist <<<")
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- pytest output (must be all green).
- `SCAN_IMPL` value and GPU name.
- Confirmation the hero + baseline checkpoints were found.
- The full `results/mechanism_report.md` content printed above.
"""
))

nb["cells"] = cells
with open("notebooks/09_mechanism.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/09_mechanism.ipynb")
