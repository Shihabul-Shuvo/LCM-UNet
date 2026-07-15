"""One-off generator for notebooks/04_gate1_baseline.ipynb.
Run with: python scripts/_build_04_gate1_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 04: Gate 1 -- Reproduce UltraLight VM-UNet on Kvasir-SEG

**This is a GPU gate (GLOBAL RULES rule 5) -- must be run in Colab.** The
vendored UltraLight_VM_UNet's PVMLayer uses `mamba_ssm.Mamba` directly,
which has no CPU fallback at all (both its fused and non-fused internal
paths require the compiled `selective_scan_cuda` extension). This cannot be
verified locally.

**Resumable.** This trains for up to 250 epochs, which will likely outlive
one Colab session. That's fine: every epoch is checkpointed
(model/optimizer/scheduler/scaler/RNG state). If the session dies, just
reopen this notebook and Run All again -- `run_one()` detects the existing
checkpoint and continues from the next epoch, not from scratch.

## IMPORTANT: no published UltraLight-on-Kvasir number exists

Gate 1 as specified says "reproduce the published Dice within ~0.5%." We
searched the vendored repo (`third_party/UltraLight-VM-UNet/`, grep for
"kvasir"/"polyp"/"CVC": zero matches) and the paper itself: **UltraLight
VM-UNet's own paper only reports results on ISIC2017/2018/PH2 (skin
lesion). It never evaluated on Kvasir-SEG (polyp).** The 86-91% Dice
numbers that exist for Kvasir-SEG in the literature belong to a
*different* model family (VM-UNet / VM-UNetV2 -- different authors,
different architecture), not UltraLight VM-UNet.

This notebook therefore cannot mechanically check "within ~0.5% of
published." It trains the real, unmodified baseline under the locked
protocol and reports the resulting Dice -- pass/fail against the ~0.5%
criterion needs a human decision about what Gate 1 should actually mean
here (see the final cell for options). Do not treat any number below as an
automatic PASS.
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
# already installed.
import subprocess

result = subprocess.run(["pip", "install", "-q", "causal-conv1d>=1.1.0", "mamba-ssm"],
                         capture_output=True, text=True, timeout=1800)
print("mamba-ssm install return code:", result.returncode)
print((result.stdout + result.stderr)[-2000:])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Confirm the scan-impl lock resolved to "cuda". Gate 1 CANNOT proceed on "ref"
# -- the vendored PVMLayer has no CPU/ref path at all.
from lcmunet.scan import SCAN_IMPL
from lcmunet import scan as scan_module

print("SCAN_IMPL =", SCAN_IMPL)
if SCAN_IMPL != "cuda":
    print("mamba-ssm import error:", repr(scan_module._CUDA_IMPORT_ERROR))
    raise RuntimeError(
        "SCAN_IMPL != 'cuda' -- Gate 1 cannot proceed. Fix the mamba-ssm "
        "build (see notebooks/01_env.ipynb) before continuing."
    )
print("GPU:", __import__("torch").cuda.get_device_name(0))
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Ensure Kvasir-SEG is present (auto-downloads if missing) and build its split.
from lcmunet.paths import get_paths
from lcmunet.data.download import ensure_kvasir
from lcmunet.data.splits import build_kvasir_split

paths = get_paths()
ensure_kvasir(paths.data_raw)
split = build_kvasir_split(paths)
print(split["counts"])
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Smoke-test the vendored, unmodified UltraLight_VM_UNet actually forward/
# backward passes on THIS machine before committing to a long training run.
import torch

from lcmunet.adapters import LogitsAdapter
from lcmunet.backbone import load_ultralight_vmunet

model = LogitsAdapter(load_ultralight_vmunet()).cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params} ({n_params/1e6:.4f} M)")

x = torch.randn(2, 3, 256, 256, device="cuda", requires_grad=True)
y = model(x)
print("output shape:", tuple(y.shape))
y.sum().backward()
print("grad finite:", torch.isfinite(x.grad).all().item())
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Gate 1 training config: built via lcmunet.experiment_matrix.build_phase1_kvasir
# (the SAME single source of truth scripts/enqueue_all.py uses for the
# Phase-1 experiment matrix -- see docs/LCM-UNet_FINAL_methodology_v4.1.md
# section 6/10.1 and the Fairness rule), NOT a separately hand-written
# RunConfig. This guarantees this Gate 1 baseline run is byte-for-byte the
# SAME config_id as Phase-1\'s "phase1_ablation_A_baseline_pvm" row -- so if
# you run this notebook before Phase-1, Phase-1\'s run_queue will see it
# already DONE and skip retraining it (no wasted GPU-hours); if you run
# Phase-1 first, running this notebook afterwards is a no-op resume, not a
# duplicate. scan_impl is resolved the same way scripts/enqueue_all.py does
# (prefers results/env.json, written by notebooks/01_env.ipynb) so this
# only matches Phase-1 when both were run on a machine where mamba-ssm
# actually built (SCAN_IMPL == "cuda") -- required anyway for
# ultralight_baseline to run at all (see the SCAN_IMPL check above).
from lcmunet.experiment_matrix import build_phase1_kvasir, resolve_scan_impl

scan_impl, scan_impl_source = resolve_scan_impl(paths)
print("scan_impl:", scan_impl, "(source:", scan_impl_source, ")")

phase1_rows = dict(build_phase1_kvasir(scan_impl))
config = phase1_rows["phase1_ablation_A_baseline_pvm"]
print("config_id:", config.config_id)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Train (or resume). Safe to re-run this cell / the whole notebook if the
# session dies -- it will pick up from the last checkpointed epoch.
from lcmunet.engine import run_one

result = run_one(config, paths=paths)
print(result)
'''
))

cells.append(nbf.v4.new_code_cell(
'''# Report the final Dice and the (non-mechanical) Gate 1 read.
test_dsc = result["test_metrics"]["dsc"]
print(f"Final Kvasir-SEG test Dice: {test_dsc:.4f} ({test_dsc*100:.2f}%)")
print()
print("No published UltraLight-on-Kvasir Dice exists to compute a ~0.5% delta")
print("against (see the first cell). For loose context only -- NOT a valid")
print("Gate 1 target, different architecture/paper -- VM-UNet reports 86.21%")
print("and VM-UNetV2 reports 90.75% DSC on Kvasir-SEG.")
print()
print(">>> PASTE THIS BACK: final test Dice =", test_dsc, "<<<")
'''
))

cells.append(nbf.v4.new_markdown_cell(
"""## What to paste back

- `SCAN_IMPL` value (must be `"cuda"`).
- The UltraLight smoke-test cell's output shape / param count / grad-finite result.
- `result` from the training cell (`completed`, `reached_epoch`, `test_metrics`).
- The final Dice number.

## Gate 1 decision needed from you / your supervisor

Because no published UltraLight-on-Kvasir number exists, "PASS/FAIL within
~0.5%" as literally specified cannot be computed. Once you have the number
above, pick how to resolve this (not decided here, per GLOBAL RULES rule 1
-- this is a methodology interpretation call, not an implementation one):

1. **Reinterpret Gate 1** as "the reproduced baseline trains stably and
   reaches a reasonable Dice for this architecture/dataset" (sanity gate,
   not a numeric-match gate), and proceed.
2. **Add ISIC2017/2018 as a true Gate 1** (UltraLight *did* publish on
   those -- 0.9091 DSC for the quadruple-parallel config in their README)
   in addition to or instead of Kvasir, since that is the comparison the
   original paper actually supports.
3. **Treat VM-UNet's 86.21% as a loose sanity anchor** (explicitly caveated
   as a different architecture) rather than a strict target.

U-Net, EGE-UNet, and MALUNet builders do NOT need Colab -- they are
CPU-runnable and already verified locally (`tests/test_comparators.py`,
forward/backward + a real 2-epoch training run on Kvasir data).
"""
))

nb["cells"] = cells
with open("notebooks/04_gate1_baseline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/04_gate1_baseline.ipynb")
