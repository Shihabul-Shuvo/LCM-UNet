"""One-off generator for notebooks/01_env.ipynb. Not part of the package;
kept here so the notebook can be regenerated deterministically if edited.
Run with: python scripts/_build_01_env_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 01: Environment Setup + Selective-Scan Lock

Run All top to bottom in Colab (T4 runtime). This notebook:
1. Installs the infra + science-stack requirements (Colab-safe -- never
   force-reinstalls torch).
2. Attempts the mamba-ssm CUDA build (`selective_scan_fn`, non-fused). If it
   fails or is flaky, `lcmunet/scan.py` falls back to a pure-PyTorch
   reference scan automatically -- either way the choice is recorded, not
   guessed.
3. Runs a numerical equivalence check (cuda vs ref) if both are available.
4. Smoke-tests the vendored, unmodified UltraLight VM-UNet backbone on a
   random batch (forward shape, param count, finite backward gradients).
5. Writes `results/env.json` on Drive: torch/CUDA/GPU/SCAN_IMPL -- the single
   recorded source of truth every downstream efficiency number depends on.

**GPU gate note (GLOBAL RULES rule 5):** whether mamba-ssm's CUDA build
actually works can only be verified by *you*, here, in Colab. Paste the
printed `SCAN_IMPL` result back -- nothing downstream proceeds without it.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Mount Drive + pull this repo fresh (no hand-copied code, ever).
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
print("DRIVE_ROOT:", DRIVE_ROOT)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Install requirements. torch/torchvision are unpinned in requirements.txt
# specifically so this is a no-op against Colab's pre-built CUDA torch.
%pip install -q -r requirements.txt
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Best-effort mamba-ssm (+ causal-conv1d) install. This is the Gate-0 GPU
# gate (methodology section 5.1 / section 13) -- allowed to fail. If it fails
# or is flaky, lcmunet/scan.py falls back to the pure-PyTorch reference scan
# and every comparison uses that instead, uniformly (methodology section 5.5).
#
# --no-build-isolation is required: both packages' setup.py does `import
# torch` at build time (to detect the CUDA/ABI version to build against), but
# pip's default isolated build sandbox does NOT have torch installed in it
# (only this outer environment does) -- without this flag the build fails
# with a generic "Getting requirements to build wheel did not run
# successfully", which hides a ModuleNotFoundError: No module named 'torch'
# a few lines up in the real log and looks like an unrelated problem.
import subprocess

newline = chr(10)
mamba_install_log = ""
try:
    subprocess.run(
        ["pip", "install", "-q", "packaging", "ninja", "wheel", "setuptools"],
        capture_output=True, text=True, timeout=300,
    )
    result = subprocess.run(
        ["pip", "install", "-q", "--no-build-isolation", "causal-conv1d>=1.1.0", "mamba-ssm"],
        capture_output=True, text=True, timeout=1800,
    )
    mamba_install_log = result.stdout[-6000:] + newline + result.stderr[-6000:]
    print(f"mamba-ssm install return code: {result.returncode}")
    print(mamba_install_log[-3000:])
except Exception as exc:
    mamba_install_log = repr(exc)
    print("mamba-ssm install raised:", exc)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Resolve and print the locked scan implementation. IMPORTANT: this import
# must happen in a FRESH kernel/process after the install cell above, since
# lcmunet.scan makes its cuda-vs-ref decision once, at import time.
from lcmunet.scan import SCAN_IMPL
from lcmunet import scan as scan_module

print("SCAN_IMPL =", SCAN_IMPL)
if scan_module._CUDA_IMPORT_ERROR is not None:
    print("mamba-ssm import error (expected if the build above failed):")
    print(repr(scan_module._CUDA_IMPORT_ERROR))

print()
print(">>> PASTE THIS BACK: SCAN_IMPL =", SCAN_IMPL, "<<<")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Numerical equivalence: cuda selective_scan_fn vs the pure-PyTorch
# reference, on a random small problem -- only meaningful if both exist.
import torch

if scan_module._cuda_selective_scan_fn is not None and torch.cuda.is_available():
    torch.manual_seed(0)
    batch, dim, dstate, length = 2, 8, 4, 16
    device = "cuda"
    u = torch.randn(batch, dim, length, device=device)
    dts = torch.rand(batch, dim, length, device=device)
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch, dstate, length, device=device)
    C = torch.randn(batch, dstate, length, device=device)
    D = torch.randn(dim, device=device)

    y_cuda = scan_module._cuda_selective_scan_fn(u, dts, A, B, C, D)
    y_ref = scan_module._selective_scan_ref(u, dts, A, B, C, D)
    max_abs_diff = (y_cuda - y_ref).abs().max().item()

    print("max |cuda - ref| =", max_abs_diff)
    assert torch.allclose(y_cuda, y_ref, atol=1e-3, rtol=1e-3), (
        "cuda and ref scans disagree beyond tolerance -- do NOT trust the "
        "cuda path until this is understood."
    )
    print("cuda ~= ref: OK")
else:
    print("Skipping cuda~=ref check: cuda scan unavailable in this environment "
          f"(SCAN_IMPL={SCAN_IMPL}).")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# GPU smoke test: random [2,C,H,W] -> vendored UltraLight VM-UNet forward,
# print output shape + param count, confirm finite backward gradients.
# This exercises the REAL backbone (needs mamba-ssm to even import), unlike
# the CPU-only shape tests in tests/test_scan.py which test lcmunet/scan.py
# in isolation.
import torch

from lcmunet.backbone import load_ultralight_vmunet

model = load_ultralight_vmunet()
n_params = sum(p.numel() for p in model.parameters())
print(f"UltraLight_VM_UNet params: {n_params} ({n_params / 1e6:.4f} M)")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

x = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
y = model(x)
print("output shape:", tuple(y.shape))
assert y.shape == (2, 1, 256, 256), f"unexpected output shape: {y.shape}"

y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all(), "non-finite/missing input gradient"
for name, p in model.named_parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"

print("Forward/backward OK. All gradients finite.")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Record everything into results/env.json on Drive -- the single source of
# truth for SCAN_IMPL and hardware that every downstream results row and
# efficiency number must be consistent with.
import json

from lcmunet.paths import get_paths
from lcmunet.env_report import write_env_json

paths = get_paths()
env_path = write_env_json(paths.results, repo_root=REPO_DIR)
print("Wrote:", env_path)
print(json.dumps(json.loads(env_path.read_text()), indent=2))
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Next step

**Do not proceed to Prompt 2 (LC-VSS implementation) until you've pasted
back, from the cell above:**
- `SCAN_IMPL` (`"cuda"` or `"ref"`)
- The full `results/env.json` contents

This decides every downstream efficiency number (methodology section 5.5 --
same scan implementation for baseline, GLGF, and LC-SS2D).
"""
))

nb["cells"] = cells
with open("notebooks/01_env.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/01_env.ipynb")
