"""One-off generator for notebooks/02_data.ipynb.
Run with: python scripts/_build_02_data_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# LCM-UNet - 02: Data Pipeline (section 7)

Run All top to bottom in Colab. This notebook:
1. Mounts Drive and pulls this repo fresh.
2. Auto-downloads Kvasir-SEG into data_raw/ (idempotent -- skips if already
   cached). CVC-ClinicDB and ISIC2017/2018 require registration, so this
   step only detects+extracts what YOU already placed in data_raw/; for
   anything missing it prints exactly what to download and where, and moves
   on to the next dataset rather than stopping the whole notebook.
3. Builds every split under splits/ (idempotent -- re-running overwrites
   with the same deterministic result, never duplicates). This is where the
   **mandatory CVC-ClinicDB sequence-level leakage guard** runs and asserts.
4. Sanity-checks DataLoaders for whichever datasets are ready.

**Action needed from you:** for any dataset reported as SKIPPED below,
follow the printed instructions, place the data in data_raw/, and re-run
this notebook (it will only redo the missing pieces).
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
print("DRIVE_ROOT:", DRIVE_ROOT)
"""
))

cells.append(nbf.v4.new_code_cell(
"""%pip install -q -r requirements.txt
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Ensure raw datasets are present under data_raw/. Kvasir-SEG auto-downloads;
# CVC-ClinicDB/ISIC2017/ISIC2018 are detected (and auto-extracted if you
# placed a matching .zip) or reported as missing with exact instructions.
from lcmunet.paths import get_paths
from lcmunet.data.download import ensure_dataset_ready
from lcmunet.data.raw_layout import DATASET_NAMES, RawDataMissingError

paths = get_paths()
ready = {}
for name in DATASET_NAMES:
    print(f"\\n=== {name} ===")
    try:
        ready[name] = ensure_dataset_ready(name, paths.data_raw)
    except RawDataMissingError:
        print(f"[SKIPPED] {name}: raw data not available yet; see instructions above.")
    except RuntimeError as exc:
        print(f"[FAILED] {name}: {exc}")

print("\\n=== raw data summary ===")
for name in DATASET_NAMES:
    status = f"{ready[name]} pairs" if name in ready else "SKIPPED / not ready"
    print(f"  {name}: {status}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Build every split (idempotent). This is where the CVC sequence-level
# leakage guard runs -- watch for the "Verify the CVC 29-sequence mapping"
# reminder and the assertion (it raises loudly if any sequence leaks across
# partitions; it does not print a soft warning and continue).
from lcmunet.data.splits import build_all_splits

results = build_all_splits(paths)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Sanity-check DataLoaders for whichever datasets have a split ready.
from lcmunet.config import RunConfig
from lcmunet.data.loaders import build_dataloaders

for name in results:
    config = RunConfig(
        run_name="data_sanity",
        model_name="ultralight_baseline",
        dataset=name,
        seed=0,
        split_file=f"splits/{name}.json",
    )
    train_loader, val_loader, test_loader = build_dataloaders(config, paths, sanity=True)
    images, masks, ids = next(iter(train_loader))
    print(
        f"{name}: train={len(results[name]['train'])} val={len(results[name]['val'])} "
        f"test={len(results[name]['test'])} | sanity batch images={tuple(images.shape)} "
        f"masks={tuple(masks.shape)}"
    )

if "kvasir_seg" in results and "cvc_clinicdb" in results:
    from lcmunet.data.loaders import build_cross_dataset_loader

    for train_name, test_name in [("kvasir_seg", "cvc_clinicdb"), ("cvc_clinicdb", "kvasir_seg")]:
        train_loader, test_loader = build_cross_dataset_loader(train_name, test_name, paths, sanity=True)
        images, masks, ids = next(iter(test_loader))
        print(f"cross-dataset train={train_name} test={test_name}: test batch images={tuple(images.shape)}")
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Confirm back

Paste back:
- The raw data summary (which datasets are ready vs. skipped).
- The full split-build output, especially the **CVC-ClinicDB counts** and
  the line `Verify the CVC 29-sequence mapping against the original
  documentation before submission.`
- Whether the CVC leakage assertion passed (no `AssertionError` raised).
- The DataLoader sanity-check batch shapes.
"""
))

nb["cells"] = cells
with open("notebooks/02_data.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/02_data.ipynb")
