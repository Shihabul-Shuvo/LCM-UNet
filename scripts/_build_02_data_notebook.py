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

Run All top to bottom in Colab. This notebook is a standalone,
cell-by-cell view of the same data pipeline notebooks/colab_runner.ipynb
runs automatically every session (lcmunet.data.prepare_all.prepare_all_datasets)
-- useful for a first-time, paste-back-each-step verification pass. Once
that's done, you never need to open this notebook again; colab_runner.ipynb
alone keeps it up to date.

Nothing is downloaded automatically -- you place each dataset's .zip
yourself under `DRIVE_ROOT/data_raw/<Name>/` (any filename; see
`lcmunet/data/download.py`'s module docstring for exactly where to get each
one: Kvasir-SEG, CVC-ClinicDB, ISIC2017, ISIC2018).

1. Mounts Drive and pulls this repo fresh.
2. Extracts + splits whichever of the four datasets already have a .zip
   placed. Idempotent -- already-prepared datasets are skipped instantly.
   This is where the **mandatory CVC-ClinicDB sequence-level leakage
   guard** runs and asserts.
3. Sanity-checks DataLoaders for whichever datasets came back PASS.

**Action needed from you:** for any dataset reported FAILED below (no .zip
found yet), follow the printed instructions to download+place it, then
re-run this notebook (it will only redo the missing pieces).
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
"""# Extract + split all four datasets in one call (idempotent -- safe and
# fast to re-run every session; nothing is downloaded here, see
# lcmunet/data/prepare_all.py / lcmunet/data/download.py).
from lcmunet.paths import get_paths
from lcmunet.data.prepare_all import prepare_all_datasets

paths = get_paths()
report = prepare_all_datasets(paths)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Sanity-check DataLoaders for whichever datasets came back PASS.
from lcmunet.config import RunConfig
from lcmunet.data.loaders import build_dataloaders

ready_datasets = [name for name, row in report.items() if row["status"] == "PASS"]

for name in ready_datasets:
    counts = report[name]["counts"]
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
        f"{name}: train={counts['train']} val={counts['val']} test={counts['test']} "
        f"| sanity batch images={tuple(images.shape)} masks={tuple(masks.shape)}"
    )

if "kvasir_seg" in ready_datasets and "cvc_clinicdb" in ready_datasets:
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
- The DATA PREPARATION SUMMARY table (PASS/FAIL per dataset, counts).
- Whether the CVC leakage assertion passed (no `AssertionError` raised) and
  the `Verify the CVC 29-sequence mapping against the original
  documentation before submission.` reminder.
- For ISIC2017, the detected bucket layout log lines and
  `data_raw/ISIC2017/isic2017_source_manifest.json`'s `source_zip` field
  (which archive was actually used).
- The DataLoader sanity-check batch shapes.
"""
))

nb["cells"] = cells
with open("notebooks/02_data.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/02_data.ipynb")
