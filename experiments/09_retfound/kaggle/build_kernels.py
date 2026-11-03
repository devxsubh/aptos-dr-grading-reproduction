"""Builds the Kaggle kernels that run Stage 7 (experiments/09_retfound/09_retfound.ipynb).

Usage:  python build_kernels.py <kaggle username>

The Stage 7 notebook is written for Colab. Here its Colab-only setup (Google Drive) is swapped for the Kaggle one
(results in /kaggle/working, images downloaded from the Hugging Face mirror, baselines read from the GitHub repository)
and the QUICK_TEST switch is set. Nothing else differs. Run order:

    dr-s7-quick    quick test of every code path (toy sizes, 4 epochs)
    dr-s7-train    the 4 training runs plus all tables, bootstrap, prediction check and figures, about 3 h on a T4 (to be confirmed by the quick test)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = sys.argv[1]
SOURCE = HERE.parent / "09_retfound.ipynb"
PREFIX = "dr-s7"

KERNELS = {
    "quick": (True, "Quick test of every code path (toy sizes, separate folder); also downloads the RETFound weights."),
    "train": (False, "The 4 ViT-Large training runs (RETFound x 3 seeds, ImageNet control x 1) and the analysis."),
}

COLAB_BLOCK_START = "# Results and the data cache live on Google Drive"
COLAB_BLOCK_END = "WORK_DIR.mkdir(parents=True, exist_ok=True)"
KAGGLE_BLOCK = '''# Kaggle: results go to /kaggle/working (this kernel's output); images are downloaded to the local disk.
WORK_DIR = Path("/kaggle/working/dr-grading")
LOCAL_DIR = Path("/tmp/dr")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
'''


def swap(text, old, new):
    assert old in text, old[:60]
    return text.replace(old, new, 1)


def build(slug, quick, role):
    nb = json.loads(SOURCE.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if COLAB_BLOCK_START in text:                                   # setup cell: Drive -> Kaggle
            i0, i1 = text.index(COLAB_BLOCK_START), text.index(COLAB_BLOCK_END)
            text = text[:i0] + KAGGLE_BLOCK + text[i1:]
        if text.lstrip().startswith("QUICK_TEST ="):
            text = swap(text, "QUICK_TEST = False", f"QUICK_TEST = {quick}")
        if "DRIVE_CACHE = WORK_DIR" in text:                            # no Drive cache on Kaggle
            text = swap(text, 'subprocess.run(["zip", "-q", "-r", "-j", str(DRIVE_CACHE), str(IMG_DIR)], check=True)', "pass")
        cell["source"] = text.splitlines(keepends=True)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"> **Kaggle kernel `{PREFIX}-{slug}`.** {role} Built by `experiments/09_retfound/kaggle/build_kernels.py` from "
        "`experiments/09_retfound/09_retfound.ipynb`; only the Drive setup and the QUICK_TEST switch differ."]})
    for n, cell in enumerate(nb["cells"]):                              # Kaggle needs a unique id on every cell
        cell["id"] = f"s5-{n:02d}"
    nb["metadata"] = {k: v for k, v in nb["metadata"].items() if k in ("accelerator", "kernelspec", "language_info")}
    out = HERE / f"{PREFIX}-{slug}"
    out.mkdir(exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = {
        "id": f"{USER}/{PREFIX}-{slug}", "title": f"{PREFIX}-{slug}", "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"{USER}/dr-s5-train"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out)


for slug, (quick, role) in KERNELS.items():
    build(slug, quick, role)
