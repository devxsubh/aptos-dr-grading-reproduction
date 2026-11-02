"""Builds the Kaggle kernels that run Stage 4e (dr-repro/stage4e_imbalance.ipynb).

Usage:  python build_kernels.py <kaggle username>

The notebook already runs on Kaggle as it is. Each kernel only sets its share of the work in the settings cell
(QUICK_TEST, TRAIN_RUNS, ANALYSE, SAVE_IMAGE_CACHE) and lists its inputs. Run order (kernels on one line can run at
the same time; each takes the outputs of the line above):

    dr-s4e-e0-quick                      quick test of every code path; downloads and caches the images
    dr-s4e-e1-focal  dr-s4e-e1-cb        the 8 training runs, about 2.5 h each on a T4
    dr-s4e-e2-final                      no training: tables, bootstrap, prediction check, post-hoc checks (CPU)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = sys.argv[1]
SOURCE = HERE.parent.parent / "stage4e_imbalance.ipynb"
PREFIX = "dr-s4e"
STAGE4_DATA = ["dr-stage4-outputs", "dr-stage4-weights"]      # Stage 4 outputs.npz; mse_s42 weights for Grad-CAM

# slug suffix: (settings, input kernels, input datasets, GPU, what it does)
KERNELS = {
    "e0-quick": (dict(QUICK_TEST=True, TRAIN_RUNS=None, ANALYSE=True, SAVE_IMAGE_CACHE=True), [], STAGE4_DATA, True,
                 "Quick test of every code path (toy sizes, separate folder); downloads and caches the images."),
    "e1-focal": (dict(QUICK_TEST=False, TRAIN_RUNS=["focal_s42", "focal_s43", "focal_s44", "ce_s42_k"], ANALYSE=False,
                      SAVE_IMAGE_CACHE=False), ["e0-quick"], [], True,
                 "Training: focal loss, seeds 42-44, and the Kaggle control ce_s42_k."),
    "e1-cb": (dict(QUICK_TEST=False, TRAIN_RUNS=["mse_cb_s42", "mse_cb_s43", "mse_cb_s44", "mse_s42_k"], ANALYSE=False,
                   SAVE_IMAGE_CACHE=False), ["e0-quick"], [], True,
              "Training: class-balanced MSE, seeds 42-44, and the Kaggle control mse_s42_k."),
    "e2-final": (dict(QUICK_TEST=False, TRAIN_RUNS=[], ANALYSE=True, SAVE_IMAGE_CACHE=False),
                 ["e0-quick", "e1-focal", "e1-cb"], STAGE4_DATA, False,
                 "No training. Tables, paired bootstrap, prediction check, control check, Mann-Whitney, retina Grad-CAM."),
}


def build(slug, settings, inputs, datasets, gpu, role):
    nb = json.loads(SOURCE.read_text())
    cell = next(c for c in nb["cells"] if "".join(c["source"]).lstrip().startswith("QUICK_TEST ="))
    text = "".join(cell["source"])
    for key, value in settings.items():
        text, n = re.subn(rf"^{key} = [^#\n]*", f"{key} = {value!r}".ljust(25) + " ", text, count=1, flags=re.M)
        assert n == 1, key
    cell["source"] = text.splitlines(keepends=True)
    full = lambda s: f"{PREFIX}-{s}"
    nb["cells"].insert(0, {"cell_type": "markdown", "id": "kaggle-intro", "metadata": {}, "source": [
        f"> **Kaggle kernel `{full(slug)}`.** {role} Built by `dr-repro/kaggle/stage4e/build_kernels.py` from "
        f"`dr-repro/stage4e_imbalance.ipynb`; only the settings cell differs. "
        f"Inputs: {', '.join([f'`{full(i)}`' for i in inputs] + [f'`{d}`' for d in datasets]) or 'none'}."]})
    out = HERE / full(slug)
    out.mkdir(exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = {
        "id": f"{USER}/{full(slug)}",
        "title": full(slug),
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [d if "/" in d else f"{USER}/{d}" for d in datasets],
        "competition_sources": [],
        "kernel_sources": [f"{USER}/{full(i)}" for i in inputs],
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"                 # needs Kaggle CLI >= 2.0
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out)


for slug, spec in KERNELS.items():
    build(slug, *spec)
