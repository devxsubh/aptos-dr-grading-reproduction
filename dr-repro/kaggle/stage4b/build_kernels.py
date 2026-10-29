"""Builds the Kaggle kernels that run Stage 4b (dr-repro/stage4b_push.ipynb), Stage 4c (dr-repro/stage4c_push.ipynb)
or Stage 4d (dr-repro/stage4d_push.ipynb).

Usage:  python build_kernels.py <kaggle username> [4b|4c|4d]      (default 4b)

The science code is taken unchanged from the notebook. Only the infrastructure is patched:

* storage: /kaggle/working/dr-grading instead of Google Drive;
* resume: every kernel first copies the finished runs (and the image cache) out of the kernels it takes as inputs,
  so the chain k0 -> k1 -> k2 -> k3 -> k4 continues where the previous kernel stopped;
* sharding: Block 1 is split by seed, Block 2 by candidate and Block 3 by candidate, so two kernels can run at once;
* a time guard that stops starting new runs before Kaggle's 12-hour limit;
* the test set stays locked (section 11 does not run) in every kernel except the last one, k4-final.

Each kernel goes to <this folder>/<slug>/ with its notebook and kernel-metadata.json, ready for `kaggle kernels push`.
"""
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = sys.argv[1]
CHAIN = sys.argv[2] if len(sys.argv) > 2 else "4b"

# slug suffix: (settings, input kernels, what it does). Inputs starting with "dr-" are full slugs of another chain.
KERNELS_4B = {
    "k0-quick": (dict(QUICK_TEST=True, RUN_BLOCKS=(1, 2, 3), OPEN_TEST=True), [],
                 "Quick test of the whole pipeline (toy models, separate folder) + downloads and caches the images."),
    "k1-s42": (dict(RUN_BLOCKS=(1,), B1_SEEDS=[42]), ["k0-quick"], "Block 1, seed 42."),
    "k1-s43": (dict(RUN_BLOCKS=(1,), B1_SEEDS=[43]), ["k0-quick"], "Block 1, seed 43."),
    "k2-a": (dict(RUN_BLOCKS=(2,), B2_CANDS=["b3_300", "b4ns_380"]), ["k1-s42", "k1-s43"],
             "Block 1 summary and adoption; Block 2 candidates b3_300 and b4ns_380."),
    "k2-b": (dict(RUN_BLOCKS=(2,), B2_CANDS=["b3_384", "v2s21k_384"]), ["k1-s42", "k1-s43"],
             "Block 1 summary and adoption; Block 2 candidates b3_384 and v2s21k_384."),
    "k3-top1": (dict(RUN_BLOCKS=(3,), B3_TOP=[0]), ["k2-a", "k2-b"], "Block 3: 5-fold CV of the best Block 2 candidate."),
    "k3-top2": (dict(RUN_BLOCKS=(3,), B3_TOP=[1]), ["k2-a", "k2-b"], "Block 3: 5-fold CV of the second-best candidate."),
    "k4-final": (dict(RUN_BLOCKS=(), OPEN_TEST=True), ["k3-top1", "k3-top2"],
                 "No training. Opens the test set once and writes the final tables."),
}
STAGE4B_DONE = ["dr-s4b-k3-top1", "dr-s4b-k3-top2"]        # their outputs hold every finished Stage 4b run
KERNELS_4C = {
    "c0-quick": (dict(QUICK_TEST=True, RUN_BLOCKS=(1, 2, 3, 4, 5), OPEN_TEST=True, PROBE_MEMORY=True), STAGE4B_DONE,
                 "Quick test of every Stage 4c code path (toy sizes) + memory and speed probe of every arm at full size."),
    "c1-ab": (dict(RUN_BLOCKS=(4,), C4_ARMS=["A_b4ns380_crop_aux", "B_convnextT448"]), STAGE4B_DONE, "Stage 4c arms A and B."),
    "c1-b2c": (dict(RUN_BLOCKS=(4,), C4_ARMS=["B2_seresnext50_512", "C_b4ns512"]), STAGE4B_DONE, "Stage 4c arms B2 and C."),
    "c1-x": (dict(RUN_BLOCKS=(5,), C4_ARMS=["X_b4ns380_eyepacs"]), STAGE4B_DONE,
             "Stage 4c external-data arm X: EyePACS-2015 pre-training, then 5-fold fine-tuning."),
    "c2-final": (dict(RUN_BLOCKS=(), OPEN_TEST=True), ["c1-ab", "c1-b2c", "c1-x"],
                 "No training. Opens the test set once: Part 0, the Stage 4b and the Stage 4c final tables."),
}
STAGE4C_DONE = ["dr-s4c-c1-ab", "dr-s4c-c1-b2c", "dr-s4c-c1-x"]   # every finished Stage 4b and 4c run
KERNELS_4D = {
    "d0-quick": (dict(QUICK_TEST=True, RUN_BLOCKS=(6,), OPEN_TEST=True), ["dr-s4c-c0-quick"],
                 "Quick test of the Stage 4d code path (toy sizes), on top of the Stage 4c quick-test runs."),
    "d1-x2": (dict(RUN_BLOCKS=(6,), OPEN_TEST=True), STAGE4C_DONE,
              "Stage 4d (post-test): arm X with a corrected fine-tune: zero-shot score, lr probe on fold 0, then 5 folds."),
}
# Kaggle datasets a kernel takes as extra inputs ("owner/name"; no owner = USER). dr-stage4-outputs holds the Stage 4
# Colab runs' outputs.npz (ce_s42-44, mse_s42-44) for Part 0 and the Stage 4 rows; the EyePACS-2015 training images
# (resized, cropped) are for Stage 4c arm X.
EYEPACS = "tanlikesmath/diabetic-retinopathy-resized"
CHAINS = {
    "4b": ("stage4b_push.ipynb", "dr-s4b", KERNELS_4B, {"k4-final": ["dr-stage4-outputs"]}, "k4-final"),
    "4c": ("stage4c_push.ipynb", "dr-s4c", KERNELS_4C,
           {"c0-quick": [EYEPACS], "c1-x": [EYEPACS], "c2-final": ["dr-stage4-outputs"]}, "c2-final"),
    "4d": ("stage4d_push.ipynb", "dr-s4d", KERNELS_4D, {"d1-x2": ["dr-stage4-outputs"]}, "d1-x2"),
}
SOURCE_NAME, PREFIX, KERNELS, DATASETS, FINAL = CHAINS[CHAIN]
SOURCE = HERE.parent.parent / SOURCE_NAME
DEFAULTS = dict(QUICK_TEST=False, RUN_BLOCKS=(1, 2, 3), B1_SEEDS=None, B2_CANDS=None, B3_TOP=None, OPEN_TEST=False,
                C4_ARMS=None, PROBE_MEMORY=False)
full_slug = lambda i: i if i.startswith("dr-") else f"{PREFIX}-{i}"

KAGGLE_STORAGE = r'''
# Kaggle: results go to /kaggle/working (this kernel's output). The finished runs and the image cache of the kernels
# this one takes as inputs are copied in first, so the chain of kernels continues where the previous one stopped.
import shutil

WORK_DIR = Path("/kaggle/working/dr-grading")
LOCAL_DIR = Path("/tmp/dr")                                    # fast scratch disk, not part of the output
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
SESSION_START = time.time()
TIME_BUDGET_H = 11.0                                           # Kaggle stops a session at 12 h


def restore_from_inputs():
    # Kernel outputs mount as .../dr-grading/; an uploaded dataset may lose that folder and start at stage4/runs/.
    inputs = Path("/kaggle/input")
    sources = sorted({p for p in inputs.glob("**/dr-grading") if p.is_dir()} |
                     {p.parent.parent for p in inputs.glob("**/stage4/runs") if p.is_dir()})
    # Finished runs first (a finished copy replaces an unfinished one), then every other missing file.
    for src in sources:
        for done in src.glob("*/runs/*/results.json"):
            dst = WORK_DIR / done.parent.relative_to(src)
            if not (dst / "results.json").exists():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(done.parent, dst, ignore=shutil.ignore_patterns("final_ema.pt"))
    for src in sources:
        for f in src.rglob("*"):
            dst = WORK_DIR / f.relative_to(src)
            if f.is_file() and f.name != "final_ema.pt" and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    n_done = len(list(WORK_DIR.glob("*/runs/*/results.json")))
    print(f"restored from {[str(s) for s in sources] or 'no inputs'}: {n_done} finished runs")


restore_from_inputs()
'''


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one {old!r}"
    return text.replace(old, new)


def build(slug, settings, inputs, role):
    nb = copy.deepcopy(json.loads(SOURCE.read_text()))
    cells = nb["cells"]
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    by_start = lambda prefix: next(c for c in cells if src(c).lstrip().startswith(prefix))
    s = {**DEFAULTS, **settings}

    # Setup: Kaggle storage instead of Google Drive.
    setup = by_start("import hashlib")
    text = src(setup)
    start = text.index("# Results and the data cache live on Google Drive")
    end = text.index("WORK_DIR.mkdir(parents=True, exist_ok=True)\n") + len("WORK_DIR.mkdir(parents=True, exist_ok=True)\n")
    set_src(setup, text[:start] + KAGGLE_STORAGE.lstrip("\n") + text[end:])

    # Settings: this kernel's share of the work.
    settings_cell = by_start("QUICK_TEST =")
    text = src(settings_cell)
    text = re.sub(r"^QUICK_TEST = .*$", f"QUICK_TEST = {s['QUICK_TEST']}", text, count=1, flags=re.M)
    text = re.sub(r"^RUN_BLOCKS = .*$", f"RUN_BLOCKS = {s['RUN_BLOCKS']!r}    # blocks this kernel may train", text,
                  count=1, flags=re.M)
    shard = (f"\n# Kaggle kernel {slug}: {role}\n"
             f"B1_SEEDS = {s['B1_SEEDS']!r}     # Block 1 seeds this kernel trains (None = all)\n"
             f"B2_CANDS = {s['B2_CANDS']!r}     # Block 2 candidates this kernel trains (None = all)\n"
             f"B3_TOP = {s['B3_TOP']!r}       # Block 3: ranks in TOP this kernel trains (None = all)\n"
             f"OPEN_TEST = {s['OPEN_TEST']}     # {'section 11 (the test set) runs' if CHAIN == '4b' else 'the test set is opened'} only in the last kernel\n")
    if CHAIN != "4b":
        shard += (f"C4_ARMS = {s['C4_ARMS']!r}     # Stage 4c arms this kernel trains (None = all)\n"
                  f"PROBE_MEMORY = {s['PROBE_MEMORY']}\n")
    text = replace_once(text, "SPLIT_SEED = 42", shard.lstrip("\n").rstrip("\n") + "\nSPLIT_SEED = 42")
    set_src(settings_cell, text)

    # Data cache: Python zip instead of the zip/unzip binaries.
    data = by_start("HF_REPO =")
    text = src(data)
    text = replace_once(text, 'subprocess.run(["unzip", "-q", "-o", str(DRIVE_CACHE), "-d", str(IMG_DIR)], check=True)',
                        "shutil.unpack_archive(DRIVE_CACHE, IMG_DIR)")
    text = replace_once(text, 'subprocess.run(["zip", "-q", "-r", "-j", str(DRIVE_CACHE), str(IMG_DIR)], check=True)',
                        'shutil.make_archive(str(DRIVE_CACHE.with_suffix("")), "zip", IMG_DIR)')
    text = text.replace("restoring cached images from Drive", "restoring cached images").replace(
        "saving a cache to Drive for next time", "saving a cache for the next kernels")
    set_src(data, text)

    # Time guard in run_experiment.
    train = by_start("def seed_everything")
    set_src(train, replace_once(src(train), "    if not allowed:\n        return None\n",
                                "    if not allowed:\n        return None\n"
                                "    if time.time() - SESSION_START > TIME_BUDGET_H * 3600:\n"
                                "        print(f\"time guard: not starting {name} ({TIME_BUDGET_H} h used); \"\n"
                                "              \"push this kernel again to continue\")\n"
                                "        return None\n"))

    # Sharding of the three blocks.
    b1 = by_start("block1_names = {}")
    set_src(b1, replace_once(src(b1), "allowed=1 in RUN_BLOCKS)",
                             "allowed=1 in RUN_BLOCKS and (B1_SEEDS is None or seed in B1_SEEDS))"))
    b2 = by_start("COMBINED = replace")
    set_src(b2, replace_once(src(b2), "allowed=2 in RUN_BLOCKS)",
                             "allowed=2 in RUN_BLOCKS and (B2_CANDS is None or cand in B2_CANDS))"))
    b3 = by_start("cv_names = {")
    set_src(b3, replace_once(src(b3), "allowed=3 in RUN_BLOCKS)",
                             "allowed=3 in RUN_BLOCKS and (B3_TOP is None or cand in [TOP[i] for i in B3_TOP]))"))

    # The test set stays locked except in the last kernel.
    final = by_start("final_rows, final_preds, final_cutoffs = [], {}, {}")
    body = src(final).split("\n", 1)[1]
    indented = "".join(("    " + line) if line.strip() else line for line in body.splitlines(keepends=True))
    set_src(final, "final_rows, final_preds, final_cutoffs, headline = [], {}, {}, None\n"
                   "if not OPEN_TEST:\n"
                   f"    print(\"Test set locked in this kernel; it is opened once, in {FINAL}.\")\n"
                   "else:\n" + indented)

    intro = {"cell_type": "markdown", "id": "kaggle-intro", "metadata": {}, "source": [
        f"> **Kaggle kernel `{PREFIX}-{slug}`.** {role} Built by `dr-repro/kaggle/stage4b/build_kernels.py` from "
        f"`dr-repro/{SOURCE_NAME}`; only storage, resume, sharding, a time guard and the test lock differ. "
        f"Inputs: {', '.join(f'`{full_slug(i)}`' for i in inputs) or 'none'}."]}
    cells.insert(0, intro)
    nb["metadata"]["accelerator"] = "GPU"

    out = HERE / f"{PREFIX}-{slug}"
    out.mkdir(exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{PREFIX}-{slug}",
        "title": f"{PREFIX}-{slug}",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [d if "/" in d else f"{USER}/{d}" for d in DATASETS.get(slug, [])],
        "competition_sources": [],
        "kernel_sources": [f"{USER}/{full_slug(i)}" for i in inputs],
        "model_sources": [],
    }, indent=2) + "\n")
    print("wrote", out)


for slug, (settings, inputs, role) in KERNELS.items():
    build(slug, settings, inputs, role)
