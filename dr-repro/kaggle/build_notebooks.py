"""Builds dr-repro/kaggle/kaggle_train.ipynb and kaggle_submit.ipynb. Shared code is written once below."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def notebook(cells):
    return {
        "cells": [
            {"cell_type": kind, "id": f"cell-{n:02d}", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for n, (kind, src) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ----------------------------------------------------------------------------------------------- shared code
FIND_DIR = r'''
def find_input_dir(*markers, env=None):
    # The folder under /kaggle/input holding all `markers`. Kaggle's mount layout differs between UI versions.
    if env and os.environ.get(env):
        return Path(os.environ[env])
    for pattern in ("*", "*/*", "*/*/*", "*/*/*/*", "*/*/*/*/*"):
        for folder in Path("/kaggle/input").glob(pattern):
            if all((folder / m).exists() for m in markers):
                return folder
    raise FileNotFoundError(f"no input folder with {markers}. Check the inputs in the notebook sidebar.")
'''

PREPROCESS = r'''
def preprocess(path, size):
    # Load -> shrink to <= 512 px -> CLAHE + blur -> square resize. Returns RGB uint8 (size, size, 3).
    # Identical in the training and the submission notebook.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    bgr = cv2.GaussianBlur(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), (5, 5), 0)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
'''

MODEL = r'''
class DRNet(nn.Module):
    # EfficientNet + the paper's head, ending in one output: the grade as a number on the 0-4 scale.
    def __init__(self, backbone="efficientnet_b3", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.Dropout(0.5), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.Dropout(0.25), nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))[:, 0]


_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
_black = -_mean / _std     # what a black pixel looks like after normalisation


def to_model_input(batch_uint8):
    # numpy (B, H, W, 3) uint8 -> normalised float tensor (B, 3, H, W) on the GPU.
    x = torch.from_numpy(batch_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
    return ((x - _mean) / _std).contiguous(memory_format=torch.channels_last)
'''

HEADER_IMPORTS = '''
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ON_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path(os.environ.get("DR_OUTPUT_DIR", "/kaggle/working" if ON_KAGGLE else "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "no GPU found (very slow!)")
print("timm", timm.__version__, "| torch", torch.__version__)
'''

# ------------------------------------------------------------------------------------------- training notebook
train = [
    ("markdown", r'''
# APTOS 2019 on Kaggle, part 1 of 2: training
### Stage 4 arm D (ordinal regression + tuned cut-offs) as a 5-fold ensemble, trained on all labelled APTOS images

APTOS 2019 is a **code competition**: Kaggle accepts only a notebook that runs **with internet off** and writes `submission.csv`. Kaggle then re-runs it on the hidden test set (about 13,000 images instead of the 1,928 public ones). Training 50 models does not belong in that notebook, so the work is split:

| Notebook | Internet | What it does | Time (T4) |
|---|---|---|---|
| **`kaggle_train.ipynb`** (this one) | **on** (ImageNet weights) | 5-fold training, 10 snapshots per fold, cut-offs tuned on out-of-fold predictions | ~3-4 h |
| `kaggle_submit.ipynb` | **off** | loads the 50 saved models + cut-offs, grades the test set, writes `submission.csv` | ~20-30 min |

**Recipe** (unchanged from the best Stage 4 arm, D: 0.914 ± 0.004 on our 550-image test split): EfficientNet-B3, ImageNet weights, CLAHE + Gaussian blur, 300 px, 60 epochs, batch 16, Adam lr 1e-4, flip + rotation augmentation, **one output trained with MSE**, 10-snapshot ensemble, cut-offs tuned for QWK. Changes for the submission:

- **5-fold cross-validation instead of one 70/15/15 split.** Every labelled image is used for training and is predicted once out-of-fold (OOF). The final ensemble is 5 folds x 10 snapshots = 50 models.
- **Folds are grouped by image hash**, so byte-identical images never sit on both sides of a fold (Stage 3 found 123 such groups). The 30 groups whose copies have **different grades are dropped**.
- **Cut-offs are tuned on all OOF predictions** (~3,600 images instead of 550 validation images). A nested estimate (cut-offs fitted without the fold they are scored on) is printed next to the in-sample one.

**How to run on Kaggle**

1. kaggle.com -> **Code -> New Notebook -> File -> Import Notebook**, upload this file.
2. Sidebar **Add Input -> Competitions -> APTOS 2019 Blindness Detection**.
3. Sidebar **Settings: Accelerator = GPU T4 x2 or P100** (one GPU is used), **Internet = On** (needs a phone-verified account).
4. **Save Version -> Save & Run All (Commit)**. It runs in the background; you can close the tab. A time guard stops before Kaggle's 12-hour limit and still saves the finished folds.
5. When the version has finished, its **Output** holds `models/` (50 files, ~2.4 GB), `meta.json` (cut-offs, settings, OOF scores) and `oof.csv`. Then run `kaggle_submit.ipynb`.

Set `QUICK_TEST = True` first for a ~5-minute check (2 folds, 2 epochs, 300 images). Do not submit the quick-test models.

*This is a reproduction study, not a clinical tool.*
'''),
    ("markdown", "## 1. Setup and settings"),
    ("code", r'''
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm.auto import tqdm

try:
    import timm
except ImportError:                      # preinstalled on Kaggle; not everywhere else
    !pip install -q timm
    import timm
''' + HEADER_IMPORTS),
    ("code", r'''
QUICK_TEST = os.environ.get("DR_QUICK_TEST") == "1"   # set True by hand for a ~5-minute pipeline check
TIME_BUDGET_H = 11.0                                  # Kaggle stops a session at 12 h; leave room for the rest


@dataclass
class Config:
    # Stage 4 arm D: the Stage 3 baseline `2_aug_lr1e-4` with an MSE (ordinal regression) output.
    image_size: int = 300
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    n_snapshots: int = 10         # weights saved at 10 evenly spaced epochs per fold
    n_folds: int = 5
    seed: int = 42
    backbone: str = "efficientnet_b3"
    use_amp: bool = True
    max_train_batches: int = 0    # 0 = all batches; >0 only for the quick test
    max_images: int = 0           # 0 = all images; >0 only for the quick test


CFG = Config()
FOLDS = list(range(CFG.n_folds))
if QUICK_TEST:
    CFG = replace(CFG, epochs=2, n_snapshots=2, max_train_batches=3, image_size=224, max_images=300)
    FOLDS = [0, 1]
if device.type != "cuda" and not QUICK_TEST:
    raise RuntimeError("No GPU. Sidebar -> Settings -> Accelerator -> GPU, then run again.")
if device.type != "cuda":
    CFG = replace(CFG, use_amp=False)       # mixed precision is a GPU feature

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(json.dumps(asdict(CFG), indent=2))
print("folds to train:", FOLDS)
'''),
    ("markdown", r'''
## 2. Data, duplicates and folds

The competition's `train.csv` (3,662 images). Images are hashed; groups of identical files whose grades disagree are dropped, and the remaining identical files share a group so that `StratifiedGroupKFold` keeps them in the same fold.
'''),
    ("code", FIND_DIR + r'''

DATA_DIR = find_input_dir("train.csv", "train_images", env="APTOS_DIR")
df = pd.read_csv(DATA_DIR / "train.csv").rename(columns={"diagnosis": "label"})
df["path"] = df["id_code"].map(lambda i: DATA_DIR / "train_images" / f"{i}.png")
print(f"data folder: {DATA_DIR}\n{len(df)} images")


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


with ThreadPoolExecutor(os.cpu_count()) as pool:
    df["hash"] = list(tqdm(pool.map(file_hash, df["path"]), total=len(df), desc="hashing"))

groups = df.groupby("hash")["label"].agg(["size", "nunique"])
conflicting = df["hash"].map(groups["nunique"]) > 1
print(f"{int((groups['size'] > 1).sum())} groups of identical images, {int((groups['nunique'] > 1).sum())} with "
      f"conflicting grades -> dropping their {int(conflicting.sum())} images")
df = df[~conflicting].reset_index(drop=True)

if CFG.max_images:                                                  # quick test only
    keep, _ = train_test_split(np.arange(len(df)), train_size=CFG.max_images, stratify=df["label"], random_state=CFG.seed)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)

df["fold"] = -1
splitter = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
for fold, (_, idx) in enumerate(splitter.split(df, df["label"], groups=df["hash"])):
    df.loc[idx, "fold"] = fold
labels = df["label"].to_numpy()
print(pd.crosstab(df["fold"], df["label"].map(dict(enumerate(CLASS_NAMES)))).reindex(columns=CLASS_NAMES).to_string())
'''),
    ("markdown", r'''
## 3. Pre-processing (unchanged from Stage 3)

CLAHE on the LAB lightness channel (clip 2.0, 8x8 tiles), 5x5 Gaussian blur, square resize to 300 px. Computed once and kept in memory (~1 GB). The submission notebook uses the same function.
'''),
    ("code", PREPROCESS + r'''

images = np.empty((len(df), CFG.image_size, CFG.image_size, 3), dtype=np.uint8)


def fill(i):
    images[i] = preprocess(df["path"].iat[i], CFG.image_size)


t0 = time.time()
with ThreadPoolExecutor(os.cpu_count()) as pool:            # OpenCV releases the GIL, so threads are enough
    list(tqdm(pool.map(fill, range(len(df))), total=len(df), desc="pre-processing"))
print(f"{images.shape}, {images.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s")
'''),
    ("markdown", r'''
## 4. Model, augmentation and cut-offs

Same network and augmentation as Stage 4. A regression output becomes a grade through four cut-offs: rounding uses 0.5 / 1.5 / 2.5 / 3.5; the tuned rule moves each cut-off on a 0.01 grid to maximise QWK (coordinate search, three passes).
'''),
    ("code", MODEL + r'''

def augment_batch(x):
    # Random horizontal flip + random rotation (0-360 degrees), as in Stage 3. Empty corners stay black.
    n = x.shape[0]
    x = x - _black
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    angle = torch.rand(n, device=x.device) * 2 * math.pi
    cos, sin, zero = angle.cos(), angle.sin(), torch.zeros_like(angle)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)    # (n, 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (x + _black).contiguous(memory_format=torch.channels_last)
'''),
    ("code", r'''
ROUNDING = np.array([0.5, 1.5, 2.5, 3.5])


def qwk(y, pred, k=NUM_CLASSES):
    # Quadratic weighted kappa from the confusion matrix (same value as sklearn's cohen_kappa_score, much faster).
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    weights = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    denom = (weights * expected).sum()
    return float(1 - (weights * cm).sum() / denom) if denom > 0 else 0.0


def apply_cutoffs(score, cutoffs):
    return np.digitize(score, cutoffs).astype(int)


def fit_cutoffs(score, y, step=0.01, passes=3):
    # Coordinate search: move one cut-off at a time to the grid value with the best QWK, keeping them ordered.
    cutoffs = ROUNDING.copy()
    best = qwk(y, apply_cutoffs(score, cutoffs))
    for _ in range(passes):
        for i in range(len(cutoffs)):
            lo = cutoffs[i - 1] + step if i > 0 else -0.5
            hi = cutoffs[i + 1] - step if i < len(cutoffs) - 1 else NUM_CLASSES - 0.5
            for value in np.arange(lo, hi + 1e-9, step):
                trial = cutoffs.copy()
                trial[i] = value
                score_trial = qwk(y, apply_cutoffs(score, trial))
                if score_trial > best:                       # strictly better: ties keep the current cut-off
                    best, cutoffs = score_trial, trial
    return cutoffs
'''),
    ("markdown", r'''
## 5. Training

Per fold: train on the other four folds for 60 epochs; after every epoch predict the held-out fold (logged with rounding, for monitoring only). At the 10 snapshot epochs (6, 12, ..., 60) the weights are saved and the held-out predictions kept. A fold's OOF score is the mean of its 10 snapshots, the same averaging the submission notebook applies to the test set.
'''),
    ("code", r'''
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, indices, cfg, batch_size=64):
    model.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        x = to_model_input(images[indices[start:start + batch_size]])
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.use_amp):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def train_one_epoch(model, optimizer, scaler, cfg, train_idx, rng):
    model.train()
    order = rng.permutation(train_idx)
    n_batches = len(order) // cfg.batch_size                                   # drop the last partial batch
    if cfg.max_train_batches:
        n_batches = min(n_batches, cfg.max_train_batches)
    total_loss, seen = 0.0, 0
    for b in tqdm(range(n_batches), desc="  training", leave=False):
        idx = np.sort(order[b * cfg.batch_size:(b + 1) * cfg.batch_size])      # sorted = faster memory reads
        x = augment_batch(to_model_input(images[idx]))
        y = torch.from_numpy(labels[idx]).to(device).float()

        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.use_amp):
            out = model(x)
        loss = F.mse_loss(out.float(), y)                                      # ordinal regression

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                             # so clipping sees real gradient sizes
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y)
        seen += len(y)
    return total_loss / seen


def train_fold(fold, cfg):
    # Train one fold; save its snapshot weights and return (snapshot file names, OOF score, log).
    train_idx, val_idx = np.flatnonzero(df["fold"] != fold), np.flatnonzero(df["fold"] == fold)
    print(f"\n=== fold {fold} ===  train {len(train_idx)}  held out {len(val_idx)}")
    seed_everything(cfg.seed + fold)
    model = DRNet(cfg.backbone, pretrained=True).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    rng = np.random.default_rng(cfg.seed + fold)

    snapshot_epochs = {round(cfg.epochs * k / cfg.n_snapshots) for k in range(1, cfg.n_snapshots + 1)}
    snap_files, snap_val, log = [], [], []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, scaler, cfg, train_idx, rng)
        val_out = predict(model, val_idx, cfg)
        val_qwk = qwk(labels[val_idx], apply_cutoffs(val_out, ROUNDING))
        log.append({"fold": fold, "epoch": epoch, "train_mse": loss, "val_qwk_rounding": val_qwk})
        mark = ""
        if epoch in snapshot_epochs:
            name = f"fold{fold}_ep{epoch:02d}.pt"
            torch.save(model.state_dict(), MODEL_DIR / name)
            snap_files.append(name)
            snap_val.append(val_out)
            mark = "  snapshot saved"
        print(f"  epoch {epoch:2d}  train MSE {loss:.3f}  held-out QWK {val_qwk:.3f}  {time.time() - t0:.0f}s{mark}", flush=True)

    oof_score = np.mean(snap_val, axis=0)
    print(f"fold {fold}: snapshot-ensemble QWK {qwk(labels[val_idx], apply_cutoffs(oof_score, ROUNDING)):.3f} (rounding)")
    del model, optimizer
    torch.cuda.empty_cache()
    return snap_files, oof_score, log
'''),
    ("markdown", r'''
## 6. Train the folds

About 35-45 minutes per fold on a T4. Before each fold the time guard checks that the average fold time still fits into `TIME_BUDGET_H`; if not, it stops and the submission uses the folds that finished.
'''),
    ("code", r'''
df["oof_score"] = np.nan
model_files, logs, done_folds = [], [], []
start = time.time()
for fold in FOLDS:
    elapsed_h = (time.time() - start) / 3600
    if done_folds and elapsed_h + elapsed_h / len(done_folds) > TIME_BUDGET_H:
        print(f"time guard: {elapsed_h:.1f} h used, another fold would pass {TIME_BUDGET_H} h. Stopping here.")
        break
    files, oof_score, log = train_fold(fold, CFG)
    model_files += files
    logs += log
    done_folds.append(fold)
    df.loc[df["fold"] == fold, "oof_score"] = oof_score
    pd.DataFrame(logs).to_csv(OUTPUT_DIR / "train_log.csv", index=False)       # progress survives a crash
print(f"\n{len(done_folds)} folds, {len(model_files)} models, {(time.time() - start) / 3600:.2f} h")
'''),
    ("markdown", r'''
## 7. Cut-offs from the out-of-fold predictions

Three numbers on the same OOF predictions:

- **rounding**: no tuning (arm C's rule);
- **tuned, nested**: cut-offs fitted on the other folds' OOF predictions, applied to this fold. This is the honest estimate;
- **tuned, in-sample**: cut-offs fitted and scored on all OOF predictions (optimistic). These are the cut-offs the submission uses.

The hidden test set is known to differ from the training set in its grade mix, so expect the leaderboard score to be lower than any of these.
'''),
    ("code", r'''
oof = df[df["fold"].isin(done_folds)]
y, score = oof["label"].to_numpy(), oof["oof_score"].to_numpy()
cutoffs = fit_cutoffs(score, y)
scores = {"oof_qwk_rounding": qwk(y, apply_cutoffs(score, ROUNDING)), "oof_qwk_tuned_in_sample": qwk(y, apply_cutoffs(score, cutoffs))}

if len(done_folds) > 1:
    nested = np.empty(len(oof), dtype=int)
    for fold in done_folds:
        held = (oof["fold"] == fold).to_numpy()
        nested[held] = apply_cutoffs(score[held], fit_cutoffs(score[~held], y[~held]))
    scores["oof_qwk_tuned_nested"] = qwk(y, nested)

print(f"OOF images: {len(oof)}")
for k, v in scores.items():
    print(f"  {k:26s} {v:.4f}")
print("cut-offs:", np.round(cutoffs, 2))

pred = apply_cutoffs(score, cutoffs)
print("\nconfusion matrix (rows = true, columns = predicted, tuned in-sample):")
print(pd.crosstab(pd.Series(y, name="true").map(dict(enumerate(CLASS_NAMES))),
                  pd.Series(pred, name="pred").map(dict(enumerate(CLASS_NAMES))))
      .reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0).to_string())
'''),
    ("markdown", r'''
## 8. Save what the submission notebook needs

`meta.json` holds the settings, the cut-offs and the list of model files; `oof.csv` the per-image OOF scores. Together with `models/` they form this notebook's output, which `kaggle_submit.ipynb` takes as an input.
'''),
    ("code", r'''
meta = {
    "config": asdict(CFG),
    "quick_test": QUICK_TEST,
    "folds": done_folds,
    "models": model_files,
    "cutoffs": [round(float(c), 4) for c in cutoffs],
    **{k: round(v, 4) for k, v in scores.items()},
    "timm_version": timm.__version__,
    "torch_version": torch.__version__,
}
(OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
oof[["id_code", "label", "fold", "hash", "oof_score"]].to_csv(OUTPUT_DIR / "oof.csv", index=False)

size_gb = sum(f.stat().st_size for f in MODEL_DIR.glob("*.pt")) / 1e9
print(json.dumps({k: v for k, v in meta.items() if k != "models"}, indent=2))
print(f"\n{len(model_files)} model files, {size_gb:.2f} GB in {MODEL_DIR}")
if QUICK_TEST:
    print("\nQUICK TEST: these models are only for checking the pipeline. Set QUICK_TEST = False for the real run.")
'''),
]

# ----------------------------------------------------------------------------------------- submission notebook
submit = [
    ("markdown", r'''
# APTOS 2019 on Kaggle, part 2 of 2: submission
### Grades the test set with the 50 models from `kaggle_train.ipynb` and writes `submission.csv`

This notebook trains nothing. It loads the snapshot weights and the cut-offs saved by the training notebook, pre-processes each test image the same way (CLAHE + blur, 300 px), averages the regression output of every model, and turns the mean into a grade with the OOF-tuned cut-offs.

**How to submit**

1. Run `kaggle_train.ipynb` as a committed version first (Save & Run All) and wait until it has finished.
2. **Code -> New Notebook -> File -> Import Notebook**, upload this file.
3. Sidebar **Add Input**: **Competitions -> APTOS 2019 Blindness Detection**, and **Your Work -> Notebooks ->** the training notebook (its output, with `meta.json` and `models/`).
4. Sidebar **Settings: Accelerator = GPU**, **Internet = Off** (Kaggle rejects code-competition submissions with internet on).
5. **Save Version -> Save & Run All**. Check the log: it prints the grade distribution and writes `/kaggle/working/submission.csv` (`id_code,diagnosis`).
6. Competition page -> **Late Submission** (or **Submit Prediction**) -> choose this notebook, the version, and `submission.csv`. Kaggle re-runs it on the full hidden test set (~13,000 images), which takes ~20-30 minutes, and shows the public and private scores. The competition has closed, so the score is not ranked on the leaderboard.

Nothing here downloads anything, so it runs with internet off: `timm` is preinstalled on Kaggle, and the network is built with `pretrained=False` before the saved weights are loaded.
'''),
    ("code", r'''
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from tqdm.auto import tqdm
''' + HEADER_IMPORTS),
    ("markdown", "## 1. Find the competition data and the trained models"),
    ("code", FIND_DIR + r'''

COMP_DIR = find_input_dir("test.csv", "test_images", env="APTOS_DIR")
WEIGHTS_DIR = find_input_dir("meta.json", "models", env="DR_MODEL_DIR")
meta = json.loads((WEIGHTS_DIR / "meta.json").read_text())
cfg = meta["config"]
cutoffs = np.array(meta["cutoffs"])
print(f"competition data: {COMP_DIR}\nmodels:           {WEIGHTS_DIR}")
print(f"{len(meta['models'])} models from folds {meta['folds']}, {cfg['backbone']} at {cfg['image_size']} px, "
      f"cut-offs {cutoffs.tolist()}")
print({k: v for k, v in meta.items() if k.startswith("oof_")})
if meta["quick_test"]:
    print("\nWARNING: these models come from a QUICK_TEST run of the training notebook. Do not submit this.")
if meta["timm_version"] != timm.__version__:
    print(f"note: trained with timm {meta['timm_version']}, running {timm.__version__}")
'''),
    ("markdown", "## 2. Load the models"),
    ("code", MODEL + r'''

models = []
for name in tqdm(meta["models"], desc="loading models"):
    model = DRNet(cfg["backbone"], pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "models" / name, map_location="cpu"))
    models.append(model.to(device).to(memory_format=torch.channels_last).eval())
use_amp = device.type == "cuda"
'''),
    ("markdown", r'''
## 3. Grade the test set

Images are pre-processed in chunks with all CPU cores, then every model scores every image (mixed precision on the GPU). An image that cannot be read gets a black picture instead of stopping the run: a crash during Kaggle's hidden re-run would give no score at all.
'''),
    ("code", PREPROCESS + r'''

def safe_preprocess(path):
    try:
        return preprocess(path, cfg["image_size"])
    except Exception as err:                     # never let one bad file sink the whole submission
        print(f"warning: {path.name}: {err!r}, using a black image")
        return np.zeros((cfg["image_size"], cfg["image_size"], 3), dtype=np.uint8)


@torch.no_grad()
def ensemble_score(batch_uint8, batch_size=64):
    # Mean regression output of all models, one number per image.
    total = torch.zeros(len(batch_uint8), device=device)
    for start in range(0, len(batch_uint8), batch_size):
        x = to_model_input(batch_uint8[start:start + batch_size])
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            for model in models:
                total[start:start + len(x)] += model(x).float()
    return (total / len(models)).cpu().numpy()


test = pd.read_csv(COMP_DIR / "test.csv")
paths = [COMP_DIR / "test_images" / f"{i}.png" for i in test["id_code"]]
scores = np.empty(len(test))
CHUNK = 512
t0 = time.time()
with ThreadPoolExecutor(os.cpu_count()) as pool:            # OpenCV releases the GIL, so threads are enough
    for start in tqdm(range(0, len(paths), CHUNK), desc="test chunks"):
        batch = np.stack(list(pool.map(safe_preprocess, paths[start:start + CHUNK])))
        scores[start:start + len(batch)] = ensemble_score(batch)
print(f"{len(test)} test images, {time.time() - t0:.0f}s")
'''),
    ("markdown", "## 4. Write `submission.csv`"),
    ("code", r'''
submission = pd.DataFrame({"id_code": test["id_code"], "diagnosis": np.digitize(scores, cutoffs).astype(int)})
submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
assert len(submission) == len(test) and submission["diagnosis"].between(0, NUM_CLASSES - 1).all()

counts = submission["diagnosis"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
print(pd.DataFrame({"grade": CLASS_NAMES, "images": counts.values, "share": (counts / len(submission)).round(3).values}).to_string(index=False))
print(f"\nwrote {OUTPUT_DIR / 'submission.csv'}")
submission.head()
'''),
]

for name, cells in (("kaggle_train.ipynb", train), ("kaggle_submit.ipynb", submit)):
    (OUT / name).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT / name)
