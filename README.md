# Reproducing Diabetic Retinopathy Grading on APTOS 2019

A research-reproduction study of **Chilukoti et al., "A reliable diabetic retinopathy grading via transfer learning and ensemble learning with quadratic weighted kappa metric"**, *BMC Medical Informatics and Decision Making* 24:37 (2024), [open access](https://doi.org/10.1186/s12911-024-02446-x). The paper fine-tunes an ImageNet-pretrained EfficientNet-B3 with CLAHE + Gaussian-blur preprocessing and a 10-snapshot ensemble, and reports **QWK 0.954** on APTOS 2019 (0.967 with EyePACS pre-training). It has no public code, so we re-implemented it from the text, measured the gap, and tested nine hypotheses on top of it, each with numeric predictions committed before its run.

**Headline.** The paper's recipe, run as published, gives QWK **0.81–0.82**. With a lower learning rate and augmentation our baseline is **0.897 ± 0.004**; the best strict-protocol result is **0.918 [0.900, 0.935]**. The paper's 0.954 was not reproduced. The one idea that helped consistently is using the *order* of the grades (ordinal regression 0.914, soft ordinal labels 0.916, about 26–32% fewer large errors); everything else landed between 0.90 and 0.92, within run-to-run noise (about ±0.01).

**Team:** Subham Mahapatra, Divyanshi Sachan, Vivek, Shivansh Gupta

> This is a reproduction study on a public research dataset. **No clinical claims are made.**

## Read first

| | |
|---|---|
| Full project report (base paper, reproduction, every hypothesis with values, failure analysis, conclusion) | [`docs/reports/Full_Project_Report.pdf`](docs/reports/Full_Project_Report.pdf) |
| Experiment index and per-stage summaries | [`experiments/README.md`](experiments/README.md) |
| Earlier reports and slides | [`docs/reports/`](docs/reports/), [`docs/slides/`](docs/slides/) |
| Phase 1 literature review | [`docs/literature/`](docs/literature/) |
| Why we think 0.954 is out of reach under a strict protocol | [`docs/gap-analysis/`](docs/gap-analysis/) |

## Results at a glance

Test set: 550 images (stratified 70/15/15 split, seed 42), never used for a decision; the 95% bootstrap intervals cover test-set noise only. QWK is the mean over seeds, with validation-tuned cut-offs unless noted.

| # | Experiment | Hypothesis | Test QWK | Verdict |
|---|---|---|---|---|
| 01 | [Baseline reproduction](experiments/01_baseline_reproduction/results.md) | Paper recipe as published | 0.81–0.82 | Gap of about 0.13 to the paper |
| 01 | | Paper recipe + lr 1e-4 + augmentation (**our baseline**) | 0.897 ± 0.004 | reference |
| 02 | [Ordinal regression](experiments/02_ordinal_regression/results.md) | H1: grade as an ordered number | 0.914 ± 0.004 | Supported, modest |
| 03–04 | [Levers, backbones, ensembles](experiments/04_ensembles_and_eyepacs/results.md) | H4: close the gap with recipe and ensembles | **0.918** [0.900, 0.935] | Helps; gap of 0.036 remains |
| 05 | [EyePACS fine-tune](experiments/05_eyepacs_finetune/results.md) | H5: EyePACS failed only because of a high lr | OOF 0.906 | Rejected |
| 06 | [Imbalance losses](experiments/06_imbalance_losses/results.md) | H2 focal loss; H3 class-balanced ordinal | 0.911; 0.908 | No gain; weak |
| 07 | [Soft ordinal labels](experiments/07_soft_labels/results.md) | H6: Gaussian soft labels | 0.916 ± 0.005 | Ties H1; fewest large errors |
| 08 | [Blur-aware labels](experiments/08_blur_aware/results.md) | H7: wider label for blurry images | 0.909 ± 0.006 | Not supported |
| 09 | [RETFound backbone](experiments/09_retfound/results.md) | H8: retina pre-training beats ImageNet | 0.899 ± 0.006 | Not supported |
| 10 | [Stacked recipe](experiments/10_stacked_recipe/results.md) | H9: Stage 4b levers stack on soft labels | 0.903 ± 0.002 | Not supported |

The baseline scored 0.897 on Colab (3 seeds) and 0.908 when re-run on Kaggle (seed 42), so differences below about 0.01 are not resolvable.

## Repository structure

```
.
├── README.md
├── AGENTS.md, CLAUDE.md        # project rules for coding agents
├── requirements.txt
├── experiments/                # one self-contained folder per experiment, numbered in run order
│   ├── README.md               # index, per-stage summaries, how executed.ipynb is built
│   ├── 01_baseline_reproduction/  02_ordinal_regression/  03_levers_and_backbones/  04_ensembles_and_eyepacs/
│   ├── 05_eyepacs_finetune/       06_imbalance_losses/    07_soft_labels/           08_blur_aware/
│   ├── 09_retfound/               10_stacked_recipe/
│   │   each: hypothesis.md · NN_name.ipynb · results.md · results/ · kaggle/ (builder, kernels, executed.ipynb)
├── docs/
│   ├── reports/                # Full_Project_Report.pdf, earlier reports, full_report.html, latex/ sources
│   ├── slides/                 # presentations
│   ├── literature/             # Phase 1 literature review
│   └── gap-analysis/           # the three gap analyses and their source notes
└── tools/
    └── attach_kaggle_outputs.py  # rebuilds executed.ipynb (cells + outputs) from a Kaggle kernel log
```

## How to reproduce

**Requirements.** A free GPU (Colab or Kaggle T4). `pip install -r requirements.txt` for a local environment; Colab and Kaggle already have most of it, and the notebooks install `timm` themselves. The images are downloaded by the notebooks, so nothing needs to be prepared by hand.

**Baseline (experiments 01 and 02, Colab).** Open Colab, **File → Upload notebook**, pick [`experiments/01_baseline_reproduction/01_baseline_reproduction.ipynb`](experiments/01_baseline_reproduction/01_baseline_reproduction.ipynb), set **Runtime → T4 GPU**, **Run all**, and allow Drive access. The first run downloads APTOS (about 15 minutes) and caches it in Drive; each experiment then takes 25–35 minutes. Finished runs are reloaded and an interrupted run resumes. Set `QUICK_TEST = True` for a two-minute check.

**Later experiments (Kaggle).** Each stage notebook runs as is on Colab or Kaggle. To run it as a chain of Kaggle kernels, generate and push them (Kaggle CLI 2.x, which also selects the T4):

```bash
python experiments/07_soft_labels/kaggle/build_kernels.py <kaggle-username>
kaggle kernels push -p experiments/07_soft_labels/kaggle/dr-s5-quick   # quick test first
kaggle kernels push -p experiments/07_soft_labels/kaggle/dr-s5-train   # then the full run
```

Experiments 08 to 10 read the Stage 5 soft-label runs from the output of the `dr-s5-train` kernel (attach it as a kernel input); experiments 07 and 10 read the Stage 4e control runs from [`experiments/06_imbalance_losses/results/runs/`](experiments/06_imbalance_losses/results/runs/), which the notebooks fetch from this repository when it is not on disk. Stages 4b to 4d use `kaggle/build_kernels.py` in [`03_levers_and_backbones`](experiments/03_levers_and_backbones/kaggle/) with a stage argument (`4b`, `4c`, `4d`).

**Re-running only the analysis.** Predictions of every run are saved as `outputs.npz` (weights are not committed). The analysis cells of each notebook rebuild the tables, bootstrap intervals and prediction checks from those files without a GPU.

**Hypotheses first.** The `hypothesis.md` of each experiment was committed before its first run; `git log -- experiments/NN_name/hypothesis.md` shows the order.

## Data

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection): 3,662 fundus photographs graded 0 to 4 (No DR 1805, Mild 370, Moderate 999, Severe 193, PDR 295). Images are **not** in this repository because the competition rules do not allow redistribution; the notebooks download a Hugging Face mirror ([`sngsfydy/aptos`](https://huggingface.co/datasets/sngsfydy/aptos)).

## Deviations from the paper

The paper leaves these unstated, so we assumed them (full checklist in [`docs/reports/Stage2_3_Report.pdf`](docs/reports/Stage2_3_Report.pdf)):

- **Split:** stratified 70/15/15, seed 42 (the paper states a split only for EyePACS).
- **Loss:** cross-entropy (implied by the softmax output). **Batch size:** 16. **CLAHE:** clip 2.0, 8×8 tiles on LAB luminance, then a 5×5 Gaussian blur. **Gradient clipping 0.1** read as global L2-norm clipping.
- **Checkpoint rule:** we report the final epoch, the best-validation epoch and both snapshot ensembles.
- The EyePACS pre-training behind the paper's 0.967 was only tried as 5 epochs (experiments 04 and 05).
- Our first local run used 224 px and 30 epochs (Apple M2, 8 GB); all Colab and Kaggle results use the paper's 300 px, 60 epochs.

## Limitations

- Single dataset, no external validation, and a small test set (29 Severe and 44 PDR images), so per-grade recall is noisy.
- Most settings have 3 seeds or fewer; bootstrap intervals cover test-set sampling noise only.
- Duplicate images with conflicting grades add label noise; the random split leaks 29 test images into training (re-scoring without them moves QWK by at most 0.007).
- The test set was opened once (experiment 04). Every later test number, in experiments 05 to 10, is post-test and labelled as such.
- Baselines come from Colab and later runs from Kaggle; the same seed differed by up to 0.011 QWK between platforms.
- Experiment 09 used a third-party copy of the RETFound weights and an untuned fine-tuning recipe.
- No licence has been chosen for this repository yet.

## Citation

```bibtex
@article{chilukoti2024,
  author  = {Chilukoti, Sai Venkatesh and Shan, Liqun and Tida, Vijay Srinivas and Maida, Anthony S. and Hei, Xiali},
  title   = {A reliable diabetic retinopathy grading via transfer learning and ensemble learning with quadratic weighted kappa metric},
  journal = {BMC Medical Informatics and Decision Making},
  volume  = {24},
  pages   = {37},
  year    = {2024},
  doi     = {10.1186/s12911-024-02446-x}
}
```
