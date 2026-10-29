# Reproducing Diabetic Retinopathy Grading on APTOS 2019

A research-reproduction study of **Chilukoti et al., "A reliable diabetic retinopathy grading via transfer learning and ensemble learning with quadratic weighted kappa metric"**, *BMC Medical Informatics and Decision Making* 24:37 (2024), [open access](https://doi.org/10.1186/s12911-024-02446-x). The paper fine-tunes an EfficientNet-B3 with CLAHE + Gaussian-blur preprocessing and a snapshot ensemble, reporting **QWK 0.954** on APTOS 2019 (ImageNet init) and **0.967** (EyePACS init + ensemble).

The paper has no public code, so the method was re-implemented from the text. This repository covers Stage 2 (base-paper selection and reproducibility checklist), Stage 3 (baseline reproduction), Stage 4 (ordinal regression) and Stages 4b–4d (closing the gap to 0.954 under a strict protocol).

**Current best (strict protocol, test set opened once): QWK 0.918 [0.900, 0.935]**, a 3-member 5-fold ensemble ([Stage 4c](#stages-4b-and-4c-closing-the-gap-to-the-paper)). The paper's 0.954 lies outside this interval.

**Team:** Shivansh, Divyanshi, Vivek, Shubham

**Report:** [`Stage2_3_Report.pdf`](Stage2_3_Report.pdf) (IEEE format, Stages 2 and 3)

> This is a reproduction study. No clinical claims are made.

## Results (paper protocol: 300 px, 60 epochs, batch 16, Colab T4)

Test set: 550 images (stratified 70/15/15 split, seed 42), never used for any decision. "Single" is the epoch with the best validation QWK; "vote" is the 10-snapshot majority vote. One training run per setting.

| Experiment | Changes vs. paper | QWK single / vote | Vote 95% CI | Macro AUC (vote) | Severe recall (vote) | PDR recall (vote) |
|---|---|---|---|---|---|---|
| Paper (reported) | | 0.954 | | | | |
| Iteration 1: paper recipe | none (lr 1e-3) | 0.824 / 0.813 | | | | |
| Iteration 1 | lr 1e-4 | 0.846 / 0.884 | | | | |
| `1_crop_aug_lr1e-4` | lr 1e-4, augmentation, retina crop | 0.883 / 0.893 | 0.862–0.918 | 0.934 | 0.24 | 0.57 |
| **`2_aug_lr1e-4`** | **lr 1e-4, augmentation** | **0.908 / 0.898** | 0.868–0.921 | 0.940 | 0.28 | 0.59 |
| `3_crop_aug_lr1e-3` | augmentation, retina crop (paper lr) | 0.838 / 0.831 | 0.805–0.856 | 0.924 | 0.00 | 0.11 |
| `4_aug_lr1e-4_wce` | + class-weighted CE | 0.878 / 0.897 | 0.867–0.921 | 0.942 | **0.48** | 0.55 |
| `5_aug_lr1e-4_cosine` | + cosine lr decay | 0.906 / 0.883 | 0.848–0.909 | 0.935 | 0.28 | 0.52 |
| `6_aug_lr1e-4_wce_cosine` | + both | 0.893 / 0.884 | 0.852–0.909 | 0.942 | 0.24 | 0.55 |

### Key findings

1. **The paper's recipe gives QWK 0.81–0.82**, well below the reported 0.954. At lr 1e-3, validation QWK repeatedly collapses and Severe NPDR is never predicted.
2. **Augmentation plus lr 1e-4** is the best setting (`2_aug_lr1e-4`, 0.908 / 0.898). It is the baseline for Stage 4.
3. **Retina crop and cosine decay did not help.** Class weighting raised Severe recall from 28% to 48% but cost Moderate recall, so QWK did not move.
4. **All lr 1e-4 variants are statistically tied** (overlapping bootstrap intervals, about ±0.03). The paper's 0.954 lies above every interval.
5. **The model ranks severity well but decides poorly:** macro AUC is about 0.94 while macro-F1 is about 0.65, and the remaining errors sit in the rare grades.
6. **APTOS contains 123 groups of byte-identical images**, 30 of them with conflicting grades. 29 of the 550 test images (5.3%) have a twin in the training set.

## Stage 4: ordinal regression

**Hypothesis** ([`dr-repro/stage4_hypothesis.md`](dr-repro/stage4_hypothesis.md), committed before any Stage 4 run): replacing the five-class cross-entropy output with ordinal regression raises QWK and reduces large grading errors. Ordinal regression here means one output, trained with mean-squared error against the grade and decoded with cut-offs tuned on the validation set. It should help because the loss then matches the ordered grades and the quadratic penalty inside QWK.

[`dr-repro/stage4_ordinal.ipynb`](dr-repro/stage4_ordinal.ipynb) tests this against the `2_aug_lr1e-4` baseline. Four arms separate the loss from the decision rule. There are three training seeds on the Stage 3 split, plus one pair of runs on de-duplicated data.

**Results** ([`dr-repro/stage4_results.md`](dr-repro/stage4_results.md), raw files in [`dr-repro/results/stage4/`](dr-repro/results/stage4/)); 10-snapshot ensemble, mean ± std over 3 seeds:

| Arm | QWK | Severe recall | PDR recall | Errors ≥ 2 grades | Referable sensitivity / specificity / AUC |
|---|---|---|---|---|---|
| A: cross-entropy, majority vote (baseline) | 0.897 ± 0.004 | 0.28 | 0.61 | 21.7 | 0.946 / 0.904 / 0.979 |
| B: cross-entropy, tuned cut-offs | 0.907 ± 0.007 | 0.44 | 0.57 | 15.7 | 0.982 / 0.879 / 0.979 |
| C: ordinal, rounding | 0.896 ± 0.001 | 0.47 | 0.17 | 17.0 | 0.900 / 0.948 / 0.984 |
| **D: ordinal, tuned cut-offs (hypothesis)** | **0.914 ± 0.004** | 0.40 | 0.58 | 16.0 | 0.963 / 0.909 / 0.984 |

- **The main prediction (P1) was not confirmed.** D − A is +0.016 and positive for every seed and for the de-duplicated data, but its paired-bootstrap interval excludes 0 for only 1 of 3 seeds.
- **Five of eight predictions were confirmed:** fewer far errors (−26%), higher Severe recall (+13 points), slightly lower PDR recall, an unchanged referable AUC, and the same sign on de-duplicated data.
- **Neither part works alone.** The ordinal loss with plain rounding ties the baseline, because the regression output shrinks towards the mean grade: PDR recall drops to 0.17. The tuned cut-offs undo this shrinkage.
- **The gain appears only in the ensemble.** As single models, all arms tie at about 0.90.
- **The first Grad-CAM figure shows a padding artefact** of EfficientNet-B3 at 300 px (the peak is always in the top-left corner), not lesions.
- **Post-hoc checks, pending a re-run with no training:** a soft-vote baseline to separate "averaging vs. voting" from the loss, and Grad-CAM restricted to the retina.

## Stages 4b and 4c: closing the gap to the paper

**Plans** (each written before its runs): [`dr-repro/stage4b_plan.md`](dr-repro/stage4b_plan.md), [`dr-repro/stage4c_plan.md`](dr-repro/stage4c_plan.md). **Notebooks:** [`dr-repro/stage4b_push.ipynb`](dr-repro/stage4b_push.ipynb), [`dr-repro/stage4c_push.ipynb`](dr-repro/stage4c_push.ipynb), run as a chain of Kaggle T4 kernels generated by [`dr-repro/kaggle/stage4b/build_kernels.py`](dr-repro/kaggle/stage4b/build_kernels.py) (22–23 Oct 2026). **Results for both stages:** [`dr-repro/stage4c_results.md`](dr-repro/stage4c_results.md), raw files in [`dr-repro/results/stage4b/`](dr-repro/results/stage4b/).

Protocol: ImageNet-pretrained backbones, APTOS data only, the same 550 test images, every choice made on validation or out-of-fold (OOF) predictions, and the test set opened once, after all Stage 4b and 4c training had finished.

- **Stage 4b, Block 1** (300 px, 2 seeds): peak lr 3e-4 (+0.017 val QWK), strong augmentation (+0.013), label cleaning (+0.010) and Ben Graham pre-processing (+0.014) were adopted; 30 epochs (−0.006) was not.
- **Stage 4b, Blocks 2–3:** B4 Noisy Student 380 px and B3 384 px were the best backbones. 5-fold ensembles of the two, blended: OOF 0.925, test **0.916**.
- **Stage 4c** added diverse members by a greedy rule fixed in advance: B4-NS with retina crop and an auxiliary head, ConvNeXt-Tiny 448, SE-ResNeXt50 512, B4-NS 512, and B4-NS after 5 epochs of EyePACS-2015 pre-training (external data, never in the headline). Only B4-NS 512 was selected: OOF 0.928, test **0.918 [0.900, 0.935]**.

| Model | Test QWK | Accuracy |
|---|---|---|
| Stage 4 A: cross-entropy, vote (seed 42) | 0.898 | 0.829 |
| Stage 4b headline: B4-NS 380 + B3 384 | 0.916 | 0.802 |
| **Stage 4c headline: + B4-NS 512** | **0.918** | 0.807 |
| *Optimistic, not a result:* subset and cut-offs both chosen on test | 0.928 | 0.791 |
| Paper (reported) | 0.954 | 0.942 |

- **Four of seven Stage 4c predictions were confirmed.** Architecture diversity (Q2), EyePACS pre-training (Q4) and the optimistic rows (Q7) were not confirmed.
- **Protocol choices we can measure do not explain the gap.** Choosing cut-offs, members and subsets on the test set reaches only 0.928, and no model here exceeds 0.83 accuracy against the paper's 0.942.
- **Test images with an identical twin in train + val do not inflate the scores** (QWK with and without them differs by ≤ 0.007).

## Stage 4d: corrected EyePACS fine-tune (post-test)

[`dr-repro/stage4d_plan.md`](dr-repro/stage4d_plan.md), [`dr-repro/stage4d_push.ipynb`](dr-repro/stage4d_push.ipynb), [`dr-repro/stage4d_results.md`](dr-repro/stage4d_results.md). Written and run after the test set was opened, so it never replaces the Stage 4c headline.

It tested whether the EyePACS arm failed because fine-tuning restarted at a high learning rate. It did not: a low lr (3e-5, no warm-up) removed the mid-training dip but ended lower (OOF 0.906 vs 0.913 for the original arm and 0.920 for the ImageNet start). The 5-epoch EyePACS weights are a worse starting point than Noisy Student ImageNet weights at every lr tried. Two of seven predictions were confirmed, plus one trivially; the headline is unchanged.

## Gap analysis and open work

- [`reports/Closing the APTOS 2019 QWK gap.md`](reports/Closing%20the%20APTOS%202019%20QWK%20gap.md) (before Stage 4c) and [`reports/APTOS QWK gap after Stage 4c.md`](reports/APTOS%20QWK%20gap%20after%20Stage%204c.md) review the literature and the paper's own tables; source notes are in [`research_notes/`](research_notes/). The second report finds that the paper's EyePACS confusion matrices do not reproduce its reported QWKs, and proposes a post-test experiment that splits before vs. after offline augmentation with the paper's recipe. That experiment has not been run.
- Stage 4 post-hoc checks (soft-vote baseline, retina-restricted Grad-CAM) are still pending.

## Repository layout

```
.
├── Stage2_3_Report.pdf                          # Stage 2 + 3 report (IEEE format)
├── DR_Grading_Stage2_3_Presentation.{pdf,pptx}  # Stage 2 + 3 slides
├── Diabetic Retinopathy Grading Phase 1 (1).pdf # Stage 1 literature review
├── reports/                 # gap analyses written before Stage 4c and after it
├── research_notes/          # source notes behind the two gap analyses
├── stage4_runs/             # Stage 4 Colab runs: config, log, outputs.npz per run (weights are not committed)
└── dr-repro/
    ├── colab_dr_grading.ipynb   # the full pipeline and all experiments, with saved outputs
    ├── colab_dr_grading.html    # the same notebook exported for reading without Jupyter
    ├── stage4_hypothesis.md     # M3: hypothesis, predictions, experiment plan, draft Introduction
    ├── stage4_ordinal.ipynb     # M4/M5: ordinal-regression experiments, ablation, statistics, failure analysis (with outputs)
    ├── stage4_results.md        # M4/M5: results, prediction check, failure analysis, limitations
    ├── results/stage4/          # Stage 4 CSVs, cut-offs and figures (from MyDrive/dr-grading/stage4/)
    ├── stage4b_plan.md          # Stage 4b: plan, selection rules and predictions for closing the gap to 0.954
    ├── stage4b_push.ipynb       # Stage 4b: protocol gap, lever screening, backbones, 5-fold CV ensemble
    ├── stage4c_plan.md          # Stage 4c: diverse members and EyePACS arm, blend rule, predictions
    ├── stage4c_push.ipynb       # Stage 4c: Stage 4b plus the new arms and the single test-set opening
    ├── stage4c_results.md       # Stage 4b + 4c: results, prediction check
    ├── stage4d_plan.md          # Stage 4d (post-test): corrected EyePACS fine-tune
    ├── stage4d_push.ipynb       # Stage 4d notebook
    ├── stage4d_results.md       # Stage 4d results, prediction check
    ├── results/stage4b/         # Stage 4b-4d CSVs and figures (from the Kaggle kernel outputs)
    ├── kaggle/stage4b/          # kernel builder and generated Kaggle kernels for Stages 4b-4d
    ├── kaggle/*.ipynb           # earlier Kaggle train/submit notebooks and their builder
    ├── archive/                 # superseded notebooks (iteration-2 run with its training logs; Kaggle iteration-1 version)
    ├── runs/  results/          # Stage 3 local run (224 px): per-run outputs, figures, duplicate audit, leakage re-scoring
    └── reports/                 # LaTeX sources of the Stage 2 + 3 report
```

`runs/`, `results/` and `reports/` were produced by the earlier local Python package (`src/`, `scripts/`), which has been replaced by the notebook. That code is still in commit `f84d254`: `git checkout f84d254 -- dr-repro/src dr-repro/scripts`.

## Running the notebook (Google Colab)

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**, pick `dr-repro/colab_dr_grading.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all** and allow Google Drive access.

The first run downloads APTOS from the Hugging Face mirror [`sngsfydy/aptos`](https://huggingface.co/datasets/sngsfydy/aptos) (about 15 minutes) and caches it in Drive. Each experiment takes 25–35 minutes on a T4. Finished experiments are loaded from `MyDrive/dr-grading/outputs/` instead of retrained, and an interrupted one resumes after its last completed epoch. Set `QUICK_TEST = True` for a two-minute check.

## Data

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection): 3,662 fundus photographs graded 0 to 4 (No DR 1805, Mild 370, Moderate 999, Severe 193, PDR 295). The images are **not** included in this repository because the competition rules do not allow redistribution.

## Limitations

- Most settings have one training run (Stage 4 used 3 seeds, Stage 4b Block 1 used 2); the confidence intervals cover test-set sampling noise only.
- Small test set: 29 Severe and 44 PDR images, so per-grade recall is noisy.
- The APTOS split, batch size and CLAHE parameters are not stated in the paper and were assumed.
- Duplicate images with conflicting grades add label noise; the random split leaks 29 test images into training.
- Single dataset, no external validation. EyePACS was tried only as 5 epochs of pre-training (Stages 4c-4d), so the paper's 0.967 remains out of scope.
- The test set has been opened once (Stage 4c). Any later test number, such as Stage 4d's, is post-test and labelled as such.

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
