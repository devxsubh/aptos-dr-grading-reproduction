# Reproducing Diabetic Retinopathy Grading on APTOS 2019

A research-reproduction study of **Chilukoti et al., "A reliable diabetic retinopathy grading via transfer learning and ensemble learning with quadratic weighted kappa metric"**, *BMC Medical Informatics and Decision Making* 24:37 (2024). The paper fine-tunes an EfficientNet-B3 with CLAHE + Gaussian-blur preprocessing and a snapshot ensemble, reporting **QWK 0.954** on APTOS 2019 (ImageNet init) and **0.967** (EyePACS init + ensemble).

The paper has no public code, so the method was re-implemented from the text. This repository covers Stage 2 (base-paper selection and reproducibility checklist), Stage 3 (baseline reproduction) and the diagnostic iterations that set the Stage 4 baseline.

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

## Repository layout

```
.
├── Stage2_3_Report.pdf                          # Stage 2 + 3 report (IEEE format)
├── DR_Grading_Stage2_3_Presentation.{pdf,pptx}  # Stage 2 + 3 slides
├── base paper.pdf                               # Chilukoti et al. 2024 (CC BY 4.0)
├── Diabetic Retinopathy Grading Phase 1 (1).pdf # Stage 1 literature review
├── CV_Student_Handout_Project.docx.pdf          # Course brief
├── CV_Sample_Flow.docx.pdf                      # Course worked example
└── dr-repro/
    ├── colab_dr_grading.ipynb   # the full pipeline and all experiments, with saved outputs
    ├── colab_dr_grading.html    # the same notebook exported for reading without Jupyter
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

- One training run (seed) per setting; the confidence intervals cover test-set sampling noise only.
- Small test set: 29 Severe and 44 PDR images, so per-grade recall is noisy.
- The APTOS split, batch size and CLAHE parameters are not stated in the paper and were assumed.
- Duplicate images with conflicting grades add label noise; the random split leaks 29 test images into training.
- Single dataset, no external validation, no EyePACS pre-training (the paper's 0.967 is out of scope).

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
