# Stage 3: Baseline reproduction, results

The paper's method re-implemented from its text (no public code) and run on APTOS 2019 under the paper's protocol. Method, deviations and the reproducibility checklist are in [`docs/reports/Stage2_3_Report.pdf`](../../docs/reports/Stage2_3_Report.pdf). Notebook: [`01_baseline_reproduction.ipynb`](01_baseline_reproduction.ipynb).

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

