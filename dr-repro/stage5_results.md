# Stage 5 (M4/M5): Soft Ordinal Labels, results

Hypothesis and predictions: [`stage5_hypothesis.md`](stage5_hypothesis.md). Notebook: [`stage5_soft_ordinal.ipynb`](stage5_soft_ordinal.ipynb). Kaggle kernels: [`kaggle/stage5/`](kaggle/stage5/) (run on a Kaggle T4, 2 Nov 2026). Raw files: [`results/stage5/`](results/stage5/).

Protocol identical to Stage 4: EfficientNet-B3, CLAHE + blur, 300 px, 60 epochs, batch 16, lr 1e-4, flip/rotation augmentation, 10 snapshots, seeds 42/43/44, the same 550-image test set. Only the training target changes. Arms A, B and D are the Stage 4 Colab numbers, not retrained. The test set was opened in Stage 4c, so these numbers drive no earlier choice.

## 1. Results (10-snapshot ensemble, mean ± std over 3 seeds)

| Arm | QWK | Accuracy | Severe recall | PDR recall | Errors ≥ 2 grades | Referable AUC |
|---|---|---|---|---|---|---|
| A: cross-entropy, vote | 0.897 ± 0.004 | 0.828 | 0.28 | 0.61 | 21.7 | 0.979 |
| B: cross-entropy, tuned cut-offs | 0.907 ± 0.007 | 0.776 | 0.44 | 0.57 | 15.7 | 0.979 |
| D: ordinal MSE, tuned cut-offs | 0.914 ± 0.004 | 0.804 | 0.40 | 0.58 | 16.0 | 0.984 |
| G: soft labels (σ 0.5), vote | 0.908 ± 0.004 | 0.839 | 0.24 | 0.58 | 19.7 | 0.983 |
| **H: soft labels (σ 0.5), tuned cut-offs** | **0.916 ± 0.005** | 0.811 | 0.39 | 0.62 | **14.7** | 0.983 |

Per seed, H scores 0.922 / 0.913 / 0.913 and G scores 0.903 / 0.911 / 0.910. The σ = 1.0 ablation (seed 42) gives H 0.914 against 0.922 for σ = 0.5.

## 2. Paired bootstrap (seed 42, 550 test images, 2,000 resamples)

Baselines here are the Kaggle control runs of the same seed, so every contrast is scored image by image.

| Contrast | ΔQWK | 95% interval |
|---|---|---|
| G − A | −0.006 | [−0.022, +0.010] |
| H − B | +0.006 | [−0.003, +0.015] |
| H − D | +0.014 | [−0.001, +0.031] |
| H − A | +0.013 | [−0.004, +0.032] |
| σ 1.0 − σ 0.5 (rule H) | −0.007 | [−0.018, +0.004] |

**No interval excludes 0.** The gains are suggestive but not distinguishable from test-set sampling noise. Note that the seed-42 control for A scores 0.908, above the Colab 3-seed mean of 0.897, which is why G − A is negative on seed 42 while it is +0.011 in the 3-seed means.

## 3. Prediction check

| # | Prediction | Observed | |
|---|---|---|---|
| Q1 | G − A between +0.005 and +0.025 | +0.011 | confirmed |
| Q2 | errors ≥ 2 grades of H at least 20% below A | 21.7 → 14.7 (−32%) | confirmed |
| Q3 | H ties D (abs diff < 0.01) | 0.916 vs 0.914 | confirmed |
| Q4 | Severe recall of G stays ≥ 0.23 | 0.276 → 0.241 | confirmed |
| Q5 | referable AUC of G within 0.01 of A | 0.979 → 0.983 | confirmed |
| Q6 | tuned cut-offs add more than the soft target (H − G > G − A) | +0.008 vs +0.011 | **not confirmed** |
| Q7 | σ 1.0 does not beat σ 0.5 by > 0.005 | 0.914 vs 0.922 | confirmed |
| Q8 | re-scored controls match Stage 4e | gap 0.0000 | confirmed |

Seven of eight held. Q6 failed: here the loss contributed slightly more than the decision rule, unlike Stage 4e where the decision rule mattered more. Q1 and Q3 are confirmed on 3-seed means, but the seed-42 paired intervals (section 2) are wide, so they carry less weight than the table suggests.

## 4. Failure analysis

Per-grade recall on the seed-42 test set (`results/stage5/per_class_recall.csv`):

| Grade | n | A | D | G | H |
|---|---|---|---|---|---|
| No DR | 271 | 0.989 | 0.970 | 0.982 | 0.978 |
| Mild | 56 | 0.375 | 0.589 | 0.446 | **0.268** |
| Moderate | 150 | 0.927 | 0.760 | 0.900 | 0.787 |
| Severe | 29 | 0.276 | 0.414 | 0.276 | **0.448** |
| PDR | 44 | 0.614 | 0.568 | 0.614 | **0.636** |

- **H trades Mild for Severe.** The tuned cut-offs lift Severe recall (0.28 → 0.45) and PDR, but Mild recall falls to 0.27 (from 0.38 under A). The Stage 4 ordinal arm D did not lose Mild (0.59), so this is specific to the soft-label cut-offs on this seed; where the missed Mild images go is in the confusion matrices.
- **Without tuning (G), the soft target alone does not help the rare grades.** Severe recall stays at 0.28, 0.24 on the 3-seed mean. The gain on Severe comes from the cut-offs, not from the soft labels.
- **What the soft target does buy** is fewer large errors under the vote (21.7 → 19.7) and, with tuned cut-offs, the lowest count of any arm (14.7), plus the best PDR recall (0.62) among the tuned arms.
- **Misgraded images:** the confusion matrices and eight misgraded fundus images are in `results/stage5/figures/`. Stage 4 found misgraded images to be blurrier (p = 0.0005). The Stage 5 notebook repeats that sharpness test, but its printed value was not saved to the results folder and has not been read, so no Stage 5 blur result is claimed.

## 5. Conclusion and limitations

Gaussian soft labels with validation-tuned cut-offs reach QWK 0.916 ± 0.005, level with ordinal MSE (0.914) and 0.019 above the paper's cross-entropy pipeline under our protocol, with the fewest errors of two or more grades (14.7). The difference to ordinal MSE is not statistically distinguishable (H − D = +0.014, interval [−0.001, +0.031]). The honest reading is therefore that **soft ordinal labels are an equivalent alternative to ordinal regression, not a clear improvement**. The reproduction gap to the paper's 0.954 is unchanged.

Limitations: one dataset and one split; 29 Severe and 44 PDR test images, so per-grade recall is noisy; paired bootstrap covers test-set noise on seed 42 only; baselines A, B, D come from Colab and the new runs from Kaggle (the controls differ by up to 0.011 QWK); the test set was already opened in Stage 4c, so Stage 5 is a post-test experiment. No clinical claim is made.
