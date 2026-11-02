# Stage 4e results: focal loss and class-balanced ordinal regression (post-test)

Plan and predictions: [`stage4e_plan.md`](stage4e_plan.md) (written before any run). Notebook: [`stage4e_imbalance.ipynb`](stage4e_imbalance.ipynb). Kernels: [`kaggle/stage4e/`](kaggle/stage4e/). Raw outputs: [`results/stage4e/`](results/stage4e/).

**Post-test.** The test set was opened once in Stage 4c. Stage 4e uses the same 550 test images, so every number here is a post-test number. None of them drove a choice. Cut-offs were tuned on validation only, as in Stage 4.

## What ran

Kaggle T4, the Stage 4 recipe (EfficientNet-B3, 300 px, 60 epochs, batch 16, lr 1e-4, augmentation, 10-snapshot ensemble), the Stage 3 split, and seeds 42–44. Only the loss changed.

| Kernel | Runs | Status |
|---|---|---|
| `dr-s4e-e0-quick` | quick test of every code path, image cache | complete |
| `dr-s4e-e1-focal` | `focal_s42`, `focal_s43`, `focal_s44`, control `ce_s42_k` | complete, 60/60 epochs each |
| `dr-s4e-e1-cb` | `mse_cb_s42`, `mse_cb_s43`, `mse_cb_s44`, control `mse_s42_k` | complete, 60/60 epochs each |
| `dr-s4e-e2-final` | analysis only (CPU) | complete |

The analysis re-checked that the split is identical to Stages 3 and 4, with 123 duplicate groups, 30 of them conflicting and 29 test images leaked. The Stage 4 arms A–D, recomputed from the Stage 4 outputs, match `results/stage4/` exactly.

## Results (10-snapshot ensemble, mean ± std over 3 seeds)

| Arm | Loss, decision rule | QWK | Severe recall | PDR recall | Errors ≥ 2 grades | Referable sens. / spec. |
|---|---|---|---|---|---|---|
| A | CE, majority vote | 0.897 ± 0.004 | 0.28 | 0.61 | 21.7 | 0.946 / 0.904 |
| A_soft | CE, averaged probabilities | 0.899 ± 0.001 | 0.25 | 0.61 | 21.3 | 0.949 / 0.904 |
| B | CE, tuned cut-offs | 0.907 ± 0.007 | 0.44 | 0.57 | 15.7 | 0.982 / 0.879 |
| **E_vote** | **focal, majority vote** | 0.901 ± 0.004 | 0.24 | 0.58 | 21.0 | 0.945 / 0.914 |
| **E_soft** | **focal, averaged probabilities** | 0.903 ± 0.001 | 0.23 | 0.60 | 20.7 | 0.948 / 0.915 |
| **E_tuned** | **focal, tuned cut-offs** | 0.911 ± 0.003 | **0.54** | 0.53 | **10.0** | 0.982 / 0.889 |
| C | MSE, rounding | 0.896 ± 0.001 | 0.47 | 0.17 | 17.0 | 0.900 / 0.948 |
| D | MSE, tuned cut-offs (Stage 4 best) | **0.914 ± 0.004** | 0.40 | 0.58 | 16.0 | 0.963 / 0.909 |
| **F_round** | **class-balanced MSE, rounding** | 0.888 ± 0.011 | 0.45 | 0.22 | 18.0 | 0.873 / 0.956 |
| **F_tuned** | **class-balanced MSE, tuned cut-offs** | 0.908 ± 0.001 | 0.46 | 0.55 | 16.0 | 0.943 / 0.915 |

Single models: every arm lies between 0.896 and 0.902 (`results_mean_over_seeds.csv`), as in Stage 4.

Per seed (ensemble):

| Arm | QWK 42 / 43 / 44 | Severe recall 42 / 43 / 44 | Errors ≥ 2 grades 42 / 43 / 44 |
|---|---|---|---|
| D | 0.910 / 0.917 / 0.914 | 0.45 / 0.34 / 0.41 | 17 / 15 / 16 |
| E_tuned | 0.909 / 0.915 / 0.910 | 0.66 / 0.52 / 0.45 | 9 / 10 / 11 |
| F_tuned | 0.908 / 0.908 / 0.907 | 0.45 / 0.52 / 0.41 | 18 / 15 / 15 |

Figures: [`recall_per_grade.png`](results/stage4e/figures/recall_per_grade.png), [`confusion_matrices.png`](results/stage4e/figures/confusion_matrices.png), [`validation_curves.png`](results/stage4e/figures/validation_curves.png).

## Paired bootstrap (ensemble, pooled over 3 seeds, 95% CI)

| Contrast | ΔQWK | 95% CI | Per seed |
|---|---|---|---|
| E_vote − A | +0.004 | [−0.003, +0.011] | −0.001 / +0.008 / +0.004 |
| E_soft − A_soft | +0.004 | [−0.003, +0.012] | +0.004 / +0.004 / +0.005 |
| E_tuned − B | +0.004 | [−0.003, +0.013] | +0.006 / +0.012 / −0.005 |
| D − E_tuned | +0.003 | [−0.007, +0.012] | +0.000 / +0.003 / +0.005 |
| F_tuned − D | −0.006 | [−0.014, +0.000] | −0.002 / −0.009 / −0.008 |
| F_round − C | −0.008 | **[−0.018, −0.001]** | −0.003 / −0.021 / −0.001 |
| F_tuned − A | +0.010 | [−0.005, +0.026] | +0.010 / +0.015 / +0.006 |
| D − A_soft | +0.015 | **[+0.001, +0.030]** | +0.012 / +0.018 / +0.015 |

## Prediction check

| # | Prediction | Observed | Result |
|---|---|---|---|
| Q1 | Focal does not change QWK under the same rule (both differences < 0.01) | E_vote − A +0.004; E_tuned − B +0.004 | **confirmed** |
| Q2 | Focal raises Severe recall under the vote by ≥ 0.05 | 0.28 → 0.24 | **not confirmed** |
| Q3 | Ordinal still beats focal (D > E_tuned) | +0.003 [−0.007, +0.012] | **confirmed** in sign only; a tie |
| Q4 | Class balancing raises Severe recall on top of ordinal by ≥ 0.05 | 0.40 → 0.46 (+0.06) | **confirmed** on the mean, but per seed it is 0 / +0.17 / 0 |
| Q5 | Class balancing costs no QWK (difference < 0.01) | −0.006 [−0.014, +0.000] | **confirmed**, though negative in all three seeds |
| Q6 | Errors ≥ 2 grades rise by no more than 10% | 16.0 → 16.0 | **confirmed** |
| Q7 | Class balancing lifts PDR recall under rounding by ≥ 0.15 | 0.17 → 0.22 | **not confirmed**; QWK falls (−0.008, CI excludes 0) |
| Q8 | Sanity: Kaggle controls within 0.015 of Colab (seed 42, A and D) | A +0.011, D −0.002 | **confirmed** |

Six of eight hold. Q2 and Q7 fail. Q3 holds in sign only, and Q4 rests on one seed.

## Control check (platform)

| Arm, seed 42, ensemble | Colab (Stage 4) | Kaggle (retrained) | Δ |
|---|---|---|---|
| A | 0.898 | 0.908 | +0.011 |
| A_soft | 0.898 | 0.905 | +0.007 |
| B | 0.903 | 0.916 | +0.013 |
| C | 0.897 | 0.898 | +0.001 |
| D | 0.910 | 0.908 | −0.002 |

Q8 passes, but the CE controls moved up by about 0.01, which is larger than the focal-vs-CE contrasts (+0.004). This only strengthens Q1. Against the same-platform CE control, focal seed 42 is lower (E_vote 0.896 vs 0.908). The MSE control did not move. Against it, F_tuned seed 42 ties D (0.908 vs 0.908).

## What this answers

**Phase 1 asked whether class-balanced ordinal learning improves recognition of the rare grades without costing QWK.** By the reading rule fixed in the plan (Q4 and Q5 both hold), the answer is *yes for Severe NPDR*. The evidence is weak, though. The +0.06 gain comes from one seed: 5 more of 29 Severe images found in seed 43, and none in the other two. QWK falls in every seed (−0.006 on average). Referable sensitivity drops from 0.963 to 0.943. PDR recall does not rise. We therefore keep **ordinal regression with tuned cut-offs (D)** as the recommended strategy. Class balancing is at best an optional trade toward Severe recall.

**Focal loss changes nothing under the paper's decision rule.** It ties CE under the vote, under averaging and under tuned cut-offs. Under the vote it does not raise Severe recall either (0.28 → 0.24). This matches the +0.003 reported by Huang et al. (2023) on EyePACS.

**The decision rule matters more than the loss.** Every arm with validation-tuned cut-offs (B, D, E_tuned, F_tuned) reaches 0.907–0.914. Every arm with a vote, averaging or rounding (A, A_soft, E_vote, E_soft, C, F_round) reaches 0.888–0.903. Among the tuned arms, the loss moves QWK by at most 0.007.

**Not predicted, exploratory:** focal loss with tuned cut-offs (E_tuned) makes the fewest errors of two or more grades in every seed (9 / 10 / 11, against 15–18 for B, D and F_tuned). It also has the highest Severe recall (0.54) and referable sensitivity 0.982, at QWK 0.911. The cost is Moderate recall (0.64 vs 0.80 for D) and accuracy (0.755 vs 0.804). This was not a prediction and the test set had already been opened, so it is a lead for future work, not a finding.

## Post-hoc checks (from the Stage 4 models, no retraining)

- **Image quality** (arm D, seed 42): misgraded images are blurrier (median Laplacian variance 50.9 vs 62.8, Mann-Whitney p = 0.0005, n = 109 vs 441) and slightly darker (87.8 vs 92.9, p = 0.046). [`posthoc_image_quality.csv`](results/stage4e/posthoc_image_quality.csv)
- **Retina-restricted Grad-CAM** (`mse_s42`, 300 px, [`gradcam_retina.png`](results/stage4e/figures/gradcam_retina.png)): in the raw maps, the single top-left feature cell holds 27–77% of the map (median 46%) across 8 images. That is the padding artefact found in Stage 4. Masking feature cells outside the retina removes it in 6 of 8 images, and the warm regions then fall on exudates and haemorrhages. In the other 2 images the fundus fills the frame, so the corner cell counts as retina and a small corner blob remains. The maps are qualitative only. [`posthoc_gradcam_corner_share.csv`](results/stage4e/posthoc_gradcam_corner_share.csv)
- **Soft-vote baseline:** see [`stage4_posthoc_results.md`](stage4_posthoc_results.md). This kernel reproduces it: D − A_soft = +0.015 [+0.001, +0.030], against [+0.002, +0.030] locally. The point estimate is identical; the CI ends differ only because the bootstrap draws different resamples. The report quotes the local figure.

## Limitations

- Post-test, on one split, with three seeds. The test set has 29 Severe images, so one image is 3.4 points of Severe recall, and the Q4 gain is five images in one seed.
- Stage 4e ran on Kaggle, while Stage 4 ran on Colab. The controls pass Q8, but the CE arms shifted by about +0.01.
- Focal was tested only at γ = 2 without α, and class balancing only with n^−1/2 weights. No tuning was done, by design.
- No clinical claims are made.

Kaggle GPU used: about 2 × 2.5 h for the training kernels, plus about 25 min for the quick test.
