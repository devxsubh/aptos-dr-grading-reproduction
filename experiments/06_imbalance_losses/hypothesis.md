# Stage 4e: Focal Loss and Class-Balanced Ordinal Regression (plan and predictions)

*Written on 1 Nov 2026, **before** any Stage 4e run. The test set was opened once in Stage 4c ([`experiments/04_ensembles_and_eyepacs/results.md`](../04_ensembles_and_eyepacs/results.md)), so this is a **post-test** experiment. Its test numbers are reported for comparison only, they drive no choice, and they never replace the Stage 4c headline. Notebook: [`experiments/06_imbalance_losses/06_imbalance_losses.ipynb`](06_imbalance_losses.ipynb). Kernels: [`experiments/06_imbalance_losses/kaggle/build_kernels.py`](kaggle/build_kernels.py). The thresholds below are copied into the notebook's `PREDICTIONS` before the first push, so the Kaggle version timestamp records them.*

## 1. Why this stage exists

Our Phase 1 review ended with a research question and a comparison of five training strategies:

> *Can class-balanced ordinal learning improve Severe NPDR and PDR recognition while reducing clinically large grading errors?*
> 01 Standard CE · 02 Weighted CE · 03 Focal loss · 04 Ordinal-aware loss · 05 Class-balanced + ordinal-aware

Stages 3 and 4 tested 01 (CE), 02 (weighted CE, Stage 3 `4_aug_lr1e-4_wce`) and 04 (ordinal MSE, Stage 4 arms C and D). **03 and 05 were never run.** Stage 4e runs them under the Stage 4 protocol, so they sit in the same table as arms A–D.

## 2. Design (fixed now)

**Protocol: identical to Stage 4.** EfficientNet-B3 (ImageNet), the paper's 3-layer head, CLAHE + blur, 300 px, 60 epochs, batch 16, Adam lr 1e-4, weight decay 1e-4, gradient clip 0.1, flip + rotation augmentation, 10 snapshots. Same Stage 3 split (stratified 70/15/15, seed 42) and training seeds 42, 43, 44. The best epoch is chosen on validation QWK with the model's default rule. Cut-offs are tuned on validation only. The only change is the loss.

| Run | Loss | Output | Arms (decision rules) |
|---|---|---|---|
| `focal_s{42,43,44}` | **Focal loss**, γ = 2, no α (Lin et al. 2017). α class-weighting is the weighted-CE axis, already tested in Stage 3. | 5 classes | **E_vote**: 10-snapshot majority vote (the paper's rule, like A) · **E_soft**: averaged probabilities, arg-max (like A_soft) · **E_tuned**: expected grade + tuned cut-offs (like B) |
| `mse_cb_s{42,43,44}` | **Class-balanced MSE**: each sample's squared error is weighted by w_c ∝ n_c^−1/2 for its grade c, scaled so the mean weight over training images is 1. These are the same weights as Stage 3's weighted CE, so one variable changes. | 1 number | **F_round**: rounding (like C) · **F_tuned**: tuned cut-offs (like D) |
| `ce_s42_k`, `mse_s42_k` | Stage 4 `ce_s42` and `mse_s42` retrained on Kaggle | as Stage 4 | Platform control: Stage 4 ran on Colab, Stage 4e runs on Kaggle |

The Stage 4 runs (A, A_soft, B, C, D) are **reused, not retrained**. They come from the Kaggle dataset `dr-stage4-outputs`.

**Post-hoc checks, carried over from Stage 4 section 12** (these need the images, so they run in the analysis kernel): a Mann-Whitney test of sharpness and brightness for misgraded vs. correct test images (arm D, seed 42, ensemble), and Grad-CAM restricted to the retina for the `mse_s42` best-validation weights.

**Statistics:** the mean over 3 seeds, ensemble rows, main split. A paired bootstrap over the 550 test images (2,000 resamples), pooled over seeds, gives one 95% interval per contrast, plus per-seed intervals. |Δ| < 0.01, or an interval that includes 0, is read as "not distinguishable".

**Budget:** 8 training runs at about 35 min each on a Kaggle T4, split across two kernels that run at the same time. That is about 5 GPU-hours, plus about 1 h for the quick test and the analysis kernel.

## 3. Predictions (ensemble, mean over 3 seeds, main split)

Stage 4 reference values: A 0.897 (Severe recall 0.28, PDR 0.61, errors ≥ 2: 21.7) · B 0.907 · C 0.896 (Severe 0.47, PDR 0.17) · D 0.914 (Severe 0.40, PDR 0.58, errors ≥ 2: 16.0).

| # | Prediction | Basis |
|---|---|---|
| Q1 | **Focal does not change QWK under the same rule:** abs(E_vote − A) < 0.01 **and** abs(E_tuned − B) < 0.01 | Focal vs. CE on EyePACS: +0.003 (Huang et al. 2023). Weighted CE did not move QWK in Stage 3. |
| Q2 | **Focal raises Severe recall under the paper's vote:** Severe recall of E_vote − A ≥ +0.05 | Focal down-weights the easy, mostly No-DR images, which pushes predictions away from the majority grades. |
| Q3 | **Ordinal still beats focal:** D − E_tuned > 0 | MSE beat focal by +0.018 on EyePACS (Huang et al. 2023). Stage 4: D > B. |
| Q4 | **Class balancing raises Severe recall on top of ordinal:** Severe recall of F_tuned − D ≥ +0.05 | The Phase 1 research question. Risk: the tuned cut-offs maximise QWK and may undo the shift. |
| Q5 | **Class balancing costs no QWK:** abs(F_tuned − D) < 0.01 | Tuned cut-offs already correct for imbalance in D (Stage 4: D − C = +0.018). |
| Q6 | **Large errors do not increase:** errors ≥ 2 grades of F_tuned ≤ 1.10 × D | Weights of n^−1/2 are mild: about 0.7 for No DR up to 2.1 for Severe. |
| Q7 | **Class balancing counteracts regression shrinkage under rounding:** PDR recall of F_round − C ≥ +0.15 | C's PDR recall of 0.17 comes from outputs shrinking towards the mean grade. Larger weights on rare high grades pull them back. |
| Q8 | *Sanity check, not a hypothesis.* The Kaggle controls land within 0.015 QWK of the Colab originals (seed 42, ensemble: A and D) | Seed-to-seed std in Stage 4 is about 0.004–0.007. |

**How we will read the results.** If Q4 and Q5 hold, class-balanced ordinal learning improves rare-grade recognition at no QWK cost, and the Phase 1 question is answered "yes" for Severe NPDR. If Q4 fails, the tuned cut-offs already absorb what class balancing does. Ordinal regression with tuned cut-offs (D) then stays our recommended strategy, and the answer is "no added benefit". If Q8 fails, every Stage 4e vs. Stage 4 contrast is flagged as confounded by platform, and the Kaggle controls are used as the reference instead. Every prediction is reported as it falls, including failures.

## 4. Kernels

| Kernel | Work | Inputs |
|---|---|---|
| `dr-s4e-e0-quick` | Quick test of every code path (toy sizes, separate folder). Downloads the images and caches them for the other kernels. | `dr-stage4-outputs`, `dr-stage4-weights` |
| `dr-s4e-e1-focal` | `focal_s42`, `focal_s43`, `focal_s44`, `ce_s42_k` | `dr-s4e-e0-quick` |
| `dr-s4e-e1-cb` | `mse_cb_s42`, `mse_cb_s43`, `mse_cb_s44`, `mse_s42_k` | `dr-s4e-e0-quick` |
| `dr-s4e-e2-final` | No training. Tables, bootstrap, prediction check, control check, Mann-Whitney test, retina Grad-CAM. | the three kernels above, `dr-stage4-outputs`, `dr-stage4-weights` |

`dr-stage4-weights` is a private dataset holding only `mse_s42_best_val_weights.pt`, for the Grad-CAM check.
