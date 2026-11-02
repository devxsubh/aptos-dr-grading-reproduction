# Stage 6 (M4/M5): Blur-aware soft labels, results

Hypothesis and predictions: [`stage6_hypothesis.md`](stage6_hypothesis.md). Notebook: [`stage6_blur_aware.ipynb`](stage6_blur_aware.ipynb). Kaggle kernels: [`kaggle/stage6/`](kaggle/stage6/) (T4, 2 Nov 2026). Raw files: [`results/stage6/`](results/stage6/).

Protocol identical to Stage 5 (EfficientNet-B3, soft labels, 300 px, 60 epochs, 10 snapshots, seeds 42/43/44, same 550 test images). Only the per-image label width changes: sigma_i = 0.7 − 0.4·r_i, with r_i the sharpness rank among the training images (0 = blurriest). Training sigma ranged 0.30–0.70 with mean 0.500. Baseline = the Stage 5 soft-label runs, not retrained. Post-test experiment: it drives no earlier choice.

## 1. Results (10-snapshot ensemble, mean ± std over 3 seeds)

| Arm | QWK | Accuracy | Severe recall | PDR recall | Errors ≥ 2 grades | Referable AUC |
|---|---|---|---|---|---|---|
| Stage 5 G: fixed sigma, vote | 0.908 ± 0.004 | 0.839 | 0.24 | 0.58 | 19.7 | 0.983 |
| Stage 5 H: fixed sigma, tuned | **0.916 ± 0.005** | 0.811 | 0.39 | 0.62 | 14.7 | 0.983 |
| Stage 6 G: blur-aware, vote | 0.907 ± 0.006 | 0.832 | 0.26 | 0.59 | 18.7 | 0.980 |
| Stage 6 H: blur-aware, tuned | 0.909 ± 0.006 | 0.779 | **0.48** | 0.60 | 14.7 | 0.980 |

Per seed, Stage 6 H scores 0.914 / 0.912 / 0.902 against 0.922 / 0.913 / 0.913 for Stage 5 H. The shuffled-sigma control (seed 42) scores H 0.920, G 0.906.

## 2. Error rate by sharpness of the test image

Share of test images graded wrongly, by thirds of test-image sharpness (10-snapshot ensemble, mean over seeds):

| Arm | blurriest third | middle third | sharpest third |
|---|---|---|---|
| Stage 5 G (fixed) | 0.226 | 0.191 | 0.067 |
| Stage 6 G (blur-aware) | 0.224 | 0.213 | 0.067 |
| Stage 5 H (fixed) | 0.262 | 0.235 | 0.071 |
| Stage 6 H (blur-aware) | **0.317** | 0.266 | 0.080 |
| control H (shuffled sigma, seed 42) | 0.279 | 0.230 | 0.076 |

Blur does predict errors strongly: under the baseline, the blurriest third is wrong 26% of the time against 7% for the sharpest. But training with blur-dependent label width did **not** reduce the error rate on blurry images.

## 3. Paired bootstrap (seed 42, 550 test images)

| Contrast | ΔQWK | 95% interval |
|---|---|---|
| H(blur-aware) − H(Stage 5) | −0.008 | [−0.019, +0.002] |
| G(blur-aware) − G(Stage 5) | +0.010 | [+0.0004, +0.022] |
| H(shuffled) − H(blur-aware) | +0.006 | [−0.004, +0.019] |

Only one interval excludes 0: the vote arm G gains +0.010 over its Stage 5 counterpart on seed 42. It is not supported by the 3-seed means (0.907 vs 0.908), so we do not claim it.

## 4. Prediction check

| # | Prediction | Observed | |
|---|---|---|---|
| R1 | QWK of H_b within 0.01 of H | 0.909 vs 0.916 (−0.007) | confirmed |
| R2 | error share in the blurriest third falls by ≥ 2 points | 0.262 → 0.317 (+0.055) | **not confirmed** |
| R3 | errors ≥ 2 grades of H_b ≤ 1.10 × H | 14.7 → 14.7 | confirmed |
| R4 | shuffled control is not better on the blurriest third | real 0.273, shuffled 0.279 | confirmed |
| R5 | sharpest third not hurt (≤ +1 point) | 0.071 → 0.080 | confirmed |
| R6 | referable AUC within 0.01 | 0.983 → 0.980 | confirmed |
| R7 | sanity: blurriest third has more errors than the sharpest | 0.262 vs 0.071 | confirmed |

Six of seven held, but the one that failed is the prediction the hypothesis stands on (R2). R1, R3, R5, R6 and R7 only say that nothing got worse or that the data behave as expected.

## 5. Conclusion and limitations

**The hypothesis is not supported.** Making the soft label wider for blurry training images did not lower the error rate on blurry test images (it rose from 26% to 32% under the tuned rule), and QWK did not improve (0.909 vs 0.916, interval including 0). The one clearly different number is Severe recall under the tuned rule (0.39 → 0.48), but with a seed std of 0.07–0.09 and a Mild/accuracy cost (accuracy 0.811 → 0.779), we do not claim a benefit. The shuffled control is no worse than the blur-aware run on the blurriest third (R4), which also means we cannot attribute any effect to blur.

A plausible reason, not tested here: the blurriest images are hard because the lesions are not visible, and no choice of target softness can restore that information; wider labels only make the model less sure about them. Blur is a strong marker of difficulty in the data but not something this form of label smoothing can fix. Image restoration or quality-based exclusion at inference would be different hypotheses.

Limitations: one sharpness measure (Laplacian variance); a single width schedule (0.3–0.7); the paired bootstrap covers test-set noise on seed 42 only; seed std ≈ 0.005 QWK is comparable to every effect seen here; post-test experiment; no clinical claim.
