# Stage 6 (M3): Hypothesis and Prediction. Blur-aware soft labels

*Written on 2 Nov 2026, **before** any Stage 6 code was run. Commit this file before running the Stage 6 notebook, and copy the thresholds in section 4 into its `PREDICTIONS` cell. Stages 2–5 stay in the repository untouched. Stage 6 is a post-test experiment (the test set was opened in Stage 4c), so its numbers are reported for comparison and drive no earlier choice.*

## 1. Starting point

The baseline is Stage 5 arm H: the paper's EfficientNet-B3 pipeline with Gaussian soft-label targets (σ = 0.5 for every image) and validation-tuned cut-offs. Mean over 3 seeds, 550 test images: QWK 0.916 ± 0.005, Severe recall 0.39, PDR recall 0.62, 14.7 errors of two or more grades. Its three per-seed runs (`soft_s42`, `soft_s43`, `soft_s44`) are reused from the `dr-s5-train` Kaggle kernel, not retrained.

## 2. Evidence that motivates the change

1. **Misgraded images are blurrier.** Stage 4 found sharpness (Laplacian variance) lower for misgraded than for correctly graded test images (Mann-Whitney p = 0.0005, [`stage4_results.md`](stage4_results.md)). This is the only image-quality signal in our results that predicts errors.
2. **A blurry fundus photo shows less of the lesions that define the grade.** Microaneurysms and small haemorrhages separate Mild from Moderate, and are the first thing lost to blur. The label on such an image is still a single hard grade, so the model is asked to be fully confident where the evidence is weak.
3. **Stage 5 gave us a tool for exactly this.** A soft label with width σ says "this grade, give or take σ grades". Stage 5 used one σ for every image. The ablation σ = 0.5 vs 1.0 showed that the average width matters little (0.922 vs 0.914 on seed 42), so the *distribution* of σ across images is the free variable.
4. **No APTOS paper in our literature search modelled image quality explicitly** (searches on 2 Nov 2026 found none). The Phase 1 review listed quality-aware grading under "reliability" work on other datasets.

## 3. Hypothesis (one change)

> **Making the soft-label width depend on image sharpness (a wider label for blurrier training images, a narrower one for sharp images, with the average width kept at σ = 0.5) will lower the error rate on blurry test images and keep or raise QWK, because a blurry image carries less evidence for its exact grade, so the model should not be pushed to be fully certain about it.**

**What changes:** only the per-image σ. Each training image gets σ_i = 0.7 − 0.4·r_i, where r_i ∈ [0, 1] is the image's sharpness rank among the training images (r = 0 the blurriest, r = 1 the sharpest). So σ runs from 0.7 (blurriest) to 0.3 (sharpest) with mean 0.5, the same average as Stage 5. Sharpness is the variance of the Laplacian of the grey image inside the retina (pixels brighter than 10), computed on the cached 512 px images before CLAHE.

**What stays fixed:** everything in Stage 5 (model, split, 300 px, 60 epochs, batch 16, lr 1e-4, augmentation, 10 snapshots, seeds 42/43/44, cut-offs tuned on validation only, same two decision rules).

**Control that separates blur-awareness from "varying σ":** the same set of σ_i values assigned to the training images in random order (`blur_shuffled_s42`, seed 42). If the shuffled control does as well as the real one, the benefit (if any) is regularisation by heterogeneous labels, not blur.

## 4. Predictions (written before running)

G_b and H_b are the new arms (blur-aware soft labels with the vote, and with tuned cut-offs). Comparisons are against Stage 5 arms G and H, 10-snapshot ensemble, mean over 3 seeds, unless stated.

| # | Prediction | Counts as confirmed if |
|---|---|---|
| R1 | QWK does not change much overall | abs(H_b − H) < 0.01 |
| R2 | The error rate on the blurriest third of test images falls | share of misgraded images (any error) in the blurriest test tercile: H_b at least 2 points below H, mean over seeds |
| R3 | Large errors do not increase | errors of ≥ 2 grades of H_b ≤ 1.10 × H (14.7 → ≤ 16.2) |
| R4 | The benefit is specific to blur | shuffled-σ control is **not** better than real blur-aware on the blurriest tercile (seed 42): error share of control ≥ that of H_b |
| R5 | Sharp images are not hurt | error share in the sharpest tercile of H_b ≤ H + 1 point |
| R6 | Ranking quality unchanged | referable AUC within 0.01 of H |
| R7 | *Sanity, not a hypothesis.* Test images are indeed harder when blurry | under H, the error share in the blurriest tercile exceeds the sharpest tercile |

We expect **R1 to be the main risk in the opposite direction**: with a seed spread of about ±0.005 and Stage 5's effect sizes, a null result on QWK is the most likely outcome. If R2 fails and R1 holds, the honest conclusion is that blur-dependent label softness does not help, and the Stage 4 blur correlation is a property of the data, not something training can fix.
