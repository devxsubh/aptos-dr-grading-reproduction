# Stage 7 (M4/M5): A retina-pretrained backbone (RETFound), results

Hypothesis and predictions: [`stage7_hypothesis.md`](stage7_hypothesis.md). Notebook: [`stage7_retfound.ipynb`](stage7_retfound.ipynb). Kaggle kernels: [`kaggle/stage7/`](kaggle/stage7/) (T4, 2–3 Nov 2026). Raw files: [`results/stage7/`](results/stage7/).

ViT-Large/16 at 224 px, soft-label loss (sigma 0.5), AdamW (peak lr 3e-4, layer-wise decay 0.65, weight decay 0.05, 3 warm-up epochs then cosine), drop-path 0.2, 25 epochs, batch 16, 5 snapshots. Runs: RETFound weights × 3 seeds, and the same ViT-L from ImageNet-21k weights × 1 seed (seed 42) as the control. The baseline is the Stage 5 EfficientNet-B3 (arm H), not retrained. Weights: the ungated copy `bitfount/RETFound_MAE` (colour-fundus checkpoint, CC-BY-NC-4.0). Post-test experiment: it drives no earlier choice.

## 1. Results (snapshot ensemble, mean ± std over seeds)

| Backbone | Arm | QWK | Accuracy | Severe recall | PDR recall | Errors ≥ 2 grades | Referable AUC |
|---|---|---|---|---|---|---|---|
| EfficientNet-B3, ImageNet (Stage 5, 3 seeds) | G: vote | 0.908 ± 0.004 | 0.839 | 0.24 | 0.58 | 19.7 | 0.983 |
| EfficientNet-B3, ImageNet (Stage 5, 3 seeds) | **H: tuned** | **0.916 ± 0.005** | 0.811 | 0.39 | 0.62 | 14.7 | 0.983 |
| ViT-L, ImageNet (control, 1 seed) | G: vote | 0.886 | 0.818 | 0.34 | 0.59 | 22 | 0.980 |
| ViT-L, ImageNet (control, 1 seed) | H: tuned | 0.910 | 0.811 | 0.48 | 0.61 | 17 | 0.980 |
| ViT-L, **RETFound** (3 seeds) | G: vote | 0.893 ± 0.008 | 0.823 | 0.16 | 0.60 | 24.7 | 0.986 |
| ViT-L, **RETFound** (3 seeds) | H: tuned | 0.899 ± 0.006 | 0.772 | 0.44 | 0.48 | 16.3 | 0.986 |

RETFound per seed (H): 0.892 / 0.904 / 0.900. Seed 42 is the only seed with a control: RETFound 0.892 against ImageNet ViT-L 0.910.

## 2. Paired bootstrap (seed 42, 550 test images)

| Contrast | ΔQWK | 95% interval |
|---|---|---|
| RETFound − ImageNet ViT-L (tuned) | −0.018 | [−0.039, +0.004] |
| RETFound − ImageNet ViT-L (vote) | +0.010 | [−0.020, +0.040] |
| RETFound ViT-L − EfficientNet-B3 (tuned) | **−0.030** | **[−0.051, −0.009]** |

Only the last interval excludes 0: on seed 42, the EfficientNet-B3 is better than the RETFound ViT-L.

## 3. Prediction check

| # | Prediction | Observed | |
|---|---|---|---|
| S1 | RETFound beats ImageNet ViT-L by ≥ +0.01 QWK (tuned, seed 42) | 0.892 vs 0.910 (−0.018) | **not confirmed** |
| S2 | rare-grade recall gain ≥ +0.05 | 0.548 → 0.371 (−0.178) | **not confirmed** |
| S3 | RETFound within 0.015 QWK of the CNN | 0.899 vs 0.916 (−0.017) | not confirmed |
| S4 | errors ≥ 2 grades ≤ 1.10 × the CNN | 14.7 → 16.3 | not confirmed |
| S5 | seed std of RETFound tuned QWK > 0.005 | 0.0060 | confirmed |
| S6 | referable AUC ≥ CNN − 0.005 | 0.983 → 0.986 | confirmed |
| S7 | sanity: mirror weights differ from ImageNet | relative difference 2.0 | confirmed |

Three of seven held (S5, S6, S7), but none of them is the hypothesis itself. The two that carry it (S1, S2) both failed, and in the opposite direction.

## 4. Per-grade recall (seed 42, tuned arm)

| Grade | n | EffNet-B3 | ViT-L ImageNet | ViT-L RETFound |
|---|---|---|---|---|
| No DR | 271 | 0.978 | 0.974 | 0.970 |
| Mild | 56 | 0.268 | 0.250 | 0.250 |
| Moderate | 150 | 0.787 | 0.847 | 0.867 |
| Severe | 29 | 0.448 | 0.483 | **0.241** |
| PDR | 44 | 0.636 | 0.614 | **0.500** |

RETFound is better on the common middle grade (Moderate 0.87) but worse on exactly the rare grades the hypothesis was about. On seed 42 it finds only 7 of 29 Severe images. Its referable-DR AUC is the best of the three (0.986), so it ranks patients well but its grade decisions for the rare high grades are worse.

## 5. Conclusion and limitations

**The hypothesis is not supported.** Retina-specific pre-training did not beat ImageNet pre-training at the same architecture (0.892 vs 0.910 on seed 42, interval including 0), did not raise the rare-grade recall (it lowered it), and the best retina ViT-L (mean 0.899) stayed below the EfficientNet-B3 pipeline (0.916; on seed 42 significantly so). In our setting on APTOS, pre-training domain is not the bottleneck.

Things we did not test and cannot rule out, so this is not a verdict on RETFound in general:
- **Under-fitting from our recipe.** RETFound's training loss ended much higher than the ImageNet ViT-L's (0.69 vs 0.56 after 25 epochs). Layer-wise decay 0.65 over 24 blocks leaves the early layers almost frozen, and we did no learning-rate search for either ViT. A recipe tuned for RETFound could behave differently.
- **Short schedule** (25 epochs), 5 snapshots instead of 10, and 224 px (ViT limit) against 300 px for the CNN. Small lesions may need the higher resolution.
- **Control with one seed.** The comparison that isolates pre-training uses a single ImageNet ViT-L run, so S1 rests on one pair of runs. The seed std (0.006) is larger than the observed gap's half.
- **Weights from a third-party mirror**, not the official gated file. The notebook checked only that the patch-embedding weights differ from ImageNet (relative difference exactly 2.00, which we did not investigate further). The different training curve (validation QWK 0.64 after epoch 1 against 0.78) shows the initialisation is different, but we cannot confirm it is the official RETFound checkpoint.
- Single dataset, 29 Severe and 44 PDR test images, post-test experiment, no clinical claim.
