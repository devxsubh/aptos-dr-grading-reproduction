# Stage 8 (M4/M5): Stacking the Stage 4b levers on soft labels, results

Hypothesis and predictions: [`stage8_hypothesis.md`](stage8_hypothesis.md). Notebook: [`stage8_stacked_recipe.ipynb`](stage8_stacked_recipe.ipynb). Kaggle kernels: [`kaggle/stage8/`](kaggle/stage8/) (T4). Raw files: [`results/stage8/`](results/stage8/).

A **single** soft-label EfficientNet-B3 (sigma 0.5) in Stage 4b's 20-epoch recipe (1 warm-up epoch, cosine decay, EMA 0.998 of the weights, 8-view test-time augmentation). `R0` = that recipe alone; `FULL` = R0 + peak lr 3e-4 + strong augmentation + Ben Graham pre-processing + removal of 44 contradictory-duplicate training images. Three seeds each, plus four leave-one-out runs (seed 42). Baselines are not retrained: the Stage 5 soft-label runs (60 epochs, 10-snapshot ensemble) and the paper-style cross-entropy control `ce_s42_k` (Kaggle, seed 42). Post-test experiment: it drives no earlier choice.

## 1. Results (test set, mean ± std over seeds)

| Model | Arm | QWK | Accuracy | Severe recall | PDR recall | Errors ≥ 2 grades | Referable AUC |
|---|---|---|---|---|---|---|---|
| A: CE control (Kaggle, seed 42) | vote | 0.908 | 0.842 | 0.28 | 0.61 | 20 | 0.975 |
| Stage 5: 60 epochs, 10 snapshots | G: vote | 0.908 ± 0.004 | 0.839 | 0.24 | 0.58 | 19.7 | 0.983 |
| Stage 5: 60 epochs, 10 snapshots | **H: tuned** | **0.916 ± 0.005** | 0.811 | 0.39 | 0.62 | 14.7 | 0.983 |
| R0: cheap recipe, single model | G: arg-max | 0.903 ± 0.003 | 0.832 | 0.39 | 0.54 | 20.0 | 0.981 |
| R0: cheap recipe, single model | H: tuned | 0.904 ± 0.003 | 0.768 | 0.54 | 0.52 | 17.0 | 0.981 |
| FULL: stacked recipe, single model | G: arg-max | 0.895 ± 0.008 | 0.828 | 0.20 | 0.59 | 21.0 | 0.982 |
| FULL: stacked recipe, single model | H: tuned | 0.903 ± 0.002 | 0.808 | 0.47 | 0.48 | 16.7 | 0.982 |

FULL per seed (H): 0.902 / 0.901 / 0.906. R0 per seed (H): 0.906 / 0.906 / 0.901.

## 2. Leave-one-out (seed 42; one lever removed from FULL)

| Run | Test QWK (H, TTA) | Cross-fitted validation QWK (TTA) |
|---|---|---|
| FULL | 0.902 | 0.917 |
| FULL − lr 3e-4 | 0.900 | 0.917 |
| FULL − strong augmentation | 0.906 | 0.911 |
| FULL − Ben Graham | **0.916** | 0.917 |
| FULL − label cleaning | **0.915** | **0.922** |
| R0 (all four removed) | 0.906 | 0.903 |

Single runs, so differences of about ±0.01 are noise. The two removals that raise test QWK are Ben Graham and label cleaning.

## 3. Paired bootstrap (seed 42, 550 test images)

| Contrast | ΔQWK | 95% interval |
|---|---|---|
| FULL − Stage 5 (H) | −0.020 | [−0.039, −0.003] |
| FULL − A (CE control) | −0.006 | [−0.022, +0.009] |
| R0 − Stage 5 (H) | −0.015 | [−0.028, −0.004] |
| FULL − R0 | −0.004 | [−0.025, +0.015] |

## 4. Prediction check

| # | Prediction | Observed | |
|---|---|---|---|
| T1 | FULL H test QWK ≥ 0.920 | 0.903 | **not confirmed** |
| T2 | FULL H − R0 H ≥ +0.005 | −0.001 | **not confirmed** |
| T3 | R0 H within 0.01 of Stage 5 H | 0.904 vs 0.916 (−0.012) | not confirmed |
| T4 | errors ≥ 2 grades of FULL H ≤ 16.2 | 16.7 | not confirmed |
| T5 | ≥ 3 of 4 leave-one-out runs have lower cross-fitted val QWK than FULL | 1 of 4 (only dropping augmentation) | not confirmed |
| T6 | seed 42: FULL H − A ≥ +0.015 | −0.006 | **not confirmed** |
| T7 | TTA raises FULL H QWK by ≥ +0.002 | +0.0013 | not confirmed |
| T8 | sanity: cleaning drops 40–90 images | 44 | confirmed |

Only the sanity check held. The hypothesis is not supported, and the stacked recipe is *worse* than Stage 5 in a way that the paired bootstrap resolves (interval excludes 0).

## 5. Conclusion and limitations

**The levers did not stack, and they did not carry over.** The four changes that helped in Stage 4b (each by +0.010 to +0.017 on cross-fitted validation QWK, measured on an ordinal-regression model with 2 seeds) added nothing here: FULL scores 0.903, the cheap recipe alone 0.904, and both are below the Stage 5 snapshot ensemble (0.916). In the leave-one-out runs, removing Ben Graham or label cleaning *raised* test QWK to about 0.916, so on this model and split those two levers may even hurt, though with one seed per run this is suggestive only.

Two separate things are going on:
1. **The cheap recipe costs about 0.012 QWK.** R0 (a single 20-epoch EMA model) is 0.012 below Stage 5 (60 epochs, ensemble of 10 snapshots); this is a model-averaging and training-length effect, not a lever effect, and the price of being three times cheaper.
2. **The levers' Stage 4b gains did not transfer.** They were measured on validation with an ordinal MSE head. Here, FULL's cross-fitted validation QWK (0.917) is higher than R0's (0.903), as in Stage 4b, but this does not carry over to the test set (0.903 vs 0.904). Either the validation gain is partly selection optimism, or the levers' effects overlap with what soft labels and EMA already provide.

The practical ceiling for a single EfficientNet-B3 on this split therefore stays at about 0.90–0.92. The best single-configuration result remains Stage 5 H (0.916), and the best overall result remains the Stage 4c ensemble (0.918).

Limitations: leave-one-out runs have a single seed each; seed std ≈ 0.003–0.008 is comparable to the effects; the single-model vs. snapshot-ensemble comparison confounds recipe length with ensembling; the 550-image test set resolves differences only down to about ±0.01; post-test experiment; no clinical claim.
