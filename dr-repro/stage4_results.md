# Stage 4 (M4/M5): Results, Prediction Check and Failure Analysis

*Results of [`stage4_ordinal.ipynb`](stage4_ordinal.ipynb), run on a Colab T4 on 18 Oct 2026: 8 runs of 60 epochs, with one disconnect (`mse_s43` resumed after epoch 17). The predictions being checked are in [`stage4_hypothesis.md`](stage4_hypothesis.md), committed before any run. Raw numbers are in [`results/stage4/`](results/stage4/).*

## 1. Reproducibility check

`ce_s42` re-trains the Stage 3 baseline `2_aug_lr1e-4` on the same split. It gave QWK **0.908** (single) and **0.898** (vote), the same as in Stage 3, so the Stage 4 baseline is the Stage 3 baseline.

## 2. Main result (ensemble of 10 snapshots, mean ± std over seeds 42, 43, 44)

| Arm | Trained with | Decision rule | QWK | Accuracy | Macro-F1 | Severe recall | PDR recall | Errors ≥ 2 grades | Ref. sensitivity | Ref. specificity | Ref. AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | cross-entropy | majority vote | 0.897 ± 0.004 | **0.828** | **0.647** | 0.276 | **0.606** | 21.7 | 0.946 | 0.904 | 0.979 |
| B | cross-entropy | expected grade + tuned cut-offs | 0.907 ± 0.007 | 0.776 | 0.594 | 0.437 | 0.568 | **15.7** | **0.982** | 0.879 | 0.979 |
| C | MSE (ordinal) | rounding | 0.896 ± 0.001 | 0.796 | 0.584 | **0.471** | 0.167 | 17.0 | 0.900 | **0.948** | **0.984** |
| **D (hypothesis)** | **MSE (ordinal)** | **tuned cut-offs** | **0.914 ± 0.004** | 0.804 | 0.633 | 0.402 | 0.576 | 16.0 | 0.963 | 0.909 | **0.984** |

Without an ensemble (best-validation epoch only), all four arms are tied: QWK 0.896 (A), 0.897 (B), 0.897 (C) and 0.899 (D).

**D − A per seed** (ensemble, paired bootstrap 95% CI): +0.012 [−0.003, +0.030], +0.023 [+0.006, +0.045], +0.013 [−0.006, +0.033]. On de-duplicated data: +0.010 [−0.004, +0.025]. All four differences are positive; one interval excludes 0.

D (0.914) is the best result of the project, but still well below the paper's 0.954.

## 3. Were the predictions right?

| # | Prediction | Observed | Verdict |
|---|---|---|---|
| P1 | QWK +0.01 to +0.03, CI excludes 0 in ≥ 2 of 3 seeds | +0.016; CI excludes 0 in 1 of 3 seeds | **Not confirmed.** The size and direction are as predicted, but the significance criterion fails. |
| P2 | Errors of ≥ 2 grades fall by ≥ 25% | 21.7 → 16.0 (−26%) | Confirmed (just) |
| P3 | Severe recall +10 points or more | 0.276 → 0.402 (+12.6) | Confirmed |
| P4 | Risk: PDR recall falls | 0.606 → 0.576 | Confirmed (small fall) |
| P5 | Referable AUC changes by ≤ 0.01 | 0.979 → 0.984 | Confirmed, though the +0.005 is consistent across all seeds |
| P6 | Decision rule alone (B − A) < half of D − A | +0.009 vs. +0.016 | **Not confirmed** (by 0.001): the decision rule alone gives more than half of the gain |
| P7 | Ordinal loss beats A even with rounding (C > A) | −0.001 | **Not confirmed** |
| P8 | Same sign of D − A on de-duplicated data | +0.016 main, +0.010 de-duplicated | Confirmed |

Five of eight predictions were confirmed. The central claim (P1) was not confirmed under our own pre-registered test.

## 4. What the ablation shows

1. **Neither part works alone.** The ordinal loss with plain rounding (C) ties the baseline. Tuned cut-offs on the cross-entropy model (B) gain about +0.009, but no seed is significant. Only the combination (D) gives the largest gain. **D − C is significant in 2 of 3 seeds and on de-duplicated data**, so the cut-offs are essential for the MSE model.
2. **Why rounding fails: the regression output is compressed.** With rounding, PDR recall collapses to 0.17. The tuned ensemble cut-offs for D are about [0.3, 1.3, 2.3, 2.9] against rounding's [0.5, 1.5, 2.5, 3.5]. The top cut-off moves down by about 0.6 grades in every seed, so PDR images get outputs well below 4. That is the expected effect of minimising squared error on imbalanced, noisy labels: predictions shrink towards the mean grade (about 1.1). The tuned cut-offs undo this shrinkage. This mechanism is behind P4 and P7.
3. **D trades accuracy for QWK.** Accuracy (0.828 → 0.804) and macro-F1 (0.647 → 0.633) fall while QWK rises. In the seed-42 confusion matrices, Severe→Moderate errors fall from 15 to 7, but Moderate→Severe rises from 5 to 19 and Mild→Moderate from 25 to 38. D makes more one-grade errors and fewer far errors. QWK rewards that trade; accuracy does not. For screening it means more referrals: referable sensitivity rises from 0.946 to 0.963, and specificity barely changes (0.904 → 0.909).
4. **The gain appears only in the ensemble.** As single models the arms are tied (D − A = +0.002 averaged over seeds). Two explanations are possible: averaging 10 regression outputs reduces noise better than a vote, or arm A is held back by the paper's majority vote rather than by its loss. These two explanations are not the same. The post-hoc control `A_soft` (averaged probabilities, arg-max) in section 12 of the notebook separates them. **Its result is pending a re-run of the notebook (no retraining).**

## 5. Failure analysis (arm D, seed 42, ensemble)

- **Label noise sets a floor.** 7 of the 550 test images belong to a group of identical images with conflicting grades. Both A and D get 6 of these 7 wrong (86%), against 16–19% on the other images. No model can be right on both copies of such an image, so these errors are not the model's fault.
- **Errors are concentrated in blurrier photos.** Median sharpness (Laplacian variance) is 62.8 for correctly graded images, 52.1 for one-grade errors and 48.1 for errors of two or more grades. Brightness differs less (92.9 vs. 87.8 and 88.4). The significance test (Mann-Whitney) is in the post-hoc section, pending the re-run.
- **The first Grad-CAM figure is an artefact, not an explanation.** For all eight images, including the correct ones, the heat map peaks in the black top-left corner. With ImageNet weights, the top-left cell of EfficientNet-B3's 10 × 10 feature map is 30–45× stronger than the median cell, even for a uniform grey image. At 300 px, the stride-2 convolutions pad the top and left edges of the image but not the bottom and right ones. That cell dominates after the map is scaled by its maximum. Inside the retina, the faint warm areas do fall on lesions (e.g. exudates in the correct Severe cases). The post-hoc section restricts the map to the retina before scaling, and reports how much of each raw map sits in that corner. This is a lesson for the paper: **saliency maps need sanity checks before they are read as explanations.**
- **Remaining confusions are between neighbours.** Under D, the largest error cells are Mild→Moderate and Moderate→Severe, and 12 PDR images are still called Moderate in seed 42. Clinically these are the difficult boundaries: they depend on counting lesions and on new-vessel growth, which is hard to see at 300 px.

## 6. Limitations

- **Single dataset and split.** One dataset (APTOS 2019), one stratified split, three training seeds and no external validation. The paired bootstrap covers test-set noise; three seeds give only a rough view of training noise.
- **Small test classes.** The test set has 29 Severe images, so one image is 3.4 percentage points of Severe recall.
- **Small tuning set.** The cut-offs are tuned on 550 validation images and move by up to 0.6 between seeds.
- **The comparison is not limited to the loss.** The ensemble comparison changes both the loss and the way snapshots are combined (vote vs. average). This is addressed post hoc, not by design.
- **Grad-CAM is qualitative**, and the raw maps are affected by the padding artefact.
- **No clinical claims are made.** Higher referable sensitivity on a curated competition dataset says nothing about screening performance in practice.
