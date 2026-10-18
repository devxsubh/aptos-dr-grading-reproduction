# Stage 4b: Closing the Gap to the Paper (plan and predictions)

*Written on 19 Oct 2026, before any Stage 4b run. Commit it before running [`stage4b_push.ipynb`](stage4b_push.ipynb). This is an engineering study that sits next to the Stage 4 hypothesis test; it does not replace it.*

## 1. Goal

The paper reports QWK **0.954** on APTOS 2019 with an ImageNet-pretrained EfficientNet-B3. Our best result so far is **0.914**: Stage 4 arm D (ordinal regression with tuned cut-offs) as a 10-snapshot ensemble, averaged over 3 seeds. Stage 4b asks two questions:

1. **How much of the 0.04 gap comes from the evaluation protocol rather than the model?** The paper does not state its split or how it chose the epoch. A model chosen on the test set looks better than one chosen on a separate validation set.
2. **Which training choices close the gap, and by how much each?** We test them one at a time, then together, and try to reach or pass 0.954 under a strict protocol: **ImageNet weights only, APTOS data only, the same 550 test images as Stages 3 and 4.**

## 2. The rules (fixed before running)

- **The test set is locked.** Every choice below is made on the validation set (Blocks 1 and 2) or on out-of-fold predictions (Block 3). Test predictions are saved, but they are shown only in the final section, once.
- **The selection rules are written into the notebook before training,** so no choice is made after seeing a test score.
- **Screening metric: cross-fitted validation QWK, with test-time augmentation (TTA).** Cut-offs are tuned on 4/5 of the validation images and applied to the other 1/5, in turn, so a run is never scored on the images that set its cut-offs.
- **Each run uses its final EMA weights.** There is no best-epoch selection, which removes one source of optimistic noise.
- **The headline number is fixed in advance:** the blend of the two Block 3 cross-validation ensembles, with TTA and cut-offs tuned on out-of-fold predictions. Every other row of the final table is reported too.
- **Leakage is reported.** QWK is also given without the test images that have an identical copy in the training data.

## 3. Part 0: protocol gap (no training, uses the Stage 4 outputs)

For the Stage 4 cross-entropy (paper-style) and ordinal runs, we recompute test QWK under three optimistic protocols:

- **model picked on the test set:** the best of the 11 saved models (best-validation epoch plus 10 snapshots);
- **cut-offs tuned on the test set;**
- **both together.**

None of these is a valid result. Together they show how much a protocol alone can add.

**Prediction:** picking on the test set adds +0.005 to +0.015, tuning cut-offs on it adds +0.005 to +0.015, and both together add +0.01 to +0.03. The protocol alone is therefore unlikely to explain the whole gap.

## 4. Block 1: cheap levers (EfficientNet-B3, 300 px, 2 seeds each, ~20 min per lever)

**New base recipe R0:** the Stage 4 ordinal model trained for 20 epochs instead of 60, with a 1-epoch warm-up, cosine learning-rate decay to 0, and an exponential moving average (EMA) of the weights (decay 0.998). The rest (Adam, lr 1e-4, batch 16, CLAHE, augmentation, paper head with 1 output, MSE) is unchanged.

| Lever (one change from R0) | Why it might help | Predicted Δ val QWK |
|---|---|---|
| R0 itself vs. Stage 4 D (single) | the EMA and a decaying lr give a smoother final model, in a third of the time | ±0.005 (mainly a time saving) |
| `lr3e-4` (peak lr 3e-4) | warm-up and decay make a higher peak safe | 0 to +0.005 |
| `strong_aug` (adds zoom 0.9–1.15×, ±5% shift, brightness ×0.8–1.2, gamma 0.8–1.25) | APTOS photos vary in camera and exposure | +0.003 to +0.010 |
| `clean_labels` (drop images whose identical copies have different grades from training) | removes about 60 contradictory labels | 0 to +0.005 |
| `30_epochs` | more training | ±0.003 |
| `graham` (Ben Graham pre-processing instead of CLAHE) | popular in the Kaggle competition | ±0.005 (no consistent gain expected) |

**Adoption rule:** a lever is adopted if its mean gain over the 2 seeds is at least **+0.003**. All adopted levers are combined for Block 2.

**TTA (8 views: 4 rotations × flip)** is measured on every run at no cost. Predicted gain: +0.002 to +0.008.

## 5. Block 2: resolution and backbone (adopted levers, 1 seed, ~1.5 h)

| Candidate | Change | Predicted Δ val QWK vs. B3 at 300 px |
|---|---|---|
| `b3_300` | combined recipe at the paper's size (also checks that adopted levers add up) | reference |
| `b3_384` | higher resolution: microaneurysms (Mild) are only a few pixels wide at 300 px | +0.005 to +0.015 |
| `b4ns_380` | EfficientNet-B4, Noisy Student weights (ImageNet + unlabelled JFT) | +0.005 to +0.015 |
| `v2s21k_384` | EfficientNetV2-S, ImageNet-21k pre-training | +0.005 to +0.015 |

**Selection rule:** the 2 candidates with the highest cross-fitted validation QWK (TTA) go to Block 3.

Using Noisy Student and ImageNet-21k weights goes beyond "ImageNet-1k only". This will be stated wherever the result is reported. No fundus images other than APTOS are used.

## 6. Block 3: 5-fold cross-validation ensemble (top 2 candidates, ~3–5 h)

- **Folds:** 5 folds on train + validation (3,112 images), stratified by grade, with identical images kept in the same fold.
- **Ensemble:** each candidate gives 5 models, 10 in total. Their regression outputs are averaged (blended).
- **Cut-offs** are tuned on the 3,112 out-of-fold predictions. That is 5.7× more images than the 550 used in Stage 4, so the cut-offs should be more stable.
- **Prediction:** the blend gains +0.01 to +0.02 over the best single Block 2 model.

## 7. Overall prediction and honest expectation

- **Final test QWK:** we expect **0.925–0.945**.
- **Reaching 0.954 or more:** we put this at about 20–30%. With ImageNet weights only, strong single models on random APTOS splits usually reach about 0.92–0.93. The test 95% interval is about ±0.025 wide, so the paper's number may still fall inside our interval even if our point estimate is lower.
- **If we do not reach it:** the conclusion is that the remaining gap cannot be closed with documented methods under a strict protocol. Part 0 shows how much of the gap a protocol alone can explain. That is a legitimate and useful finding for a reproduction study.

## 8. Compute budget (Colab T4)

| Block | Runs | Time |
|---|---|---|
| Part 0 | none | seconds |
| Block 1 | 12 | about 2 h |
| Block 2 | 4 | about 1.5 h |
| Block 3 | 10 | about 3–5 h (depends on which candidates win) |

Every run checkpoints after each epoch to `MyDrive/dr-grading/stage4b/`, so the blocks can be spread over several Colab sessions. "Run all" always continues where it stopped.
