# Stage 8 (M3): Hypothesis and Prediction. Stacking the proven training levers on soft labels

*Written **before** any Stage 8 code was run. Commit this file before running [`stage8_stacked_recipe.ipynb`](stage8_stacked_recipe.ipynb), and copy the thresholds in section 4 into its `PREDICTIONS` cell. Stages 2–7 stay in the repository untouched. Stage 8 is a post-test experiment (the test set was opened in Stage 4c); its numbers drive no earlier choice.*

## 1. Starting point

Stages 5–7 tested ideas about the loss, blur and the backbone, and everything landed within about 0.01 QWK of the baseline: the Stage 5 soft-label model (arm H, EfficientNet-B3, 60 epochs, 10 snapshots) scores **0.916 ± 0.005** on the 550 test images, and the paper-style cross-entropy baseline scores 0.897 (Colab, 3 seeds) or 0.908 (Kaggle re-run, seed 42). The platform and re-run noise is about ±0.01.

The larger gains in our history came from *training-recipe* choices, not from new ideas (Stage 4b, measured one at a time on cross-fitted validation QWK with test-time augmentation, 2 seeds, on an ordinal-regression model): peak lr 3e-4 (+0.017), Ben Graham pre-processing (+0.014), strong augmentation (+0.013) and removing contradictory duplicate labels (+0.010). Stage 4c then used them, but only inside a 5-fold ensemble of larger backbones (test QWK 0.918). They have never been combined with soft labels on a single model, and their effects on test were never measured individually.

## 2. Evidence that motivates the change

1. **Each lever helped on its own** in Stage 4b's screening, by more than any Stage 5–7 idea moved the test score.
2. **The levers address different failure causes.** A higher peak lr (with warm-up and decay) makes optimisation reach a better minimum. Strong augmentation (zoom, shift, brightness, gamma) fights over-fitting to camera and exposure differences. Ben Graham pre-processing removes lighting differences and keeps local lesion detail. Label cleaning removes about 60 contradictory training images. Being different, their gains may partly add.
3. **Soft labels reduce large errors** (Stage 5: errors of ≥ 2 grades 21.7 → 14.7), which is complementary to what the recipe levers do (smoother optimisation, better input).
4. **The 20-epoch recipe is three times cheaper** (Stage 4b's R0: warm-up, cosine decay, EMA of the weights, TTA), so a stacked model costs less than the 60-epoch, 10-snapshot Stage 5 model.

## 3. Hypothesis (one change, with its ablation)

> **Training the soft-label EfficientNet-B3 with the four Stage 4b levers together (peak lr 3e-4, strong augmentation, Ben Graham pre-processing, contradictory duplicates removed from training), inside the 20-epoch EMA recipe and scored with test-time augmentation, will raise single-model test QWK above the Stage 5 baseline of 0.916, because the levers fix separate causes of error and their gains partly add up.**

**Arms.**

| Run | Recipe | Seeds | Role |
|---|---|---|---|
| `R0_s{42,43,44}` | Stage 4b's R0 (20 epochs, 1 warm-up epoch, cosine decay, EMA 0.998, lr 1e-4, CLAHE, base augmentation) with soft labels (sigma 0.5), no lever | 3 | control: does the cheap recipe alone match Stage 5? |
| `FULL_s{42,43,44}` | R0 + peak lr 3e-4 + strong augmentation + Ben Graham + label cleaning | 3 | **the hypothesis** |
| `minus_{lr,aug,graham,clean}_s42` | FULL with one lever removed | 1 each | which levers carry the gain (leave-one-out) |

**What stays fixed:** EfficientNet-B3, 300 px, batch 16, weight decay 1e-4, gradient clipping 0.1, soft labels (sigma 0.5), the split (stratified 70/15/15, seed 42), the 550 test images. **Single model:** the final EMA weights, no snapshot ensemble; each model is scored with 8-view TTA. Two decision rules as before: **G** arg-max, **H** expected grade with cut-offs tuned on the validation set only.

## 4. Predictions (written before running)

Test set, mean over 3 seeds, rule H unless stated. Stage 5 H = 0.916 ± 0.005, errors of ≥ 2 grades 14.7; paper-style baseline A on Kaggle, seed 42 = 0.908.

| # | Prediction | Counts as confirmed if |
|---|---|---|
| T1 | The stacked recipe beats the Stage 5 baseline | FULL H test QWK ≥ 0.920 |
| T2 | The levers add something on top of the cheap recipe | FULL H − R0 H ≥ +0.005 |
| T3 | The cheap recipe alone matches Stage 5 | abs(R0 H − 0.916) < 0.01 |
| T4 | Large errors do not increase | errors of ≥ 2 grades of FULL H ≤ 16.2 (1.10 × 14.7) |
| T5 | The gain is spread over levers, not one | in the leave-one-out runs (seed 42), at least 3 of 4 have lower cross-fitted validation QWK than FULL |
| T6 | A clear gain over the paper-style baseline | seed 42: FULL H test QWK − A (Kaggle control, 0.908) ≥ +0.015 |
| T7 | TTA helps | mean test QWK of FULL H with TTA − without ≥ +0.002 |
| T8 | *Sanity, not a hypothesis.* Label cleaning removes the expected number of images | between 40 and 90 training images dropped |

**Honest expectation.** The Stage 4b gains were measured one at a time, on an ordinal-regression model, on validation data, and may overlap; they could also be partly used up by soft labels and EMA. We expect FULL H around 0.92–0.93 but put the chance of reaching 0.925 at about 35%, and the chance that FULL is *not* distinguishable from Stage 5 on the seed-42 paired bootstrap at about 50%. If T1 and T2 fail, the conclusion is that the recipe levers do not stack, and the single-model ceiling on this split is about 0.92. The paper's 0.954 would still be unexplained.
