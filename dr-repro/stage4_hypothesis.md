# Stage 4 (M3): Hypothesis, Prediction and Introduction

*Written on 18 Oct 2026, before any Stage 4 code was run. Commit this file before running `stage4_ordinal.ipynb`, so the history shows that the predictions came first. If the team changes a number below, change it here and in the `PREDICTIONS` cell of the notebook before training.*

## 1. Starting point (the baseline)

The Stage 4 baseline is the best Stage 3 setting, `2_aug_lr1e-4`: the paper's EfficientNet-B3 pipeline at 300 px, 60 epochs and batch 16, with random flip/rotation augmentation and lr 1e-4. On the 550-image test set (seed 42) it gave:

| | QWK | Macro-F1 | Macro AUC | Severe recall | PDR recall | Errors of ≥ 2 grades |
|---|---|---|---|---|---|---|
| Single model (best val epoch) | 0.908 | 0.684 | 0.933 | 0.34 | 0.68 | 21 |
| 10-snapshot vote | 0.898 | 0.653 | 0.940 | 0.28 | 0.59 | 22 |

## 2. Evidence that motivates the change

1. **The model ranks severity well but turns that ranking into poor grades.** Macro AUC is 0.94 but macro-F1 is only 0.65. Its outputs order patients correctly, but the arg-max over five independent classes rarely picks the rare grades.
2. **Most errors are near-misses on an ordered scale.** About 96% of test images are within one grade. Severe cases are mostly called Moderate (15 of 29), and Mild cases are mostly called Moderate (25 of 56).
3. **Re-weighting classes moved errors but did not reduce them** (Stage 3 experiment 4). Severe recall rose from 28% to 48%, Moderate recall fell, and QWK stayed at 0.897. Re-weighting tells the model which classes matter more. It does not tell it that the classes are ordered.
4. **Cross-entropy ignores the order of the grades.** Calling a Severe case "No DR" costs the same loss as calling it "PDR", yet QWK, and the clinic, treat these mistakes very differently. Our Phase 1 review named this as Gap 3: *ordinal information is underused*.

## 3. Hypothesis (one change)

> **Replacing the five-class cross-entropy output with an ordinal regression output (one number, trained with mean-squared error against the grade 0–4, and turned into a grade by four cut-off points tuned on the validation set) will raise QWK and reduce large grading errors, because the loss then matches the ordered structure of the grades and the quadratic penalty that QWK uses.**

**What changes:** the last layer of the head (512 → 1 instead of 512 → 5), the loss (MSE instead of cross-entropy) and the rule that turns the output into a grade (validation-tuned thresholds instead of arg-max). These three parts are one change because none of them works without the others: a single-number output needs a regression loss and needs thresholds.

**What stays fixed:** data, split (stratified 70/15/15, split seed 42), CLAHE pre-processing, 300 px, augmentation, backbone, the rest of the head, Adam (lr 1e-4, weight decay 1e-4), gradient clipping 0.1, batch 16, 60 epochs and 10 snapshots. The best epoch is chosen on validation QWK in both arms. The test set is used for nothing but the final numbers.

**Why it should work (mechanism):**

- **The loss matches the metric.** MSE penalises an error of *d* grades by *d²*, which is exactly the quadratic weight inside QWK. Cross-entropy penalises every wrong class the same.
- **The decision rule handles imbalance without re-weighting.** Thresholds tuned on validation can move the Moderate/Severe boundary to wherever it maximises agreement, instead of relying on the arg-max, which favours frequent classes.
- **Snapshots average better.** Averaging one number over 10 snapshots is a natural ensemble. A majority vote over five classes throws away how confident each snapshot was.

## 4. Predictions (written before running)

All comparisons are **ordinal (D) vs. baseline (A)** on the 10-snapshot ensemble, averaged over 3 training seeds (42, 43, 44) on the same split, unless stated otherwise.

| # | Prediction | Counts as confirmed if |
|---|---|---|
| P1 | QWK improves by a small but real amount | mean ΔQWK is between +0.01 and +0.03, **and** the paired-bootstrap 95% CI of ΔQWK excludes 0 for at least 2 of 3 seeds |
| P2 | Large errors become rarer | errors of ≥ 2 grades fall by at least 25% |
| P3 | Severe NPDR is found more often | Severe recall rises by at least 10 percentage points |
| P4 | *Risk:* PDR is under-called, because regression pulls predictions towards the middle grades | PDR recall falls (D < A) |
| P5 | Ranking quality does not change, only the decisions do | referable-DR AUC (grade ≥ 2) differs by at most 0.01 |
| P6 | Most of the gain needs ordinal *training*, not just a better decision rule | the post-hoc rule on the old model (B − A) gains less than half of D − A |
| P7 | The ordinal loss helps even without tuned thresholds | plain rounding of the regression output (C) beats A on QWK |
| P8 | The result is not an artefact of duplicate leakage | on de-duplicated data (seed 42), D − A has the same sign as in P1 |

Being wrong is allowed. If P1 fails, the most likely reasons are that thresholds tuned on 550 validation images over-fit (29 Severe images decide one cut-off), or that regression compresses the extreme grades (P4) by more than it helps the middle.

## 5. Experiment plan (M4)

**Four decision rules, two trained models per seed.** Rules B and D add no training, because they reuse the saved outputs.

| Arm | Trained with | Output turned into a grade by | Question it answers |
|---|---|---|---|
| A | cross-entropy (baseline) | arg-max (ensemble: majority vote, as in the paper) | the reference |
| B | cross-entropy (same model as A) | expected grade Σ k·p(k), validation-tuned thresholds | is it only the decision rule? |
| C | MSE regression | rounding to the nearest grade | is it only the loss? |
| D | MSE regression (same model as C) | validation-tuned thresholds | **the hypothesis** |

**Runs:** 2 models × 3 seeds = 6 runs, plus 1 seed of each on de-duplicated data (P8). That is 8 runs of about 30 minutes each on a Colab T4.

**De-duplication (P8 only):** remove every image whose byte-identical copies carry different grades (30 groups), and keep one image of each remaining identical group. Then split as before.

**Metrics.** These are QWK (primary), accuracy, macro-F1, per-grade recall, errors of ≥ 2 grades, the confusion matrix, and the clinical screening view: referable DR (grade ≥ 2) sensitivity, specificity and AUC. The course asks for sensitivity and AUC, not only accuracy. For a fair AUC across both models, the score for "grade ≥ k" is Σ_{j≥k} p(j) for cross-entropy and the regression output itself for MSE.

**Statistics:**

- A **paired bootstrap** (2,000 resamples of the same test images) for each difference such as D − A.
- Mean ± standard deviation over 3 seeds, so that training randomness is visible.
- With 29 Severe test images, one image is 3.4 percentage points of recall, so P3 needs at least 3 more Severe images found.

**Failure analysis (M5):**

- confusion matrices for A and D;
- Grad-CAM on missed Severe cases and on the largest errors;
- error rates on images that have conflicting duplicates;
- whether errors concentrate in blurry or dark photos.

## 6. Draft Introduction (for the IEEE paper)

Diabetic retinopathy (DR) is a leading cause of preventable vision loss, and screening programmes grade fundus photographs on a five-step scale, from no DR to proliferative DR. Automated grading with convolutional networks can support screening where specialists are scarce, and transfer learning from ImageNet has made strong models trainable on public datasets of a few thousand images, such as APTOS 2019 [aptos2019].

Chilukoti *et al.* [chilukoti2024] report a quadratic weighted kappa (QWK) [cohen1968kappa] of 0.954 on APTOS 2019 with an ImageNet-pretrained EfficientNet-B3 [tan2019efficientnet], CLAHE pre-processing [zuiderveld1994clahe] and a snapshot ensemble [huang2017snapshot]. No code was released. Our re-implementation, using the published hyper-parameters, reached a QWK of 0.81–0.82 on a stratified held-out test set. A diagnosis of the training logs led to a lower learning rate and augmentation, which raised this to 0.90–0.91, but no further variant (retina cropping, class-weighted loss, cosine decay) moved it beyond test-set noise.

These experiments point to one specific weakness. The network ranks images by severity well (macro AUC 0.94) but converts that ranking into poor grades (macro-F1 0.65), and its errors are concentrated between neighbouring grades, especially Severe NPDR predicted as Moderate. The model is trained, like the base paper, as a five-way classifier with cross-entropy. That objective treats the grades as unordered labels, even though the disease and the evaluation metric are ordinal. Class re-weighting, a common response to the rarity of Severe and proliferative cases [lin2017focal], shifted errors between grades without reducing them.

We therefore test one change: replacing the classification output with ordinal regression, a single output trained with squared error against the grade and decoded with thresholds chosen on the validation set. Squared error mirrors the quadratic penalty of QWK, and the tuned thresholds handle class imbalance without re-weighting. We stated our predictions before running any experiment, and we separate the effect of the training objective from that of the decision rule with a four-arm ablation, three training seeds and paired bootstrap tests.

Our contributions are:

1. an open re-implementation of the base paper with a documented reproduction gap and its diagnosis;
2. a pre-registered, single-change test of ordinal regression against the reproduced baseline;
3. an ablation that separates the loss from the decision rule;
4. an evaluation that reports per-grade recall, referable-DR sensitivity, specificity and AUC, and a failure analysis, rather than accuracy alone.

No clinical claims are made.

*New references to add to `refs.bib` when the paper is written:* ordinal classification by binary decomposition (Frank & Hall, ECML 2001) and rank-consistent ordinal regression for neural networks (CORAL; Cao, Mirjalili & Raschka, *Pattern Recognition Letters*, 2020), as related work on ordinal outputs.
