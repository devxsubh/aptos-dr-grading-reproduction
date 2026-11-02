# Stage 5 (M3): Hypothesis, Prediction and Introduction. Soft ordinal labels

*Written on 2 Nov 2026, **before** any Stage 5 code was run. Commit this file before running [`stage5_soft_ordinal.ipynb`](stage5_soft_ordinal.ipynb), so the history shows that the predictions came first. If the team changes a number below, change it here and in the `PREDICTIONS` cell of the notebook before training.*

Stages 2–4e stay in the repository as the record of what we tried. Stage 5 is a new line of work that reuses their protocol, split and baseline numbers. It does not overwrite them.

## 1. Starting point

The baseline is the Stage 4 arm A: the paper's EfficientNet-B3 pipeline at 300 px, 60 epochs, batch 16, with flip/rotation augmentation, lr 1e-4 and five-class cross-entropy, scored by the paper's 10-snapshot majority vote. Mean over 3 seeds on the 550-image test set:

| | QWK | Severe recall | PDR recall | Errors of ≥ 2 grades | Referable AUC |
|---|---|---|---|---|---|
| A: cross-entropy, vote | 0.897 | 0.28 | 0.61 | 21.7 | 0.979 |
| B: cross-entropy + tuned cut-offs | 0.907 | 0.44 | 0.57 | 15.7 | 0.979 |
| D: ordinal MSE + tuned cut-offs (Stage 4 hypothesis) | 0.914 | 0.40 | 0.58 | 16.0 | 0.984 |

## 2. Evidence that motivates the change

1. **Stage 4 showed that using the order of the grades helps** (D − A = +0.017), but ordinal MSE has two weak points. A single regression output shrinks towards the middle grades, so PDR recall under plain rounding fell to 0.17 (arm C). And one number cannot say *how unsure* the model is between two grades.
2. **Stage 4e showed that the decision rule matters more than the loss.** Every arm with validation-tuned cut-offs scored 0.907–0.914, and every vote or rounding arm 0.888–0.903. A loss that keeps five probabilities but still knows the grades are ordered can use both.
3. **Focal loss tied cross-entropy** (Stage 4e), so re-focusing on hard examples is not the missing ingredient. Class weighting moved errors between grades without reducing them (Stage 3).
4. **Almost all errors are one grade off, and some of those labels are themselves uncertain.** About 96% of test images are within one grade, and APTOS has 30 groups of identical images with conflicting grades. A hard one-hot target says "Moderate is 100% right and Severe is 100% wrong", which APTOS's own labels do not support.
5. **The literature points the same way.** Ordinal losses with five outputs (CORN, [DR-Net v2](https://arxiv.org/html/2511.14398v1)) and ordinal regression on ResNet50 ([arXiv 2511.14398](https://papers.cool/arxiv/2511.14398)) are reported for APTOS, but none of these reports a controlled comparison with plain cross-entropy under one protocol, which is what Stage 5 adds.

## 3. Hypothesis (one change)

> **Replacing the one-hot cross-entropy target with a *Gaussian soft-label* target over the grades (each image's label becomes a bell-shaped distribution centred on its true grade, σ = 0.5, so a neighbouring grade is penalised less than a distant one) will raise QWK and reduce large grading errors, because the loss then knows that Moderate is closer to Severe than to No DR, while the network keeps five probabilities and can still output a confident, well-separated prediction for rare grades.**

**What changes:** only the training target and loss. The head still has five outputs and the model is still scored by the same two rules as arms A and B (majority vote, and expected grade with validation-tuned cut-offs).

**What stays fixed:** data, split (stratified 70/15/15, seed 42), CLAHE pre-processing, 300 px, augmentation, backbone, head, Adam (lr 1e-4, weight decay 1e-4), gradient clipping 0.1, batch 16, 60 epochs, 10 snapshots, training seeds 42/43/44. The best epoch is chosen on validation QWK. The test set is used for nothing but the final numbers.

**Why it should work (mechanism):**

- **The target carries the order.** With σ = 0.5 a Moderate image gets about 79% on Moderate and about 10% on each neighbour, and about 0 on the far grades. A near-miss costs little and a far miss costs a lot, which is the shape of the quadratic penalty inside QWK.
- **Labels are noisy at grade borders.** Soft targets stop the network from being pushed to 100% confidence on images whose true grade is arguable, which should reduce the over-fitting seen in Stage 3 (training loss falling to 0.12 while validation QWK stays flat).
- **Five outputs, no shrinkage.** Unlike MSE, the rare grades keep their own output and are not pulled to the mean.

**Ablation (separate arm):** σ = 1.0 (a wider bell: about 40% on the true grade and 24% on each neighbour) on seed 42, to test whether more smoothing keeps helping or blurs the borders.

## 4. Predictions (written before running)

Comparisons use the 10-snapshot ensemble, mean over 3 seeds on the same split. G is soft labels with the vote (compare with A), H is soft labels with tuned cut-offs (compare with B and D).

| # | Prediction | Counts as confirmed if |
|---|---|---|
| Q1 | Soft labels raise QWK under the paper's vote | mean G − A is between +0.005 and +0.025 |
| Q2 | Large errors become rarer | errors of ≥ 2 grades of H are at least 20% below A (21.7 → ≤ 17.4) |
| Q3 | Soft labels tie the Stage 4 ordinal regression | abs(H − D) < 0.01. We expect that both exploit the order of the grades, so they should be roughly equal. |
| Q4 | Severe recall does not collapse under the vote | Severe recall of G ≥ 0.23 (A is 0.28; smoothing could pull mass away from rare grades) |
| Q5 | Ranking quality does not change, only the decisions do | referable-DR AUC of G differs from A by at most 0.01 |
| Q6 | Most of the gain still comes from the decision rule | (H − G) > (G − A): tuned cut-offs add more than the soft target itself |
| Q7 | Over-smoothing does not help | QWK of σ = 1.0 does not exceed σ = 0.5 by more than 0.005 (seed 42, ensemble, rule H) |
| Q8 | *Sanity check, not a hypothesis.* The notebook's re-scoring of the saved Kaggle control runs reproduces the Stage 4e values | seed-42 ensemble QWK of A and D within 0.015 of the numbers in `results/stage4e` |

Being wrong is allowed. The most likely failure is Q1: with only 29 Severe and 44 PDR test images and a Stage 4 seed spread of about ±0.004–0.007, a +0.005 to +0.01 change may be indistinguishable from noise. If Q1 and Q3 both fail, the honest conclusion is that the ordered structure was already captured by Stage 4's regression and soft labels add nothing.

## 5. Experiment plan (M4)

| Run | Loss | Seeds | Output turned into a grade by |
|---|---|---|---|
| `soft_s{42,43,44}` | Gaussian soft-label CE, σ = 0.5 | 42, 43, 44 | **G**: 10-snapshot majority vote · **H**: expected grade, validation-tuned cut-offs |
| `soft_sigma1_s42` | Gaussian soft-label CE, σ = 1.0 | 42 | same two rules (ablation, Q7) |

Arms A, B and D are **not retrained**. Their mean-over-seeds numbers come from `results/stage4/`, and the seed-42 Kaggle control runs (`ce_s42_k`, `mse_s42_k` in `results/stage4e/runs/`, same split) give the paired bootstrap. Four runs of about 30 minutes each on a Colab T4, about 2 hours in total, and the notebook resumes after a disconnect.

**Statistics:** a paired bootstrap over the 550 test images (2,000 resamples) for G − A, H − B and H − D on seed 42, plus the 3-seed mean ± std. An interval that includes 0, or |Δ| < 0.01, is read as "not distinguishable".

## 6. Draft Introduction (M3)

Diabetic retinopathy (DR) is graded on an ordered five-step scale, from No DR to proliferative DR, and automatic grading is judged with quadratic weighted kappa (QWK), which punishes a mistake by the square of its distance. Chilukoti et al. (2024) report QWK 0.954 on APTOS 2019 with an ImageNet-pretrained EfficientNet-B3, CLAHE pre-processing and a snapshot ensemble. Reproducing this method under the stated settings gave QWK 0.81–0.82, and with a lower learning rate and augmentation 0.90, which leaves a gap that we attribute to unreported protocol details. Our earlier experiments showed that the order of the grades is the useful signal: ordinal regression with tuned cut-offs reached 0.914, while focal loss and class weighting did not help. Regression has a weakness, though: it compresses the rare extreme grades. We therefore ask whether keeping five probabilities but training them with Gaussian soft-label targets, which encode the order directly in the loss, gives the benefit of ordinal regression without the shrinkage. We test this single change under one fixed protocol, three seeds and paired bootstrap intervals, and we report failures as well as gains.
