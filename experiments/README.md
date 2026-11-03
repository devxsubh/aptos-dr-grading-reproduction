# Experiments

One folder per experiment, numbered in the order the experiments were run. Every folder has the same layout:

```
NN_name/
├── hypothesis.md      # the hypothesis and numeric predictions, committed BEFORE the first run
├── NN_name.ipynb      # the notebook (source cells; Colab-run notebooks keep their saved outputs)
├── results.md         # results, prediction check, failure analysis
├── results/           # CSVs, figures, per-run logs and kernel logs
├── kaggle/            # kernel builder and the generated kernels; executed.ipynb = cells + outputs of the run
└── runs/ , archive/   # only where an experiment has them (raw run files, superseded notebooks)
```

| # | Folder | Stage | Experiment | Ran on | Notebook with outputs | Headline (test QWK) |
|---|---|---|---|---|---|---|
| 01 | [`01_baseline_reproduction`](01_baseline_reproduction/) | 2–3 | The paper's method, re-implemented, then tuned | Colab, local M2 | [`01_baseline_reproduction.ipynb`](01_baseline_reproduction/01_baseline_reproduction.ipynb) (saved outputs) | 0.81–0.82 as published, 0.897 baseline |
| 02 | [`02_ordinal_regression`](02_ordinal_regression/) | 4 | Ordinal regression (H1) | Colab | [`02_ordinal_regression.ipynb`](02_ordinal_regression/02_ordinal_regression.ipynb) (saved outputs) | 0.914 |
| 03 | [`03_levers_and_backbones`](03_levers_and_backbones/) | 4b | Training levers, resolution, backbones, 5-fold CV (H4) | Kaggle | none recoverable¹ | levers +0.010 to +0.017 (validation) |
| 04 | [`04_ensembles_and_eyepacs`](04_ensembles_and_eyepacs/) | 4c | Diverse members, EyePACS arm, test set opened once (H4) | Kaggle | none recoverable¹ | **0.918** (best overall) |
| 05 | [`05_eyepacs_finetune`](05_eyepacs_finetune/) | 4d | Corrected EyePACS fine-tune (H5) | Kaggle | none recoverable¹ | OOF 0.906 (rejected) |
| 06 | [`06_imbalance_losses`](06_imbalance_losses/) | 4e | Focal loss and class-balanced ordinal (H2, H3) | Kaggle | [`kaggle/dr-s4e-*/executed.ipynb`](06_imbalance_losses/kaggle/) | 0.911 / 0.908 |
| 07 | [`07_soft_labels`](07_soft_labels/) | 5 | Gaussian soft ordinal labels (H6) | Kaggle | [`kaggle/dr-s5-train/executed.ipynb`](07_soft_labels/kaggle/dr-s5-train/executed.ipynb) | 0.916 |
| 08 | [`08_blur_aware`](08_blur_aware/) | 6 | Blur-aware label width (H7) | Kaggle | [`kaggle/dr-s6-train/executed.ipynb`](08_blur_aware/kaggle/dr-s6-train/executed.ipynb) | 0.909 |
| 09 | [`09_retfound`](09_retfound/) | 7 | Retina-pretrained ViT-L, RETFound (H8) | Kaggle | [`kaggle/dr-s7-train/executed.ipynb`](09_retfound/kaggle/dr-s7-train/executed.ipynb) | 0.899 |
| 10 | [`10_stacked_recipe`](10_stacked_recipe/) | 8 | Stage 4b levers stacked on soft labels (H9) | Kaggle | [`kaggle/dr-s8-train/executed.ipynb`](10_stacked_recipe/kaggle/dr-s8-train/executed.ipynb) | 0.903 |

¹ Experiments 03–05 ran on a teammate's Kaggle account. Their notebooks, generated kernels, result CSVs and figures are here, but their kernel logs were not available to us, so no executed notebook could be built.

**About `executed.ipynb`.** Kaggle returns a kernel's log and saved files but not the executed notebook. `executed.ipynb` therefore contains the cells exactly as they ran with their outputs put back by [`tools/attach_kaggle_outputs.py`](../tools/attach_kaggle_outputs.py): printed text comes from the kernel log (kept in `results/kernel_logs/`), tables and figures from the files the cell saved. A banner at the top of each file says so.

The summaries below were written stage by stage; each links to its experiment folder.

## Stage 4: ordinal regression

**Hypothesis** ([`experiments/02_ordinal_regression/hypothesis.md`](02_ordinal_regression/hypothesis.md), committed before any Stage 4 run): replacing the five-class cross-entropy output with ordinal regression raises QWK and reduces large grading errors. Ordinal regression here means one output, trained with mean-squared error against the grade and decoded with cut-offs tuned on the validation set. It should help because the loss then matches the ordered grades and the quadratic penalty inside QWK.

[`experiments/02_ordinal_regression/02_ordinal_regression.ipynb`](02_ordinal_regression/02_ordinal_regression.ipynb) tests this against the `2_aug_lr1e-4` baseline. Four arms separate the loss from the decision rule. There are three training seeds on the Stage 3 split, plus one pair of runs on de-duplicated data.

**Results** ([`experiments/02_ordinal_regression/results.md`](02_ordinal_regression/results.md), raw files in [`experiments/02_ordinal_regression/results/`](02_ordinal_regression/results/)); 10-snapshot ensemble, mean ± std over 3 seeds:

| Arm | QWK | Severe recall | PDR recall | Errors ≥ 2 grades | Referable sensitivity / specificity / AUC |
|---|---|---|---|---|---|
| A: cross-entropy, majority vote (baseline) | 0.897 ± 0.004 | 0.28 | 0.61 | 21.7 | 0.946 / 0.904 / 0.979 |
| B: cross-entropy, tuned cut-offs | 0.907 ± 0.007 | 0.44 | 0.57 | 15.7 | 0.982 / 0.879 / 0.979 |
| C: ordinal, rounding | 0.896 ± 0.001 | 0.47 | 0.17 | 17.0 | 0.900 / 0.948 / 0.984 |
| **D: ordinal, tuned cut-offs (hypothesis)** | **0.914 ± 0.004** | 0.40 | 0.58 | 16.0 | 0.963 / 0.909 / 0.984 |

- **The main prediction (P1) was not confirmed.** D − A is +0.016 and positive for every seed and for the de-duplicated data, but its paired-bootstrap interval excludes 0 for only 1 of 3 seeds.
- **Five of eight predictions were confirmed:** fewer far errors (−26%), higher Severe recall (+13 points), slightly lower PDR recall, an unchanged referable AUC, and the same sign on de-duplicated data.
- **Neither part works alone.** The ordinal loss with plain rounding ties the baseline, because the regression output shrinks towards the mean grade: PDR recall drops to 0.17. The tuned cut-offs undo this shrinkage.
- **The gain appears only in the ensemble.** As single models, all arms tie at about 0.90.
- **The first Grad-CAM figure shows a padding artefact** of EfficientNet-B3 at 300 px (the peak is always in the top-left corner), not lesions.
- **Post-hoc checks** ([`experiments/02_ordinal_regression/results_posthoc.md`](02_ordinal_regression/results_posthoc.md), [`experiments/06_imbalance_losses/results.md`](06_imbalance_losses/results.md)): averaging the CE snapshots' probabilities (A_soft, 0.899) does not close the gap. D − A_soft = +0.015 [+0.002, +0.030], so the gain comes from the loss and cut-offs, not from averaging instead of voting. Misgraded images are blurrier (p = 0.0005). Restricting Grad-CAM to the retina removes the corner artefact in 6 of 8 images.

## Stage 4e: focal loss and class-balanced ordinal regression (post-test)

**Plan** ([`experiments/06_imbalance_losses/hypothesis.md`](06_imbalance_losses/hypothesis.md), predictions Q1–Q8 written before any run), **notebook** [`experiments/06_imbalance_losses/06_imbalance_losses.ipynb`](06_imbalance_losses/06_imbalance_losses.ipynb), Kaggle kernels from [`experiments/06_imbalance_losses/kaggle/`](06_imbalance_losses/kaggle/), **results** [`experiments/06_imbalance_losses/results.md`](06_imbalance_losses/results.md), raw files in [`experiments/06_imbalance_losses/results/`](06_imbalance_losses/results/). Phase 1 proposed five training strategies, and focal loss and class-balanced ordinal regression had not been run. Stage 4e runs both under the Stage 4 protocol (3 seeds). Only the loss changes. It ran after the test set was opened, so its numbers drive no choice.

| Arm (10-snapshot ensemble, mean of 3 seeds) | QWK | Severe recall | PDR recall | Errors ≥ 2 grades |
|---|---|---|---|---|
| A: cross-entropy, vote | 0.897 | 0.28 | 0.61 | 21.7 |
| E_vote: focal, vote | 0.901 | 0.24 | 0.58 | 21.0 |
| E_tuned: focal, tuned cut-offs | 0.911 | 0.54 | 0.53 | 10.0 |
| **D: ordinal MSE, tuned cut-offs** | **0.914** | 0.40 | 0.58 | 16.0 |
| F_tuned: class-balanced MSE, tuned cut-offs | 0.908 | 0.46 | 0.55 | 16.0 |

- **Focal loss ties cross-entropy** under every decision rule (+0.004, all CIs include 0) and does not raise Severe recall under the vote.
- **Class balancing raises Severe recall on top of ordinal** (0.40 → 0.46) at −0.006 QWK. That meets the pre-registered bar, but the gain comes from one seed (0 / +0.17 / 0 by seed), and referable sensitivity falls. **D stays the recommended strategy.**
- **The decision rule matters more than the loss.** Every tuned-cut-off arm scores 0.907–0.914, and every vote or rounding arm 0.888–0.903.
- Six of eight predictions held. Q2 (focal raises Severe recall) and Q7 (class balancing fixes PDR under rounding) failed. Kaggle re-runs of two Stage 4 models land within 0.015 of Colab.

## Stages 4b and 4c: closing the gap to the paper

**Plans** (each written before its runs): [`experiments/03_levers_and_backbones/hypothesis.md`](03_levers_and_backbones/hypothesis.md), [`experiments/04_ensembles_and_eyepacs/hypothesis.md`](04_ensembles_and_eyepacs/hypothesis.md). **Notebooks:** [`experiments/03_levers_and_backbones/03_levers_and_backbones.ipynb`](03_levers_and_backbones/03_levers_and_backbones.ipynb), [`experiments/04_ensembles_and_eyepacs/04_ensembles_and_eyepacs.ipynb`](04_ensembles_and_eyepacs/04_ensembles_and_eyepacs.ipynb), run as a chain of Kaggle T4 kernels generated by [`experiments/03_levers_and_backbones/kaggle/build_kernels.py`](03_levers_and_backbones/kaggle/build_kernels.py) (22–23 Oct 2026). **Results for both stages:** [`experiments/04_ensembles_and_eyepacs/results.md`](04_ensembles_and_eyepacs/results.md), raw files in [`experiments/03_levers_and_backbones/results/`](03_levers_and_backbones/results/).

Protocol: ImageNet-pretrained backbones, APTOS data only, the same 550 test images, every choice made on validation or out-of-fold (OOF) predictions, and the test set opened once, after all Stage 4b and 4c training had finished.

- **Stage 4b, Block 1** (300 px, 2 seeds): peak lr 3e-4 (+0.017 val QWK), strong augmentation (+0.013), label cleaning (+0.010) and Ben Graham pre-processing (+0.014) were adopted; 30 epochs (−0.006) was not.
- **Stage 4b, Blocks 2–3:** B4 Noisy Student 380 px and B3 384 px were the best backbones. 5-fold ensembles of the two, blended: OOF 0.925, test **0.916**.
- **Stage 4c** added diverse members by a greedy rule fixed in advance: B4-NS with retina crop and an auxiliary head, ConvNeXt-Tiny 448, SE-ResNeXt50 512, B4-NS 512, and B4-NS after 5 epochs of EyePACS-2015 pre-training (external data, never in the headline). Only B4-NS 512 was selected: OOF 0.928, test **0.918 [0.900, 0.935]**.

| Model | Test QWK | Accuracy |
|---|---|---|
| Stage 4 A: cross-entropy, vote (seed 42) | 0.898 | 0.829 |
| Stage 4b headline: B4-NS 380 + B3 384 | 0.916 | 0.802 |
| **Stage 4c headline: + B4-NS 512** | **0.918** | 0.807 |
| *Optimistic, not a result:* subset and cut-offs both chosen on test | 0.928 | 0.791 |
| Paper (reported) | 0.954 | 0.942 |

- **Four of seven Stage 4c predictions were confirmed.** Architecture diversity (Q2), EyePACS pre-training (Q4) and the optimistic rows (Q7) were not confirmed.
- **Protocol choices we can measure do not explain the gap.** Choosing cut-offs, members and subsets on the test set reaches only 0.928, and no model here exceeds 0.83 accuracy against the paper's 0.942.
- **Test images with an identical twin in train + val do not inflate the scores** (QWK with and without them differs by ≤ 0.007).

## Stage 4d: corrected EyePACS fine-tune (post-test)

[`experiments/05_eyepacs_finetune/hypothesis.md`](05_eyepacs_finetune/hypothesis.md), [`experiments/05_eyepacs_finetune/05_eyepacs_finetune.ipynb`](05_eyepacs_finetune/05_eyepacs_finetune.ipynb), [`experiments/05_eyepacs_finetune/results.md`](05_eyepacs_finetune/results.md). Written and run after the test set was opened, so it never replaces the Stage 4c headline.

It tested whether the EyePACS arm failed because fine-tuning restarted at a high learning rate. It did not: a low lr (3e-5, no warm-up) removed the mid-training dip but ended lower (OOF 0.906 vs 0.913 for the original arm and 0.920 for the ImageNet start). The 5-epoch EyePACS weights are a worse starting point than Noisy Student ImageNet weights at every lr tried. Two of seven predictions were confirmed, plus one trivially; the headline is unchanged.

## Stages 5–8: four more hypotheses (post-test)

Each stage has a hypothesis file with numeric predictions committed **before** its first run, a notebook (run on a Kaggle T4 through the `build_kernels.py` in each experiment's `kaggle/` folder), and a results file with the prediction check. All four ran after the test set was opened (Stage 4c), so none of their numbers drove an earlier choice. Test set, 550 images, tuned-cut-off rule; baseline A (paper pipeline + augmentation + lr 1e-4, cross-entropy, vote) is 0.897 ± 0.004 on Colab, and 0.908 when re-run on Kaggle (seed 42), so platform and re-run noise is about ±0.01.

| Stage | Hypothesis | Test QWK | Predictions held | Verdict |
|---|---|---|---|---|
| [5](07_soft_labels/hypothesis.md) | Gaussian soft ordinal labels (sigma 0.5) | **0.916 ± 0.005** (3 seeds) | 7 of 8 | Ties ordinal regression (0.914); fewest large errors (14.7 vs 21.7). Not significant on the seed-42 paired bootstrap. |
| [6](08_blur_aware/hypothesis.md) | Label width that grows with image blur | 0.909 ± 0.006 | 6 of 7 | Not supported: error rate on the blurriest third rose (0.26 → 0.32) |
| [7](09_retfound/hypothesis.md) | Retina-pretrained ViT-L (RETFound) beats ImageNet at the same architecture | 0.899 ± 0.006 | 3 of 7 | Not supported: lower than the ImageNet ViT-L control (0.910) and the EfficientNet-B3 |
| [8](10_stacked_recipe/hypothesis.md) | Stage 4b levers (lr 3e-4, strong augmentation, Ben Graham, label cleaning) stack on soft labels | 0.903 ± 0.002 | 1 of 8 | Not supported: no better than the cheap recipe alone (0.904), below Stage 5 |

Results files: [`experiments/07_soft_labels/results.md`](07_soft_labels/results.md), [`experiments/08_blur_aware/results.md`](08_blur_aware/results.md), [`experiments/09_retfound/results.md`](09_retfound/results.md), [`experiments/10_stacked_recipe/results.md`](10_stacked_recipe/results.md); raw files in each experiment's `results/` folder.

- **Only the order of the grades helped.** Ordinal regression (Stage 4) and soft ordinal labels (Stage 5) both reach about 0.914–0.916 and cut errors of two or more grades by 26–32%, at the cost of accuracy and Mild recall.
- **Blur predicts errors but training cannot fix it:** 26% of the blurriest third of test images are graded wrongly against 7% of the sharpest.
- **Everything lands between 0.90 and 0.92** across loss, input handling, backbone and recipe, with differences about the size of the noise. The paper's 0.954 is still not reproduced.
- **RETFound caveat:** the official weights are gated, so Stage 7 used a third-party copy and an untuned fine-tuning recipe. Read it as "not helpful in our setup".

## Gap analysis and open work

- [`docs/gap-analysis/01-closing-the-gap.md`](../docs/gap-analysis/01-closing-the-gap.md) (before Stage 4c) and [`docs/gap-analysis/02-after-stage4c.md`](../docs/gap-analysis/02-after-stage4c.md) review the literature and the paper's own tables; source notes are in [`docs/gap-analysis/notes/`](../docs/gap-analysis/notes/). The second report finds that the paper's EyePACS confusion matrices do not reproduce its reported QWKs, and proposes a post-test experiment that splits before vs. after offline augmentation with the paper's recipe. That experiment has not been run.
- [`docs/gap-analysis/03-beyond-stage4d.md`](../docs/gap-analysis/03-beyond-stage4d.md) (notes in [`docs/gap-analysis/notes/`](../docs/gap-analysis/notes/)) estimates that every remaining strict lever is worth ≤ +0.004 OOF.

