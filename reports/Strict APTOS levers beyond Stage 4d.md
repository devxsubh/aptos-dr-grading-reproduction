# Remaining strict levers stay below detection

No untried lever that fits the strict protocol is expected to add more than about **+0.004 OOF QWK** to our blend on its own. Stacked together, the plausible ones add roughly **+0.003 to +0.008 OOF**. That moves the expected true test QWK from about 0.918 to about **0.92–0.925**. The best value per Kaggle T4 GPU-hour is, in order:

1. Inference and decoding fixes on the saved fold checkpoints: nested cut-offs and multi-scale TTA, under 0.5 h.
2. A short 640 px FixRes-style fine-tune of the B4-NS 512 folds, about 1 h.
3. A B4-NS 380 member trained without Ben Graham pre-processing, for diversity, about 1.5 h.
4. A properly re-tuned ConvNeXt or SE-ResNeXt run, about 4 h for a screen plus 5 folds. This is the only lever with a credible path to a member that is both strong and decorrelated from the blend.

Fold-safe pseudo-labelling of the 1,928 unlabelled Kaggle test images is downgraded to **+0.000 to +0.004**. Its leaderboard gains came from adapting to the shifted Kaggle test distribution, and our hold-out does not have that shift. In-domain self-supervised pre-training measured about zero on APTOS. Label-noise methods lost QWK when tested on real DR noise. Detection is the binding constraint, not ideas. On the 550-image test set the minimum detectable paired improvement at 80% power is about **0.007–0.018**. On the 3,112-image OOF it is about **0.003–0.008**, before seed noise is added. So no remaining lever can be confirmed on the test set, and even OOF can rank only +0.005-sized levers, and only with several seeds per arm. The test set was opened in Stage 4c, so any new test number is post-test, must be labelled as such, must not drive a choice, and must be read as "not distinguishable" when |Δ| < 0.01. Nothing here bears on clinical performance.

## Our 0.928 OOF already sits at the published strict ceiling

Every carefully evaluated result that uses only natural-image weights and APTOS labels lands between 0.92 and 0.934:

- Fang, Kornblith and Schmidt fine-tuned 19 ImageNet models over a per-model grid of optimiser, learning rate and weight decay. The best, ConvNeXt-Tiny, reached **0.9297** on a local 80/20 split, and that split was also their selection split ([Fang et al. 2023](https://ar5iv.labs.arxiv.org/html/2301.04644)).
- nnMobileNet reports **0.934** under 5-fold CV, with a 1000-epoch recipe ([Zhu et al.](https://arxiv.org/html/2306.01289v4)).
- BloodAxe's 2019-only 4-fold SE-ResNeXt50 models scored 0.921–0.923 ([BloodAxe repo](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)).
- SSiT uses a split the same shape as ours (70/15/15, about 550 test images) and self-supervised pre-training on EyePACS, which is off-protocol for us. It still reaches only **0.9297 ± 0.0029** ([SSiT](https://arxiv.org/html/2210.10969v5)).

Retinal-domain pre-training therefore does not visibly raise the APTOS ceiling. Our own Stage 4d result points the same way: a corrected low-learning-rate EyePACS fine-tune reached OOF 0.906, below the Noisy Student ImageNet start at every learning rate tried (`dr-repro/stage4d_results.md`).

Label noise does not explain the ceiling, but it shapes which levers can work. APTOS was graded by a single person. Four ophthalmologists re-graded it, and the original labels matched their consensus on **90.6%** of the 3,178 images where they agreed ([Lin et al., npj Digit Med 2025](https://www.nature.com/articles/s41746-024-01424-x)). Our duplicate audit found 30 of 123 byte-identical groups with conflicting grades. Copy-against-copy QWK in those groups is only 0.77 (`dr-repro/results/duplicates.json`).

A simulation in the notes asks what QWK a perfect predictor would score against labels with independent ±1-grade noise. The answer is about 0.97–0.98, so noise alone does not cap scores near 0.93. The plateau seen across strong teams more likely reflects images that are genuinely ambiguous between grades 1, 2 and 3. For reference, individual retina specialists score 0.82–0.91 QWK against an adjudicated reference on EyePACS-type images ([Krause et al.](https://arxiv.org/pdf/1710.01711)). Our model sits above that band against a single grader's labels. That fits a model that has learned this one grader's convention, which is a statement about agreement with one grader, not about clinical accuracy.

Part of the 0.928 is optimism from our own procedure. A simulation sized to APTOS (in the notes; synthetic Gaussian-noise model) shows that tuning four cut-offs on the 3,112 OOF scores inflates OOF QWK by about **+0.002**. The honest OOF is therefore nearer 0.926. Ensemble size is also close to exhausted. In the one controlled DR curve, ensembling ResNet-50 runs gained only **+0.005 QWK from 1 to 10 models**, with no further gain at 15, and 70% of the gain had arrived by 3 models ([Huang et al.](https://ar5iv.labs.arxiv.org/html/2110.14160)). Our blend already averages 15 fold-models whose predictions correlate at 0.98–0.99. In our own data, a new seed of the same recipe barely decorrelates (ρ 0.992–0.994). A crop or pre-processing variant is moderate (ρ 0.981). A different architecture family decorrelates most (ρ 0.957–0.959) (`dr-repro/results/stage4b/stage4c_members_oof.csv`). The member that would actually move the blend is one that is both decorrelated (ρ ≤ 0.97) and strong (OOF ≥ 0.92). No arm so far has been both.

## A recipe fix, not a new backbone, is the only lever with room above +0.005

Only one controlled sweep gives APTOS QWK for many ImageNet backbones. In it, the gap from ConvNeXt-Tiny (0.9297) down to EfficientNet-B4 (0.9211) is just **+0.009**, and everything newer than VGG falls in a 0.92–0.93 band ([Fang et al.](https://ar5iv.labs.arxiv.org/html/2301.04644)). No stated-protocol APTOS QWK exists for B5/B6/B7 Noisy Student, EfficientNetV2-M/L, ConvNeXt V2, ConvNeXt-S/B IN-22k, Swin V2, MaxViT, EVA-02 or RegNetY. A low-data fine-tuning benchmark finds that ConvNeXt, RegNet and EfficientNet do consistently well across domains, while attention models "tend to perform poorly under low data finetuning" ([arXiv 2406.05612](https://arxiv.org/abs/2406.05612)). Our own estimates for a straight swap, with no APTOS measurement behind any of them, are:

| Backbone swap | Expected single-model QWK change |
|---|---|
| B5-NS 456 | +0.000 to +0.005 |
| EfficientNetV2-M | −0.01 to +0.005 (V2-S already scored only 0.910 for us) |
| Swin, MaxViT or DeiT III at ≤384 px | −0.02 to +0.005 |

The anomaly worth chasing is how badly our non-EfficientNet arms did. Fang's tuned ConvNeXt-T beat B4 by +0.009. Our ConvNeXt-T 448 trailed B4-NS 380 by **0.023** (OOF 0.897 vs 0.920). That relative shortfall of about 0.03 is larger than any backbone effect in the literature. Our SE-ResNeXt50 512 (0.916) likewise sits about 0.007 below BloodAxe's 0.9225–0.9231 for the same family under similar CV ([BloodAxe repo](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)).

Three confounds were changed together in both failed arms, against the EfficientNet members:

- **Learning rate.** Both used the shared AdamW 3e-4 recipe tuned for EfficientNet. In Fang's Kaggle runs at 512 px, the same augmentation gave ResNet-50 **0.861 at lr 1e-3 vs 0.895 at 1e-4**, a 0.034 swing. Inception-ResNet-v2 was flat across those two settings. The best learning rate also moved by 10× with augmentation strength ([Fang et al.](https://ar5iv.labs.arxiv.org/html/2301.04644)).
- **Loss.** Both used SmoothL1, while all three blend members use MSE. On EyePACS, a controlled loss ablation found **MSE beat SmoothL1 by +0.012 QWK** and beat cross-entropy by +0.020 ([Huang et al., Diagnostics 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)).
- **Crop.** Both used the retina crop.

ConvNeXt has a further mismatch: it ran IN-1k 224 px weights at 448 px with no resolution warm-up.

A cheap one-fold screen tests these together. It crosses lr {5e-5, 1e-4} × layer-wise lr decay {0.75, 1.0} × {Ben Graham, plain crop}, with MSE, on ConvNeXt-Small IN-22k-384 weights at 384 px, at about 20–30 min per configuration. Stage 4d showed that fold-to-fold noise is about ±0.01, so a single fold can only detect recovery of the large ~0.02–0.03 shortfall, not fine differences between configurations. Promote one configuration to 5 folds only if its fold-0 score is within 0.005 of B4-NS on the same fold. Pixel-scaled timm throughput puts ConvNeXt-Small at 384 px at about 1.2× B4-NS 380, or roughly 1.7 h for 5 folds. These are unmeasured estimates ([timm benchmark](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv)).

A recovered SE-ResNeXt or ConvNeXt at OOF ≥ 0.92 and ρ ≈ 0.96 is the one plausible way to add +0.003 to +0.005 to the blend. Our existing SE-ResNeXt already added +0.0025 at OOF 0.916.

Protocol status needs an explicit decision before widening the backbone search:

- **Clearly in protocol:** supervised IN-1k and IN-21k/22k weights.
- **Already accepted, same kind:** Noisy Student weights. These were trained with ImageNet labels plus pseudo-labelled unlabelled JFT images ([Xie et al.](https://arxiv.org/abs/1911.04252v4)), and they are in our blend. DINOv2/v3 is the same kind of thing: unlabelled curated web images.
- **Further out:** CLIP, OpenCLIP and SigLIP, which learn from web image-text pairs.

Within the clearly compliant set, "better ImageNet weights" is a real lever. In Fang's local-split experiments, a RandAug-pre-trained ResNet-50 beat the torchvision checkpoint by +0.009 at fixed downstream augmentation ([Fang et al.](https://ar5iv.labs.arxiv.org/html/2301.04644)). In the grey zone, CLIP-L/14@336 gave the highest APTOS QWK in Fang's study, **0.9337**, about +0.013 over plain B4. At 336 px it would cost about 4–6 T4-hours for 5 folds (unmeasured). DINOv2-L beats RETFound on APTOS AUROC (0.952 vs 0.944) ([Hou et al.](https://arxiv.org/pdf/2502.06289)), but no DINOv2 or DINOv3 APTOS QWK exists. Both would be diversity bets with a QWK prior of about 0.90–0.93, and both should be reported in a separate row if used.

## Inference, decoding and resolution are cheap, and mostly make the number honest

**Decoding.** Two decoding changes cost nothing on the GPU and should come first.

The first is nested per-fold cut-offs. Fit the cut-offs on four folds and apply them to the fifth. This removes the roughly +0.002 OOF optimism, so the reported OOF goes *down*, but the decision metric becomes honest. Use coordinate or grid search with restarts. In the simulation, a single Nelder-Mead start from the default cut-offs was **about 0.005 worse**, because QWK is piecewise constant in the cut-offs.

The second change is a fact to apply, not a lever. Cut-off decoding depends only on the ranking of the scores. So any strictly monotone recalibration of the regression score, such as temperature scaling, Platt or isotonic, cannot change QWK once the cut-offs are re-fitted. Theory agrees that the QWK-optimal prediction is an affine stretch of the regression function ([Nedel'ko 2018](https://mathizv.isu.ru/en/article?id=843)). In the simulation, though, four free cut-offs beat a two-parameter affine rescale (0.931 vs 0.912). Keep the free cut-offs.

Learned blending also loses. No APTOS stacker beat a simple mean: BloodAxe reported that "None of KNN, LogisticRegression, Trees didn't work" ([7th place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058)), and another top team's extra-trees stacker came out "a tiny bit worse on private" ([9th place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)).

**TTA.** Test-time augmentation is close to saturated. The controlled EyePACS curve reaches +0.003 at 15 views, and 5 views get about 90% of that ([Huang et al.](https://ar5iv.labs.arxiv.org/html/2110.14160)). Our 8-view TTA added +0.003 OOF on average across members, and only +0.0012 to the test blend (`dr-repro/results/stage4b/stage4c_final_test_results.csv`). The one untried variant is multi-scale TTA: run the 380/384 checkpoints at about 1.15–1.2× their training size and average with the native size. It needs no training and takes about 0.3–0.5 h of T4 inference (an unmeasured estimate). Expect −0.002 to +0.003. This and every checkpoint-based lever below assume the 15 fold checkpoints are still retained in the Kaggle outputs, which has not been verified.

**Resolution.** Resolution is the training-side lever with the most fundus evidence. On EyePACS, ResNet-50 gained **+0.011 from 512 to 768 px**, but the 768 px run took twice as long (11 h 15 min vs 5 h 16 min) ([Huang et al., Diagnostics 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)). Referable-DR AUROC plateaus at 0.963 / 0.970 / 0.968 / 0.971 across 256–1024 px ([Sci Rep 2024](https://www.nature.com/articles/s41598-024-82884-9)). Our own B4-NS 380→512 step gave only +0.004, which fits a flattening curve on APTOS's mixed-quality images.

A 3–5-epoch fine-tune of the existing B4-NS 512 fold checkpoints at 640 px, with low learning rate, costs about 55 minutes. Expect +0.000 to +0.004, and watch 0↔1 confusions on OOF as well as QWK. Full training at 768–1024 px costs 2.3–4× the 512 run per epoch, and no APTOS result shows a benefit.

**Pre-processing.** Pre-processing diversity has one APTOS precedent. In the controlled EyePACS ablation, Ben Graham pre-processing *lowered* kappa by 0.005 and CLAHE by 0.007 against plain background removal ([Huang et al.](https://ar5iv.labs.arxiv.org/html/2110.14160)). APTOS top teams mostly found heavy pre-processing neutral or harmful. The 4th place is the instructive case: two Ben-processed models with *lower* CV (about 0.927 vs 0.933) still lifted its ensemble by +0.001 on the private leaderboard ([Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)).

All three of our blend members share Ben Graham pre-processing, which our own validation preferred by +0.014 at 300 px. A crop-only B4-NS 380 member trained with MSE is the mirror image of that 4th-place mix. Expect ρ of about 0.96–0.98 with the blend and +0.001 to +0.004 OOF, at about 87 minutes. Adopt it only if ρ < 0.975.

No APTOS ablation exists for green-channel input, colour constancy, CLAHE parameter sweeps or vessel filters, so leave them for last.

**Loss heads.** Loss-head changes are nearly free if they ride along on a retrain that is happening anyway. No source compares CORN, CLM, SORD, EMD or ranking losses head-to-head with tuned MSE on APTOS. A 2026 APTOS benchmark across five seeds found ordinal losses ahead of cross-entropy by only +0.004 to +0.007 Kendall τb, about one standard deviation. Enforced unimodality was clearly worse ([Stelter et al., NLDL 2026](https://proceedings.mlr.press/v307/stelter26a.html)).

The best-supported extra head is a 5-way softmax trained with cross-entropy plus Galdran's distance-aware cost term. That term added **+1.5 to +3.2 QWK points over classification baselines** on EyePACS ([Galdran et al., MICCAI 2020](https://ar5iv.labs.arxiv.org/html/2010.00291)). Decode it by expected grade and blend it with the MSE score on inner OOF. Anchor expectations on our measured +0.003 from the plain auxiliary cross-entropy head; the realistic range is +0.000 to +0.003.

Balanced-MixUp gained +0.006 to +0.009 QWK over instance sampling on EyePACS ([Galdran et al., MICCAI 2021](https://ar5iv.labs.arxiv.org/html/2109.09850)). It has a negative DR precedent elsewhere, so its APTOS prior spans −0.003 to +0.004.

No DR evidence exists for SAM, SWA, layer-wise learning-rate decay on EfficientNets, or freeze-BN. SWA largely duplicates our EMA of 0.998, and SAM doubles the cost of each step.

## Unlabelled images and label cleaning promise less than their leaderboard reputation

**Pseudo-labelling.** Pseudo-labelling is the lever whose expected value shrank most in this round. Only the **1,928 public-test images** can be downloaded. The roughly 13k-image re-run set was never released, and Kaggle said it does not plan to release the solution file ([Kaggle thread 97606](https://www.kaggle.com/competitions/aptos2019-blindness-detection/discussion/97606)).

Every reported gain, +0.001 to +0.009 on the private leaderboard, was measured on the same shifted distribution as the images being pseudo-labelled. In the training data, image size and black-border cropping are "strongly indicative of diagnosis", but not in the public test data ([thread 100815](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815)). The one team that pseudo-labelled train-distribution images inside CV found it "either did not improve much or only a tiny bit" ([Psi, 9th place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)). The 4th place "tried pseudo once but didn't get good result" ([Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)).

Our OOF and our hold-out both come from the train distribution, so the realistic prior is **+0.000 to +0.004** at 2.5–5 T4-hours. If it is run, four conditions make it fold-safe:

1. Only fold-k teachers label the pool for fold-k students.
2. Targets are soft regression values.
3. Cut-offs are re-fitted on OOF only, never on the pool.
4. The 1,928 images are first hashed against all 3,662 training images, because duplicates cross the train/test boundary ([thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)). Any image matching one of the 550 hold-out images is removed.

The result belongs in a separate row: "strict labels + unlabelled APTOS test images (transductive)".

**Self-supervised pre-training on APTOS images.** This has a direct, near-identical measurement. On a 70/15/15 APTOS split, ImageNet initialisation followed by DINO on APTOS scored κ **0.894 vs 0.893** for ResNet-50 and **0.896 vs 0.896** for DeiT-S, over 5 runs ([Matsoukas et al.](https://arxiv.org/html/2303.07034v3)). A DINO run over about 5k images would take an estimated 10–11 T4-hours before any fine-tuning. Skip it.

**Label-noise methods.** Classic label-noise methods are mismatched to our evaluation, because our test labels share the training labels' single-grader noise. On real EyePACS noise, scored against the original labels, cleanlab lowered held-out QWK from 0.650 to 0.604 (dropping flagged images), 0.611 (relabelling) and 0.587 (soft labels) ([hkoul2000, unreviewed](https://github.com/hkoul2000/trustworthy-diabetic-retinopathy)). Ordinal label correction at the dataset's native noise level slightly worsened MAE, 0.671 → 0.683–0.689 ([ORDAC](https://arxiv.org/html/2509.02351)). Cleaning toward expert consensus helped only when the evaluation labels were also consensus labels: referable-DR AUC rose 0.785 → 0.810 ([Lin et al.](https://www.nature.com/articles/s41746-024-01424-x)). Those consensus labels do not appear to be public.

What survives is mild target softening. Galdran's ordinal label smoothing beat cross-entropy by +0.045 κ on a private screening set ([Galdran et al., TVST 2020](https://pubmed.ncbi.nlm.nih.gov/32832207/)). One APTOS Kaggle team found that ±0.3 jitter on regression targets "greatly reduced overfitting" ([Tymchenko et al.](https://ar5iv.labs.arxiv.org/html/2003.02261)). For our MSE pipeline, the cheap candidates are:

- fold-safe self-distillation with targets 0.5–0.7 × label + the remainder × OOF prediction;
- target jitter;
- down-weighting, rather than dropping, images whose OOF residual exceeds 1.5 grades.

Each is worth about +0.000 to +0.003 for one roughly 87-minute run.

**Near-duplicates.** An unverified audit (snippet only) reports 14 near-duplicate pairs beyond the exact copies ([Fire-Dev2 audit](https://github.com/Fire-Dev2/aptos-leakage-audit)). That implies only 3–4 pairs crossing our dev/hold-out boundary, and an effect below 0.002. A perceptual-hash pass over the train and Kaggle-test images takes a few CPU-minutes. It is hygiene, and it is a prerequisite for pseudo-labelling. It should not be used to re-split the locked test set.

**Full-data refit.** A refit on train plus validation has no published comparison against a fold ensemble. It would remove the OOF predictions our cut-offs depend on. Report it only as an unselected post-test row, if at all.

## The test set can confirm only about +0.01, and OOF needs repeated seeds

The arithmetic of detection decides what the remaining budget can buy. At n = 550 with our grade mix (271/56/150/29/44), the standard error of a single model's QWK is **0.009–0.011**. The Fleiss-Cohen-Everitt large-sample formula ([Fleiss, Cohen & Everitt 1969](https://www.ime.usp.br/~abe/lista/pdf5BEy4kEmzd.pdf)) and the bootstrap on the Stage 4 test predictions agree to the third decimal. That matches our reported interval of [0.900, 0.935].

For comparing two models on the same images, the paired standard error of the difference matters more. For real model pairs at ρ ≥ 0.99, it is about 0.0025–0.0065, so the minimum detectable improvement at 80% power is **0.007–0.018** on the test set. On a 3,112-image OOF it is about **0.003–0.008**. A realistic diffuse improvement of +0.005 has only **20% power at n = 550** and about 77% at n = 3,112. These figures come from the project's own calculation on `stage4_runs/mse_s{42,43,44}/outputs.npz`, which re-reads already-opened Stage 4 test predictions only to measure noise. This matches the field-wide finding that in half of the Kaggle medical-imaging challenges studied, evaluation noise exceeded the winner's margin ([Varoquaux & Cheplygina 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663/)).

Training-seed noise sits on top of that. Across three seeds on the fixed test set, QWK had SD 0.004, about 17% of the test-bootstrap variance. That fits Bouthillier et al.'s finding that model initialisation contributes less than half the variance of data bootstrapping ([Bouthillier et al., MLSys 2021](https://ar5iv.labs.arxiv.org/html/2103.03098)). The seed SD on our 3,112-image OOF has never been measured, and a plausible range is 0.002–0.004. To rank one lever against baseline at 80% power, comparing seed-averaged OOF means unpaired, takes:

| True gain | σ_seed 0.002 | σ_seed 0.003 | σ_seed 0.004 |
|---|---|---|---|
| +0.003 | 7 seeds per arm | 16 | 28 |
| +0.005 | 3 seeds per arm | 6 | 10 |

Using the same folds, seeds and data order in both arms (common random numbers) lowers these counts, but not by an order of magnitude. Even at the optimistic end, three seeds in each of two B4-NS 380-class arms is 5 new runs, about 7–9 T4-hours. That is roughly the whole ~9-hour budget planned after Stage 4c, spent to *verify* a single +0.005 lever. So the budget can pay for adding blend members under a lenient rule, but not for claiming that any lever works.

The current greedy rule ("adopt if ΔOOF ≥ +0.002") lets through about 29% of levers whose true effect is zero. That is an acceptable cost when adding blend members, because a null member costs about nothing in expectation. It is not acceptable as evidence. The rule to pre-register before the next run:

1. **Primary metric.** Nested-cut-off OOF QWK with fixed TTA.
2. **Adopting a blend member.** Require ΔOOF ≥ +0.002 and a paired OOF bootstrap with P(Δ > 0) ≥ 0.80, following Bouthillier's probability-of-outperforming criterion.
3. **Claiming a lever works.** Require the same sign in at least 2 of 2 seed repeats, a mean Δ ≥ +0.003, and a lower 90% paired-bootstrap bound above zero.
4. **Test set.** Report any further test number as **post-test**, with its 95% CI, the paired Δ against the Stage 4c headline, and the expected detectable difference of about 0.008–0.018 next to it.

A true +0.004 gain has roughly a **20–25% chance of producing a lower test score than 0.918**. That is a further reason the test number must not steer anything.

## What to run, ranked by expected gain per T4-hour

All gains are priors (grade C/D) relative to the 0.928 blend, and costs are scaled from our measured T4 times.

| Rank | Lever | Protocol status | Expected ΔOOF (blend) | T4 cost | Can the gain be detected? |
|---|---|---|---|---|---|
| 0 | Nested per-fold cut-offs, coordinate search with restarts; perceptual-hash audit | Strict | −0.002 on reported OOF (more honest) | CPU only | Not a gain; fixes the metric |
| 1 | Multi-scale TTA on saved fold checkpoints | Strict | −0.002 to +0.003 | 0.3–0.5 h | OOF, weakly; never on test |
| 2 | 640 px FixRes fine-tune of B4-NS 512 folds | Strict | +0.000 to +0.004 | ~1 h | OOF, weakly |
| 3 | Crop-only (no Ben Graham) B4-NS 380 with MSE + CE/cost-sensitive auxiliary head | Strict | +0.001 to +0.004 | ~1.5 h | OOF, marginal |
| 4 | Second seed of B3 384 (cheapest member) | Strict | +0.000 to +0.002 | ~1.1 h | Measures σ_seed, which every later decision needs |
| 5 | ConvNeXt-S IN-22k / SE-ResNeXt50 re-tune (MSE, lr 5e-5–1e-4, layer-wise decay, Ben vs plain): fold-0 screen, then 5 folds | Strict (supervised IN-22k) | +0.000 to +0.005 | ~2 h screen + ~2 h | Screen detects only large recoveries; 5-fold OOF needs ≥ 2 seeds |
| 6 | Fold-safe soft pseudo-labels on the 1,928 Kaggle test images | Separate transductive row | +0.000 to +0.004 | 2.5–5 h | No |
| 7 | B5-NS 456 member | Same status as the B4-NS already used | +0.001 to +0.004 | ~3 h | No |
| 8 | Self-distillation, target jitter, conflict-aware weighting | Strict | +0.000 to +0.003 each | ~1.5 h each | No |
| Low | CLIP-L/14@336 or DINOv2-B; Balanced-MixUp; SAM/SWA; full-data refit; green channel/CLAHE/colour constancy; stacking | Grey zone or strict | −0.003 to +0.008 | 1.5–6 h | No; weak, negative or no APTOS evidence |
| Skip | In-domain SSL on APTOS; cleanlab/co-teaching relabelling; B6/B7; 768–1024 px full training | — | ≈ 0 or negative | 3–11 h | — |

If the plan is about 8 T4-hours, ranks 0–5 fit (about 0.5 + 1 + 1.5 + 1.1 + 4 h). Running the B3 seed early gives the seed SD before the ConvNeXt decision is made. If ranks 1–3 and 5 all land, the plausible result is an OOF of about 0.930–0.934 (honest, nested), with an expected true test QWK of about 0.92–0.925. A post-test check of that would show a difference from 0.918 that the 550 images cannot separate from zero.

## Conclusion

Under the strict protocol, the question has turned from "which lever?" to "which measurement?". Our blend already matches the best natural-image results published on local APTOS splits. The last real modelling opportunity is narrow: the ConvNeXt and SE-ResNeXt arms were probably sunk by a shared EfficientNet recipe, SmoothL1 loss and a retina crop, not by their architecture. Fixing that recipe is the only route found to a strong, decorrelated member. Everything else is worth thousandths. Pseudo-labelling's reputation rests on leaderboard gains that came from adapting to a shifted test distribution, which does not carry over to a random hold-out.

The more durable output of the remaining hours is methodological. Measuring the seed SD on OOF, switching to nested cut-offs, and pre-registering a detection-aware decision rule would turn the final write-up from "0.918 vs 0.954" into a calibrated claim. That claim is that this protocol has a true ceiling near 0.925–0.93, and that its 550-image test set cannot resolve differences below about 0.01. That framing also strengthens the reproduction's central finding: a gap of 0.036 is two to four times larger than anything the strict levers or the test set's own noise can produce.
