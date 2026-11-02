# Ensembling, TTA, multi-seed, preprocessing headroom and detectability on APTOS 2019 (strict protocol, after Stage 4d)

Scope notes (from the assignment, not a source): strict protocol, 3,112 CV images plus a 550-image locked test set. Best strict result is test QWK 0.918 [0.900, 0.935] and OOF 0.928, from a 3-member blend of 5-fold ensembles (B4-NS 380, B3 384, B4-NS 512) with 8-view TTA and OOF-tuned cut-offs. This file does not repeat the earlier notes:
- `APTOS QWK gap after Stage 4c/remaining_levers.md` covers pseudo-labelling, B5/B6/B7, foundation models, losses, SWA and FixRes.
- `Closing the APTOS 2019 QWK gap/aptos_top_solutions.md` covers the full top-solution write-ups, stacking and thresholds.
- `Closing the APTOS 2019 QWK gap/evaluation_protocols.md` covers CI widths of about ±0.02 and selection-on-test optimism.

This file adds:
- controlled ensemble-size and TTA-view curves;
- the project's own TTA and correlation measurements, re-read from its CSVs;
- preprocessing evidence beyond Ben Graham;
- full-data refits;
- a quantitative power analysis on the project's own Stage 4 test predictions;
- a decision rule.

Labels used throughout:
- **[Kaggle-LB]**: Kaggle public or private leaderboard (hidden test set).
- **[local split]**: a single held-out split of the 3,662 training images.
- **[CV/OOF]**: cross-validation or out-of-fold.
- **[EyePACS]**: a different dataset.

**About the project-calculation lines.** Lines marked "project calculation" come from my script `/private/tmp/claude-501/-Users-subhammahapatra-Downloads-Projects-SubhamPortfolio-dr-grading-reproduction/7182d768-cc02-4bff-8102-56b6fe59189c/scratchpad/scripts/qwk_power.py`. It was run on `stage4_runs/mse_s{42,43,44}/outputs.npz`: the Stage 4 arm-D MSE runs, 10-snapshot mean, cut-offs tuned on validation, the same 550-image test split, 4,000 bootstrap resamples. These are **not literature findings**. The Stage 4 test images were already opened in Stage 4, so this re-use makes no decision for the project. It only measures noise.

## 1. Ensemble size and diversity: what 1 → 3 → 8+ models bought, and which kind of diversity decorrelates members

### Takeaway
Going from one model to a large ensemble is worth only about +0.003 to +0.009 QWK on APTOS:
- [Kaggle-LB]: about +0.003 to +0.009.
- In the one controlled DR curve [EyePACS, 42,670 test images]: +0.005, saturating at 10 models (1→3 models gives +0.0034, 3→10 gives +0.0016 more).

Our blend already holds 15 fold-models at member correlation 0.98-0.99, so we sit on the flat part of that curve. Diversity rankings:
- In our own data, a different architecture family decorrelates far more (ρ 0.957-0.959) than a preprocessing or crop variant (ρ 0.981) or a new seed (ρ 0.992-0.994).
- The only APTOS evidence that preprocessing diversity pays is the 4th place's +0.001 [Kaggle-LB] from swapping in two Ben-processed models.
- Simple or trimmed means matched or beat stacking in every APTOS report that compared them.

### Cited Findings
- **Controlled ensemble-size curve, grade B (transfer).** ResNet-50, EyePACS, 42,670-image test set, quadratic kappa, members differing by training run. 1 model 0.8581; 2: 0.8596; 3: 0.8615; 5: 0.8617; 10: 0.8631; 15: 0.8631. "+0.50 points at 10 models, with no further gain at 15." — [Huang et al., "Identifying the key components in ResNet-50 for DR grading", arXiv 2110.14160](https://ar5iv.labs.arxiv.org/html/2110.14160)
- **1st place [Kaggle-LB].**
  - Best stage-1 single model private 0.931 (SE-ResNeXt50), against a stage-1 8-model ensemble (4 families × 2 seeds) private 0.934, and a final 0.936 after pseudo-labels plus a threshold change. — [Guanshuo Xu write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065) (via `aptos_top_solutions.md`)
  - Fang et al. instead quote a single Inception-ResNet-v2 from the winning ensemble at 0.927, which implies about +0.009. The two numbers conflict because they refer to different single models and stages. — [Fang, Kornblith & Schmidt, arXiv 2301.04644](https://arxiv.org/pdf/2301.04644)
- **4th place [Kaggle-LB].** Best single private 0.929 (B7@224). 6-EfficientNet ensemble (5 folds × 8 dihedral TTA each) private 0.932. Version 2, which swapped in two Ben-preprocessed models "with more variance", reached 0.933, even though those models' CV was lower (about 0.927 vs about 0.933 [CV]). — [Qishen Ha write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- **37th place [local split + Kaggle-LB].** Single EfficientNet B0-B5 scored 0.919-0.926 on a 20% holdout and 0.914-0.921 private. The ensemble scored 0.932 holdout and 0.926 private, about +0.005 to +0.006 over the best single on both. — [Tahsin Mostafiz](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108017)
- **Diminishing value of preprocessing-variant ensembles [Kaggle-LB].** "Ensembling models with slightly different preprocessing had big impact on our LB score in the early days. However, as our individual models improved, the improvement of this technique decreased." — [Toumbourou](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947)
- **Stacking vs mean.**
  - 7th place: "Stacking. None of KNN, LogisticRegression, Trees didn't work." — [BloodAxe write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058)
  - 9th place: an extra-trees stacker "worked well on both CV as well as public LB, but a tiny bit worse on private". — [Psi](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
  - 17th place used linear stacking at private 0.9305 (local CV 0.9346), with no mean baseline reported. — [Xuan Cao](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
  - 58th place [local split, 736 images]: a 0.25-trimmed mean of 20 models scored 0.971 vs 0.968 for the plain mean. — [Tymchenko et al., arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
  - 46th place ensembled only models with prediction correlation below 0.9. — [Valan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119)
- **Project's own member correlations and blend gains [CV/OOF, 3,112 images].** Correlation with B4-NS 380 / gain added to the blend:

  | Member | Correlation | Blend gain |
  |---|---|---|
  | B3 384 | 0.984 | (start pair) |
  | B4-NS 380 + retina crop + aux head (a preprocessing/crop variant) | 0.981 | +0.0014 |
  | B4-NS 512 | 0.989 | +0.0035 |
  | SE-ResNeXt50 512 | 0.957 | +0.0025 |
  | ConvNeXt-T 448 | 0.959 | +0.0009 |
  | B4-NS 380 EyePACS-init | 0.975 | +0.0007 |

  — local file `dr-repro/results/stage4b/stage4c_members_oof.csv`
- **Seed and snapshot correlations (project calculation, Stage 4 test outputs, 550 images).**
  - Between seeds 42/43/44 of the same recipe, the continuous predictions correlate at 0.992-0.994, and 7-10% of images change grade.
  - Last snapshot vs 10-snapshot mean: ρ 0.984, with 8.4% of grades changed and +0.007 QWK (paired SE 0.006).
  - Seed 42 vs mean of 3 seeds: ρ 0.998, +0.005 QWK (paired SE 0.0045).
  - Mean of 2 seeds vs mean of 3 seeds: ρ 0.9994, −0.001 (paired SE 0.0024).
  - So the third seed of a seed-average adds nothing measurable.
- **Deep-ensemble theory.** Random initialisation alone gives functionally diverse members, and ensembling helps most when members are decorrelated. — [Fort et al., "Deep Ensembles: A Loss Landscape Perspective", arXiv 1912.02757](https://alphaxiv.org/abs/1912.02757) (cited in `remaining_levers.md`)

### Inferences
- **Where we are on the size curve.** Huang's EyePACS curve gets 70% of its 10-model gain by 3 models. Our blend is 3 recipes × 5 folds. Further members of the same family will give about +0.000-0.002 OOF, matching our observed +0.001-0.0035 per added arm.
- **Seeds vs architecture vs preprocessing.**
  - In our data, a new seed (ρ ≈ 0.993) is the least decorrelating option.
  - A crop/preprocessing variant (ρ ≈ 0.98) is moderate.
  - A different family (ρ ≈ 0.96) is the most decorrelating, but the family members were weaker (OOF 0.897-0.916), so they added no more than B4-NS 512 did.
  - The useful member is one that is both decorrelated (ρ ≤ 0.97) and strong (OOF ≥ 0.92). None of our arms has both so far.
- **The one untried preprocessing-diversity member with APTOS precedent is a "no Ben Graham" member.** Our pipeline adopted Ben Graham in Stage 4b (+0.014 val at 300 px; README line 68), so all three blend members share it. The 4th place mixed raw and Ben models. Most top teams trained on crop-only or raw images. A raw-crop B4-NS 380 member is therefore the mirror image of the 4th-place recipe. Expected effect: ρ about 0.96-0.98 with the blend, single OOF perhaps 0.91-0.92, blend gain about +0.001 to +0.004 at about 87 T4-min. This is an analogy, not a measurement.
- **Blending method.** At ρ 0.98-0.99 and 3 members, any learned weights (ridge, rank, stacker) carry more variance than the possible gain. Keep the simple mean. A trimmed mean only matters with 10+ members.

### Gaps
- No APTOS source gives a controlled 1/3/5/10-member curve under CV. The Huang curve is on EyePACS with ResNet-50.
- No study I found compares the prediction correlation of preprocessing-diverse vs architecture-diverse members on fundus images. The searches returned only segmentation (DiPE, arXiv 2210.12388) and generic overviews.

## 2. TTA: number and type of views, multi-scale TTA, reported gains

### Takeaway
TTA on DR grading is worth about +0.000 to +0.007 QWK per model:
- In the one controlled curve [EyePACS], +0.003 at 15 views, with about 90% of it reached by 5 views.
- In the project's own OOF data, mean about +0.003, ranging from −0.0003 for B4-NS 380 to +0.0066 for B3 384.
- On our test set it added +0.0012 to the headline blend.
- One APTOS team saw TTA slightly hurt (0.971 → 0.969 [local split]).

Our 8-view TTA is already at saturation. I found no APTOS evidence for multi-scale TTA, so it is a cheap, untested, inference-only probe.

### Cited Findings
- **Controlled TTA-view curve, grade B (transfer).** ResNet-50, EyePACS 42,670 test images. 1 view 0.8581; 2: 0.8593; 3: 0.8601; 5: 0.8609; 10: 0.8603; 15: 0.8611. Best gain +0.0030. — [Huang et al., arXiv 2110.14160](https://ar5iv.labs.arxiv.org/html/2110.14160)
- **58th place [local split, 736 images].** The trimmed-mean ensemble scored 0.971 without TTA and 0.969 with 10-view TTA. — [Tymchenko et al., arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- **TTA used by top APTOS teams (no ablations).**
  - 2nd place: flip TTA. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
  - 4th place: 8 dihedral. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
  - 5th place: 2×. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
  - 9th place: 4×. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
  - 17th place: 8×. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- **Project's own TTA deltas [CV/OOF 3,112].** oof_qwk_cf_tta − oof_qwk_cf:

  | Member | TTA delta |
  |---|---|
  | B4-NS 380 | −0.0003 |
  | B3 384 | +0.0066 |
  | B4-NS 380 crop+aux | +0.0028 |
  | ConvNeXt-T 448 | +0.0048 |
  | SE-ResNeXt50 512 | +0.0043 |
  | B4-NS 512 | +0.0013 |
  | EyePACS-init B4-NS | +0.0032 |
  | Mean | +0.0032 |

  — local file `dr-repro/results/stage4b/stage4c_members_oof.csv`
- **Project test [local split, 550].** Blend B4-NS 380 + B3 384: 0.9143 without TTA vs 0.9155 with TTA (+0.0012). — local file `dr-repro/results/stage4b/stage4c_final_test_results.csv`
- **General TTA analysis.** Even when TTA improves net accuracy, it "can change many correct predictions into incorrect predictions". A learned aggregation over views gives "consistent improvements" over simple averaging. The abstract gives no magnitudes. — [Shanmugam et al., ICCV 2021, arXiv 2011.11156](https://arxiv.org/abs/2011.11156)
- **DRAC22 OCTA DR grading (transfer).** Flip TTA was part of an ablation (QWK 0.827 → 0.891 with ensemble + pseudo-labels + TTA + post-processing), but the extracted table does not let the TTA share be isolated. — [arXiv 2210.09558](https://ar5iv.labs.arxiv.org/html/2210.09558)

### Inferences
- **More views of the same scale.** Going from 8 to 16 views (dihedral × 2 small scales or crops) should give ≤ +0.001 OOF, by the Huang plateau and our already-8-view pipeline.
- **Multi-scale TTA.** Inferring the 380/384 checkpoints at 1.1-1.2× their training size (about 420-460 px), averaged with the native size, is a FixRes-like probe at about 10-20 min of T4 inference. It needs no training. Expect −0.002 to +0.003. It is worth one OOF check because it is nearly free, but it must be decided on OOF only.
- **Learned TTA aggregation (Shanmugam).** With 8 views and ρ ≈ 0.99 between views, learned weights would mostly fit noise on 3,112 images. Not recommended.

### Gaps
- No APTOS or DR study isolates multi-scale TTA, or views beyond dihedral, under CV.
- Shanmugam et al.'s per-dataset magnitudes were not read (abstract only).

## 3. Preprocessing variants not yet tried (green channel, circular crop + per-image normalisation, CLAHE settings, colour constancy, vessel/lesion filters)

### Takeaway
The best controlled evidence (ResNet-50 on 42,670 EyePACS test images) finds that both Ben Graham (−0.005) and CLAHE (−0.007) **lower** kappa against plain background removal. Most APTOS top teams found heavy preprocessing neutral or harmful [Kaggle-LB/CV].

I found **no** credible APTOS ablation for green-channel input, colour constancy (grey-world or Retinex), vessel- or lesion-enhancing filters, or alternative CLAHE settings. The only APTOS green-channel result is a confounded few-shot study at QWK 0.57-0.70.

Expect any new preprocessing to change single-model OOF by about ±0.01, and to be useful mainly as a diversity source for a blend (≤ +0.004).

### Cited Findings
- **Controlled preprocessing ablation, grade B (transfer).** ResNet-50, EyePACS test (42,670 images):

  | Preprocessing | Test kappa |
  |---|---|
  | Background removal + resize (default) | 0.8310 |
  | Graham | 0.8260 (the text says 0.8227, which conflicts with Table 5) |
  | CLAHE | 0.8238 |

  The authors: "Although lesions become more recognizable with the application of the two preprocessing methods, they are not helpful for DR grading". — [Huang et al., arXiv 2110.14160](https://ar5iv.labs.arxiv.org/html/2110.14160)
- **APTOS top teams, preprocessing outcomes.**
  - 2nd place: "Ben's color preprocessing ... didn't boost score".
  - 7th place: Ben's, CLAHE-based, red-channel dropping and spherical unwarping "All of them failed to outperform using images as-is".
  - 4th place: Ben models had CV about 0.927 vs about 0.933 raw [CV], yet helped the ensemble by +0.001 [Kaggle-LB].
  - 58th place: switching from circle crop / Ben's to no preprocessing plus hard augmentation helped on both CV and LB.

  — [2nd](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926), [7th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058), [4th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987), [58th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108007) (all via `aptos_top_solutions.md`)
- **Heavy preprocessing on the public LB.** "Heavy preprocessing may reduce the performance of EfficientNet (0.811 -> 0.732)" at 256 px [Kaggle-LB public]. — [Quan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104818)
- **Per-image normalisation (a crude colour-constancy step).** The 9th place used "subtract the image's own mean, divide by its own std" on about half its models, with circle crop (90% in training, 95% at inference). No ablation is given. — [Psi](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- **Green channel on APTOS (weak, confounded).**
  - A 5-way 5-shot prototypical network on APTOS (2,929/733 split) combined circle crop with an enhanced green channel. QWK rose from 0.5712 to 0.6990. Crop and green channel were not separated, and the regime is few-shot, far from ours. — [Coscitech (UMRI) 2025](https://ejurnal.umri.ac.id/index.php/coscitech/article/view/11126)
  - An ordinal-regression pipeline with green channel + noise masking + CLAHE reports APTOS QWK 0.8992, with no baseline without that preprocessing. — [arXiv 2511.14398](https://arxiv.org/html/2511.14398v1)
- **CLAHE + per-channel enhancement across 27 CNNs.** The authors report MobileNet on the green channel with CLAHE as "sufficiently fast and accurate". Accuracy only, no kappa. — [KMITL repository](https://dspace.kmitl.ac.th/handle/123456789/12402)
- **The target paper's own preprocessing.** CLAHE + Gaussian blur at 300 px, with CLAHE parameters not stated. — [Chilukoti et al., PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) (via `evaluation_protocols.md`)
- **Colour constancy, vessel enhancement.** Searches for colour-constancy and vessel-enhancement preprocessing for DR grading returned no study with a QWK ablation. The vessel-filter papers found are about vessel *segmentation*, for example morphological + Gabor. — [arXiv 1908.04123](https://arxiv.org/pdf/1908.04123)

### Inferences
- **Our strict pipeline already uses Ben Graham** (adopted at +0.014 val, 300 px, 2 seeds; README). The literature points the other way on large data (Huang −0.005 [EyePACS]; 4th-place −0.006 CV). So the most informative untried variant is the **simplest**: retina crop + resize, no Ben, normalised with dataset statistics or per-image mean/std. It is useful as a blend member (Section 1), not as a replacement, because our own validation preferred Ben at 300 px.
- **Green-channel-only input** throws away the red and blue channels that an ImageNet backbone expects. Replicating the green channel to three channels breaks the pretrained colour statistics. No APTOS evidence supports it. Low priority.
- **CLAHE settings.** Chilukoti's CLAHE parameters are unstated. Huang found CLAHE (default settings) worse than none [EyePACS]. Sweeping clipLimit or tile size would spend GPU-hours on a lever with negative prior evidence. Low priority, except possibly as one CLAHE member for diversity.
- **Colour constancy (grey-world / Shades-of-Grey).** Our augmentation already randomises colour heavily (Stage 4b strong augmentation, +0.013). Since Ben Graham's local-mean subtraction already removes illumination, an explicit colour-constancy step is likely redundant with Ben (ρ near 1). No evidence either way.

### Gaps
- No APTOS QWK ablation exists in the sources found for green channel, colour constancy, CLAHE parameter sweeps or vessel/lesion filters. Any claim about them is a prior.
- The IET Image Processing 2024 "preprocessing-based ensemble" DR paper ([DOAJ record](https://doaj.org/article/723bf6c9b0934e4fb53de857d5620ac8)) may contain per-preprocessing kappas, but only its title or abstract was seen.

## 4. Full-data refit (train + val) after CV vs fold-model ensembles

### Takeaway
I found **no** published DR, APTOS or controlled image-classification comparison of a full-data refit against a K-fold model average. The sources are practitioner opinion and an echo-state-network paper that does not declare a winner.

APTOS top teams used fold ensembles; none reported a refit. For our protocol a refit has three costs:
- it removes the OOF predictions our cut-offs are tuned on;
- its training length cannot be validated;
- it gives up the free 5-model ensemble.

Its upside is +25% data per model (3,112 vs 2,490 images). That is worth something for a single model, but probably less than the 5-fold average already gives.

### Cited Findings
- **Echo-state-network study.** It compares retraining on all data, averaging fold models and the best-validated fold. Retraining uses all the data, but the tuned hyperparameters "may no longer be optimal", while averaging "adds regularization and stability". No definitive winner is reported. — [arXiv 2006.11282](https://arxiv.org/pdf/2006.11282)
- **Practitioner positions.**
  - Retraining on all data "usually yields a more robust model" — [PyTorch forum](https://discuss.pytorch.org/t/what-is-the-proper-processing-applying-kfold/37444).
  - CV metrics are an "assumed estimate of full-data performance" — [Hugging Face forum](https://discuss.huggingface.co/t/re-training-nlp-model-with-training-and-validation-dataset-after-validation-has-been-done/12246).
  - Fold averaging means "each model is trained on an incomplete dataset" — [MaxEnt group](https://groups.google.com/g/Maxent/c/ocS2lcKephA).
  - None of these give measurements.
- **APTOS practice.** The 4th (5 folds × 6 models), 9th (4 folds × 3 bags) and 5th (5-fold) places all submitted fold ensembles. — [4th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987), [9th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990), [5th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
- **Fold average vs single model in the project [local split, 550].**
  - B4-NS 380, a single model with validation cut-offs: 0.9055.
  - The same recipe as a 5-fold CV ensemble with OOF cut-offs: 0.9093.
  - The paired difference to the headline shrinks from +0.0127 to +0.0088.

  — local file `dr-repro/results/stage4b/stage4c_final_test_results.csv`, `stage4c_paired_bootstrap.csv`

### Inferences
- **Expected gain of a refit.** Learning-curve slopes for APTOS at 2,490 → 3,112 images are not published. In our own runs, the step from one model to a 5-fold average was +0.004 test and the step from 1 to 3 seeds was +0.005 (project calculation). That is about the same size as a plausible +25%-data gain. A 5-seed full-data refit ensemble might therefore match or slightly beat the 5-fold average, by about 0.000 to +0.003. It cannot be validated, though, because no OOF exists for the refit models.
- **Cut-off transfer risk.** Full-data models trained for the same epochs see 25% more steps and may be calibrated differently from the fold models. Cut-offs tuned on the fold OOF may then not transfer. A safe variant is "5 full-data refits with different seeds, same epochs and lr schedule as the fold models, cut-offs from the fold OOF", reported as a separate row. It costs as much as one 5-fold run (about 87 min for B4-NS 380).
- **Recommendation.** Do not use a refit as the main lever. It is unverifiable on OOF and has no evidence of a gain above about 0.003. If GPU time remains at the end, a refit row can be reported after the test set is opened once, clearly labelled as not selected on OOF.

### Gaps
- No controlled comparison of refit vs fold ensemble was found for any medical-imaging classification task.

## 5. Statistics: SE of QWK at n ≈ 550, minimum detectable improvement, seeds/folds needed, decision rule

### Takeaway
At n = 550 with the APTOS grade mix (271/56/150/29/44), the SE of a single model's QWK is about 0.009-0.011. The Fleiss-Cohen-Everitt formula and the bootstrap agree to the third decimal.

What matters for comparing two models is the **paired** SE of the difference. That is about 0.0025-0.0065 for real model pairs at ρ 0.99+, so the minimum detectable improvement at 80% power is about **0.007-0.018 on the 550-image test**. On 3,112 OOF images it is about **0.003-0.008**, before training-seed noise is added.

Our last test improvement (+0.003 [−0.003, +0.009]) is therefore far below what the test set can detect. Levers worth +0.003-0.005 can be ranked only on OOF, with at least 2-3 seeds per arm under common random numbers, and even then only at modest power.

### Cited Findings
- **Asymptotic variance of weighted kappa (Fleiss, Cohen & Everitt 1969).** The earlier formulas were wrong (conservative). The correct large-sample variance is:

  Var(κ_w) = [Σ_ij p_ij (w_ij − (w̄_i· + w̄_·j)(1 − κ_w))² − (κ_w − p_e(1 − κ_w))²] / [n (1 − p_e)²]

  where w̄_i· = Σ_j w_ij p_·j and w̄_·j = Σ_i w_ij p_i·. — [Fleiss, Cohen & Everitt 1969 (PDF)](https://www.ime.usp.br/~abe/lista/pdf5BEy4kEmzd.pdf)
- **Sample-size planning.** The variance takes the form Q/N, so the N needed for a target SE C is Q/C². — [Cantor, sample size for kappa](https://www.ime.usp.br/~abe/lista/pdfGSoh9GPIQN)
- **Small-n caveats.** The asymptotic variance becomes unreliable at small n or as κ → 1, and the bootstrap is suggested instead. — [Vierkant, SAS SUGI22](https://support.sas.com/resources/papers/proceedings/proceedings/sugi22/STATS/PAPER295.PDF). The minimum N for valid weighted-kappa inference is about 20-100 depending on the number of categories (rule of thumb 2k²). — [Cicchetti & Fleiss](https://conservancy.umn.edu/bitstream/handle/11299/98489/v01n2p195.pdf?sequence=1)
- **Comparing dependent kappas (same images, two models).** Bootstrap methods (McKenzie et al. 1996; Vanbelle & Albert 2008) and weighted least squares (Barnhart & Williamson 2002) exist. The bootstrap is the standard route for correlated kappas. — [Vanbelle 2017, PMC5600130](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5600130/)
- **Sources of variance in ML benchmarks (Bouthillier et al., MLSys 2021).**
  - "Bootstrapping data stands out as the most important source of variance". "Model initialization generally is less than 50% of the variance of bootstrap, on par with the visit order of stochastic gradient descent." Hyperparameter-search choice adds about as much variance as initialisation.
  - Comparison error rates: a single-point comparison gives about 10% false positives and about 75% false negatives. Comparing means over k = 50 gives under 5% false positives and about 90% false negatives at their δ = 1.9952σ threshold. Their probability-of-outperforming test gives about 5% false positives and about 30% false negatives.
  - Recommended rule: A beats B if P(A > B) − CI_min > 0.5 (significant) **and** P(A > B) + CI_max > γ = 0.75 (meaningful), with percentile-bootstrap CIs. Randomising more sources (data splits, seeds, hyperparameter search) per run gives a better estimator at a 51× compute saving vs the ideal estimator.

  — [Bouthillier et al., arXiv 2103.03098](https://ar5iv.labs.arxiv.org/html/2103.03098)
- **CV-based tests.** The naive t-test over resamples or folds inflates Type I error because it underestimates variance (Dietterich 1998). The Nadeau-Bengio corrected resampled t-test inflates the variance by (1/J + n_test/n_train). For k-fold, Corani et al. use ρ = 1/k. — [correctR vignette](https://ftp.fau.de/cran/web/packages/correctR/vignettes/correctR.html)
- **Training-repeat SD on an APTOS 70/15/15 split (≈ 550 test).** SSiT reports fine-tuning SDs over 5 repeats of 0.20-0.63 kappa points (0.002-0.006) for SSL inits and up to 2.17 for random init. — [SSiT, arXiv 2210.10969](https://arxiv.org/pdf/2210.10969) (via `evaluation_protocols.md`)
- **Test-set size guidance.** "At least several hundreds samples, ideally thousands or more". In 4 of 8 Kaggle medical-imaging challenges, "the evaluation noise is worse than the winner gap". — [Varoquaux & Cheplygina 2022, PMC9005663](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663/) (via `evaluation_protocols.md`)
- **Project calculation: single-model SE at n = 550.** Test grades 271/56/150/29/44.

  | Seed | QWK | Bootstrap SE | Fleiss-Cohen-Everitt SE |
  |---|---|---|---|
  | 42 | 0.9094 | 0.0105 | 0.0106 |
  | 43 | 0.9172 | 0.0093 | 0.0093 |
  | 44 | 0.9145 | 0.0093 | 0.0093 |

  95% bootstrap CI half-width ≈ ±0.019-0.021. This matches the project's reported [0.900, 0.935] for 0.918.
- **Project calculation: paired SE of real differences at n = 550.** MDE80 = 2.80 × SE (two-sided α = 0.05, power 0.80).

  | Pair | ρ | Grades changed | ΔQWK | Paired SE | MDE80 |
  |---|---|---|---|---|---|
  | Seed vs seed | 0.992-0.994 | 7-10% | | 0.0053-0.0065 | 0.015-0.018 |
  | Seed 42 vs 3-seed mean | 0.998 | 3.1% | +0.0052 | 0.0045 | 0.0127 |
  | 2-seed mean vs 3-seed mean | 0.9994 | 1.8% | | 0.0024 | 0.0068 |

  The project's own Stage 4c paired bootstraps imply SE ≈ 0.003 for the headline-vs-two-member-blend comparison (+0.0027 [−0.0033, +0.0094]). — local file `dr-repro/results/stage4b/stage4c_paired_bootstrap.csv`
- **Project calculation: scaling to OOF size.** This is the realistic, diffuse improvement (3-seed mean vs one seed, true Δ = +0.0052), resampled at larger n from the same 550-image pool (an approximation of a 3,112-image OOF):

  | n | Paired SE | MDE80 | Power to detect +0.005 |
  |---|---|---|---|
  | 550 | 0.0044 | 0.0124 | 0.20 |
  | 1,100 | 0.0032 | 0.0090 | 0.35 |
  | 3,112 | 0.0019 | 0.0053 | 0.77 |
- **Project calculation: best-case "minimal change" improvements.** B fixes m of A's errors by one grade and changes nothing else.
  - At n = 550, a +0.003 improvement made by fixing 4-6 images has power 0.35-0.63.
  - +0.0055 (10 images) has power 0.86.
  - At n = 3,112, +0.002-0.003 has power ≥ 0.84.
  - Real levers change 2-10% of grades in both directions, so they sit near the diffuse case above, not this best case.
- **Project calculation: training-seed SD on the fixed 550 test.** Stage 4 arm-D 10-snapshot ensembles, seeds 42/43/44: QWK 0.9094 / 0.9172 / 0.9145, SD 0.0040. The seed variance (1.6e-5) is about 17% of the test-bootstrap variance (about 9.4e-5), which fits Bouthillier's "less than 50%".

### Inferences
- **Rule-of-thumb formulas for this dataset.**
  - SE(QWK) ≈ 0.0097 × √(550/n).
  - SE(ΔQWK) for two strong models at ρ ≈ 0.99 ≈ 0.003-0.006 × √(550/n).
  - MDE80 ≈ 2.8 × SE(ΔQWK).
  - At n = 550, MDE80 ≈ 0.008-0.018. At n = 3,112 OOF, MDE80 ≈ 0.003-0.008.
  - On the test set, only gains of about ≥ +0.01 can be confirmed, which none of the remaining levers is expected to give.
- **Seeds needed to rank levers on OOF (inference).** With seed-averaged OOF means compared unpaired, n per arm = 2·((z_{α/2} + z_β)·σ_seed / Δ)², with z sum = 2.80. σ_seed on 3,112-image OOF is not measured. The plausible range from SSiT and our 550-image SD of 0.004 is 0.002-0.004.

  | σ_seed | Δ = +0.003 | Δ = +0.005 |
  |---|---|---|
  | 0.002 | 7 seeds/arm | 3 seeds/arm |
  | 0.003 | 16 | 6 |
  | 0.004 | 28 | 10 |

  Pairing (same folds, same seeds, same data order across the two arms: "common random numbers") removes part of σ_seed. Bouthillier's evidence says randomising splits as well gives a better estimator. On a T4 budget, levers of +0.003 cannot be ranked with high power. Levers of +0.005 need about 3-6 seeds per arm.
- **False-acceptance rate of the current greedy "+0.002 OOF" rule (inference).** If a lever has true Δ = 0, the OOF evaluation SE of the difference is about 0.0019 and the seed noise of the difference is about 0.003. The combined SE is about 0.0036, so P(Δ̂ ≥ +0.002) ≈ P(Z ≥ 0.56) ≈ 0.29. Roughly 1 in 3 null levers would be accepted. That is acceptable for adding blend members, because a null member costs about 0 in expectation. It is not acceptable for claiming a lever "works".
- **Recommended decision rule (pre-register before Stage 5).**
  1. **Primary metric**: OOF QWK on 3,112 images, cut-offs re-tuned on OOF inside each fold-out (nested), fixed TTA.
  2. **Adopt a blend member** if ΔOOF ≥ +0.002 **and** the paired bootstrap (≥ 2,000 resamples over OOF images) gives P(Δ > 0) ≥ 0.80. This is a lenient, Bouthillier-style "probability of outperforming" bar. The cost of a false accept is small.
  3. **Claim that a lever works** only if it replicates: the same sign in ≥ 2 of 2 (or ≥ 3 of 4) seed repeats with the same folds, a mean Δ ≥ +0.003, and a lower 90% paired-bootstrap bound > 0 on the pooled OOF. Alternatively P(A > B) over seed × bootstrap ≥ 0.75, with CI_min above 0.5.
  4. **Test set**: open once, report QWK with 95% CI and the paired-bootstrap Δ against the Stage 4c headline. State in advance that |Δ| < about 0.01 is "not distinguishable" and must not be read as confirmation or refutation.
  5. Report the expected test MDE (about 0.008-0.018) next to every test delta.

### Gaps
- σ_seed on the 3,112-image OOF (5-fold, per member) has not been measured. A 2-seed repeat of one member would pin it down and sharpen the seeds table.
- No literature result gives a power curve for QWK differences specifically. The figures above are from the project's own predictions, plus generic ML-benchmark studies.
- The 3,112-image scaling is approximated by resampling the 550 Stage 4 test images, so it inherits their error structure. Per-image OOF predictions of the current blend are not in the repo (only summary CSVs), so the exact OOF paired SE for Stage 4c could not be computed.

## 6. Overall: realistic best strict test QWK, and the 2-3 levers with the most expected gain per T4 GPU-hour

### Takeaway
Published ImageNet-only ceilings on local APTOS splits are about 0.92-0.93:
- Fang's best-of-grid 0.9297 [local split];
- SSiT 0.930 ± 0.003 with SSL EyePACS pre-training [local split, 550];
- BloodAxe 0.909-0.923 for 2019-only 4-fold models [CV].

Our OOF 0.928 is already at that ceiling. Ensemble, TTA, seed and preprocessing levers together plausibly add +0.002 to +0.006 OOF, which suggests a true test QWK of about 0.92-0.925. Any single 550-image draw will land within about ±0.02 of that. **A realistic strict best is therefore about 0.92-0.93 on test. Values ≥ 0.935 would most likely be favourable test noise.**

Best gain per GPU-hour within this note's scope:
1. multi-scale/extra-view TTA on existing checkpoints (inference only);
2. a no-Ben-Graham (crop-only) B4-NS 380 member for preprocessing diversity;
3. one extra seed of the strongest member, B4-NS 512.

The earlier note's soft pseudo-labelling (strict labels + unlabelled competition images) remains the largest single expected gain if it is allowed.

### Cited Findings
- **ImageNet-only local-split ceiling.** Best of 19 ImageNet models over a hyperparameter grid, ConvNeXt-Tiny 0.9297 on an 80/20 split (also the selection split). — [Fang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf)
- **SSiT on a 70/15/15 split (about 550 test), 5 repeats.** 0.9297 ± 0.0029 with self-supervised EyePACS pre-training. — [arXiv 2210.10969](https://arxiv.org/pdf/2210.10969)
- **2019-only CV.** BloodAxe 4-fold: SE-ResNeXt50 0.921-0.923, ResNeXt50 0.9234. — [BloodAxe README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **With 2015 pre-training [CV].** 4th place about 0.933; 17th place 0.9346. — [4th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987), [17th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- **Kaggle private-LB winner (external data, pseudo-labels).** 0.936. — [APTOS announcement](https://asiateleophth.org/?p=1297)
- **Size of ensemble and TTA gains.** +0.005 (ensemble) and +0.003 (TTA) on EyePACS; Graham/CLAHE −0.005 / −0.007. — [Huang et al., arXiv 2110.14160](https://ar5iv.labs.arxiv.org/html/2110.14160)
- **Measured T4 costs (5 folds × 20 epochs).** B4-NS 380 about 87 min, B3 384 about 64 min, B4-NS 512 about 138 min. — local file `dr-repro/results/stage4b/stage4c_members_oof.csv` (minutes_5_folds)

### Inferences
Ranked levers in this note's scope (all expected gains are priors, grade C/D; OOF Δ is relative to the 0.928 blend):

| Rank | Lever | Protocol | Expected ΔOOF | T4 cost | Gain per GPU-h | Notes |
|---|---|---|---|---|---|---|
| 1 | Multi-scale TTA on existing 15 fold checkpoints (native + 1.15-1.2× size, dihedral) | Strict | −0.002 to +0.003 (median about +0.001) | about 0.3-0.5 h inference | highest (about +0.002/h) but small in absolute terms | Needs saved fold checkpoints. Decide on OOF only. Revert if ≤ 0 |
| 2 | B4-NS 380 member trained on crop-only images (no Ben Graham), same recipe | Strict | +0.001 to +0.004 | about 1.5 h | about +0.0015/h | Mirrors the 4th-place raw+Ben mix. Check ρ < 0.975 with the blend before adopting |
| 3 | Second seed of B4-NS 512 (strongest member) | Strict | +0.000 to +0.002 | about 2.3 h | about +0.0005/h | Seed ρ ≈ 0.993. Also measures σ_seed on OOF, which the decision rule needs |
| (ref) | Soft pseudo-labels on 1,928 APTOS test images (from `remaining_levers.md`) | Strict labels + unlabelled competition images | +0.002 to +0.008 | about 2.5-5 h | about +0.001/h | Largest absolute prior. Separate reporting row |
| low | Full-data refit, green channel, CLAHE sweeps, colour constancy, learned TTA weights, ridge or rank stacking | Strict | about 0 to +0.003 | 1.5 h+ each | low | No APTOS evidence, or negative evidence (Huang, the 7th/9th-place stacking reports) |

- **Combined expectation.** Levers 1-3 together give about +0.002 to +0.006 OOF, taking OOF to about 0.930-0.934. With pseudo-labels it is perhaps +0.004 to +0.010. The test score would be expected to move by a similar *true* amount (about 0.918 → 0.92-0.925 expected). But the 550-image draw adds SD about 0.009 around it, and the paired MDE is about 0.008-0.018. So the test cannot confirm the improvement, and a lower test score than 0.918 is entirely possible (roughly a 20-25% chance for a true +0.004 gain, given a paired SE of about 0.0045-0.006: P(Z < −0.004/SE)).
- **Ceiling statement for the report.** Under the strict protocol (ImageNet weights, APTOS labels only), the evidence supports a true QWK ceiling of about 0.925-0.93 on local splits of APTOS 2019. A reported test value up to about 0.935 is consistent with that within sampling error. The paper's 0.954 is about 2.5-3 test SEs above it.

### Gaps
- The T4 inference time for multi-scale TTA on 15 checkpoints was not measured. The 0.3-0.5 h figure is scaled from training throughput.
- The per-fold checkpoints of the current blend are assumed to be available, but I did not verify this.
- No source gives an APTOS learning curve (QWK vs training-set size) that would let me quantify a full-data refit or the value of more labelled data.
