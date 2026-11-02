# Label-noise handling and unlabelled-data (semi-/self-supervised) levers that use only APTOS 2019 images

Scope (from the assignment, not a source). Strict protocol: ImageNet-type weights, APTOS 2019 labelled train only (3,662 rows), 70/15/15 stratified split with seed 42, and a locked 550-image test set. Best strict result so far: test QWK 0.918 [0.900, 0.935] and OOF 0.928, from a 5-fold ensemble (B4-NS 380/512 + B3 384, regression with tuned cut-offs). Our byte-identical duplicate cleaning is already adopted (+0.010 val QWK).

This file adds to `APTOS QWK gap after Stage 4c/remaining_levers.md` §1 and §4, and to `Closing the APTOS 2019 QWK gap/aptos_top_solutions.md` Q2 and Q5. Those files already list each team's pseudo-label recipe and private-LB deltas, and I do not repeat them here. What is new here:
- whether the unlabelled images can be obtained, and a duplicate hazard
- why private-LB pseudo-label gains overstate what our hold-out would show
- direct APTOS evidence on in-domain self-supervised learning
- measured outcomes of label-noise methods on real (not synthetic) DR noise
- an estimate of how noisy APTOS labels are
- near-duplicate counts beyond byte-identical copies

**How to read the numbers.** Every number is tagged:
- **[private LB]** or **[public LB]**: Kaggle leaderboards. The public LB used the 1,928 downloadable test images; the private LB used the unseen re-run set.
- **[CV]**: k-fold or out-of-fold on APTOS train.
- **[local split]**: a single hold-out drawn from APTOS train.
- **[other dataset]**: not APTOS.

Retrieval note: the Kaggle threads were read through Kaggle's discussion JSON endpoint (`discussions.DiscussionsService/GetForumTopicById`) and are cited by their human-readable URLs. Current as of 29 October 2026.

## Q1. Pseudo-labelling the unlabelled APTOS Kaggle test images: what is available, what gains were reported, and how to do it fold-safely

### Takeaway
Only the 1,928 public-test images can be downloaded. The roughly 13k-image re-run test set (which held the private LB) was never released, and Kaggle said it will not release labels.

Every reported pseudo-label gain was measured on the Kaggle test distribution, the same distribution as the images being pseudo-labelled. On that distribution the gains were +0.001 to +0.009 [private LB]. The one team that tried pseudo-labelling on train-distribution images inside CV got "a tiny bit" or nothing [CV]. Our locked test set comes from the train distribution, so the realistic prior for our OOF and test scores is about +0.000 to +0.004.

Fold safety needs four things:
1. Per-fold teachers.
2. Excluding the 550 hold-out images from every teacher and student.
3. Hashing the 1,928 images against the 3,662 train images first, because Kaggle test contains duplicates of train images.
4. Logging this as a separate "strict labels + unlabelled APTOS test images" row.

### Cited Findings
- **The unseen test set cannot be obtained.**
  - Kaggle's welcome post says "We've retained as much data as possible for the test set ... so the train set is relatively small."
  - Kernels re-ran "against an unseen test set". A participant quoted the Data page: "It will be about 13k images."
  - Asked for the labels after the competition, Kaggle's Sohier replied: "We do not have plans to release the solution file so that we can preserve the test set for future opportunities."
  - Source: [Kaggle welcome/FAQ thread 97606](https://www.kaggle.com/competitions/aptos2019-blindness-detection/discussion/97606)
- **The 1,928 test images are 15% of the test data (the public LB). The other 85% was private, and the competition was kernel-only with no internet.** — [Inoichan slides](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome); 0.936 QWK on "85% of the test set" — [Fang et al., arXiv 2301.04644](https://arxiv.org/pdf/2301.04644)
- **Train-test duplicates exist.**
  - "Duplication also occurs between train and test, and I guess some of them have different labels." — ynktk, [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)
  - Maxwell: "there are duplicated images in train and test". He also published a duplicate-list kernel (maxwell110/duplicated-list-csv-file). — [thread 97606](https://www.kaggle.com/competitions/aptos2019-blindness-detection/discussion/97606)
- **Pseudo-labelling on train-distribution images did not help.** The 9th place (Psi, private 0.931, no pseudo-labels in the final submission) wrote that pseudo-labelling out-of-fold data inside CV "either did not improve much or only a tiny bit", and pseudo-labelling the full test set "did not change public LB". — [Psi, thread 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990) (also in aptos_top_solutions.md)
- **Recipe variety among the teams that reported deltas (details in aptos_top_solutions.md Q2):**

  | Place | Label type | Training | Delta |
  |---|---|---|---|
  | 1st | Soft | 10 more epochs | +0.001 [private LB], combined with IDRiD/Messidor |
  | 2nd | All test images, no confidence filter, 2 rounds | — | "huge" [public LB], no private number |
  | 5th | Confidence-filtered | — | not reported |
  | 7th | — | Fine-tune the head only | not reported |
  | 8th | — | 3-epoch retrain | +0.002/+0.004 [private LB] |
  | 10th | Soft labels with an ImageNet restart, or hard labels with fine-tuning | — | +0.003 with own labels, +0.009 with ensemble labels [private LB] |
  | 11th | Test split into 5 parts, one per fold | 10-epoch warm start | not reported |

  Neuron Engineer, outside the gold medals: 3 rounds, 0.926 → 0.929 [private LB]. — Kaggle threads [108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065), [107926](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926), [107960](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960), [108058](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058), [108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030), [107944](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944), [107958](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107958), [107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- **The 4th place gave up on pseudo-labels:** "I've tried pseudo once but didn't get good result so I gave up". — [Qishen Ha, thread 107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- **The test images come from a shifted distribution.**
  - "Image size and black-space cropping are strongly indicative of diagnosis in the training data, but not in the public test data."
  - Over 70% of the test set is 480×640.
  - The 1st place moved the 0/1 cut-off because the public test "had fewer normal cases than train".
  - Sources: [thread 100815](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815); [Camaro, 107944](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944); [Guanshuo Xu, 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065) (via aptos_top_solutions.md)
- **No academic APTOS semi-supervised paper reports a QWK delta from pseudo-labelling the 1,928 images.** Searches for FixMatch, Mean Teacher and pseudo-label APTOS QWK found only:
  - EyePACS low-label studies, using about 1,000 labelled images (2% of EyePACS). — [ResearchGate: Deep Semi-Supervised and Self-Supervised Learning for DR Detection](https://www.researchgate.net/publication/362489955_Deep_Semi-Supervised_and_Self-Supervised_Learning_for_Diabetic_Retinopathy_Detection)
  - Semi-supervised regression benchmarks that do not include DR (UTKFace, IMDB-WIKI, STS-B). — [arXiv 2510.15266](https://arxiv.org/html/2510.15266)

### Inferences
- **Why the private-LB gains overstate our case.**
  - Pseudo-labelling the 1,928 public-test images is a form of transductive domain adaptation toward the Kaggle test distribution: different aspect ratios and class mix. That is why gains showed up on the private LB, which is the same distribution.
  - Our 550-image hold-out and our OOF are train-distribution. The only train-distribution evidence (Psi's in-CV pseudo-labelling) is about zero.
  - So the prior for our OOF is +0.000 to +0.004. That is a downgrade from the +0.002 to +0.008 in remaining_levers.md, which had no train-distribution evidence.
  - A test-set gain of that size cannot be detected on 550 images (paired-bootstrap width about ±0.006-0.009, per remaining_levers.md).
- **Fold-safe recipe (refines remaining_levers.md §1).**
  1. Hash all 1,928 Kaggle-test images: SHA-256, plus a perceptual hash such as pHash or dHash on border-cropped thumbnails (see Q5). Compare them against all 3,662 train images.
     - Drop any Kaggle-test image matching one of the 550 hold-out images. Otherwise the student trains on a teacher-labelled copy of a hold-out image. No hold-out label leaks, but it is self-training on a test image.
     - Drop, or assign to that image's fold, any image matching a dev image.
  2. For student fold k, only fold-k teachers (trained without fold k) label the pseudo-label pool. Average the members and TTA views.
     - Never use one all-fold ensemble. Its labels carry fold-k label information into fold k's student, which inflates OOF.
     - The 11th-place variant gives each fold a disjoint 1/5 of the pool. It uses fewer images, but the pseudo-label sets are independent across folds.
  3. Use soft regression targets. Optionally drop or down-weight (×0.5) the pseudo-labels within about 0.15 of a cut-off.
     - The evidence is mixed: the 2nd place used no filter, the 5th used a confidence filter, and the 10th used both soft and hard labels.
     - There is no APTOS ablation of filtering. Treat the threshold as a validation-tuned hyperparameter chosen on OOF only.
  4. Cut-offs must be re-tuned on OOF only. The pseudo-label pool has a different class mix, so never use it for threshold fitting.
- **Protocol status (not decided here).** The 1,928 images are APTOS 2019 competition images that came with the same Kaggle download. They carry no labels, and no external dataset is involved. They are still outside the 3,662 labelled rows and come from a shifted distribution. Report them as a separate row: "strict labels + unlabelled APTOS test images (transductive)".

### Gaps
- No source gives a pseudo-labelling delta on APTOS CV or on a train-distribution hold-out with everything else fixed. Psi's comment is qualitative.
- No count was found of how many of the 1,928 public-test images duplicate train images. ynktk and Maxwell confirm duplicates exist but give no number for the train-test overlap. It has to be measured locally.
- Kaggle's "about 13k" unseen-test figure is a participant quoting the Data page. I did not read the Data page itself.

## Q2. Self-supervised pre-training on APTOS-only images (SimCLR, MoCo, DINO, MAE, BYOL) on top of ImageNet initialisation

### Takeaway
There is one direct APTOS measurement, on the same 70/15/15 split ratio as ours: ImageNet init followed by DINO self-supervised pre-training on APTOS gave quadratic κ 0.894 vs 0.893 (ResNet50) and 0.896 vs 0.896 (DeiT-S) [local split, 5 runs]. In-domain SSL from random init on 30k EyePACS images was worse than ImageNet supervised init [other dataset].

The apparent "DINO beats ImageNet by 11% κ" result on APTOS uses ImageNet-pretrained SSL weights, not APTOS pre-training, against a weak 0.81 baseline. Expected gain to our 0.928 OOF: about 0.000, at a cost (about 10+ T4-hours by my estimate) that exceeds the budget. Not recommended.

### Cited Findings
- **Matsoukas et al., "Pretrained ViTs Yield Versatile Representations For Medical Images".** APTOS2019, 3,662 images split 70/15/15, 256×256 input, quadratic Cohen's κ, median ± std over 5 runs. The SSL step was DINO, 300 epochs, batch 256.

  | Initialisation | ResNet50 | DeiT-S |
  |---|---|---|
  | Random | 0.849 ± 0.022 | 0.687 ± 0.017 |
  | ImageNet supervised | 0.893 ± 0.004 | 0.896 ± 0.005 |
  | ImageNet + DINO on APTOS | 0.894 ± 0.008 | 0.896 ± 0.010 |

  [local split] — [arXiv 2303.07034v3](https://arxiv.org/html/2303.07034v3)
- **Truong et al. 2021 (MLHC).** SimCLR, SwAV and DINO, all pre-trained **on ImageNet** (no fundus SSL). Scored on the APTOS Kaggle test via the LB (a weighted public/private average), 5 folds. Full-data fine-tuning κ: ImageNet supervised 0.806, SwAV 0.829, SimCLR 0.826 (all ResNet50), DINO 0.837 (ViT-S/8, a different backbone). Random init 0.593. The largest gains come in the 50-image regime: DINO 0.660 vs 0.482. [LB-scored] — [ar5iv 2108.10048](https://ar5iv.labs.arxiv.org/html/2108.10048); [PMLR v158](https://proceedings.mlr.press/v158/truong21a/truong21a.pdf)
- **PLOS ONE 2022, "To pretrain or not?" (EyePACS-1, about 30k training images, ResNet50).** MoCo v2 pre-trained on EyePACS from random init (IWNE-CL) "was weaker than both" ImageNet-supervised and ImageNet-MoCo inits, and ImageNet-MoCo was best. The authors attribute this to dataset size and to ImageNet-style augmentations being a poor fit for retinal images. Results are plots only (AUC), with no κ. [other dataset] — [PLOS ONE pone.0274291](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274291)
- **Self-supervised results that are not APTOS-only.**
  - Multi-scale SSL on EyePACS: κ 0.932, vs 0.903 for SimCLR pre-trained on EyePACS. — [Sci. Reports 2025](https://www.nature.com/articles/s41598-025-85685-w)
  - SimCLR on real + diffusion-synthetic images: APTOS QWK 0.928 [CV, 5-fold] (from a search snippet; the synthetic images make it non-strict and uncontrolled). — [Frontiers in AI 2026](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1932259/full)
  - A fundus SSL corpus with about 35k EyePACS and about 6k APTOS images, evaluated on RetinaMNIST (not APTOS). — [arXiv 2608.26686](https://arxiv.org/html/2608.26686v1)
- **AMD progression (other task).** The SSL advantage "shrinks under fine-tuning" once labels number in the thousands. The objective matters more than corpus scale or domain. — [arXiv 2609.12834](https://arxiv.org/abs/2609.12834)
- **A general small-data result, from a search summary only (not fetched).** BEiT/SplitMask pre-training on the target data alone was competitive with ImageNet pre-training on datasets of about 8k images. That compares from-scratch to ImageNet pre-training, not ImageNet + in-domain. — [El-Nouby et al., arXiv 2112.10740](https://arxiv.org/pdf/2112.10740)

### Inferences
- The Matsoukas numbers fit our setting closely: APTOS only, the same 70/15/15 ratio, ImageNet init followed by in-domain DINO, about 2.5k train images. The measured delta is +0.001 / +0.000 κ with std 0.004-0.010. Our ensemble is stronger (0.928 vs 0.893) and uses Noisy-Student EfficientNets, whose ImageNet features were already trained on 300M extra images. In-domain SSL on about 5k images (3,112 dev + 1,928 Kaggle test) is unlikely to add anything; expect +0.000 ± 0.003.
- **Cost estimate (mine, not measured).** Our B4-NS trains at about 50 img/s at 380 px on a T4 (remaining_levers.md). At 224 px that is roughly 145 img/s. DINO with 2×224 global and 6×96 local crops, plus teacher forwards, costs about 3.5-4 view-equivalents per image. So 300 epochs × 5k images ≈ 10-11 T4-hours, before re-running the 5-fold fine-tunes. MAE is cheaper per step but needs a ViT and many epochs. Either way it exceeds the about 9 GPU-hour budget.
- **Protocol status if it were tried.** SSL uses no labels. Including dev val-fold images keeps OOF label-clean but is image-transductive. The 550 hold-out images must be excluded from the SSL pool. The 1,928 Kaggle-test images raise the same reporting question as in Q1.

### Gaps
- No APTOS-only MAE, BYOL, MoCo or SimCLR on top of ImageNet init with QWK was found. No in-domain SSL on top of Noisy-Student EfficientNets was found on any DR dataset.
- I did not verify the El-Nouby result beyond the search summary.

## Q3. Label-noise methods for ordinal DR labels: co-teaching, DivideMix, robust losses, label smoothing, confident learning / cleanlab, re-weighting

### Takeaway
No co-teaching, DivideMix, GCE, SCE or bootstrapping ablation exists on APTOS with real (unsimulated) noise. The APTOS noisy-label papers inject synthetic symmetric noise and report accuracy only.

Where label-noise methods were tested on real DR noise and scored against the original labels, they **did not help or hurt**:
- cleanlab drop, relabel or soft labels on EyePACS: QWK 0.650 → 0.604 / 0.611 / 0.587
- ordinal label correction on Kaggle DR at τ=0: MAE 0.671 → 0.683-0.689

Methods that did help were mild target-softening: non-uniform (ordinal) label smoothing, +0.045 κ over CE [other dataset]; the 58th place's ±0.3 regression-target jitter; and conflict-aware duplicate cleaning (our +0.010, a Kaggle participant's +0.010 CV). Gains from cleaning toward expert consensus appear only when the evaluation labels are also cleaned (npj Digit Med: APTOS referable-DR AUC 0.785 → 0.810 vs consensus labels).

Our test labels share the original grader's noise, so expect ≤ +0.003 from any further noise method.

### Cited Findings
- **APTOS with synthetic noise (CUFIT, arXiv 2412.00150).**
  - Setup: symmetric noise at 10/20/40/60% "following Co-teaching", 2,930 train / 366 eval images, DINOv2 ViT-S with a Rein adapter. Accuracy only, no QWK, no clean baseline.
  - At 10% noise: full fine-tuning 66.8%, Co-teaching 82.8%, JoCor 84.8%, CoDis 83.2%, CUFIT 84.2%.
  - At 40% noise: Co-teaching 79.5%, CUFIT 81.6%.
  - The paper also states Kaggle-EyePACS has "an estimated 30-40% label error". [local split, synthetic noise] — [arXiv 2412.00150](https://arxiv.org/html/2412.00150)
- **Ordinal Adaptive Correction (ORDAC, arXiv 2509.02351), Kaggle DR 2015 (about 88k images, 5 folds).**
  - At τ = 0, i.e. the dataset's own noise, normal training MAE was 0.6712. ORDACC (label correction) gave 0.6893 and ORDACR 0.6828, so correction was slightly worse.
  - It helped only under injected asymmetric noise: at τ = 0.4, MAE 0.860 → 0.744.
  - No QWK reported. [other dataset] — [arXiv 2509.02351](https://arxiv.org/html/2509.02351)
- **cleanlab on real DR noise, evaluated against the original labels** (hkoul2000, a GitHub research project; EyePACS-2015 "colored resized", ConvNeXt; the planned APTOS run was never executed):

  | Variant | Training examples | Held-out QWK |
  |---|---|---|
  | Baseline | 23,014 | 0.6499 |
  | Drop flagged | 19,148 | 0.6038 |
  | Relabel | 23,014 | 0.6110 |
  | Soft labels | 23,014 | 0.5874 (best ECE: 0.024 vs 0.042) |

  The same project's leakage check: image-level vs patient-level split QWK 0.6426 vs 0.6476. [other dataset, unreviewed] — [hkoul2000/trustworthy-diabetic-retinopathy](https://github.com/hkoul2000/trustworthy-diabetic-retinopathy)
- **cleanlab on APTOS, evaluated against expert consensus** (Lin et al., npj Digital Medicine 8:10, 2025).
  - Original labels were "annotated by a single individual". Four ophthalmologists re-graded on the ICDR scale, and labels with more than 75% agreement were kept: 3,178 of 3,662 images.
  - The original labels matched the consensus at accuracy 0.906.
  - Iterative cleanlab (Swin Transformer, 5-fold CV, CE, 10 epochs) raised label accuracy to 0.928 after six rounds. The DQS-guided stop gave 0.924 after three rounds. Docta reached 0.912 and Fastdup 0.901.
  - Referable-DR AUC rose 0.785 ± 0.019 → 0.810 ± 0.021 (p = 0.036), and accuracy improved significantly (p = 0.006). No QWK was reported.
  - [local CV, binary task, consensus-label evaluation] — [Lin et al. 2025, Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11701072/fullTextXML); [Nature page](https://www.nature.com/articles/s41746-024-01424-x)
- **Non-uniform (Gaussian, ordinal) label smoothing** (Galdran et al., TVST 2020; private Mexican screening set, 46,865 images, ResNet50/101, ImageNet init).

  | Model | CE | Standard LS | N-ULS |
  |---|---|---|---|
  | ResNet50 quad-κ ×100 | 73.17 | 71.26 | 77.69 (p < 0.025) |
  | ResNet101 quad-κ ×100 | 71.98 | 74.52 | 77.19 |

  σ was set so that about 95% of the mass falls on neighbouring grades. The method is motivated by graders disagreeing mainly between adjacent grades. [other dataset] — [PMC7414697 via Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7414697/fullTextXML); [PubMed 32832207](https://pubmed.ncbi.nlm.nih.gov/32832207/)
- **Cost-sensitive regularisation.** The atomic-sub-task model, which integrates expert-disagreement data, "implicitly account[s] for inherent noise in DR grade annotations". It was tested on top of CE, focal loss and N-ULS (search snippet; numbers not retrieved). — [Galdran et al., cost-sensitive regularization](https://eprints.bournemouth.ac.uk/34882/1/paper_1508.pdf)
- **APTOS competition evidence on target noise and cleaning.**
  - 58th place: random label noise of ±0.3 on the regression target "greatly reduced overfitting" (aptos_top_solutions.md). — [Tymchenko et al., arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
  - jionie: "I have used the cleaned dataset to train, better cv: 0.9189->0.9284, but lower lb: 0.782->0.769" [CV vs public LB]. It is unclear whether "cleaned" meant dropping duplicates or conflicting labels; jionie later "just keep[s] all". — [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)
  - Robust-loss and label-smoothing reports on the private/public LB (78th, 37-50th): label smoothing 0.905 vs 0.903 [private LB], "helped a lot with overfitting" (aptos_top_solutions.md Q5). — [ilovescience 108209](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108209); [Toumbourou 107947](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947)
  - Mixup is a poor fit, because grading is non-linear ("when you mix grade 1 and 3 ... ophthalmologists will make a diagnosis as grade 3 or higher"). — Maxwell, [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)
- **QMix** (mixed label and image-quality noise, retinal diagnosis) is evaluated on DRTiD, DeepDRiD, EyeQ and DRAC, not APTOS. — [arXiv 2404.05169](https://arxiv.org/pdf/2404.05169)
- **No co-teaching or DivideMix result on real APTOS/EyePACS DR noise with QWK was found** (searched; CUFIT covers synthetic noise only).

### Inferences
- **Why classic noisy-label methods are a poor fit here.**
  - They are designed for high, mostly class-flip noise, where the clean test labels differ from the noisy train labels. Our test labels come from the same single-grader process as the train labels.
  - MSE regression already estimates the conditional mean grade, which averages out symmetric adjacent-grade noise. Noise mainly adds variance and overfitting.
  - Sample selection (co-teaching, DivideMix's GMM split) on about 2.5k images per fold would discard hard, borderline examples. Those are the grade 1/2/3 cases that QWK depends on.
  - The two real-noise tests found (cleanlab on EyePACS, ORDAC at τ = 0) both lost score against the original labels.
- **The cheap variants worth one run each** (strict; one training run each, about 87 min for B4-NS 380):
  - (a) **Soft-target self-distillation**: target = α·label + (1−α)·fold-safe OOF prediction, α ≈ 0.5-0.7. Already proposed in remaining_levers.md §4, and it combines with Q1 at no extra cost.
  - (b) **Regression-target jitter** of ±0.2-0.3 uniform, the 58th-place trick.
  - (c) **Conflict-aware loss weighting**: down-weight images whose fold-safe OOF residual exceeds 1.5 grades, rather than dropping them.

  Expected Δ OOF for each: +0.000 to +0.003, no APTOS ablation. N-ULS's +0.045 is against CE classification. Our regression baseline already captures most of the ordinal benefit (BloodAxe found regression > classification by 0.001-0.002 [CV], aptos_top_solutions.md Q5).
- **cleanlab relabelling toward a "truer" label** would be expected to lower QWK against the original-grader test labels, as on EyePACS. It is only worth doing as a separate "consensus-label" evaluation, which we do not have for APTOS. The npj re-grading labels do not appear to have been released; I found no download.
- **Drop or keep duplicates with conflicting grades.** We already dropped the 30 conflicting byte-identical groups. jionie's CV-up/LB-down result shows that cleaning raises a train-distribution score, and our hold-out is train-distribution. Our +0.010 is consistent with that and remains valid under our protocol. Hold-out images themselves are not cleaned: per our duplicates.json, 5 of the 29 hold-out rows with identical train images have a conflicting grade, so they stay as they are.

### Gaps
- No QWK for co-teaching, DivideMix, GCE, SCE, bootstrapping, ELR or cleanlab on APTOS with real noise.
- The npj APTOS consensus labels and the number of APTOS images cleanlab flagged are not reported or released, as far as I could find.
- The ±0.3 jitter result has no number attached.

## Q4. How noisy are APTOS labels, and what QWK ceiling does that imply?

### Takeaway
APTOS was graded by a single grader. Against a 4-ophthalmologist consensus (kept for only 3,178 of 3,662 images), the original labels are 90.6% accurate, so about 9% disagree. Byte-identical copies within APTOS disagree on 24% of within-group pairs. Their copy-vs-copy QWK is 0.77, on a grade-2-heavy subset.

Specialists grading against an adjudicated reference reach QWK 0.82-0.91 (EyePACS-type images). A simulation (mine) puts the ceiling at about 0.98 for a perfect predictor against labels with 9.4% adjacent-grade noise, so noise alone does not cap us near 0.93. The practical ceiling seen in competition CV is about 0.93-0.935 (with 2015 pre-training).

### Cited Findings
- **Single grader.** "The original labels for these datasets were annotated by a single individual" (APTOS and EyePACS). Four-ophthalmologist re-grading kept 3,178 of 3,662 images at more than 75% agreement. Original-label accuracy vs consensus was 0.906. No inter-grader κ was reported. — [Lin et al., npj Digit Med 2025](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11701072/fullTextXML)
- **Our own duplicate audit** (project file `dr-repro/results/duplicates.json` and `duplicate_rows.csv`; computed in this session):
  - 3,662 rows, 3,534 unique images, 123 duplicate groups (128 extra rows), 30 groups with conflicting grades.
  - Grade spans in the conflicting groups: 21 differ by 1 grade, 8 by 2, 1 by 3.
  - Grade pairs: (2,3) ×9, (2,4) ×8, (1,2) ×5, (0,1) ×4, (3,4) ×3, (1,4) ×1.
  - Within-group pairwise disagreement: 32 of 134 pairs (23.9%).
  - QWK between first and second copies: 0.772. The duplicate rows are skewed to grade 2 (130 of 246 copies), so κ is depressed by low variance.
  - In the faithful 550-row test split, 29 rows had an identical image in train, 5 of them with a conflicting grade. — local project files (no URL)
- **Kaggle participants found similar counts.** ynktk: "I found 268 duplicated images in train data, and 64 among them have different labels" (64/268 ≈ 24%, method unstated). Maxwell, using manojprabhaakr's similar-image kernel: "there are 143 duplicated images in train data". — [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)
- **9th place.** Setting about 30 test labels from their duplicate images in train "dropped our score by nearly 0.01 on LB", attributed to "a lot of labels are inconsistent". [public LB] — [Psi, thread 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- **Human agreement.**
  - Krause et al. 2018 (Ophthalmology): QWK vs adjudicated retina-specialist consensus was 0.82-0.91 for individual retina specialists, 0.80-0.84 for ophthalmologists and 0.84 for the algorithm.
  - Earlier literature cited 0.40-0.65 inter-grader κ.
  - Individual specialists had 74.4-82.1% sensitivity for moderate-or-worse DR.
  - [other dataset] — [arXiv 1710.01711](https://arxiv.org/pdf/1710.01711); [PubMed 29548646](https://pubmed.ncbi.nlm.nih.gov/29548646/)
- **Kaggle EyePACS label error** is "estimated 30-40%" (claim in CUFIT, citing earlier work). — [arXiv 2412.00150](https://arxiv.org/html/2412.00150)
- **Competition ceiling evidence.** Top-team 2019-train CV saturates at about 0.933-0.935 even with 2015 pre-training [CV], and the private LB maximum is 0.936 [private LB] (aptos_top_solutions.md Q4). — [Qishen Ha 107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987); [Xuan Cao 107916](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)

### Inferences
- **Simulation (mine, illustrative).** Draw true grades with the APTOS class prior (1,805 / 370 / 999 / 193 / 295) and n = 3,112. Then shift each label ±1 grade with probability p. A perfect predictor of the true grade scores QWK against the noisy labels of about 0.989 (p = 5%), 0.980 (p = 9.4%), 0.973 (p = 13%) and 0.958 (p = 20%).
  - If the 9.4% consensus disagreement were the only error, the ceiling would be about 0.98, far above 0.928.
  - Real noise is concentrated on ambiguous images, and some conflicts span 2+ grades (9 of 30 groups). That lowers the ceiling.
  - The duplicate test-retest rate (24% of pairs) implies a per-label inconsistency of about 13% if errors were independent (from 2p(1−p) ≈ 0.24), giving a ceiling of about 0.97.
- **What actually limits us.** The ~0.93 plateau seen across strong teams [CV] is probably set by irreducible ambiguity in borderline grade 1/2/3 images, and by labels that are noisy *and* predictable only from the same grader's habits. It is not set by a hard noise ceiling. Noise-robust training does not remove that ambiguity. Our 0.928 OOF is already above the 0.80-0.91 human-vs-adjudicated range, which fits a model that learned the single grader's own convention.
- **Implication for 0.954.** Label noise does not make 0.954 impossible in principle (the simulated ceiling is about 0.97-0.98). But no strict APTOS method in any source reaches it with k-fold CV. This supports the earlier conclusion that the paper's number reflects split or duplicate handling (paper_forensics.md, aptos_top_solutions.md Q4).

### Gaps
- No inter-grader κ for APTOS: the npj paper reports accuracy only. The re-graded labels do not appear to be public.
- The simulation assumes independent adjacent-grade noise. The real noise structure (which grades, which images) is unknown.

## Q5. Near-duplicate (not just byte-identical) images in APTOS: how many, and how they affect training and evaluation

### Takeaway
Byte-identical groups (123 groups, 30 conflicting) are most but not all of the problem. One audit with perceptual hashing plus pixel correlation found 134 exact and 14 near-duplicate pairs: 131 groups covering 270 images (7.4%), 37 with conflicting grades. It estimates that 5.3-6.0% of test images have a copy in train under random splits. Kaggle participants counted 143 to 268 duplicated train images depending on the matcher.

So our byte-level grouping probably leaves about 8-14 near-duplicate pairs (about 7 extra conflicting groups) ungrouped. That could put a handful of near-copies across our dev/hold-out boundary: an expected QWK effect under 0.002. It is still worth closing with a perceptual-hash pass before any new run, and before pseudo-labelling (Q1).

### Cited Findings
- **Fire-Dev2/aptos-leakage-audit** (GitHub; read from search snippets only, because the repo returned HTTP 404 when fetched in this session):
  - "134 exact and 14 near-duplicate pairs form 131 groups covering 270 images (7.4%)", with "37 contain conflicting grades".
  - Random splits put a training copy in the test set for 6.0% of test images at 80/20 and 5.3% at 70/30.
  - Method: exact hashes, perceptual and difference hashes, and pixel correlation of border-cropped thumbnails.
  - "The first rating was AI-assisted; 20 of the pairs were also rated by two people, who agreed on all of them."
  - Source: [Fire-Dev2/aptos-leakage-audit](https://github.com/Fire-Dev2/aptos-leakage-audit) (unverified; snippet only)
- **Kaggle counts.** ynktk: 268 duplicated train images, 64 with different labels (example: 14e3f84445f7 = grade 3 vs f0f89314e860 = grade 4, "the same but have different labels"). Maxwell: 143 duplicated train images via manojprabhaakr's "similar-duplicate-images-in-aptos-data" kernel. Duplicates also span train and test. — [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856); [thread 97606](https://www.kaggle.com/competitions/aptos2019-blindness-detection/discussion/97606)
- **Our audit is byte-level (SHA-based):** 123 groups and 30 conflicting. In the faithful split, 29 of 550 test rows had a byte-identical train copy, 5 of them conflicting. — local `dr-repro/results/duplicates.json`
- **Independent byte-level audit, same numbers.** "123 exact-duplicate groups, 30 of which have byte-identical images with conflicting labels"; "the random split leaks 29 test images into training". — [devxsubh/aptos-dr-grading-reproduction (our repo)](https://github.com/devxsubh/aptos-dr-grading-reproduction) (search snippet)
- **Looser perceptual hashing over-matches.** A 256-bit dHash with a Hamming threshold on EyePACS-2015 dropped 2,252 of 35,126 images and found 167 conflicting clusters. Its author warns that conflicting labels in a near-duplicate cluster "do not automatically prove which label is correct". [other dataset] — [hkoul2000](https://github.com/hkoul2000/trustworthy-diabetic-retinopathy)
- **Effect on scores.**
  - Cleaning raised CV 0.919 → 0.928 but lowered public LB 0.782 → 0.769 (jionie). [CV vs public LB] — [thread 98856](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/98856)
  - Copying duplicate labels to test images cost about 0.01 [public LB] (Psi). — [thread 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
  - A patient-level vs image-level split made no measurable QWK difference on EyePACS (−0.005, CI [−0.035, +0.041]). [other dataset] — [hkoul2000](https://github.com/hkoul2000/trustworthy-diabetic-retinopathy)
  - A related open-source fix: "keep conflicting-label duplicates out of separate splits". — [MedVisionAI PR #12](https://github.com/OmkarJayvantJadhav/MedVisionAI/pull/12)

### Inferences
- **Size of the residual near-duplicate risk for us.** Take the audit at face value: about 14 near-duplicate pairs beyond exact ones, in 3,662 images.
  - With a random 15% hold-out, about 14 × 2 × 0.15 × 0.85 ≈ 3-4 pairs would straddle dev and hold-out.
  - Even if all were mispredicted by one grade, a 550-image QWK moves by well under 0.005. More realistically, near-copies help the model only slightly: under 0.002.
  - The effect on OOF is similar in size, through fold-crossing near-duplicates.
  - So near-duplicates do not explain any meaningful part of our gap to 0.954. They are a hygiene item.
- **Recommended hygiene (cheap, about minutes of CPU).**
  - Compute pHash/dHash on border-cropped 64-px thumbnails, plus pixel correlation for confirmation, over the 3,662 train and the 1,928 Kaggle-test images.
  - Use a conservative threshold, and inspect the matches by eye: loose thresholds over-match, as in the 2,252-image EyePACS case.
  - Add confirmed near-duplicate groups to the fold-grouping key. Report the hold-out QWK with and without the affected hold-out rows. Do **not** re-split the locked test set: this is an analysis, not a decision.
- **Duplicates affect pseudo-labelling.** Any Kaggle-test image that near-matches a hold-out image must be excluded from the pseudo-label pool (Q1, step 1).

### Gaps
- The Fire-Dev2 counts (14 near pairs, 37 conflicting groups, 270 images) could not be verified: the repo returned 404 on fetch.
- No count of Kaggle-test ↔ train duplicates was found.
- No source measures the QWK effect of near-duplicate (vs exact) grouping on APTOS.
