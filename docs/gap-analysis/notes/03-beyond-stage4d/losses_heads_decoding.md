# Loss, head, schedule and decoding levers for strict APTOS 2019 QWK (beyond Stage 4d)

Scope and baseline (from the assignment, not from a source). Our best strict result is test QWK 0.918 [0.900, 0.935] and OOF 0.928. It comes from 5-fold B4-NS 380 + B3 384 + B4-NS 512 models with single-output MSE/SmoothL1 regression and OOF-tuned cut-offs. Already measured in-house:
- Ordinal regression with tuned cut-offs beats CE plus vote by +0.016.
- An auxiliary CE head at weight 0.3 adds +0.003 OOF.
- 30 epochs are no better than 20.

This file adds only what the earlier notes do not cover:
- `Closing the APTOS 2019 QWK gap/ordinal_training_techniques.md` already covers the Vargas CLM +0.084 vs CE at 128 px, the Beckham–Pal squared-error reformulation, Galdran's non-uniform label smoothing (NULS), De Fauw's kappalogclipped loss and +0.005 from ranked decoding, BloodAxe's 0.002 difference between threshold sources, the 1st place's 0.5→0.7 cut-off change, and the UWF mixup negative.
- `APTOS QWK gap after Stage 4c/remaining_levers.md` already covers pseudo-labels, FixRes, B5, soft-label self-distillation and seeds.

Evidence grades:
- **A**: controlled ablation on APTOS with a stated split.
- **B**: controlled ablation on another fundus DR benchmark (EyePACS, Messidor, DDR, IDRiD).
- **C**: confounded result, unclear split, or competition/leaderboard result.
- **D**: theory, my own simulation, or general ML.

Leaderboard numbers are labelled LB. All others are local split or CV.

## 1. Ordinal methods beyond MSE regression (CORAL, CORN, CLM, SORD, EMD, QWK loss, Poisson/binomial unimodal, ranking): which have head-to-head APTOS numbers against plain regression?

### Takeaway
I found **no source that compares any ordinal loss head-to-head with plain MSE regression on APTOS using QWK**.

The cleanest direct regression comparison is on EyePACS (grade B). There, **MSE beat CE by +0.020 QWK, beat SmoothL1 by +0.012, beat a pure kappa loss by +0.046 and beat kappa+CE by +0.019** (ResNet-50, 512 px, official 42,670-image test set). SmoothL1, our alternative arm loss, came out measurably worse than MSE.

On APTOS itself, a 2026 multi-loss benchmark (ResNet-50, 5 seeds) found that ordinal losses (ordinal encoding, Lp soft labels, Wasserstein-unimodal, mean-variance, exp-MSE) beat CE by only +0.004 to +0.007 Kendall τb, which is within about 1 SD. Enforced unimodality (UnimodalNet) was clearly worse. The study reports no QWK and has no MSE arm.

The best realistic use of these losses in our pipeline is **ensemble diversity, not a single-model gain**. A swap is unlikely to beat MSE by more than about +0.003 OOF.

### Cited Findings
- **Loss ablation, EyePACS (grade B, strongest regression-vs-others evidence found).**
  - Setup: ResNet-50, ImageNet weights, 512×512, SGD 1e-3, batch 16, 25 epochs, official split 35,126 / 10,906 / 42,670.
  - Test QWK: CE 0.8032, focal (γ=2) 0.8059, kappa loss 0.7775, kappa + CE 0.8050, MAE 0.7679, SmoothL1 0.8117, **MSE 0.8235**. Validation followed the same order (MSE 0.8207, SmoothL1 0.8094).
  - The MSE gain over CE replicated on other datasets and backbones. Messidor-2: +0.85 points. DDR: +1.30 points. Six other backbones: +1.1 to +1.9 points (MobileNet, DenseNet-121, ResNeXt-50, ResNet-101, ViT-S, ViT-HS).
  - Regression was decoded by clip to [0, 4] and round, with no threshold tuning. — [Huang et al., Identifying the Key Components in ResNet-50 for DR Grading, Diagnostics 2023 (PMC10216935)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Ordinal-loss benchmark on APTOS, 2026 (grade A for ordinal-vs-CE, but no QWK and no MSE arm).**
  - Setup: ResNet-50, ImageNet weights, GDRBench splits with an extra 80:20 train/val split, 100 epochs, Adam 1e-3, batch 64, 5 seeds. Loss hyperparameters tuned with Optuna on validation.
  - APTOS Kendall τb:
    | Loss | Kendall τb |
    |---|---|
    | CE | 0.849 ± 0.012 |
    | Ordinal encoding (Frank & Hall) | 0.856 ± 0.007 |
    | Binomial CE | 0.853 ± 0.006 |
    | UnimodalNet | 0.816 ± 0.007 |
    | Lp soft labels | 0.854 ± 0.004 |
    | Wasserstein-Unimodal-Wasserstein | 0.854 ± 0.013 |
    | Mean-variance loss | 0.856 ± 0.007 |
    | exp-MSE (proposed) | 0.853 ± 0.012 |
  - APTOS AMAE: CE 0.505, exp-MSE 0.471, MVL 0.484.
  - Averaged over five DR sets, exp-MSE is best on AMAE, AUOC and τb (τb 0.695 vs CE 0.674). The authors note "not all ordinal-aware methods outperform the nominal CE baseline: BCE, UN, and MVL yield weaker performance". — [Stelter et al., Preserving Ordinality in DR Grading through a Distribution-Based Loss Function, NLDL 2026 (PMLR v307)](https://proceedings.mlr.press/v307/stelter26a.html); [code](https://github.com/Trustworthy-AI-UU-NKI/Ordinal-DR-Grading)
- **CORN and ordinal label distribution learning on APTOS (grade C: single split, weak backbone, numbers rounded to 0.01).**
  - Setup: VGG16 at 224 px, APTOS 2,929 / 733 split, best-validation-QWK checkpoint reported. Note that this selects the checkpoint on the evaluation split.
  - QWK: CE (balanced) 0.82, focal 0.83, LDAM-DRW 0.85, **CORN 0.86**, OLDL symmetric 0.87, OLDL asymmetric 0.88, CAP-WAE 0.90. There is no regression baseline.
  - So CORN gave +0.04 over CE, but at an absolute level far below ours. — [CAP-WAE / Ordinal Label-Distribution Learning with Constrained Asymmetric Priors (arXiv 2509.26146)](https://ar5iv.labs.arxiv.org/html/2509.26146)
- **Poisson/binomial unimodal heads and EMD loss on Kaggle DR 2015 (grade B, results only as plots).**
  - Setup: Graham preprocessing, 256 px, a small 4.3M-parameter ResNet. Baselines were CE and squared error on (K−1)·sigmoid(f(x)). Unimodal heads were tested with CE or squared-EMD loss, with τ fixed at 1 or learned.
  - Authors' summary for DR: "In terms of QWK, only our binomial formulations appear to be competitive, but only in the argmax case do one of our methods (the binomial formulation) beat the cross-entropy baseline."
  - EMD helped accuracy for binomial. The authors suggest DR is large enough that "the deep net is able to learn reasonable and 'unimodal-like' probability distributions without it being enforced".
  - Unimodal heads do not beat CE under expectation decoding, which is the decoding analogous to our regression + cut-offs. — [Beckham & Pal, Unimodal probability distributions for deep ordinal classification, ICML 2017 (arXiv 1705.05278)](https://arxiv.org/abs/1705.05278)
- **Cost-sensitive (distance-aware) regulariser on CE / focal / NULS, EyePACS (grade B).**
  - Setup: ResNeXt-50, test on the 55k-image EyePACS test set. The regulariser penalises the expected ground-distance cost Σ_j p_j·M[y, j] and is added with λ = 10 (best in every case).
  - Test QWK:
    | Base loss | Plain | + cost-sensitive (CS) | + atomic sub-task (AST) |
    |---|---|---|---|
    | CE | 75.76 | 77.27 | 77.39 |
    | Focal | 74.72 | 77.38 | 77.94 |
    | NULS | 77.09 | 77.91 | 78.71 |
  - Messidor-2 (test only): 79.79 for NULS-AST. There is no regression baseline. Gains are +1.5 to +3.2 points over classification baselines that score about 2–5 points below Huang's MSE at a similar level. — [Galdran et al., Cost-Sensitive Regularization for DR Grading, MICCAI 2020 (arXiv 2010.00291)](https://ar5iv.labs.arxiv.org/html/2010.00291)
- **CORAL head ablation (grade C/D; the paper itself returned 404, so only a search snippet was seen).** In a MobileNetV3 DR-triage study with 3-fold CV, the CORAL model averaged QWK 0.9019, while "MobileNetV3 (No CORAL)" reached validation QWK 0.9058. The CORAL head did not help. The dataset was not confirmed. — [arXiv 2602.21943 (search excerpt)](https://arxiv.org/pdf/2602.21943)
- **Evidential ordinal threshold heads (grade C: mixed APTOS + Messidor-2 + EyePACS subset, split not stated).**
  - Setup: ConvNeXt-Base at 512 px, AdamW, cosine with warm-up, EMA, early stopping on validation QWK.
  - QWK: K−1 binary threshold heads with BCE 0.918; + learned query pooling 0.914; + evidential Dirichlet heads with KL annealing 0.940. In the ablation, removing KL annealing dropped QWK to 0.90.
  - There is no CE, MSE or CORAL baseline, and test selection is unclear. — [Uncertainty-Aware Ordinal Deep Learning for Cross-Dataset DR Grading (arXiv 2602.10315)](https://arxiv.org/html/2602.10315)
- **Pure QWK loss is worse than MSE at scale.** Kappa loss scored 0.7775 vs MSE 0.8235 on EyePACS ([Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)). This matches De Fauw's finding that pure kappa loss is unstable (earlier notes).

### Inferences
- With MSE beating SmoothL1 by +0.012 on EyePACS, our SmoothL1 arms may be slightly handicapped as single models. They may still be useful as diverse blend members. A cheap check is to retrain the SmoothL1 arm with MSE and compare per-arm OOF (grade D).
- Ordinal-vs-CE gains on APTOS (Stelter: +0.004 to +0.007 τb) are about the same size as the in-house regression-vs-CE gain of +0.016 QWK, or smaller. Nothing found suggests CORN, CLM, SORD or EMD would beat a tuned MSE + cut-off model by more than about +0.003 OOF. A loss swap mainly changes which images are mis-ranked, which can help blend diversity at r ≈ 0.98–0.99.
- How each would plug in:
  - **CORN / CORAL** (K−1 = 4 sigmoid logits): decode the score as Σ_k P(y > k), which is already a continuous expected grade, then run our cut-off tuner on it unchanged.
  - **CLM**: one scalar plus learned thresholds. Our cut-off tuner replaces the learned thresholds at decode time.
  - **Binomial / SORD / NULS / EMD on a 5-way softmax**: decode with the expectation Σ_k k·p_k and tune cut-offs.

  Each can share the backbone with the MSE head as an extra head (see Section 2), which avoids a separate training run.
- Enforced unimodality (UnimodalNet, Poisson) has negative or neutral evidence on DR. Skip it.

### Gaps
- No APTOS QWK for CORAL, CORN, CLM, SORD, EMD or ranking losses (for example a pairwise margin loss on the regression score) against MSE on the same split. I found no ranking-loss DR paper.
- The Stelter tables report AMAE, AUOC, τb and balanced accuracy, not QWK, and have no MSE regression arm.
- The full text of the mobile CORAL paper (404) and the Nirthika and de la Torre kappa-loss papers (noted earlier) were not read.

## 2. Hybrid heads (regression + classification + ordinal multi-task, as used by top APTOS 2019 teams): what gains were reported?

### Takeaway
Hybrid heads were used by top APTOS 2019 teams, but **no team published an isolated gain**. Examples include BloodAxe's regression + auxiliary loss, a 28th-place BCE + clipped MSE (CV 0.936), and the 1st place's Huber + GeM (earlier notes).

Controlled fundus evidence on combining losses is weak or negative:
- Kappa + CE scored 0.8050 vs MSE 0.8235 on EyePACS.
- A regression + classification feature-fusion model reported QWK 0.90 on APTOS with no ablation.

Our own +0.003 OOF from an auxiliary CE head is the best measured number available. Expect an extra ordinal head (CORN or cost-sensitive softmax) on top of it to add about +0.000 to +0.003. Its main value would be a second decodable score for blending.

### Cited Findings
- **Kappa + CE combined vs single losses (grade B).** On EyePACS, kappa + CE scored 0.8050 test vs CE 0.8032 and kappa alone 0.7775, all below MSE at 0.8235. Combining a kappa objective with CE recovers CE-level performance but does not reach regression. — [Huang et al. (PMC10216935)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Multitasking regression + classification model on APTOS (grade C/D).** Separate regression and classification models had their features concatenated into an MLP. Weighted kappa was reported as 0.90 on APTOS (search summary), with no single-head ablation seen. — [Multitasking Deep Learning Model for Detection of Five Stages of DR (arXiv 1907.12508)](https://arxiv.org/abs/1907.12508)
- **A competitor's account (grade D, secondary).** Regression with MSE "gave the best results", a Wasserstein-style loss did not beat it, and others "had success with ordinal classification, or using a mixed loss function". — [DagsHub write-up (search summary)](https://www.dagshub.com/AmbaniDev/detection-1)
- **Three-head design on APTOS (grade C).** CAP-WAE uses classification, asymmetric-Gaussian dispersion and ordinal-regression heads on VGG16. It reaches QWK 0.90 on APTOS vs 0.86 for CORN alone and 0.88 for OLDL-AS alone. Single split, 224 px, best-QWK checkpoint. — [arXiv 2509.26146](https://ar5iv.labs.arxiv.org/html/2509.26146)
- **Earlier notes (not re-researched).** BloodAxe's best single models were plain regression (CV 0.9234), despite regression + auxiliary-loss variants existing. De Fauw's EyePACS 5th place used the "kappalogclipped" hybrid.

### Inferences
- Concrete plug-in for our pipeline, all on one shared backbone:
  - Head A: MSE scalar (main).
  - Head B: 5-way softmax with CE + Galdran's cost-sensitive term or NULS, at weight about 0.3.
  - Optionally head C: CORN with 4 logits.

  At decode time, build candidate scores (s_A, E_B = Σ k·p_k, s_C = Σ P(y > k)). Choose a convex combination per fold on inner OOF, then tune cut-offs. Our measured +0.003 for the CE auxiliary head is the anchor. Replacing plain CE in head B with CE + CS (worth +1.5 to +2.7 points on EyePACS over CE) may raise head B's own decodable QWK. It is unknown whether it raises the blend.
- Hybrid heads cost about 0 extra GPU time. They are the cheapest way to get loss diversity without separate runs, but diversity inside one backbone is lower than across runs.

### Gaps
- I could not fetch any top-10 APTOS 2019 write-up with a hybrid-head ablation (Kaggle discussion pages do not render). I found no published number for "regression + classification + ordinal" vs regression-only on APTOS.

## 3. Regularisation and schedule (mixup/CutMix for ordinal targets, label smoothing, SWA, SAM, layer-wise lr decay, fine-tune length, progressive resizing, gradual unfreezing, freeze-BN): any APTOS evidence?

### Takeaway
There is controlled fundus evidence (EyePACS, grade B) for only a few of these:
- **cosine decay vs constant lr: +0.003 test QWK** (multi-step +0.003, exponential −0.005);
- **augmentation stack: +0.039 over none**, of which flip + rotation + crop + colour is best and adding every augmentation is slightly worse;
- **resolution: 512→768 gives +0.011 and →1024 gives +0.013**;
- **class-balanced sampling hurts**, instance sampling is best;
- **Balanced-MixUp: +0.006 to +0.009 QWK over instance sampling**;
- **Graham preprocessing: −0.005 vs plain** on EyePACS.

I found **no DR evidence at all for SAM, SWA, layer-wise lr decay, gradual unfreezing or freeze-BN**. Plain mixup on DR has one negative result, noted earlier. For regression targets, the principled variant is label-similarity-weighted mixup (C-Mixup), which is untested on DR.

Our recipe already has cosine, warm-up, EMA and strong augmentation, so most of the measured gains are already banked. The open items are a priori worth about 0 to +0.003 each.

### Cited Findings
- **Learning-rate schedule (grade B, MSE, 512 px, EyePACS test).**
  | Schedule | Val QWK | Test QWK |
  |---|---|---|
  | Constant | 0.8207 | 0.8235 |
  | Multi-step [15, 20] ×0.1 | 0.8297 | 0.8264 |
  | Exponential γ = 0.9 | 0.8214 | 0.8185 |
  | Cosine | 0.8269 | 0.8267 |

  — [Huang et al. (PMC10216935)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Augmentation (grade B).** Test QWK: none 0.7923, flip 0.8125, flip + rotation 0.8272, flip + rotation + crop + colour jitter 0.8310 (best). With every augmentation in the study: 0.8252. — [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Preprocessing (grade B).** Test QWK: default 0.8310, + Graham 0.8260 (the text says 0.8227), + CLAHE 0.8238. Both enhancements slightly hurt a ResNet-50 on EyePACS. — [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Resolution (grade B).** Test QWK 0.6388 / 0.7435 / 0.8032 / 0.8137 / 0.8164 at 128 / 256 / 512 / 768 / 1024 px. The 768 px run took 11 h 15 m vs 5 h 16 m at 512. — [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Sampling (grade B).** Instance-balanced sampling gave the highest validation kappa. Class-balanced and progressively balanced sampling overfit (figure only) ([Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)). In Galdran's Balanced-MixUp study (EyePACS test, ResNeXt-50), class-based sampling scored 74.45 vs instance-based 80.17 QWK, a gap of −5.7 points. — [Galdran et al., Balanced-MixUp for Highly Imbalanced Medical Image Classification, MICCAI 2021 (arXiv 2109.09850)](https://ar5iv.labs.arxiv.org/html/2109.09850)
- **Balanced-MixUp (grade B).**
  - Method: mix an instance-sampled batch with a class-balanced batch, using λ ~ Beta(α, 1).
  - EyePACS test QWK, ResNeXt-50: instance sampling 80.17; Balanced-MixUp α = 0.1 / 0.2 / 0.3 gave 80.35 / **80.78** / 80.62.
  - MobileNetV2: 78.75 → 79.61 (α = 0.1); α = 0.3 fell to 78.34.
  - Focal loss scored 79.73 and CB loss 79.19, both below instance sampling. There is no plain-mixup row. The authors say the results "do not represent the state-of-the-art". — [arXiv 2109.09850](https://ar5iv.labs.arxiv.org/html/2109.09850)
- **C-Mixup for regression (grade D for DR).** Plain mixup on regression labels "can result in arbitrarily incorrect labels". C-Mixup samples mixing partners with probability based on label similarity (a Gaussian kernel on label distance). It reports +6.56% in-distribution improvement over the best prior method across eleven regression datasets, none of them DR. — [Yao et al., C-Mixup, NeurIPS 2022 (arXiv 2210.05775)](https://ar5iv.labs.arxiv.org/html/2210.05775)
- **Ensemble and TTA increments at a well-tuned baseline (grade B).** On EyePACS, 10 TTA views moved test QWK 0.8581 → 0.8603 (+0.002), and a 10-model ensemble moved it 0.8581 → 0.8631 (+0.005), saturating by about 10–15. — [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **SAM / SWA / LLRD / freeze-BN / gradual unfreezing.** Targeted searches found no DR grading study reporting a QWK effect. SAM appears only in retinal vessel segmentation (DRIVE). The closest DR paper uses EMA without ablating it. — [search: SAM in a vessel-segmentation Sensors 2024 paper, via DOAJ](https://doaj.org/article/419074fae33f4679993247330c660555); [arXiv 2602.10315](https://arxiv.org/html/2602.10315)

### Inferences
- **Already banked.** Cosine, warm-up, EMA and strong augmentation are in the recipe. "30 epochs did not help vs 20" agrees with the small schedule effects measured on EyePACS (±0.003).
- **Graham preprocessing is worth a re-check as a diversity axis.** It lost 0.005 on EyePACS for ResNet-50. A plain-crop arm (or CLAHE) of an existing backbone could be both comparable and less correlated (grade D; the earlier notes record a DrHB LB swing).
- **Balanced-MixUp / C-Mixup plug-in.** For the MSE head, mix images and regress to λ·y_i + (1−λ)·y_j. Draw partners either from a class-balanced sampler (Balanced-MixUp, α ≈ 0.1–0.2) or from a label-distance kernel (C-Mixup, which favours adjacent grades). Keep α small. The measured upside is +0.6 to +0.9 points over instance sampling on EyePACS, against the UWF negative noted earlier. An APTOS prior is about −0.003 to +0.004 (grade D). Cut-offs must be refit because mixing shrinks the score distribution towards the middle grades.
- **SAM (ρ ≈ 0.05) on the B3/B4 heads** roughly doubles step cost (two forward-backward passes), so about +60–140 min per arm. With no DR evidence it is a pure prior. SWA over the last epochs is nearly redundant with EMA 0.998. Layer-wise lr decay and freeze-BN matter mostly for very small batches or foundation-model fine-tunes. At batch 16 with Noisy-Student EfficientNets, I would rank all four below the Section 2 heads and pseudo-labelling.
- **Higher resolution.** EyePACS shows a +0.011 gain from 512 → 768 for ResNet-50. Our own B4 380 → 512 gain was +0.004, which fits the curve flattening. A 640–768 px short FixRes fine-tune (already in remaining_levers.md) is the schedule lever with the most fundus evidence behind it.

### Gaps
- No APTOS or EyePACS ablation of SAM, SWA, layer-wise lr decay, gradual unfreezing, freeze-BN, CutMix or label smoothing on a regression target.
- Huang et al. did not test warm-up, optimisers other than SGD, mixup, SWA or EMA. Balanced-MixUp did not report a plain-mixup row or the input resolution.

## 4. Knowledge distillation (ensemble to single model, self-distillation) on APTOS or fundus data

### Takeaway
No APTOS or fundus paper shows a distilled single model reaching or beating its teacher ensemble on QWK under a clean protocol:
- The one APTOS ensemble-to-student QWK pair (teacher 0.975, MobileNetV2 student 0.938) has a teacher number that looks non-strict.
- Self-knowledge distillation for DR was evaluated on Messidor and IDRiD and reported mainly as accuracy.

For our goal (raising ensemble QWK), distillation is useful only as the soft-target mechanism inside fold-safe pseudo-labelling or label softening (remaining_levers.md). It is not useful as compression.

### Cited Findings
- **Ensemble teacher → MobileNetV2 student on APTOS (grade C).** The ensemble reports 95.8% accuracy / QWK 0.975 and the distilled student 90.0% / 0.938. The split was not verified. A 0.975 teacher is far above the 0.93–0.94 Kaggle private-LB ceiling, which suggests a lenient protocol. — [BRAC University repository](https://dspace.bracu.ac.bd/items/87e00522-386b-42a3-ba25-0b7523df3119)
- **Self-knowledge distillation for DR grading (grade C/D).** Evaluated on Messidor and IDRiD, reported mainly as accuracy. The full text returned 403, so no kappa or gain was seen. — [Luo et al., Automatic DR Grading via Self-Knowledge Distillation, Electronics 2020](https://www2.mdpi.com/2079-9292/9/9/1337)
- **Teacher choice matters (earlier notes).** On APTOS (balanced accuracy), a ConvNeXt-Base teacher gave better students than RETFound, and RETFound distillation could fall below no distillation. — [Informatica 15303](https://www.informatica.si/index.php/informatica/article/view/15303)
- **Self-supervised + KD on EyePACS (grade D).** It reports 79.1% linear-probe accuracy with an EMA teacher, and no kappa. — [search summary of Hajighasemlou et al. 2024](https://par.nsf.gov/servlets/purl/10489604)

### Inferences
- Plug-in that could raise QWK. For each fold k, form soft targets y_soft = α·y + (1−α)·ŷ_teacher(fold-k OOF ensemble), with α ≈ 0.5–0.7. Train the MSE head on y_soft. Optionally add the 1,928 unlabelled APTOS Kaggle-test images with ŷ_teacher targets. This is ensemble → single-model distillation used as label denoising. It must use fold-k teachers only, as remaining_levers.md explains. The expected gain is a +0.000 to +0.004 OOF prior with no fundus evidence.
- Because the APTOS labels are noisy (earlier notes: conflicting duplicates), distillation on train labels is more likely to help than distillation as compression. It can still at best match the ensemble's ranking. It cannot add information beyond the teacher.

### Gaps
- No fundus paper with a QWK ablation of ensemble → single-model distillation or born-again self-distillation on a held-out split. The MDPI self-KD full text was not accessible.

## 5. Decoding: cut-off optimisation (Nelder-Mead vs coordinate descent vs per-fold averaging), expected-grade decoding, temperature/calibration effects, and threshold overfitting on about 550 test images

### Takeaway
Theory says the QWK-optimal point prediction is an **affine transform of the regression function that matches the target's mean and variance**. This explains why MSE outputs, which are shrunk towards the mean, need stretched cut-offs.

Since cut-off decoding depends only on rank, **QWK after cut-off tuning does not change under any strictly monotone recalibration of a regression score**. Temperature or isotonic calibration therefore cannot raise our QWK unless the cut-offs are not refit. For a softmax expectation, temperature changes the score non-monotonically across images, so cut-offs must always be refit after it.

My APTOS-sized simulation (grade D) gives these sizes:
- tuning 4 cut-offs on 3,112 OOF scores inflates OOF QWK by about **+0.002**;
- applying them to a 550-image test set costs about **0.001** vs population-optimal cut-offs;
- the 550-image sampling SD is about **0.008**;
- cut-offs tuned on the test set itself would inflate test QWK by about **+0.007**;
- Nelder-Mead from the default start was about **0.005 worse** than coordinate search, because QWK is piecewise constant in the cut-offs.

### Cited Findings
- **Analytical QWK-optimal decision (grade D, theory, abstract only).** The optimal decision function under QWK "is obtained by a linear transformation of the regression function", with coefficients set so the prediction's mean and variance match the target's. The author notes that "the variance of the forecast is artificially made equal to the variance of the original value". — [Nedel'ko, Bulletin of Irkutsk State University, Series Mathematics, 2018, vol. 23, pp. 36–45](https://mathizv.isu.ru/en/article?id=843)
- **Expectation vs argmax decoding (grade B).** Beckham & Pal define the "expectation trick", E[k] under p(y|x), as an alternative to argmax. On DR, a unimodal binomial head beat CE only under argmax decoding. Under expectation decoding, CE was already competitive. — [arXiv 1705.05278](https://arxiv.org/abs/1705.05278)
- **No threshold tuning yet regression wins (grade B).** Huang et al. decoded MSE by clip + round (cut-offs 0.5/1.5/2.5/3.5) and still beat CE by +0.020 on EyePACS. Tuned cut-offs would add to this; the earlier notes report +0.005 from De Fauw. — [Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10216935/)
- **Generic optimism of tuning on the evaluation data (grade D).** Optimisation "amplifies overfitting", so a single CV estimate for optimised models is optimistic ([Tušar et al. 2017](https://kt.ijs.si/bernard/publication/tusar-2017-study/)). Nested schemes pick thresholds on inner OOF predictions and leave the outer fold untouched ([nestkit docs](https://nestkit.readthedocs.io/en/latest/architecture.html)).
- **Ord-MAP (grade D).** This is an ordinal alternative to MAP that minimises expected absolute-distance loss and is reported to beat MAP on ordinal tasks. It targets MAE, not QWK. — [Delgado 2025, UAB Divulga](https://ddd.uab.cat/pub/uabdivulga/uabdivulga_a2025m9/uabdivulga_a2025m9a6iENG.pdf)
- **Earlier notes (not repeated):** De Fauw's +0.005 from expected-score + fitted cut-offs, BloodAxe's 0.002 difference between threshold sources, and the APTOS 1st place's 0.5 → 0.7 cut-off move.

### Inferences
- **Simulation (my own, grade D).** Script: `thrsim.py`, in the session scratchpad.
  - Setup: APTOS class priors (1805/370/999/193/295). Score s = 0.25 + 0.85·y + N(0, 0.47²), which imitates a shrunk MSE output at population QWK about 0.933. Cut-offs are fitted on 3,112 "OOF" samples and scored on 550 "test" samples. 60 repeats.
  - Results:
    | Quantity | QWK |
    |---|---|
    | Naive 0.5/1.5/2.5/3.5 cut-offs on test | 0.897 |
    | Coordinate-descent cut-offs fitted on OOF, test | 0.931 ± 0.008 |
    | Population-optimal cut-offs, test | 0.932 ± 0.008 |
    | Nelder-Mead from default start, test | 0.927 ± 0.010 |
    | Variance-matching affine + fixed 0.5 cut-offs, test | 0.912 |
    | Oracle cut-offs fitted on the test set itself | 0.939 |
  - In-sample optimism on OOF: 0.9354 vs 0.9332 (+0.002).
  - Population-optimal cut-offs in this simulation were (0.98, 1.42, 2.66, 3.14). The lowest cut-off rose and the top one fell, the same pattern as our tuned top cut-off near 2.95 and BloodAxe's 3.39.
  - Implications:
    1. Our 0.928 OOF probably carries about +0.002 of cut-off optimism. Per-fold nested cut-offs (fit on 4 folds, apply to the 5th) would remove it.
    2. The test-set CI half-width (about ±0.017) is dominated by sampling noise (SD about 0.008), not by threshold transfer (about 0.001). No decoding change below about +0.01 can be confirmed on 550 images.
    3. A paper that tunes cut-offs on its own test set gains about +0.007 for free. This is relevant to the 0.954 paper target.
    4. Use coordinate or grid search (or Nelder-Mead with several restarts from a coordinate-search solution), not a single Nelder-Mead start.
    5. The 2-parameter variance-matching affine did *worse* than 4 free cut-offs here (0.912 vs 0.931). The Nedel'ko result holds for continuous predictions; with discrete grades and unequal class gaps, the free cut-offs capture class-specific shrinkage. So do not replace tuned cut-offs with affine rescaling.
- **Per-fold vs pooled cut-offs.** Pooled-OOF fitting uses 3,112 points and has low variance. Averaging 5 per-fold cut-off vectors is about as stable and also gives an honest nested OOF. The difference in final test QWK should be about 0.001, given the transfer cost above (grade D).
- **Calibration / temperature.** For the regression score, any monotone recalibration (Platt, isotonic, temperature on a sigmoid output) is absorbed by refitting cut-offs. It changes QWK only if the cut-offs are not refit, so it is not a lever. For a softmax auxiliary head, temperature changes E[k] in a way that depends on each image's distribution shape. Temperature can therefore change the ranking, and it should be tuned on inner OOF jointly with the blend weight between s_MSE and E_softmax. The expected effect is small (≤ +0.002, grade D, no source).
- **Expected-grade vs argmax.** For any softmax-type head (CE, NULS, CORN, binomial), decode with the expectation plus tuned cut-offs. Argmax discards the shrinkage correction. The in-house +0.016 (regression + cut-offs vs CE + vote) is consistent with this.
- **Overall ranking of this file's levers for the strict pipeline.** These are priors, not measurements.

  | Lever | How it plugs in | Expected Δ OOF (blend) | Cost |
  |---|---|---|---|
  | Nested per-fold cut-offs plus coordinate search with restarts | Decoding only | −0.002 on reported OOF (more honest); ±0.001 test | ~0 |
  | Extra CE + cost-sensitive (or NULS) softmax head and/or CORN head; blend s_MSE with E[k] on inner OOF | Shared backbone, one retrain per arm | +0.000 to +0.003 | 1 retrain per arm |
  | Retrain SmoothL1 arms with MSE | Loss change | +0.000 to +0.003 per arm | 1 retrain |
  | Fold-safe soft-label distillation (α·y + (1−α)·ŷ_teacher) | Combine with pseudo-labels | +0.000 to +0.004 | See remaining_levers.md |
  | Balanced-MixUp / C-Mixup on the MSE head, small α | Training | −0.003 to +0.004 | 1 retrain |
  | Plain-crop (non-Graham) arm for diversity | Preprocessing | +0.000 to +0.003 | 1 arm |
  | SAM / SWA / layer-wise lr decay / freeze-BN | Training | ~0 to +0.002 | SAM ×2 step cost |

  Even if all of them work, these add about +0.005 to +0.008 OOF. That does not close a 0.036 gap to 0.954, which again points to protocol differences rather than missing losses or decoders.

### Gaps
- No source compares coordinate descent, Nelder-Mead and OptimizedRounder on DR data, or measures cut-off optimism on APTOS. The numbers above come from a synthetic Gaussian-noise simulation whose noise model (homoscedastic, linear shrinkage) is an assumption.
- No closed-form Bayes-optimal decoder for expected QWK given a posterior was found, beyond Nedel'ko's regression result. QWK's denominator depends on the prediction marginals, so per-image expected-cost decoding is not exactly QWK-optimal.
- No DR study of temperature scaling or calibration effects on QWK. Galdran's multi-head calibration work was not fetched.
