# Inflation mechanisms behind high APTOS 2019 DR-grading results (QWK / accuracy)

Context (from the project, not an external source): strict-protocol ensemble gives test QWK 0.918 (95% CI 0.900-0.935) and accuracy 0.81. Tuning on test raises this to at most 0.928. Removing the 36 test images that have an identical twin in train moves QWK by less than 0.007. The target is QWK 0.954 and accuracy about 0.94.

Note on the target paper: Chilukoti et al. 2024 (BMC Med Inform Decis Mak 24:37) reports exactly **QWK 0.954 / accuracy 0.942 on APTOS with EfficientNet-B3, without EyePACS pretraining and without an ensemble**. This is very probably the paper being reproduced. Its APTOS split, test-set definition, checkpoint selection and number of runs are **not reported** (details below).

## 1. Splitting after augmentation or oversampling (augmented copies in both train and test)

### Takeaway
In medical imaging, augmentation-before-split is a recognised form of leakage, but I found **no study that measures it in isolation**. The closest measurements come from the same "same source image in train and test" mechanism (slice-level or image-level splits): 5-30 points of accuracy in OCT and 29-55 points in brain MRI. For APTOS, offline class-balancing oversampling followed by a random split would put a sibling copy in train for almost every minority-class test image. That plausibly explains a +10-point accuracy gap on its own (my inference).

### Cited Findings
- Yagis et al. 2021 (Sci Rep, brain MRI): splitting at the 2D-slice level rather than by subject raised average slice-level test accuracy by 30% on OASIS, 29% on ADNI, 48% on PPMI and 55% on a local Parkinson's dataset. On randomly labelled OASIS data, slice-level splitting still gave about 96% accuracy, against 50% for subject-level splitting. The paper also describes the "late split" mechanism: if augmentation is done before the test set is separated, augmented copies of one image appear in both train and test, which inflates performance. — [Yagis et al., PMC8604922](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8604922/); [Sci Rep](https://www.nature.com/articles/s41598-021-01681-w)
- Tampu et al. 2022 (Scientific Data, OCT): improper (per-image) splits inflate performance by **0.07 to 0.43 MCC (accuracy +5% to +30%)**. Per dataset, per-image vs per-subject: AIIMS +0.08 MCC (accuracy about +0.04), Srinivasan +0.43 MCC (accuracy about +0.30), Kermany v2 +0.12 (accuracy about +0.10), Kermany v3 +0.07 (accuracy about +0.07). Kermany v2's original split has 92% of test images from subjects also in train. The paper names augmentation leakage (different augmented versions of one image in train and test) as a mechanism but **did not quantify it**. All reviewed works using per-image or original-Kermany splits reported accuracy above 95%. — [Tampu et al., PMC9500039](https://pmc.ncbi.nlm.nih.gov/articles/PMC9500039/); [arXiv 2202.12267](https://arxiv.org/pdf/2202.12267)
- Migraine MRI preprint (2026): SMOTE applied globally before partitioning "resulted in artificially inflated accuracy values". Restricting augmentation to training folds dropped performance substantially and exposed accuracy vs macro-F1 discrepancies. — [arXiv 2605.23453](https://arxiv.org/pdf/2605.23453)
- Leakage cannot happen when augmentation is applied on the fly inside the training loop. It can happen when augmentations are precomputed and serialised to disk before the split, which is common in Keras-style "augment the folder, then split" pipelines (vendor guide, not peer reviewed). — [LatticeFlow](https://latticeflow.ai/news/engineers-guide-to-data-leakage)
- The RIDGE checklist (medical segmentation) requires augmentation to be applied only after splitting into train/val/test. — [RIDGE, arXiv 2401.08847](https://arxiv.org/pdf/2401.08847)
- Kapoor and Narayanan (Patterns 2023): leakage affects 294 papers across 17 fields (22 reviews), classified into 8 leakage types, including "no clean separation of training and test" and "duplicates in datasets". The 2022 preprint version said 329 papers from 20 reviews. — [Kapoor & Narayanan, PMC10499856](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10499856/); [arXiv 2207.07048](https://ar5iv.labs.arxiv.org/html/2207.07048)

### Inferences
- APTOS class counts are roughly 1805/370/999/193/295 for grades 0-4. Balancing to about 1805 per class by offline augmentation means about x4.9 for grade 1, x1.8 for grade 2, x9.4 for grade 3 and x6.1 for grade 4. After a random 80/20 split of that augmented pool, the probability that a test copy has at least one sibling in train is 1 - 0.2^(k-1). That is about 0.998 for k=5, so nearly every minority-class test image is "seen". These are exactly the classes (1, 3, 4) where strict models make most of their errors. The fix would raise accuracy far more than QWK, which already saturates. This is consistent with the target's 0.942 accuracy against 0.954 QWK. **Estimate, not measured.**
- By the same logic, augmentation leakage pushes test accuracy towards training accuracy (often above 0.95 after 30+ epochs). That is a large enough mechanism to give +10 accuracy points and +0.03-0.04 QWK together. None of the other quantified mechanisms here is large enough on its own.
- The project could measure this directly: build the augmented pool offline, split at random, and compare against the clean split. That would be the first APTOS-specific number.

### Gaps
- I found no paper that quantifies augmentation-before-split leakage specifically for fundus or DR grading.
- Chilukoti et al. 2024 do not say whether preprocessing or augmentation was done before or after splitting.

## 2. Near-duplicates beyond exact duplicates; patient (left/right eye) leakage in APTOS

### Takeaway
APTOS has no patient IDs or eye pairing, so patient-level leakage cannot be checked or controlled. I found no published, quantified APTOS near-duplicate or patient-leakage analysis. The one fundus near-duplicate study found (pooled ODIR/RFMiD/APTOS) saw a negligible metric change (≤0.0007). That matches the project's own under-0.007 result for exact twins.

### Cited Findings
- A 2026 Diagnostics study pooled 11,223 fundus images from ODIR-5K, RFMiD and APTOS 2019. A perceptual-hash sweep found 7 near-duplicate pairs, 3 of which crossed the train/test split. Excluding the affected test images moved no metric by more than 0.0007. The authors note the cohort "is not duplicate-free". APTOS was eventually dropped for a resolution confound. — [Diagnostics 16(19):3109](https://doi.org/10.3390/diagnostics16193109)
- Kaggle participant (top 3%, 76/2943): "Almost 10% of the public test data were part of the train." Overwriting predictions with the train labels of the leaked images made results "drop significantly", which the author read as evidence that the training annotations "were pretty bad". In other words, duplicated images carried inconsistent grades. — [MamatShamshiev GitHub](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- Other participants removed duplicated training images and published deduplicated label files. — [Kaggle solution repos via search, e.g. tahsin314](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection)
- APTOS images come from multiple clinics, cameras and time periods. — [Fang et al. 2023, arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- In EyePACS, pairing left and right eyes of the same patient (as an input feature) is one of the components that raises ResNet-50 kappa. This shows strong between-eye correlation, so random image-level splits on patient-linked data leak patient information. — [Huang et al. 2023, Diagnostics, PMC10216935](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10216935/)

### Inferences
- APTOS 2019 does not release patient or eye IDs. If left and right eyes of the same patient are both in the set, random splits put correlated pairs on both sides. The size of this effect on APTOS is unknown and cannot be measured without IDs. Because the project removed exact twins with under 0.007 change, near-duplicates of the same type probably account for only a few thousandths of QWK (estimate).
- Fellow-eye leakage in other modalities is large (the OCT/MRI subject-level numbers in Section 1). In fundus images, though, the two eyes look different, unlike adjacent MRI slices, so the effect is probably much smaller (estimate).

### Gaps
- No published APTOS-specific near-duplicate graph (pHash/SSCD) with a before/after QWK was found.
- I could not verify how many APTOS images are same-patient fellow eyes.

## 3. Validation-as-test, checkpoint/hyperparameter selection on test, small test sets, best-of-N runs

### Takeaway
Even careful benchmarks on APTOS report the **maximum over a hyperparameter grid on a single local split with no separate test set**, and still top out at 0.93 QWK for ImageNet models. Small APTOS test sets (about 550-730 images) give sampling SDs of roughly 0.009 QWK. Seed SDs are 0.003-0.006 QWK. Picking the best of many runs or splits is therefore worth about +0.01 to +0.02 (estimate), which is consistent with the project's measured +0.006 to +0.010.

### Cited Findings
- Fang, Kornblith & Schmidt 2023 ("Does progress on ImageNet transfer to real-world datasets?"): APTOS was split 80/20 (2,930 / 732, class balanced) because competition test labels are hidden. The reported value is the **"best performing model ... across our hyperparameter grid"** on that 20% split. The grid covers 3 optimiser/scheduler settings, learning rates 1e-1 to 1e-5, 5 weight decays and 30 epochs; I estimate roughly 60 configurations, but the paper gives no count. No seeds or variances are reported. — [arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- Fang et al. QWK on that split: AlexNet 0.8835, ResNet-50 0.9215, EfficientNet-B0 0.9195, EfficientNet-B4 0.9211, DenseNet-121 0.9287, ViT-B/16 0.9262, ConvNeXt-tiny 0.9297 (best ImageNet model). The best CLIP model (CLIP-L14@336) reached 0.9337. No EfficientNet-B3 entry exists. — [arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- Fang et al.: increasing colour and affine augmentation alone accounts for a **+0.03 QWK** improvement. This is a legitimate, non-leaking lever of the same size as the remaining gap. — [arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- SSiT (Huang et al., JBHI 2024): APTOS split at random 70/15/15 (about 549 test images). Fine-tuning was repeated 5 times. Kappa SD across runs was 0.29-0.63 points (0.003-0.006) for SSL models and up to 2.17 points for random init. SSiT ViT-S scored 92.97 ± 0.29. The best other models scored DiRA RN50 92.34, DenseCL RN50 91.15 and MAE 90.97, all pretrained on EyePACS. There is no ImageNet baseline, and the paper does not say how checkpoints were selected. — [SSiT arXiv 2210.10969](https://arxiv.org/html/2210.10969v5)
- Varoquaux (2017): cross-validation error bars are underestimated, and standard error across folds understates them. The example is about ±10% accuracy for 100 samples. — [Varoquaux, arXiv 1706.07581](https://ar5iv.arxiv.org/html/1706.07581)
- Varoquaux & Cheplygina (npj Digit Med 2022): systematic biases enter at every stage, including evaluation and publication incentives. — [PMC9005663](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663)
- Kaggle threshold (kappa) optimisation was worth about +0.003 on the private LB and was unreliable on the public LB (one participant's estimate). — [MamatShamshiev GitHub](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)

### Inferences
- With about 550 test images, the sampling SD of QWK is about 0.009. This matches the project's CI half-width of about 0.017. Reporting the best of N random splits gives an expected maximum of about +1.16 SD for N=5 (+0.010), +1.54 SD for N=10 (+0.014) and +1.87 SD for N=20 (+0.017). Best of N seeds on a fixed split (seed SD about 0.005) gives only about +0.006 to +0.009. **Estimates from order statistics of a normal, not measured.**
- Selection effects (best split, best seed, best epoch on test, thresholds) stack only partially, because they are correlated. A plausible combined ceiling is about +0.02 over a strict mean, still short of +0.036 on its own. This matches the project's measured ceiling of 0.928.

### Gaps
- No APTOS study found reports QWK as a distribution over random splits.

## 4. Training on full data / testing on a subset; Kaggle public and private test; label noise

### Takeaway
Local APTOS holdouts give much higher QWK than Kaggle's hidden test. A top Kaggle team scored **about 0.97 on a 736-image local holdout** but 0.925 on Kaggle's private test and about 0.83 on the public test. A local-split number of 0.954 is therefore not comparable to the Kaggle 0.936. Training on all 3,662 images and "testing" on any subset is test-on-train and would give near-training accuracy (inference).

### Cited Findings
- Tymchenko et al. 2020 (Kaggle rank 54/2943): pretrained on EyePACS 2015 (35,126 images) and trained on APTOS 2019 + IDRiD (413) + MESSIDOR (1,200, Google relabel). **Local holdout of 736 APTOS 2019 training images: QWK 0.971 without TTA, 0.969 with TTA.** Kaggle "validation" (public LB): 0.818-0.827. Kaggle test (private): 0.9247-0.9255. They report the Kaggle test as about 13,000 images; Fang et al. describe the private LB as 85% of the test set. The two sources disagree on size, but the hidden test is larger than any local split. The authors note kappa is "a volatile metric". — [Tymchenko et al., arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- Fang et al. 2023: the top Kaggle submission scored **0.936 QWK on the private LB** (85% of the test set) and used 88,702 extra images from the 2015 Kaggle competition. Their own ResNet-50 and Inception-ResNet-v2 scored 0.896 on the private LB, and a tuned Inception-ResNet-v2 with extra interventions scored 0.927. For Kaggle submissions they trained on all available data, including the 2015 data. — [arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- The APTOS / Asia Pacific Tele-Ophthalmology Society page lists Guanshuo Xu first with a top score of 0.9361, then 0.9343, 0.9337 and 0.9336. — [asiateleophth.org](https://asiateleophth.org/?p=1297)
- A 37th-place EfficientNet ensemble reported private QWK of about 0.926. A single EfficientNet-B3 write-up reports private QWK 0.894. — [tahsin314 GitHub](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection); search-result summary of an EfficientNet-B3 GitHub write-up (not opened)
- The public test release is 1,928 images (a 2025 paper: "3,662 training, 1,928 test images", with QWK 0.923 ± 0.008 in CV and 0.919 on the held-out test). — [search result, Springer s42452-025-07998-9](https://link.springer.com/article/10.1007/s42452-025-07998-9) (snippet only; I have not verified it is the same paper)
- Label noise: clinicians often disagree and labelling schemes differ across datasets, so Tymchenko et al. used label smoothing. Overwriting predictions on leaked public-test duplicates with train labels made the score worse, which points to inconsistent labels between duplicates. — [Tymchenko et al.](https://ar5iv.labs.arxiv.org/html/2003.02261); [MamatShamshiev GitHub](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)

### Inferences
- A **local APTOS holdout can legitimately reach about 0.97 QWK** when the model has EyePACS pretraining and external data (Tymchenko). So 0.954 on a local split is not impossible without leakage if EyePACS or other external DR data is used. Chilukoti's own EyePACS-pretrained variant scored 0.956. For ImageNet-only pretraining, the best careful number found is 0.93 (Fang, best of grid).
- Local splits of the APTOS training set are "easier" than Kaggle's hidden test, likely because of distribution shift between the train and test clinics or cameras. A local-split 0.954 and a Kaggle-private 0.936 measure different things.
- The 62 conflicting-grade duplicate images put a ceiling on accuracy in strict splits. An accuracy of 0.94 implies the model reproduces label noise on the test images, which points to memorisation (Section 1) or test-in-train (inference).
- Mechanism of +0.03 to +0.04 QWK that is not leakage: EyePACS pretraining and external data (Tymchenko 0.97 local; Chilukoti 0.954 → 0.956 with EyePACS), plus stronger colour and affine augmentation (+0.03 in Fang).

### Gaps
- I could not open the Kaggle discussion threads directly (Kaggle forum pages were not fetched), so no thread-level duplicate counts are cited here.
- The exact private-test size is conflicting: about 13,000 (Tymchenko) against "85% of the test set" (Fang).

## 5. Papers reporting QWK ≥ 0.94 or accuracy ≥ 0.90 on APTOS 2019 5-class, and careful benchmarks

### Takeaway
The high claims found (≥0.95 QWK / ≥0.94 accuracy) **do not report their split or selection protocol**, and none is a verified leak-free result. Careful benchmarks with stated protocols land at **0.92-0.934 QWK** (Fang best-of-grid 0.930 ImageNet / 0.934 CLIP; SSiT 0.930; nnMobileNet 0.925-0.934 as cited). The only ≥0.95 local number with a known protocol (Tymchenko 0.97) used EyePACS pretraining plus external data and a single 736-image holdout.

### Cited Findings
- **Chilukoti et al. 2024**, BMC Med Inform Decis Mak 24:37 (EfficientNet-B0-B6, VGG and ResNet; B3 best). APTOS: no EyePACS pretraining, no ensemble: **accuracy 0.942, QWK 0.954**. With EyePACS pretraining: 0.954 / 0.956. With EyePACS + ensemble: 0.954 / **0.967**. The protocol is **not stated**: no APTOS split sizes, no statement of which data the APTOS numbers are computed on, no early-stopping or best-epoch description, single numbers with no seeds or variance. The only split described is EyePACS (24,590 / 5,268 / 5,268). Their comparison table cites nnMobileNet at 0.934 (0.925 in the text, which is inconsistent) and SSiT at 0.925. — [PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Tymchenko et al. 2020**: local holdout of 736 images, QWK 0.969-0.971, with EyePACS pretraining + IDRiD + MESSIDOR. On the Kaggle private test it scored 0.925. — [arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- **Fang et al. 2023** (careful, but best-of-grid on the reported split): ImageNet models 0.8835-0.9297, CLIP up to 0.9337. — [arXiv 2301.04644](https://arxiv.org/html/2301.04644)
- **SSiT (Huang et al. 2024)**: 70/15/15 split, 5 runs, 0.9297 ± 0.0029 (ViT-S, EyePACS SSL pretraining). — [arXiv 2210.10969](https://arxiv.org/html/2210.10969v5)
- A 2025 transfer-learning study (arXiv 2507.17121, per search snippet) reports **accuracy 89.60%, QWK 0.930** for 5-class. Its 98.50% is for binary. Other papers advertising about 98% "APTOS accuracy" are binary or use other preprocessing; for example, one DenseNet121 study with CLAHE + super-resolution reports 98.36% accuracy, task granularity not verified. — [arXiv 2507.17121](https://arxiv.org/html/2507.17121v1) (snippet-level); search results
- A 2025 EfficientNet-B0 study: QWK 0.923 ± 0.008 in CV and 0.919 on a held-out test (snippet-level). Its protocol and numbers are close to the project's strict result. — [Springer s42452-025-07998-9](https://link.springer.com/article/10.1007/s42452-025-07998-9)
- Cross-dataset sanity: a model trained on EyePACS+DDR gets QWK 0.865 on APTOS. A 2026 APTOS-trained model gets 0.92 kappa in-domain (mean over 5 seeds). — [Sensors 26(14):4515](https://doi.org/10.3390/s26144515); [ScienceDirect S2405959526001098](https://www.sciencedirect.com/science/article/pii/S2405959526001098)
- A review notes that reported APTOS accuracies are high while DDR results fall to about 80%, and calls for multi-centre validation. — [arXiv 2511.11065](https://arxiv.org/pdf/2511.11065)

### Inferences
- The project's strict ensemble at 0.918 (CI 0.900-0.935) agrees statistically with every careful protocol found: Fang 0.92-0.93 (single models, best of grid), SSiT 0.930, the 2025 EfficientNet-B0 at 0.919-0.923, and Kaggle private 0.925-0.936 with external data. The 0.954 / 0.942 target sits above every protocol-stated ImageNet-only result.
- Ranking of mechanisms that could close +0.026 to +0.036 QWK and +10 accuracy points:
  1. Augmentation or oversampling before the split (large, can explain both gaps; not measured for DR).
  2. Training on all data and evaluating on a subset (test-on-train; explains everything).
  3. Undisclosed external DR pretraining (EyePACS), which is legitimate but changes the claim; worth about +0.02 to +0.05 on local splits based on Tymchenko and Chilukoti (estimate).
  4. Stronger augmentation (+0.03 in Fang; legitimate).
  5. Selection effects (+0.01 to +0.02 combined; project measured +0.006 to +0.010).
  6. Duplicates and near-duplicates (≤0.007 measured).
- An accuracy of 0.942 alongside QWK 0.954 is the strongest signal for mechanism 1 or 2. Legitimate improvements (pretraining, augmentation) raise QWK on careful splits but have not been shown to give 5-class APTOS accuracy above about 0.85-0.90.

### Gaps
- RETFound (Zhou et al., Nature 2023) APTOS results could not be retrieved in this session. From memory they are reported as AUROC rather than QWK, but that is **unverified** and not cited here.
- I did not find a "Fang et al. 2023" DR benchmark other than Fang, Kornblith & Schmidt (NeurIPS 2023, ImageNet transfer), which is the one used above.
- The search surfaced no other peer-reviewed paper with a stated protocol and QWK ≥ 0.94 on ImageNet-only APTOS 5-class. Many such claims exist in lower-tier venues but were not checked one by one.
