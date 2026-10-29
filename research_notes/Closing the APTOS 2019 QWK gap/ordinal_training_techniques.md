# Training and decision techniques that raise QWK on ordinal DR grading

Scope: techniques that could plausibly add +0.005 to +0.03 QWK to an EfficientNet + MSE-regression + tuned-cut-off pipeline on APTOS 2019. Our current strict OOF QWK is about 0.925 and the target is 0.954.

Evidence grades used below:
- **A**: a controlled comparison on a proper held-out test set, repeated runs.
- **B**: a controlled comparison on validation or CV only, or on a leaderboard with a single run.
- **C**: competition write-up or README numbers with confounded changes (no clean ablation).
- **D**: a claim with no ablation, or an abstract with no numbers.

Read this first, because it affects every number below. On APTOS 2019, the public leaderboard, the private leaderboard and local CV disagree a lot. Public LB was about 0.80–0.83 while private LB was about 0.92–0.94 for the same models ([Inoichan deck](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome); [4uiiurz1 11th place](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)). In one top solution, adding external data *lowered* CV and *raised* LB ([BloodAxe repo](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)). So a gain measured only on a leaderboard, or only on in-distribution CV, may not carry over to a strict OOF protocol.

## Loss functions: MSE vs SmoothL1/Huber, ordinal methods (CORAL, CORN, cumulative link, SORD), direct QWK losses, regression + classification heads, focal loss

### Takeaway
No source I found shows a clean, held-out gain for CORAL, CORN or SORD over MSE regression on APTOS. Top APTOS 2019 finishers mostly used plain MSE regression. The strongest controlled evidence for an ordinal loss is on EyePACS at 128 px from a weak baseline: a cumulative link model + QWK loss gave +0.084 over softmax cross-entropy, but it was not compared with MSE regression. Treat loss swaps as a source of ensemble diversity, not as a reliable single-model gain at our level.

### Cited Findings
- **Cumulative link model (CLM) + QWK loss vs nominal softmax CE, Kaggle EyePACS 2015 (grade A for CLM vs CE).** Images were 128×128. The training set (17,563 image pairs, 10% of patients held out for validation) was evaluated on the Kaggle test set (26,788 pairs), with 5 runs per configuration. QWK was 0.498 ± 0.013 for the nominal CE baseline and 0.582 ± 0.016 for CLM clog-log + QWK loss, a gain of +0.084 that the paper reports as 16.8%. The link functions are nearly tied: clog-log 0.582 ± 0.016, probit 0.580 ± 0.021, logit 0.579 ± 0.009. Caveat: there is no MSE-regression baseline, and absolute QWK is far below modern levels. — [Vargas et al., Cumulative link models for deep ordinal classification (arXiv 1905.13392)](https://ar5iv.labs.arxiv.org/html/1905.13392)
- **Squared error on the expected class vs CE vs Cheng's ordinal binary encoding, Kaggle DR 2015 (grade B, validation only).** The "fix a" squared-error formulation "consistently achieves a much higher QWK than cross-entropy and the ordinal encoding scheme proposed by Cheng". A learnable-'a' variant with a sigmoid did better still. The results are shown only as validation-set plots, with no test-set numbers and no significance tests. The appendix also tests direct QWK optimisation with cold and warm starts. — [Beckham & Pal, A simple squared-error reformulation for ordinal classification (arXiv 1612.00775)](https://ar5iv.labs.arxiv.org/html/1612.00775)
- **Direct QWK loss vs ordinal-regression loss depends on model size (grade D from my reading; only the abstract/snippet was seen).** Nirthika et al. (2020) compared CE, direct QWK optimisation and OR-based losses on DR and prostate-cancer images. They report that "directly optimizing QWK is the better choice when small networks are used" while "for large networks OR based loss function gives better performance". I did not see their effect sizes. — [Nirthika et al., via rer.jfn.ac.lk](https://rer.jfn.ac.lk/items/dcddfe65-01e6-4c80-8fab-0a168aed20c3/full)
- **Original weighted-kappa loss (de la Torre et al.).** It is described secondhand as a continuous QWK used as a training loss and compared with log-loss on three datasets, including DR, with "reduced overfitting and training time". I did not access the primary paper or its numbers. — [cited in Vargas et al. (arXiv 1905.13392)](https://arxiv.org/pdf/1905.13392)
- **Kaggle DR 2015, 5th place (De Fauw) (grade C).** The final loss was a combined "kappalogclipped" loss (kappa + log-loss). Pure kappa loss was unstable early in training. MSE regression, tried briefly at the end, "got somewhat similar performance". No numbers were given for the comparison. — [Jeffrey De Fauw, Detecting diabetic retinopathy in eye images](https://defauw.ai/diabetic-retinopathy-detection/)
- **CORAL and CORN on DR (grade D for QWK).** The AOR-DR paper benchmarks CORAL and CORN on APTOS, Messidor, DDR and DeepDR. It reports accuracy, F1, sensitivity and specificity, not QWK. The authors say k-rank methods "struggle with the long-tailed distribution of DR data", and CORAL and CORN are close to each other on most metrics. — [AOR-DR (arXiv 2507.04978)](https://arxiv.org/pdf/2507.04978)
- **ViT + CORN on RetinaMNIST: QWK 0.8017.** There is no CORAL or MSE baseline, so no gain can be computed (grade D). — [search summary of rer.jfn.ac.lk / arXiv results](https://rer.jfn.ac.lk/items/dcddfe65-01e6-4c80-8fab-0a168aed20c3)
- **SORD (soft ordinal labels).** The original CVPR 2019 paper evaluates image-quality ranking, age estimation, horizon-line regression and monocular depth, not DR. It reports results "competitive with" specialised methods. I found no DR or APTOS QWK numbers for SORD. — [Diaz & Marathe, Soft Labels for Ordinal Regression, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html)
- **Non-uniform (ordinal-aware) label smoothing vs CE on DR (grade B).** Galdran et al. report quadratic kappa of 73.17 for CE and 77.69 for non-uniform label smoothing (P < 0.025). Plain label smoothing "generally harmed" performance. The baseline is CE classification, not MSE regression. — [Galdran et al., Non-uniform Label Smoothing for DR Grading (PMC7414697)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7414697/)
- **APTOS 2019, 7th/10th place (BloodAxe, ods.ai) (grade C).** Runs were labelled as classification, regression, or regression with an auxiliary loss. Options included `clipped_mse` and `focal_kappa` as a classification criterion. Ordinal training scripts (`train_ord.py`) exist but no results are reported for them. The best single models reported were regression models, e.g. reg_resnext50_rms with CV 0.9234 ± 0.0035. — [BloodAxe/Kaggle-2019-Blindness-Detection](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Top APTOS 2019 solutions used MSE regression.** The 11th place solution used MSE for both first- and second-level models (private 0.930) ([4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)). A gold-medal solution used a single-output EfficientNet-B5 regression (private 0.927 for EXP_765) ([DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)). A silver solution used EfficientNet-B2/B5 regression (private 0.924) ([Inoichan deck](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)).
- **Recent APTOS ordinal-regression paper: QWK 0.8992 (grade D).** This is a validation-only number from a frozen ResNet50 + MSE head at 224 px. There is no baseline and no ablation, so it is not useful as a gain estimate. — [Kumar et al., Stage-Aware Diagnosis of DR via Ordinal Regression (arXiv 2511.14398)](https://arxiv.org/html/2511.14398v1)
- **CAP-WAE (NeurIPS 2025).** It uses a direction-aware ordinal loss whose soft labels penalise under-grading more. It claims state-of-the-art QWK on "public DR benchmarks", but the abstract gives no numbers or datasets (grade D). — [NeurIPS 2025 abstract](https://nips.cc/virtual/2025/124883)

### Inferences
- The +0.084 CLM gain is relative to nominal CE at 128 px. It says nothing about CLM vs MSE regression at 0.92+ QWK. The squared-error paper suggests that expectation/regression-style losses already capture most of the ordinal benefit over CE.
- Our observed shrinkage of the top cut-off (about 2.95) is consistent with MSE on an imbalanced target. BloodAxe's optimised top threshold was 3.39, not 3.5, which suggests the same effect ([BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)). Cut-off tuning already corrects most of this shrinkage at decode time. A loss change is therefore more likely to change *which* images are ranked wrongly (useful for diversity) than to add a large single-model gain.
- The most plausible single-model loss gains are an ordinal/expectation classification head (CLM or softmax-expectation) or non-uniform label smoothing, trained alongside the regression head as a multi-task model. I found no APTOS-specific ablation, so a +0.005 gain is speculative.

### Gaps
- I found no paper with a head-to-head QWK comparison of MSE vs SmoothL1/Huber on APTOS or EyePACS.
- I found no source reporting QWK for CORAL vs CORN vs MSE on the same DR split.
- I found no APTOS ablation of focal loss vs MSE.
- I could not read the de la Torre and Nirthika primary papers, so their effect sizes are missing.

## Threshold / cut-off optimisation (coordinate search, Nelder-Mead, OptimizedRounder) and avoiding overfitting

### Takeaway
Tuned cut-offs reliably add a small amount: about +0.005 in De Fauw's EyePACS work and a few thousandths in APTOS write-ups. The APTOS 2019 winner's last-minute threshold change was one of the gains it reported. Tuned cut-offs do not transfer perfectly across distributions.

### Cited Findings
- **Kaggle DR 2015, 5th place (grade B/C).** Plain argmax "did quite well but is unstable". Decoding the softmax into an expected score with ranked cut-offs fitted by scipy `minimize` took some single models "from 0.83 to 0.835", i.e. about +0.005. — [De Fauw](https://defauw.ai/diabetic-retinopathy-detection/)
- **APTOS 2019, 1st place (grade C; read through a search-result excerpt, not the full page).** "Hours before the deadline" the author changed the QWK thresholds from [0.5, 1.5, 2.5, 3.5] to [0.7, 1.5, 2.5, 3.5], and private LB improved to 0.936. — [1st place solution summary, Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- **APTOS 2019, BloodAxe (grade C).** Thresholds optimised on OOF predictions were [0.527, 1.477, 2.428, 3.394]. A two-model ensemble scored 0.834 with OOF-tuned thresholds and 0.836 with thresholds tuned on APTOS-2015, a difference of 0.002. — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Silver solution.** Thresholds were "manually optimized", with no numbers. — [Inoichan deck](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)
- **Model selection.** A 37th-place author found that kappa "is unstable and often doesn't seem to be correlated with val_loss" and selected models on val_loss instead. — [tahsin314 repo, via search summary](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection)

### Inferences
- The 1st-place change moved only the lowest cut-off, from 0.5 to 0.7. Our tuned cut-offs moved the *top* cut-off a lot (about 2.95). Each pipeline's optimal cut-offs depend on how its own outputs are shrunk.
- To keep cut-offs from overfitting, fit them in nested form: tune on OOF folds other than the one being scored, then apply them to the held-out fold. Our "strict OOF" protocol already does something like this. The BloodAxe result (0.002 between threshold sources) suggests the variance from cut-off source is on the order of 0.002. This is an inference from one data point.

### Gaps
- I found no study comparing coordinate search, Nelder-Mead and OptimizedRounder on DR with nested CV.
- I found no published estimate of how much optimistic bias in-sample cut-off tuning adds to APTOS OOF QWK.

## Handling label noise and class imbalance in APTOS (duplicates with conflicting labels, label smoothing, mixup/cutmix, oversampling)

### Takeaway
The evidence here is thin. Ordinal-aware (non-uniform) label smoothing has the best controlled support (+4.5 kappa points over CE on DR). Plain label smoothing and mixup each have a measured *negative* result on DR. I found no primary source that quantifies APTOS duplicates or their effect on QWK.

### Cited Findings
- **Label smoothing on DR (grade B).** Plain label smoothing "generally harmed the performance of the CNN". Non-uniform label smoothing raised quadratic kappa from 73.17 to 77.69 over CE (P < 0.025). — [Galdran et al. (PMC7414697)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7414697/)
- **Mixup on DR (grade B, ultra-wide-field fundus images, not APTOS).** Adding mixup lowered kappa from 79.91 to 79.24. The authors state that "both pseudo-labeling and mixup compromise the DR grading performance". — [Source-free active domain adaptation for UWF DR grading (arXiv 2309.10619)](https://arxiv.org/pdf/2309.10619)
- **MixUp + transfer learning paper on APTOS (grade D).** It reports kappa 0.776, but I saw no with/without-mixup ablation. — [Springer 2026](https://link.springer.com/article/10.1007/s10791-026-10299-3)
- **Oversampling, EyePACS 5th place (grade D).** Oversampling the small classes gave "more stable updates and better, quicker training". No kappa number was given. — [De Fauw](https://defauw.ai/diabetic-retinopathy-detection/)
- **Data cleaning (grade C, confounded).** A gold-medal APTOS run combined 2015 and 2019 data that had been "cleaned", used StratifiedKFold and multi-resolution training, and reached LB 0.818 vs 0.808 for an earlier run. Cleaning cannot be separated from the other changes. — [DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)
- **Duplicates and leakage, general.** Retinal dataset audits flag duplicate and near-duplicate images as sources of bias and train/test leakage. I saw this only in a search summary, not in a specific primary source for APTOS. — (no primary source verified)

### Inferences
- With conflicting-label duplicates in APTOS, the cleaner fixes are (a) grouping duplicates into the same fold, so OOF is not inflated and the fold split stays strict, and (b) dropping or averaging conflicting labels. I have no measured QWK effect for either.
- Mixup on a regression target blends grades. The one DR ablation found it slightly harmful, so it is a low-priority experiment.

### Gaps
- I found no primary source counting APTOS 2019 duplicates or conflicting labels, or measuring the QWK effect of removing them. The Kaggle discussions on this exist, but I could not fetch them.
- I found no APTOS ablation of CutMix or oversampling.

## Training schedule tricks: SWA, EMA, longer training at higher resolution, progressive resizing, layer-wise lr, freezing BN

### Takeaway
Resolution, augmentation strength and 2015-data pretraining have the clearest reported effects. Stronger augmentation added about +0.03 QWK in a controlled APTOS study. 2015 pretraining and multi-resolution training gave LB gains, but confounded ones. I found no DR-specific QWK numbers for SWA, EMA, layer-wise lr or frozen BN.

### Cited Findings
- **Augmentation strength, APTOS 2019 (grade B).** Fang et al. used an 80/20 class-balanced local split (2,930 train / 732 eval). Stronger colour and affine augmentation alone gave about +0.03 QWK for ResNet-50. Training at 512 px with extra augmentation and the 2015 data raised private LB to about 0.89–0.90. Local validation QWK ranged from 0.8835 (AlexNet) to 0.9297 (ConvNeXt-tiny) at 256 px. ResNet-50 scored 0.9215 and EfficientNet-B4 was "slightly worse". Architecture changes after VGG gave "no significant gains". — [Fang et al., Does progress on ImageNet transfer to real-world datasets? (arXiv 2301.04644)](https://arxiv.org/html/2301.04644)
- **Augmentation level, BloodAxe (grade C, LB).** SE-ResNeXt50 with GAP scored hard 0.797, medium 0.802 and light 0.801. Pooling scored GAP 0.802, GWAP 0.794 and max 0.787. — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Resolution / multi-resolution, DrHB (grade C, LB).** Going from 224 to 352 px with 2015 pretraining did not change the single-model LB (0.725 → 0.727), and the 5-fold ensemble *dropped* 0.808 → 0.785. Combined 2015+2019 data with progressive 224→352→448 training reached CV 0.8216 / LB 0.818. A 456 px variant scored LB 0.821. B4 with 224→380 and RAdam + Lookahead reached CV 0.831 / LB 0.816 / private 0.927. — [DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)
- **Resolution, BloodAxe (grade C).** ResNeXt50 at 768 px had CV 0.9200 ± 0.0044 / LB 0.803. Most runs used 512 px. — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Resolution, EyePACS 5th place (grade D).** Inputs of 120 px gave "around 0.70 kappa", and the main 512 px models scored about 0.82–0.835. The architectures also changed, so this is not a controlled comparison. — [De Fauw](https://defauw.ai/diabetic-retinopathy-detection/)
- **2015-data pretraining then fine-tuning on 2019.** This was common among top APTOS finishers: 11th place ([4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)), DrHB ([repo](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)), and silver ([Inoichan](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)). The 1st place solution used two external datasets ([Kaggle 1st place, search excerpt](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)). BloodAxe found extra data lowered CV but raised LB: reg_resnext50_rms went from CV 0.9234 / LB 0.791 without extra data to CV 0.9129 / LB 0.804 with it ([BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)).
- **Schedules used by strong solutions (grade D, no ablation).** The 11th place used SGD with cosine annealing 1e-3 → 1e-5 for 30 epochs, then RAdam for 10 epochs on pseudo-labels ([4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)). Fang et al. used AdamW with a cosine schedule ([arXiv 2301.04644](https://arxiv.org/html/2301.04644)).

### Inferences
- The BloodAxe result, where external data lowered CV and raised LB, matters for us. If the 0.954 target was measured on a different split, or with 2015 data, our strict OOF on 2019 data alone may understate real gains from 2015 pretraining, or the two numbers may not be comparable at all.
- Given Fang et al.'s +0.03 from augmentation, checking whether our augmentation is "less" or "more" by their definition (colour jitter brightness/contrast/saturation 0.2 and hue 0.1, plus rotations) is a cheap, evidence-backed experiment.

### Gaps
- I found no DR or APTOS ablation of SWA, EMA, layer-wise lr decay or frozen BN. These remain untested priors from general image classification.

## Pseudo-labelling / semi-supervised learning and knowledge distillation on fundus images

### Takeaway
Pseudo-labelling the competition test set was a standard ingredient of top APTOS 2019 and EyePACS 2015 solutions, including the APTOS winner (soft pseudo-labels). None of them published a clean ablation. The one controlled DR study I found (UWF fundus, domain adaptation) found pseudo-labelling harmful.

### Cited Findings
- **APTOS 2019, 1st place (grade C, search excerpt).** The winner added "soft pseudo-labels from public test data in a second training stage" plus two external datasets. Private LB was 0.936. — [Kaggle 1st place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- **APTOS 2019, 11th place (grade C).** First-level SE-ResNeXt50/101 and SENet154 models (MSE, 2019 + 2015 data) produced pseudo-labels as a weighted average of predictions on the full public + private test set, split into 5 folds. Second-level models, initialised from first-level weights, trained for 10 epochs on 2019 train + pseudo-labelled test data. The final scores were public 0.826 / private 0.930. No ablation of the pseudo-labelling step was reported. — [4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)
- **BloodAxe (grade D).** Some model names contain "pl1", suggesting a pseudo-labelling round, but no isolated result is reported. — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **EyePACS 2015, 5th place (grade D).** Adding test images labelled with ensemble predictions "probably helped to push me to about 0.83 for a single model". — [De Fauw](https://defauw.ai/diabetic-retinopathy-detection/)
- **Negative result (grade B, UWF fundus, not APTOS).** "Both pseudo-labeling and mixup compromise the DR grading performance." — [arXiv 2309.10619](https://arxiv.org/pdf/2309.10619)

### Inferences
- In our setting, the honest analogue of competition pseudo-labelling is to pseudo-label *unlabelled external* fundus images (e.g. the EyePACS 2015 images without using their labels) or the APTOS test images, while keeping all scored folds strictly unseen. Pseudo-labelling images that later appear in an OOF fold would leak labels.
- Self-distillation from our own ensemble into a single model could compress the small +0.001 blend gain, but would not add much beyond it.

### Gaps
- I found no APTOS or EyePACS paper reporting a held-out QWK gain from pseudo-labelling or knowledge distillation with an ablation.
- I could not read the full Kaggle 1st-place write-up (Kaggle pages did not render), so details such as soft-label weighting, number of rounds and resolution are missing.

## Increasing ensemble diversity so blending actually helps (losses, preprocessing, resolutions, architectures)

### Takeaway
Ensembles helped in every top solution, but the reported gains are small: +0.005 to +0.02 on LB, and only +0.009 in BloodAxe's 3-model case. Diversity in these solutions came from different architectures, preprocessing (Ben's vs plain crop) and training stages more than from loss changes. Fang et al. found that architecture barely matters for APTOS accuracy. That agrees with our near-perfect backbone correlation (r = 0.984): swapping backbones alone is a weak source of diversity.

### Cited Findings
- **BloodAxe ensemble (grade C, LB).** Single models scored 0.828 and 0.823. Their 2-model blend scored 0.834, and the 3-model blend 0.843, about +0.015 over the best single model. TTA added at most +0.001 (FlipLR 0.844). — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **EyePACS 5th place (grade C).** Log-mean ensembling of 2–3 models at about 0.82 each reached about 0.828 on public LB. Ensembling plus ranked decoding "pushed my models to +0.84". — [De Fauw](https://defauw.ai/diabetic-retinopathy-detection/)
- **Fold averaging (grade C).** DrHB's 5-fold average reached LB 0.808 vs 0.725 for a single fold, +0.083. LB noise was large here, given the 0.80 public vs 0.93 private gap. — [DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION). Inoichan noted that 5-fold CV models generalised better on private LB than a single fold, even when public scores were similar. — [Inoichan deck](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)
- **Ben preprocessing as a diversity axis (grade C).** EfficientNet-B5 with Ben's preprocessing scored single LB 0.755 vs 0.725 without it, but the 5-fold average scored 0.804 vs 0.808. Preprocessing changes the model but not clearly the accuracy, which is the property useful for an ensemble. — [DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)
- **Architecture matters little on APTOS (grade B).** Models after VGG showed "no significant gains" in QWK. The top LB entry was an ensemble at 0.936 whose single Inception-ResNet v2 scored 0.927, so the ensemble added about +0.009. — [Fang et al. (arXiv 2301.04644)](https://arxiv.org/html/2301.04644)
- **11th place.** Diversity came from three SE-ResNeXt/SENet backbones plus a second, pseudo-label stage, averaged at the end. — [4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)

### Inferences
- Given r = 0.984 between our two backbones, the levers most likely to lower correlation are: (1) a different *target/loss* family, such as a CLM or softmax-expectation classification head alongside MSE; (2) different *preprocessing*, such as Ben's Gaussian-blur subtraction vs CLAHE vs raw crop; (3) different *resolution/crop* regimes; (4) different *training data*, such as 2015-pretrained vs not. Only (2) and (4) have even confounded APTOS evidence. (1) is based on the idea that loss changes alter which images are ranked wrongly, and is untested here.
- Ensemble gains of +0.01 to +0.015 in competition settings came from blends of models scoring about 0.82–0.83 on public LB. At 0.925 OOF, the headroom is likely smaller.

### Gaps
- I found no study measuring correlation between DR model outputs as a function of loss, preprocessing or resolution.
- I found no reported +0.03 single-technique gain on a strict held-out APTOS protocol at a 0.92+ baseline. The 0.954 target may depend on protocol differences (external data, test split) rather than on any one technique. This should be checked against the target paper.
