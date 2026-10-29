# Evaluation protocols behind high-QWK (>= ~0.93) APTOS 2019 results

Scope note: these notes cover how high-QWK APTOS papers split data, select models, epochs and thresholds, and report results, plus published critiques and independent benchmarks. Notation: "QWK 93.4" means 0.934. Where I could read only an arXiv version, I say so. Section 4 also includes one calculation of my own on the project's saved Stage 4 outputs. It is clearly labelled and is not a literature finding.

## 1. What exactly does Chilukoti et al. 2024 say about split, validation, epoch selection, thresholds, test size and ensembling?

### Takeaway
For APTOS the paper gives **no split, no test-set size, no validation procedure, no epoch- or checkpoint-selection rule, no threshold step and no uncertainty estimate**. The only explicit three-way split (train/validation/test) it describes is for EyePACS. The headline 0.954 sits in Table 6 under "Pretrained Model = None", and the paper never defines what "None" means (ImageNet weights or random initialisation). The APTOS accuracy reported next to it (0.942) is far above the accuracy that every other high-QWK paper reports on APTOS. Taken together, the paper's protocol cannot be checked, and the 0.954 should be treated as an unverified upper value, not as a target that a strict protocol is expected to reach.

### Cited Findings
- **Bibliographic details.** BMC Medical Informatics and Decision Making 24:37 (2024), open access as PMC11323616. The abstract claims "state-of-the-art QWK scores (0.901, 0.967 and 0.944)" on EyePACS, APTOS and Messidor-2 — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Split (EyePACS only).** "We divide the data into three datasets for training, validation, and testing." "The training set contains 24,590 retinal images," and "the validation and test data sets contain 5,268 retinal images each" (35,126 images in total, which is a 70/15/15 split). The paper does not mention stratification or a seed. This is the only split it describes — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **APTOS.** The paper describes APTOS only as "a collection of 3,662 fundus photographs of patients from rural India". It gives no split method, no split sizes and no test-set size. The words "split" and "threshold" do not appear anywhere in the paper — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Epochs and checkpoints.** "The EfficientNet-B3 model is trained for 60 epochs." Strategy 1: "Weights are saved at 30 and 60 epochs, and then the two models are created." Strategy 2: "the model weights are saved at 10 intervals during training." The paper does not describe early stopping, best-validation checkpointing, or which split the reported metrics come from. Its only uses of the word "best" in a selection sense are "by choosing the parameters that give the best accuracy" and "as it achieved the best performance among the different pre-trained models used in this study". It does not say on which data those choices were made — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Ensembling (Strategy 2, the source of the 0.967).** "At the time of inference, the ten predictions are obtained using the saved model weights at ten intervals." "The final prediction is estimated to be the most common prediction among all the predictions." So the ensemble is a majority vote over 10 snapshots; tie-breaking is not described. For Strategy 1, "we aggregated the two sets of predictions to obtain the final predictions" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Output, loss and optimiser.** The output is a "Softmax activation layer to generate the probabilities for each category" (classification, not regression, so no cut-off thresholds). The optimiser is Adam; "The learning rate for all models used in this article is 0.001"; gradient clipping is 0.1 and weight decay 1e-4. The paper names no loss function and no learning-rate schedule — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Preprocessing.** Strategy 1 resizes to 150×150 with normalisation only. Strategy 2 uses "normalization and the images are resized to 300×300", with "CLAHE … in combination with Gaussian Blur" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Table 6, "Results on three datasets using proposed strategies"** (columns: Dataset, Pretrained Model, Ensemble, Acc, Prec, Rec, F1, QWK):
  - APTOS, None, No: 0.942 / 0.943 / 0.943 / 0.944 / **0.954**
  - APTOS, EyePACS, No: 0.954 / 0.952 / 0.942 / 0.954 / **0.956**
  - APTOS, EyePACS, Yes: 0.954 / 0.953 / 0.95 / 0.956 / **0.967**
  - EyePACS rows: 0.878 without the ensemble, 0.901 with it. Messidor-2 rows: 0.854 / 0.933 / 0.944

  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Meaning of "pre-trained".** "The pre-trained model in Table 6 indicates whether a model is pre-trained in the DR data set", and "we used the model trained on the Eyepacs data set to fine-tune on the Aptos and Messidor databases." "None" is never defined. The prose never mentions 0.954; it discusses only 0.956 and 0.967 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Internal inconsistencies.**
  - The EyePACS Strategy-1 ensemble is 0.867 in Table 3 but 0.901 in the abstract and Table 6.
  - Table 3 lists QWK of exactly 0 for VGG-16/19 and ResNet-18/34/50 on EyePACS.
  - Table 7 cites nnMobileNet at 0.934 on APTOS, while the text says 0.925.

  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Uncertainty, code and data.** There are no confidence intervals, standard deviations or repeated runs, and no code statement. On data: "The datasets used in this project are publicly available" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- **Duplicates.** Not mentioned — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

### Inferences
- The project's assumption of a 70/15/15 split for APTOS is borrowed from the EyePACS description. The paper never states it for APTOS, so the APTOS test size could be anything from about 366 (10%) to about 732 (20%) images, or the results could even come from the validation split.
- Nothing in the text excludes reporting the snapshot, epoch or "parameters that give the best accuracy" chosen on the reporting split. Snapshot voting with no stated selection rule leaves room for choices driven by the test set.
- **Accuracy as a check on plausibility.** Other APTOS papers that report QWK of about 0.93 report 5-class accuracy of only 85–89%: MIL-VT 85.5%, RadFuse 87.07%, nnMobileNet 89.1% (all in Section 2). Chilukoti reports 94.2% accuracy at QWK 0.954, and 95.4% at 0.967. That is 5–9 accuracy points above any other paper in this survey, which makes protocol differences (leakage, selection on the reporting split, a different label set or split) a plausible explanation alongside model quality.
- Taken at face value, Table 3's zero QWKs (models that predict a single class) and the 0.867 vs 0.901 mismatch suggest the results tables were not checked carefully.

### Gaps
- The text does not say whether the APTOS numbers come from a validation or a test split, how big that split was, or whether duplicates were removed. Only the authors could answer this, and I found no erratum, comment or author code.
- I could not see the exact context of "by choosing the parameters that give the best accuracy": the fetch tool returned only short quotes.

## 2. Other papers reporting QWK >= ~0.93 on APTOS 2019: split, test size, validation, external data and pre-training

### Takeaway
Results of 0.93 or more come from three kinds of protocol:
- **Kaggle hidden test set.** The winner reached 0.936, with about 88,702 extra EyePACS-2015 images and a large ensemble.
- **Cross-validation on the 3,662 training images.** nnMobileNet reached 0.934 with 5-fold CV.
- **Single random or stratified 80/20 or 70/15/15 splits.** RadFuse reached 0.932 (preprint); SSiT reached 0.930 ± 0.003 with EyePACS self-supervised pre-training; Fang et al. reached 0.930 as the best of a hyperparameter grid scored on the same split.

Nothing except Chilukoti exceeds about 0.936, and the careful protocols cluster at 0.92–0.93.

### Cited Findings (table)

| Paper (year, status) | APTOS QWK | Split / protocol | Test size | Separate validation for selection? | External data / pre-training |
|---|---|---|---|---|---|
| Chilukoti et al. 2024, BMC Med Inform Decis Mak (peer-reviewed) | 0.954 (Pretrained "None"); 0.956 (EyePACS init); 0.967 (EyePACS init + 10-snapshot vote) | not stated for APTOS | not stated | not stated | EyePACS-trained weights for 0.956 and 0.967; ImageNet use undefined — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) |
| Kaggle APTOS 2019 winner, Guanshuo Xu (2019 competition) | 0.9361 (top score); runners-up 0.9343, 0.9337, 0.9336 | hidden Kaggle test set; private leaderboard is 85% of it | see Q5 | not applicable (labels hidden) | 88,702 EyePACS-2015 images; L1 loss, GeM pooling, ensemble of models at different input sizes — [APTOS announcement](https://asiateleophth.org/?p=1297); method details as summarised by [Fang et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf) |
| nnMobileNet, Zhu et al. (CVPR Workshops 2024; read on arXiv 2306.01289) | 93.4 (Acc 89.1, AUC 97.8) | "Following the experimental setting of 5-fold cross-validation in [MIL-VT]" | about 732 per fold (5-fold CV of 3,662) | not stated | The authors contrast their model with others "pre-trained on large-scale external datasets" — [arXiv PDF](https://arxiv.org/pdf/2306.01289) |
| RadFuse, Mohsen, Belhaouari & Shah (2025, **arXiv preprint** 2504.15883) | 93.24 (Acc 87.07); in sensitivity sweeps the "peak" is 93.10 for step size and 93.26 for curvature | "divided into training (80%), and testing (20%)"; "we slected 10% of the training set for validation"; test holds 20% of each class | about 733 | yes for early stopping. RadEx parameters were chosen by sensitivity sweeps reported on "the APTOS-2019 dataset"; the paper does not say on which split | ImageNet init, 512 px, ResNeXt-50 — [arXiv PDF](https://arxiv.org/pdf/2504.15883) |
| SSiT, Huang et al. (IEEE TMI 2023; read on arXiv 2210.10969) | 92.97 ± 0.29 (fine-tuning, ViT-S); DiRA RN50 92.34 ± 0.36 | "randomly split into 70%/15%/15% for training/validation/testing"; "each experiment is repeated 5 times" (mean ± SD) | about 550 | yes (validation split) | self-supervised pre-training on 88,702 unlabelled EyePACS images — [arXiv PDF](https://arxiv.org/pdf/2210.10969) |
| Fang, Kornblith & Schmidt, NeurIPS 2023 Datasets & Benchmarks | local best 0.9297 (ConvNeXt-tiny); 0.88–0.93 across 19 ImageNet models; 0.9302 for ResNet-50 with RandAug | "local 80% to 20% random class-balanced train/validation split"; Table 3 gives "the best performing model … across our hyperparameter grid" on that split | about 732 (also the selection split) | **no**: one split used for selection and reporting | ImageNet only (local); 2015 data added for Kaggle submissions — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf) |
| MIL-VT, Yu et al. (MICCAI 2021) — reference point just below 0.93 | 92.0 (Acc 85.5, AUC 97.9), as cited | 5-fold CV (per nnMobileNet) | about 732 per fold | not stated | pre-trained on a "large fundus dataset" (unnamed) — [MIL-VT repo](https://github.com/greentreeys/MIL-VT); values from [nnMobileNet](https://arxiv.org/pdf/2306.01289) and [RadFuse](https://arxiv.org/pdf/2504.15883) Table 4 |
| Tymchenko, Marchenko & Spodarets (2020; arXiv 2003.02261; conference status not verified) | 0.925466 (Kaggle; rank 54 of 2,943) | Kaggle hidden test set | about 13,000 images ("APTOS 2019 … Dataset (13,000 images)") | n/a | "similar datasets with different labeling", multistage transfer — [arXiv](https://arxiv.org/abs/2003.02261) |
| Liu et al., Frontiers in Public Health 12 (2024, published 2025) | wKappa 92.3 (weighting not stated) | 10-fold CV | about 366 per fold | not stated; hyperparameters chosen by grid and random search on unstated data | ImageNet — [Frontiers](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2024.1442114/full) |
| Chaturvedi et al. (2020, **arXiv preprint** 2004.06334) — protocol example | 91.96 | 3,112 train / 550 "validation" | 550 | **no**: "The validation set, which contains 550 fundus photography images, was used for model" selection, and the same 550 images are used for reporting | ImageNet DenseNet — [arXiv PDF](https://arxiv.org/pdf/2004.06334) |
| Majumder & Kehtarnavaz (2021, arXiv 2103.04207; reported as IEEE Access) — protocol example | 0.84–0.90 (abstract: 0.90) | "randomly selecting 90% of the data for training and the remaining 10% for validation", 3 repeats | about 366 | **no**: "the model with the highest validation accuracy is saved", and results are on that same validation set | ImageNet Xception variant; EyePACS — [arXiv PDF](https://arxiv.org/pdf/2103.04207) |

Further results found only through search snippets, not fully read:
- A Frontiers in AI 2026 ensemble with SimCLR and diffusion augmentation reports 0.928 QWK under 5-fold stratified CV — [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1932259/full)
- A Springer EfficientNet-B0 paper reports QWK 0.923 ± 0.008 with "3,662 training and 1,928 test images". The 1,928-image Kaggle test set has hidden labels, so how this was scored is unclear — [Springer](https://link.springer.com/article/10.1007/s42452-025-07998-9)
- A 2026 cross-dataset paper reports about 0.92 in-domain, falling to 0.81–0.89 on unseen datasets — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959526001098)

Comparison tables that cite earlier APTOS kappas:
- RadFuse Table 4: CANet 90.0, GREEN-ResNet50 90.8, GREEN-SE-ResNeXt50 91.2, MIL-VT 92.0, Dual-Branch 89.03, ADCNet 74.78 — [RadFuse](https://arxiv.org/pdf/2504.15883)
- nnMobileNet Table 3: DLI 89.0, CANet 90.0, GREEN 90.8 / 91.2, MIL-VT 92.0, nnMobileNet 93.4 — [nnMobileNet](https://arxiv.org/pdf/2306.01289)
- Chilukoti Table 7: SSiT (Huang) 0.925, nnMobileNet 0.934 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

### Inferences
- **SSiT is the closest match to the project's protocol**: random 70/15/15 split, about 550 test images, validation available, 5 repeats. With self-supervised pre-training on all 88,702 EyePACS images it reaches 0.930 ± 0.003. The project's about 0.925 out-of-fold with ImageNet weights is therefore already in line with carefully evaluated published work.
- **The usual ceiling for ImageNet-only models on random APTOS splits is about 0.92–0.93.** Fang et al.'s 19-architecture sweep tops out at 0.9297, and that is already the best of a grid scored on the reporting split.
- **None of the ≥0.93 papers mention de-duplicating APTOS.** So the 29–36 leaked test images the project found would be present in all of their random splits and CV folds.
- **Selection on the reporting set is common in this literature.** Chaturvedi, Majumder and Fang all report the best checkpoint or hyperparameters on the same split they use for reporting. RadFuse's sweeps on an unstated split are another likely case.

### Gaps
- I could not read the full MIL-VT paper (its arXiv id was not found), so its exact fold protocol and pre-training data are second-hand.
- I could not confirm that the 0.9361 announced by APTOS is the private-leaderboard score. The announcement does not say so, but Fang et al. give 0.936 for Xu on the private leaderboard, which is consistent.
- The Springer EfficientNet-B0 and Frontiers AI 2026 papers were seen only as snippets.

## 3. Published reproductions, critiques and meta-analyses on leakage, duplicates and optimistic protocols

### Takeaway
No peer-reviewed paper I found documents APTOS-specific duplicates or measures their effect on QWK. The project's own duplicate count (123 groups, 30 with conflicting grades) appears to be new. Nearby evidence:
- **Ophthalmology imaging leakage.** Wrong splits inflate test accuracy by 5–30 points on OCT (MCC by 0.07–0.43).
- **A direct DR reproduction.** Public-data re-implementation of Gulshan et al. lost about 0.04 AUC on EyePACS and about 0.14 on Messidor-2.
- **Medical-imaging challenges.** Evaluation noise often exceeds the gap between winners, and "overfitting by observer" is common.
- **ML-based science generally.** Leakage is widespread across fields.

### Cited Findings
- **Voets, Møllersen & Bongo, PLOS ONE 2019 (peer-reviewed reproduction).** They re-implemented Gulshan et al. (JAMA 2016) on public Kaggle EyePACS and Messidor-2 data because the original code and data were unavailable. They reached AUC 0.951 (95% CI 0.947–0.956) on Kaggle EyePACS and 0.853 (0.835–0.871) on Messidor-2, against the original 0.99 on both. They blame single grades per image and different data. The 2018 preprint version reported 0.94 and 0.80, and noted the original paper "missed details regarding hyper-parameter settings for training and validation" — [arXiv 1803.04337](https://arxiv.org/pdf/1803.04337); [RePEc listing](https://ideas.repec.org/a/plo/pone00/0217541.html)
- **Tampu, Eklund & Haj-Hosseini, Scientific Data 2022 (peer-reviewed; ophthalmology OCT).** "Improperly split test sets inflated results by 0.07 up to 0.43 in terms of Matthews Correlation Coefficient (accuracy: 5% to 30%)". The inflation comes from near-identical images of the same subject falling on both sides of the split — [PMC9500039](https://pmc.ncbi.nlm.nih.gov/articles/PMC9500039)
- **Varoquaux & Cheplygina, npj Digital Medicine 2022 (peer-reviewed).**
  - Across 8 Kaggle medical-imaging challenges, "6 of the 8 challenges are in the diminishing returns category". For 4 of them "the evaluation noise is worse than the winner gap". In pneumothorax segmentation, the private set showed "an overfit larger than the winner gap". DR and APTOS were not among the challenges studied.
  - They recommend test sets of "at least several hundreds samples, ideally thousands or more" with confidence intervals, and leakage safeguards "separating out all test data from the start".
  - They warn of "overfitting by observer" through "adjusting the method to improve the observed cross-validation performance", and note that private challenge test sets expose this, "which is often not the case for published studies".
  - They flag implicit multiple comparisons from reusing the same datasets across studies.

  [PMC9005663](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663/)
- **Kapoor & Narayanan, Patterns 2023 (peer-reviewed).** Leakage was found in 294 papers across 17 fields, sorted into eight leakage types, and they propose "model info sheets". In civil-war prediction, the claimed advantage of complex models disappeared once leakage was fixed — [PMC10499856](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)
- **2025 arXiv survey of DL for DR screening (preprint).** It lists per-image rather than per-patient splits as a source of inflated metrics when both eyes of a patient cross the split. It also reports that high internal accuracy sometimes tracked camera, illumination or field of view rather than disease, and "sharp drops when re-tested on Messidor or DDR datasets after training on EyePACS/APTOS" — [arXiv 2511.11065](https://arxiv.org/pdf/2511.11065)
- **Fang et al., NeurIPS 2023.** The same ImageNet models that score up to about 0.93 on a local APTOS 80/20 split get **0.890–0.896 on the Kaggle private leaderboard**, even with 512 px inputs, more augmentation and the 2015 data. Public-leaderboard scores for the same submissions are only about 0.61–0.78 (for example ResNet-50 at 0.8952 private / 0.7531 public). Their per-dataset correlation between ImageNet accuracy and APTOS QWK is 0.06 — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf)
- **Kaggle 7th/10th-place solution (BloodAxe).** 4-fold CV QWK of about 0.92–0.97 against leaderboard scores of 0.75–0.81 (public leaderboard, judging by the values). For example, "Finetuned 806_813" has CV 0.9680 ± 0.0037 but leaderboard 0.803 — [GitHub](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Duplicates in other benchmarks.** Hashing the ISIC skin-lesion data found duplicate images with conflicting labels (the same image labelled NEV and MEL) — [ISIC forum](https://forum.isic-archive.com/t/a-list-of-duplicate-images-in-the-training-set/1141). A dermatology dataset audit reports duplicates and train–test leakage in DermaMNIST/Fitzpatrick17k and overlap across ISIC 2016–2020 — [arXiv 2401.14497](https://arxiv.org/pdf/2401.14497)
- **Other reproductions of Chilukoti et al.** Public GitHub repositories reproducing the paper exist. They appear to belong to the same project or team as this one, so they are not independent evidence. One reports the paper recipe at about 0.82 rising to 0.908 after fixes — [GitHub devxsubh](https://github.com/devxsubh/aptos-dr-grading-reproduction); [GitHub sshivanshg](https://github.com/sshivanshg/dr-grading-reproduction)
- **Leakage measured by the project itself.** In the Stage 2 faithful run, the 29 leaked test images changed QWK by under 0.001 (0.8275 on all 550, 0.8284 on the 521 non-leaked). Accuracy on the leaked images was only 65.5%, partly because of the conflicting grades. Source: local file `/Users/subhammahapatra/Downloads/dr-grading-reproduction/dr-repro/results/leakage_s2_faithful.json`

### Inferences
- **Direct duplicate leakage is unlikely to explain much of the 0.03–0.04 gap.** In the project's data the 29–36 leaked images are about 5% of the test set, and removing them barely changed QWK. Many leaked twins carry conflicting grades, so memorisation does not reliably raise test QWK. The large OCT-style inflation (Tampu) comes from many near-duplicate slices per subject, which APTOS does not have at that scale.
- **The bigger protocol levers are probably reporting-split selection, the choice of split, and an unstated or easier test set.** Fang's local-validation vs private-leaderboard gap (about 0.93 vs 0.896) shows that a local random split of the 3,662 training images is itself optimistic compared with a separate hidden test set.

### Gaps
- I found no peer-reviewed or preprint paper that systematically audits APTOS 2019 for exact or near duplicates, or measures their QWK effect, and no Kaggle discussion thread on APTOS duplicates came up in searches. The Aarhus MSc thesis that mentions "reliance on data leakage datasets" for a DR study returned 404 and could not be read.
- I found no published meta-analysis that quantifies QWK inflation from selecting on the test set in DR grading specifically.

## 4. Typical confidence intervals for QWK with about 550 test images, and how papers handle uncertainty

### Takeaway
With about 550 images and QWK around 0.91–0.93, a bootstrap 95% interval is about ±0.02 (total width 0.04). That is the same size as the gap to 0.954. Most high-QWK APTOS papers report **no** interval:
- Chilukoti, RadFuse, nnMobileNet and Fang: none.
- SSiT: SD over 5 training repeats (±0.003), which covers training noise but not test-sampling noise.

### Cited Findings
- **Reporting practice.**
  - Chilukoti et al. report no confidence intervals, SDs or repeated runs — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
  - SSiT: "each experiment is repeated 5 times and report the mean ± standard deviation". APTOS fine-tuning SDs are 0.20–0.63 kappa points for SSL methods and up to 2.17 for random init — [arXiv 2210.10969](https://arxiv.org/pdf/2210.10969)
  - Voets et al. report 95% CIs for AUC (e.g. 0.947–0.956) — [arXiv 1803.04337](https://arxiv.org/pdf/1803.04337)
  - A glaucoma-grading study reports 95% CIs for QWK — [arXiv 1810.13376](https://arxiv.org/pdf/1810.13376)
- **Method.** A SAS paper describes bootstrap CIs for kappa and weighted kappa, because asymptotic variance becomes unreliable at small sample sizes or as kappa nears 1 — [SAS SUGI22](https://support.sas.com/resources/papers/proceedings/proceedings/sugi22/STATS/PAPER295.PDF)
- **Recommendations.** Varoquaux & Cheplygina recommend "at least several hundreds samples, ideally thousands or more" plus CIs. They report that "only 6% of segmentation challenges surveyed" tested variability statistically and "15% out of 410 popular computer science papers" used a statistical test — [PMC9005663](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663/)
- **The project's own README.** 10-snapshot vote CIs on 550 test images are, for example, 0.868–0.921 and 0.862–0.918, "about ±0.03". The Stage 4b plan quotes "about ±0.025". Local files: `/Users/subhammahapatra/Downloads/dr-grading-reproduction/README.md`, `/Users/subhammahapatra/Downloads/dr-grading-reproduction/dr-repro/stage4b_plan.md`

#### Calculation on project outputs (not a literature finding)
I ran a quick check on the saved Stage 4 arm-D outputs (`stage4_runs/mse_s4*/outputs.npz`) with the script `/private/tmp/claude-501/-Users-subhammahapatra-Downloads-dr-grading-reproduction/ca81f130-82a7-4675-aff6-3e142e324b4e/scratchpad/scripts/opt.py`. It uses a 2,000-resample bootstrap, cut-offs tuned by coordinate search, and the 10-snapshot mean.

**Honest ensemble QWK (cut-offs tuned on validation) and bootstrap CI:**

| Run | Test images | QWK | 95% CI |
|---|---|---|---|
| seed 42 | 550 | 0.909 | [0.887, 0.928] |
| seed 43 | 550 | 0.917 | [0.897, 0.934] |
| seed 44 | 550 | 0.915 | [0.895, 0.932] |
| seed 42, de-duplicated | 526 | 0.926 | [0.906, 0.943] |

The de-duplicated run uses different data and may not share the same test split. The CI is about ±0.02.

**Optimism from choosing on the test set** (same three seeds):

| What is tuned on the test set | QWK gain over the honest score |
|---|---|
| Cut-offs | +0.005 to +0.011 |
| The single best of the 10 snapshots | +0.003 to +0.015 over the snapshot chosen on validation |
| The single best snapshot plus its cut-offs | up to +0.023 over the snapshot chosen on validation |

Validation QWK for the ensembles is 0.925–0.936, about 0.01–0.017 above test.

Per-epoch test QWK was not logged, so I could not measure the optimism of choosing the best of all 60 epochs on the test set. This simulates only a narrower case.

### Inferences
- At n of about 550, a reported 0.954 and a true value of about 0.93 differ by a little more than one half-width of the interval. Test-sampling noise alone could plausibly account for about 0.01–0.02 of the gap, especially if the paper used a single favourable split.
- Selection on the test set (cut-offs, snapshot or epoch) adds roughly +0.01 to +0.02 in the project's setup. Together with split noise (±0.02) and a smaller or unstated test set, protocol effects could plausibly cover about 0.02–0.04. That is most or all of the gap, but no single effect does. This is an inference, not a measured decomposition.

### Gaps
- I found no published table of QWK CI widths against test size for APTOS. The estimates above come from the project's own predictions.

## 5. What do independent, carefully evaluated settings (Kaggle private leaderboard, external validation) find for APTOS-trained models?

### Takeaway
On the hidden Kaggle test set, the best result in the world after a 2,943–3,922-participant competition, with about 88,702 extra labelled images and big ensembles, is **0.936**. Strong single models land at 0.89–0.93, and ImageNet-only models with standard tricks at about 0.89–0.90. External validation on other datasets drops further, to about 0.81–0.89. So 0.954 with a single ImageNet EfficientNet-B3 at 300 px, and 0.967 with a 10-snapshot vote, would both beat the competition winner by a wide margin.

### Cited Findings
- **Winning scores.** APTOS: "The highest score is 0.9361 closely followed by 0.9343, 0.9337 and 0.9336", from "more than 124,000 submissions from 3,922 participants" — [APTOS](https://asiateleophth.org/?p=1297). Fang et al.: "The top Kaggle submission achieves 0.936 QWK on the private leaderboard (85% of the test set)", using external data, extra augmentation, L1 loss, GeM pooling and multi-size ensembling — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf)
- **Single models in the winning ensemble.** One Inception-ResNet v2 from the winning ensemble "achieves 0.927 QWK" on its own. Fang's own ResNet-50 and Inception-ResNet v2 (512 px, more augmentation, plus 2015 data) reach 0.896 on the private leaderboard, and ResNet-152 reaches 0.890. More colour and affine augmentation alone gave +0.03 QWK — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf)
- **A published leaderboard entry.** Tymchenko et al. scored 0.925466 (rank 54 of 2,943) with multistage transfer from other DR datasets — [arXiv 2003.02261](https://arxiv.org/abs/2003.02261)
- **Test-set size, with conflicting sources.**
  - The public leaderboard is "1928 images, 15%" and the private leaderboard 85% — [Speaker Deck summary](https://speakerdeck.com/takarasawa_/aptos-2019-blindness-detection-mian-qiang-hui-zi-liao)
  - Tymchenko describes the dataset as 13,000 images — [arXiv](https://arxiv.org/abs/2003.02261)
  - SSiT says only 5,590 labelled images exist in total (3,662 + 1,928) — [arXiv 2210.10969](https://arxiv.org/pdf/2210.10969)

  These conflict. A private test set of about 11,000 images would fit 1,928 being 15% of about 12,850, but I could not confirm this.
- **Public vs private leaderboard.** Public-leaderboard QWKs are much lower than private for the same models (for example about 0.75 vs about 0.90), which points to a distribution shift between parts of the test data — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb33c53ed5b14ce9028309431f565cc-Paper-Datasets_and_Benchmarks.pdf); [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **External validation.** An APTOS-trained model reaches about 0.92 in-domain and 0.81–0.89 on unseen datasets (2026, seen only as a snippet) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959526001098). A DR survey reports sharp drops when EyePACS/APTOS models are tested on Messidor or DDR — [arXiv 2511.11065](https://arxiv.org/pdf/2511.11065)

### Inferences
- **Private-leaderboard scores are the cleanest independent benchmark** (separate hidden test set, no tuning on it). They put the best achievable APTOS QWK in 2019 at about 0.936 with external data, and about 0.89–0.93 for single models. On that scale, the project's about 0.925 out-of-fold with ImageNet init and no external data is a strong result. Reaching 0.954 under a strict protocol with ImageNet weights alone would be unprecedented in the sources found.
- Local-split QWKs (about 0.92–0.93) and private-leaderboard QWKs (about 0.89–0.90) for the same ImageNet models are not directly comparable, because the test populations differ.

### Gaps
- I could not open the Kaggle leaderboard or the winner's discussion post directly (Kaggle pages were not fetched), so the private-leaderboard figures rely on Fang et al. and the APTOS announcement.
- I found no prospective clinical validation of an APTOS-trained 5-grade model that reports QWK.
