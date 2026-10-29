# Paper forensics: Chilukoti et al. 2024 (BMC Med Inform Decis Mak 24:37), APTOS QWK 0.954 / accuracy 0.942

Paper: Chilukoti SV, Shan L, Tida VS, Maida AS, Hei X. "A reliable diabetic retinopathy grading via transfer learning and ensemble learning with quadratic weighted kappa metric." BMC Med Inform Decis Mak 2024;24:37. DOI 10.1186/s12911-024-02446-x. Received 15 Dec 2022, accepted 29 Jan 2024, published 6 Feb 2024. Full text read in full from [PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) (Europe PMC full-text XML, with tags stripped). The NSF PAR copy ([par.nsf.gov/servlets/purl/10489604](https://par.nsf.gov/servlets/purl/10489604)) is the typeset publisher PDF ("12911_2024_2446_Article.indd" in its metadata), so it is the same version, not an earlier manuscript.

Note on the user's framing: the paper itself does not use the title wording "Diabetic retinopathy detection using transfer learning...". The 0.954 is the Table 6 row "Aptos | Pretrained Model = None | Ensemble = No". The paper never defines "None". From the surrounding text it most likely means ImageNet weights, with no DR pre-training.

Labels used below: **[Stated]** = the paper's own words; **[Computed]** = my arithmetic on the paper's published tables (method given); **[Inference]** = my reasoning.

## Q1. Data, split, duplicates, preprocessing, augmentation, training, selection and evaluation: what the paper actually says

### Takeaway
For APTOS the paper gives **no split, no split ratio, no test-set size, no stratification, no validation or checkpoint-selection rule, no batch size, no epoch count, no threshold step and no duplicate handling**. The only three-way split it describes (train/validation/test of 24,590/5,268/5,268) is for EyePACS. The 3,662 APTOS images are the labelled Kaggle *training* set, so the Kaggle private test set could not have been used. The 0.954 result therefore cannot be checked from the text.

### Cited Findings
**Data and split**
- [Stated, "Dataset description"] EyePACS: "a resized version of the Eyepacs dataset for diabetic retinopathy [30] comprises 35,126 images … We divide the data into three datasets for training, validation, and testing. The training set contains 24,590 retinal images, while the validation and test data sets contain 5,268 retinal images each." 24,590 + 2 × 5,268 = 35,126, which is 70/15/15 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Dataset description"] APTOS is described only as: "the Asia Pacific Tele-Ophthalmology Society 2019 blindness detection data set (APTOS 2019 BD), which is a collection of 3,662 fundus photographs of patients from rural India … labeled by a group of trained physicians using the International Clinical Diabetic Retinopathy Disease Severity Scale". There is nothing on how it was split — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated] Messidor-2: "comprises 874 DR examinations, totaling 1,748 fundus images … acquired from 762 patients". No split is described — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, by absence] The full text never uses the words "split", "stratif*", "batch", "test set", "oversampl*", "duplicate", "threshold" or "seed". "Balanc*" appears only in the background text on EfficientNet and QWK — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) (searched the full text)
- [Stated] The APTOS reference [19]/[37] is the Kaggle competition page "kaggle.com/competitions/aptos2019-blindness-detection", "Accessed Dec 2023". The EyePACS reference [30] was "Accessed May 2022" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated] Data availability: "The datasets used in this project are publicly available and can be found at [30, 36, 37]." There is no code availability statement, and PMC marks the article "pmc-prop-has-supplement no" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

**Architecture, loss and optimisation**
- [Stated, "Final classifier modeling"] EfficientNet-B3 gives 7×7×1,536 features, then adaptive average pooling and flatten, then FC 1536→512, "dropout layer, which has a drop rate of 0.5, and the ReLU", then FC 512→512, dropout 0.25 and ReLU, then FC to "five units", then "Softmax activation layer to generate the probabilities for each category". Training is classification, not regression. The loss is never named (cross-entropy is implied) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "The proposed method"] "We trained both the feature extractor and the classifier during the training phase" (full fine-tuning) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Hyper-parameter tuning"] "we used Adam optimizer … gradient clipping with clipping equal to 0.1 and a weight decay of 10^-4. The learning rate for all models used in this article is 0.001." No schedule, batch size, epoch count for APTOS or early stopping is given — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Results and discussion"] They used PyTorch on "Google Colab Pro … RAM of 25.46 GB" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

**Ensembles, preprocessing and "augmentation"**
- [Stated, "First ensemble strategy"] "The EfficientNet-B3 model is trained for 60 epochs. Weights are saved at 30 and 60 epochs … we aggregated the two sets of predictions … only normalization and resizing are implemented. All images were resized to 150 × 150." This strategy applies only to the EyePACS Table 3 results — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Second ensemble strategy"] "we tested different data augmentations and found that CLAHE … in combination with Gaussian Blur performs well. We do normalization and the images are resized to 300 × 300 … the model weights are saved at 10 intervals during training. At the time of inference, the ten predictions are obtained using the saved model weights at ten intervals. The final prediction is estimated to be the most common prediction among all the predictions." So the ensemble is a majority vote of hard labels from 10 snapshots — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Results of model trained on three datasets"] "All of these models were pre-processed using CLAHE, Gaussian Blur, and normalization." The Table 6 results (including APTOS 0.954) therefore use the 300×300 CLAHE + blur pipeline — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, Abstract] "data augmentation techniques are used to address data limitations, improve features, and improve generalization"; [Conclusion] "data augmentation techniques such as CLAHE and Gaussian Blur are employed". No geometric augmentation (flips or rotations) is mentioned anywhere — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Impact of transfer learning…"] "we used the model trained on the Eyepacs data set to fine-tune on the Aptos and Messidor databases" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

**Metric computation**
- [Stated, "Quadratic Weighted Kappa (QWK)"] "Step 1: Calculate the expected matrix (E) taking the outer product between the actual (A) and predicted (P) label vectors … Step 2: Construct the output matrix by building a confusion matrix of actual and predicted labels … W_ij = (i−j)²/(k−1)² … QWK = 1 − num/den". This is standard Cohen's QWK on **hard predicted labels**. No rounding of a continuous output and no optimised cut-offs are mentioned. The cited source [17] is a Kaggle notebook, "aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated] The averaging used for precision, recall and F1 (macro, weighted or micro) is not stated — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

**Headline numbers, Table 6 ("Results on three datasets using proposed strategies")** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

| Dataset | Pretrained | Ensemble | Acc | Prec | Recall | F1 | QWK |
|---|---|---|---|---|---|---|---|
| Eyepacs | None | No | 0.921 | 0.923 | 0.923 | 0.924 | 0.878 |
| Eyepacs | None | Yes | 0.944 | 0.943 | 0.942 | 0.932 | 0.901 |
| **Aptos** | **None** | **No** | **0.942** | 0.943 | 0.943 | 0.944 | **0.954** |
| Aptos | Eyepacs | No | 0.954 | 0.952 | 0.942 | 0.954 | 0.956 |
| Aptos | Eyepacs | Yes | 0.954 | 0.953 | 0.95 | 0.956 | 0.967 |
| Messidor2 | None | No | 0.854 | 0.856 | 0.853 | 0.844 | 0.854 |
| Messidor2 | Eyepacs | No | 0.923 | 0.922 | 0.923 | 0.912 | 0.933 |
| Messidor2 | Eyepacs | Yes | 0.924 | 0.922 | 0.923 | 0.921 | 0.944 |

No confusion matrix, per-class metric, test size or confidence interval is given for any APTOS or Messidor row.

### Inferences
- The 0.954 row is a **single model** ("Ensemble = No") at 300 px with CLAHE + Gaussian blur and a softmax argmax. Nothing in the method explains a better result than ours: B3 at 300 px, cross-entropy, no TTA, no cut-off tuning, LR 1e-3 with Adam on full fine-tuning, and no geometric augmentation stated. If anything, our pipeline is stronger.
- Because the APTOS images number 3,662, the paper used only the public labelled training set. The Kaggle test set (whose labels were never released) was not used, so the "test" set was carved out of the 3,662 images by an unstated procedure.
- Which split (validation or test), which checkpoint (last epoch, best validation, or best on the reported split) and how many images were used for APTOS are all unknown. The project's 70/15/15 assumption for APTOS is borrowed from the EyePACS paragraph and is not stated for APTOS.
- The APTOS reference was "Accessed Dec 2023", during the 13-month review (Dec 2022 to Jan 2024), while EyePACS was accessed in May 2022. One reading is that the APTOS and Messidor-2 experiments were added in revision. This is speculative, but it fits the APTOS rows getting far less methodological detail than EyePACS.
- **References show a find-and-replace error.** Refs [17] and [30] read "https://www.Eyepacs.com/aroraaman/…" and "https://www.Eyepacs.com/tanlikesmath/…". These are Kaggle URLs with "Kaggle" replaced by "Eyepacs". This points to a global text substitution in the manuscript, which lowers confidence in its editorial care. The same substitution may also affect sentences that say "Eyepacs" where "Kaggle" was meant.

### Gaps
- There is no statement anywhere of the APTOS split, test size, stratification, seed, de-duplication, epochs, batch size or checkpoint rule. Only the authors could supply these.
- The loss function is not named (CE is implied by the softmax), and the precision/recall/F1 averaging mode is not stated.
- The Springer landing page redirected to a login wall, so I could not confirm whether open peer-review reports are posted. BMC MIDM often publishes them, and they could hold reviewer questions about the split.

## Q2. Is there public code, a notebook or a supplement, and what does it do?

### Takeaway
I found no public code, notebook or supplement. The first author's GitHub (Saivenkatesh951) has 16 public repos, and none is for DR. His portfolio lists the DR work with its headline QWKs but no code link. The only GitHub repos that implement this paper are third-party reproductions (apparently this project's own), and they report reaching about 0.82 with the paper's recipe.

### Cited Findings
- The paper has no code availability statement, and its data statement points only to the public datasets. PMC records no supplement ("pmc-prop-has-supplement no") — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- The GitHub user "Saivenkatesh951" has name "Sai venkatesh Chilukoti" and location "St. Joseph, Minnesota". St. Joseph matches co-author Tida's affiliation, College of Saint Benedict and Saint John's. Its 16 public repos are AGRIBOT, Breast_cancer, Cervical_cancer, ideal-bassoon, MultiModal-transformer, nutrition-rag, Portfolio.github.io, prostat-cancer, pytorch-MobileNet, ray (a fork), reasearch_articles, the profile repo, Saivenkatesh951.github.io, stylegan2-ada-pytorch, test (one file, text.txt) and TrainableQuantumConvolution. None contains DR code — [GitHub API: users/Saivenkatesh951/repos](https://github.com/Saivenkatesh951?tab=repositories)
- The author's portfolio says: "Aug 2023 – Aug 2024 University of Louisiana at Lafayette Research Assistant … Developed a computationally efficient transfer-learning and ensemble framework for 5-class diabetic retinopathy grading, achieving QWK scores of 0.901, 0.967, and 0.944 on EyePACS, APTOS, and Messidor-2." It gives no code link — [Saivenkatesh951/Portfolio.github.io index.html](https://github.com/Saivenkatesh951/Portfolio.github.io)
- A GitHub repository search for "chilukoti" returns only two DR repos, both reproductions: [sshivanshg/dr-grading-reproduction](https://github.com/sshivanshg/dr-grading-reproduction) and [devxsubh/aptos-dr-grading-reproduction](https://github.com/devxsubh/aptos-dr-grading-reproduction). The latter's description says: "The paper's recipe reached QWK 0.82 vs. the reported 0.954; diagnosis and fixes raise it to 0.908."
- The paper appears on PMC, NSF PAR and the CSBSJU institutional repository, but not on arXiv or Research Square — [NSF PAR](https://par.nsf.gov/biblio/10489604); [CSBSJU Digital Commons](https://digitalcommons.csbsju.edu/csci_pubs/37)

### Inferences
- With no code, none of the code-level leaks can be checked: splitting after offline augmentation, balancing by copying images before the split, picking the checkpoint on the test set, or fitting normalisation on all data. The text never says the test split was kept untouched until the end.
- The two reproduction repos appear to be this project's own (the git user is Subham-u, and the repo description matches this project's history), so they are not independent replications.

### Gaps
- PubPeer's API returned HTTP 403, so I could not confirm whether any PubPeer thread exists. Google Scholar was not queried directly.
- Kaggle notebooks by the author were not searched (Kaggle search is not reachable through these tools).

## Q3. Are the reported numbers internally consistent? (confusion matrices against tables)

### Takeaway
**No.** The tables cannot all be true at once. (a) The QWK and accuracy reported in Table 3 do not follow from the confusion matrices in Tables 4 and 5. The ensemble's published matrix gives QWK 0.919, not 0.867. EfficientNet-B5's matrix gives 0.488, not 0. ResNet-101/152 predict only class 0, so their QWK must be exactly 0, yet the table reports 0.009 and 0.067. (b) Each model's confusion matrix has a different number of images (5,008 to 5,339) and a different class mix, although the paper says the test set is 5,268 images. (c) In Table 6, including the APTOS rows, the reported F1 is higher than both precision and recall, and recall differs from accuracy. Neither is possible under any standard averaging. The APTOS 0.954 comes with no confusion matrix and cannot be checked directly. But the reporting pipeline that produced it is demonstrably inconsistent elsewhere in the same paper.

### Cited Findings
- [Stated, Table 4, "Confusion Matrix for ensembled EfficientNet-B3" (EyePACS), rows = real, columns = predicted] Rows are [3886 1 26 0 0], [169 80 91 1 0], [68 6 710 1 7], [2 0 31 93 5] and [3 0 7 3 98]. The text uses it for the worked example "FP … 169 + 68 + 2 + 3 = 242" and "TN … = 2784" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, Table 3] EffNet-B3_En: Acc 0.924, F1 0.867, Prec 0.867, QWK 0.867, Recall 0.878. EffNet-B5: Acc 0.721, QWK 0. ResNet-101: QWK 0.009. ResNet-152: QWK 0.067. Text: "The QWK score of 0.867 is obtained using the predictions of the EfficientNet model based on the first ensemble strategy" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, Table 5] All seven VGG and ResNet matrices put every image in column 0. The text says "the models only identify class 0" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, Table 2] EyePACS label counts are 25,180 / 2,443 / 5,292 / 873 / 708. These sum to 34,496, not the stated 35,126 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated, "Metric scores…"] "The AUC-ROC score for all VGG and ResNet models is 0.2 because these models can only detect one DR stage". In Table 3 it is 0.196. A constant predictor has AUC 0.5, not 0.2 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- [Stated] Section heading: "Fig. 4 Illustration of second ensemble strategy". The text, however, says "Fig. 4, which shows the confusion matrices" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

**[Computed] Recomputed from the published matrices (Tables 4 and 5)** — standard Cohen QWK, W = (i−j)²/16, E = outer(row sums, column sums)/n; weighted P/R/F1 by true-class support. Source tables: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

| Model | n in matrix | True-class counts | Acc (computed) | QWK (computed) | Table 3 Acc | Table 3 QWK |
|---|---|---|---|---|---|---|
| VGG-16 | 5,008 | 3638/363/763/131/113 | 0.726 | 0.000 | 0.732 | 0 |
| VGG-19 | 5,300 | 3887/376/793/131/113 | 0.733 | 0.000 | 0.734 | 0 |
| ResNet-18 | 5,268 | 3825/371/806/152/114 | 0.726 | 0.000 | 0.723 | 0 |
| ResNet-34 | 5,268 | 3830/391/804/138/105 | 0.727 | 0.000 | 0.724 | 0 |
| ResNet-50 | 5,268 | 3864/365/806/116/117 | 0.733 | 0.000 | 0.735 | 0 |
| ResNet-101 | 5,257 | 3848/383/783/140/103 | 0.732 | **0.000** | **0.766** | **0.009** |
| ResNet-152 | 5,167 | 3775/360/814/128/90 | 0.731 | **0.000** | 0.721 | **0.067** |
| EffNet-B0 | 5,268 | 3899/341/786/135/107 | 0.772 | 0.506 | 0.767 | 0.501 |
| EffNet-B1 | 5,266 | 3901/374/776/122/93 | 0.783 | 0.490 | 0.778 | 0.478 |
| EffNet-B2 | 5,268 | 3881/375/794/110/108 | 0.772 | 0.501 | 0.767 | 0.502 |
| EffNet-B3_30 | 5,270 | 3891/383/782/117/97 | 0.848 | **0.753** | 0.852 | **0.824** |
| EffNet-B3_60 | 5,266 | 3845/403/752/139/127 | 0.868 | 0.857 | 0.867 | 0.854 |
| EffNet-B4 | 5,339 | 3891/351/812/161/124 | 0.769 | **0.474** | 0.767 | **0.401** |
| EffNet-B5 | 5,270 | 3810/374/842/139/105 | **0.758** | **0.488** | **0.721** | **0** |
| EffNet-B6 | 5,270 | 3870/348/795/136/121 | 0.779 | 0.533 | 0.767 | 0.532 |
| EffNet-B3_En (Table 4) | 5,288 | 3913/341/792/131/111 | 0.920 | **0.919** | 0.924 | **0.867** |

- [Computed] For B3_En, computed weighted F1/P/R are 0.906/0.921/0.920, against Table 3's 0.867/0.867/0.878. Under support-weighted averaging recall must equal accuracy, yet Table 3 shows 0.924 against 0.878 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) Table 3/4
- [Computed] **Table 6 violates mathematical bounds.** For each class, F1_c ≤ (P_c + R_c)/2, so under both macro and weighted averaging F1 ≤ (P + R)/2. Under micro averaging F1 = P = R = accuracy. Table 6 breaks this in the rows APTOS/None/No (F1 0.944 > (0.943 + 0.943)/2), APTOS/EyePACS/No (0.954 > 0.947), APTOS/EyePACS/Yes (0.956 > 0.9515) and EyePACS/None/No (0.924 > 0.923). Under weighted averaging, recall must also equal accuracy, but APTOS/None/No shows 0.942 against 0.943 and APTOS/EyePACS/No shows 0.954 against 0.942. Under macro averaging on APTOS (5.3% severe, 10.1% mild), macro recall would be well below accuracy, not equal to it — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) Table 6
- [Computed] EffNet-B3_En in Table 3 (first strategy, 150 px) has QWK 0.867, while Table 6 EyePACS/None/Yes (second strategy, 300 px) has 0.901. The Table 4 matrix gives 0.919, which matches neither. Its accuracy of 0.920 is closest to Table 6 EyePACS/None/No (0.921), but that row's QWK is 0.878 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

### Inferences
- **Varying test sets per model.** The class composition of the "test" matrix changes from model to model. Class-0 counts run from 3,638 to 3,913, and totals run from 5,008 to 5,339. Several totals exceed the stated test size of 5,268. This is consistent with (i) a fresh random split per training run (no fixed seed or saved split), (ii) evaluation on different subsets (for example validation in some runs), or (iii) matrices assembled by hand from different runs. In every case "the test set" was not a single fixed hold-out across experiments. If the APTOS experiments were run the same way, the reported APTOS number may come from one favourable random split. With a test set of about 550 images, a random split alone moves QWK by about ±0.02, the same size as our 95% CI half-width.
- **Reported metrics do not follow from the reported predictions.** Values were mis-transcribed, came from different runs than the matrices, or were computed with a buggy metric routine. The repeated values within a row (B3_En: F1 = Prec = QWK = 0.867; B3_60: Prec = QWK = 0.854) and the impossible AUC of about 0.196 for constant predictors point to transcription or metric-code errors. Since the APTOS 0.954 has no matrix, nothing guarantees it is computed correctly either.
- **Accuracy plausibility.** APTOS papers with QWK around 0.93 report 5-class accuracy of about 85–89% (see project note `research_notes/Closing the APTOS 2019 QWK gap/evaluation_protocols.md`). The paper's 94.2% is about 5 points higher, and our 0.81 is in line with the field. A 94% accuracy on APTOS would need near-perfect separation of mild, moderate and severe, against known inter-grader noise. That makes a protocol artefact or a reporting error more likely than a better model.
- The EyePACS part also leaks at patient level. The Kaggle 2015 data (and its "resized" copy) contains left and right eyes of the same patient with highly correlated grades, and the paper splits per image with no patient grouping. This inflates the EyePACS numbers and the EyePACS-pretrained APTOS rows (0.956/0.967), but not the ImageNet APTOS row (0.954).

### Gaps
- There is no APTOS confusion matrix, so the 0.954 itself cannot be recomputed. Whether 0.942 accuracy and 0.954 QWK are jointly plausible cannot be tested without the matrix.
- Whether the averaging mode is "weighted" (as Table 3's recall ≈ accuracy pattern suggests for most rows) cannot be confirmed.

## Q4. Have later papers, comments or PubPeer questioned, replicated or failed to replicate these numbers?

### Takeaway
The paper has about 43 citations (Semantic Scholar). They nearly all cite it for QWK as a metric or list its 0.954/0.967 in comparison tables uncritically. I found **no independent replication** and **no published critique**, apart from one 2026 conference paper's general remark that high-QWK APTOS methods "do not verify whether their threshold selection was performed on held-out data". No erratum or correction was found.

### Cited Findings
- Semantic Scholar lists citationCount 43 (PMID 38321416, DBLP journals/midm/ChilukotiSTMH24) — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1186/s12911-024-02446-x)
- An ICATI 2026 paper (IEEE Nexo-Tech), "Explainable Diabetic Retinopathy Severity Grading Using CLAHE-Enhanced EfficientNet-B3 with Cross-Dataset Evaluation", writes: "Chilukoti et al. [15] showed ensembled EfficientNet-B3 with CLAHE achieves QWK 0.901 on EyePACS and 0.967 on AP-TOS, representing the upper bound for this architecture class". It also writes: "Methods that achieve high QWK on APTOS 2019 typically benefit from the relatively clean nature of that dataset and do not verify whether their threshold selection was performed on held-out data [15], [16]." — [DOI 10.1109/ICATI69399.2026.11604250](https://doi.org/10.1109/ICATI69399.2026.11604250) (citation context via [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1186/s12911-024-02446-x/citations?fields=title,contexts))
- A Discover Computing 2026 paper lists "Chilukoti et al. [27] 95.4" in an accuracy comparison table, apparently taking the 0.954 accuracy of the EyePACS-pretrained row — [DOI 10.1007/s10791-026-10284-w](https://doi.org/10.1007/s10791-026-10284-w) (context via Semantic Scholar)
- A 2026 ICIMTech paper on Ben Graham preprocessing cites it as the source of practice for "split … in a stratified manner, with 80% of the images for training and 20% for validation, as is the practice in dR classification studies [17]". It attributes a stratified 80/20 practice to a paper that never describes one — [DOI 10.1109/ICIMTech202671078.2026.11699053](https://doi.org/10.1109/ICIMTech202671078.2026.11699053) (context via Semantic Scholar)
- Other 2025–2026 citers (ConvNeXt ordinal, IJDDT 2026; SwinV2 "Leakage-Safe" ICCSC 2026; MethodsX 2026; Sci Rep 2026) cite it for ensembles or QWK. In the retrieved contexts none reports re-running it — [Semantic Scholar citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1186/s12911-024-02446-x/citations?fields=title,contexts)
- Public reproductions found: only the two GitHub repos named in Q2, which report about 0.82 with the paper's recipe — [devxsubh/aptos-dr-grading-reproduction](https://github.com/devxsubh/aptos-dr-grading-reproduction)

### Inferences
- The 0.954/0.967 is spreading through the literature as a "state of the art" or "upper bound" reference, while its protocol is unverified. Treat it as an unvalidated claim, not a benchmark.

### Gaps
- PubPeer could not be queried (HTTP 403). Google Scholar's citing list was not checked directly. Semantic Scholar returned contexts for only some citers.
- I could not read the full texts of citers that might re-run the method. The ICATI 2026 "CLAHE-Enhanced EfficientNet-B3" paper is the most likely candidate to report its own B3 + CLAHE numbers on APTOS, and it is worth reading.

## Q5. Do the authors' related papers (same group) use the same pipeline and report similar numbers?

### Takeaway
I found no other DR paper by this group (Chilukoti / Shan / Tida / Maida / Hei, UL Lafayette). The first author's other listed work is on differential privacy (DP-SGD), code-model poisoning and quantum convolution, so there is no sister paper to compare pipelines against.

### Cited Findings
- The author's portfolio lists the DR framework as one item, with other work on "DP-SGD-Global-Adapt-V2-S" (privacy) and attacks on code LLMs (CodeLlama, DeepSeek-Coder, StarCoder2 "75–86% ASR@1") — [Portfolio.github.io](https://github.com/Saivenkatesh951/Portfolio.github.io)
- His GitHub repos include TrainableQuantumConvolution and stylegan2-ada-pytorch (skin-lesion synthesis), with no DR repo — [GitHub](https://github.com/Saivenkatesh951?tab=repositories)
- A web search for the author names with diabetic retinopathy returned no other DR paper by this group — (search returned only unrelated epidemiology papers)
- The paper's "Directions for future work" mentions federated learning and CoAtNet as next steps. I found no follow-up publication — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

### Inferences
- With no sister paper, the pipeline cannot be cross-checked through the group's other work.

### Gaps
- A DBLP or Google Scholar author-profile check was not done, so a lesser-known follow-up (for example a thesis or workshop paper) cannot be ruled out.

## Q6. Protocol features that could inflate the APTOS result (ranked, with evidence status)

### Takeaway
None of the paper's *stated* method choices can explain 0.954. The candidates lie in what is unstated (split and selection) and in reporting reliability. Best-supported first: (1) metric or reporting errors, since the paper's tables provably don't match its own confusion matrices; (2) no fixed hold-out, with test sets that differ between runs and possibly evaluation or checkpoint choice on the reported split; (3) an unknown, possibly small or validation-based APTOS split, plus split-to-split variance; (4) possible leakage from offline "augmentation" (CLAHE or blur copies) or duplicates crossing the split. Items (3) and (4) are not evidenced, only allowed by the silence.

### Cited Findings
- The paper reports results only from one evaluation, with no seeds, repeats or CIs, and describes no checkpoint-selection rule. Snapshot weights are "saved at 10 intervals during training" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- The confusion-matrix and table mismatches and the varying test-set sizes are in Q3 (computed from [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/) Tables 3–5)
- The project's own measurement: 29 leaked duplicate test images changed QWK by under 0.001 in the Stage 2 run (local file `/Users/subhammahapatra/Downloads/dr-grading-reproduction/dr-repro/results/leakage_s2_faithful.json`, summarised in `research_notes/Closing the APTOS 2019 QWK gap/evaluation_protocols.md`)

### Inferences
1. **Reporting or metric-computation error (strongest evidence).** The paper's QWK, accuracy, P, R and F1 values provably do not follow from its own predictions in at least six of 16 EyePACS models, and Table 6 contains values that cannot hold together. An APTOS 0.954 produced by the same reporting process carries no guarantee.
2. **No fixed held-out test set.** The varying class counts per model show that the evaluation set changed from run to run. Choosing the best run or epoch by its score on that set (the paper gives no other selection rule) is the classic "selection on test" inflation. Our test-tuned upper bound of 0.928 covers cut-offs and member choice, but not picking the luckiest of several random *splits* or *training runs*.
3. **Split unknown, possibly the validation set or a small test set.** With n ≈ 366–732, the SE of QWK is about 0.01. The best of several splits or runs could plausibly add 0.01–0.02. Even so, 0.918 to 0.954 is still a stretch through variance alone, which points back to item 1.
4. **Offline "data augmentation" before the split (possible, unverified).** The paper repeatedly calls CLAHE and Gaussian blur "data augmentation … to address data limitations". If augmented copies were added to the image pool and the pool was then split at random, near-identical copies of test images would sit in training. That would raise both QWK and especially accuracy (94.2%). The text elsewhere calls them preprocessing applied to all images, so this stays a hypothesis. It is testable: split after making CLAHE and blur copies and see whether ~0.95 / 94% appears.
5. **Duplicates.** Not mentioned in the paper. The project's own measurement suggests they explain under 0.001 of QWK, so they are not a major contributor unless combined with item 4.
6. **Not an explanation:** model capacity, resolution, TTA or ordinal tricks. The paper uses a single B3 at 300 px with softmax argmax, which is weaker than our pipeline on every stated axis.
- **Practical recommendation for the reproduction:** report 0.954 as "not reproducible and internally inconsistent", citing the Q3 recomputation table. If a sensitivity analysis is wanted, run (a) evaluation on the selection split, (b) the best of N random splits or seeds, and (c) splitting after offline augmentation, and see which, if any, reaches 0.95 QWK with 94% accuracy.

### Gaps
- None of items 2–4 can be confirmed without the authors' code or answers. Contacting the corresponding author (Xiali Hei, xiali.hei@louisiana.edu, as listed in the paper) for the APTOS split, test size, checkpoint rule and confusion matrix is the only direct route.
