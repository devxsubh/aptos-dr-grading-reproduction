# EyePACS 2015 to APTOS 2019 transfer: recipes, hyperparameters, expected gains

Source note: most primary material below is the text of the Kaggle APTOS 2019 discussion write-ups and their comment threads. Kaggle renders these pages with JavaScript, so they were read through Kaggle's own discussion API (`GetForumTopicById`), which returns the author's raw markdown. Every quote is from the author's own post or reply unless marked otherwise. Context: in APTOS 2019 the public LB was about 0.80-0.85 and the private LB about 0.92-0.94, because the public test split had a different distribution. So public-LB deltas are noisy and not on the same scale as your OOF QWK.

Note on the leaderboard: the original 2nd and 3rd place teams, and one of the 5th/6th, were removed after the private LB was released ([Kaggle thread 108689](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108689)). So the "2nd place after cleaning" is Mykhailo Matviiv's team, whose write-up is titled "4th place solution (2nd after LB cleaning)". No public write-ups exist for the removed 2nd/3rd teams.

## Q1. How top APTOS 2019 solutions used the 2015 data

### Takeaway
All top solutions used the 2015 data, and most used both 2015 train (35,126) and 2015 test (53,576 labelled) images. They split about evenly between (a) pre-train on 2015, then fine-tune on 2019 and (b) training on combined 2015+2019 data. Those who pre-trained did far more 2015 epochs than 5 (15-80). The one solution that gives both learning rates fine-tuned at **1e-5**, 50x below its 5e-4 pre-training lr and 30x below your 3e-4.

### Cited Findings
**1st, Guanshuo Xu (private 0.936): combined data, no pre-train/fine-tune split**
- First tried "2015 data (both train and test) as train set, and 2019 train data as validation set", but "the validation results and public LB were very different". Final choice: "combine the whole 2015 and 2019 data as train set, and solely relied on public LB for validation." — [1st place write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- No special preprocessing ("just plain resizing"). Input 512 for inception_resnet_v2, inception_v4 and seresnext50; 384 for seresnext101. Loss `nn.SmoothL1Loss()`. GeM pooling. Heavy augmentation (rotate 180, scale/shear/shift 0.2, colour jitter, blur/sharpen). — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Optimizer and schedule (author's reply in comments): "I used Adem [Adam] with lr=0.0002 and trained 30-50 epochs, depending on the LB performance, then lowered to 0.00002 and continued for 5-10 more epochs." Epoch count chosen by public LB with a step size of 5. — [same, comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Stage-1 results (combined 2015+2019 training): single models private 0.923-0.931, 8-model ensemble private 0.934. Stage 2 added soft pseudo-labelled public test, IDRiD (labels smoothed with stage-1 predictions) and Messidor (labels bounded by group means of stage-1 predictions), trained "10 more epoch" → private 0.935, then 0.936 with threshold [0.7,1.5,2.5,3.5]. — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)

**"4th place (2nd after LB cleaning)", Mykhailo Matviiv + aispiriants: pre-train then fine-tune**
- "We used full-size images from 2015 train dataset to pretrain our models. Models trained on old data only gave us ~0.73-0.75 on public LB." Data listed: current train, old competition test. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- EfficientNet B3 @300, B4 @460, B5 @456. "Pretrain on old data (~80 epochs on b3, ~15 epochs on b5)", then "Train on current competition data (~50 epochs on b3, ~15 epochs on b5)". Ben's colour preprocessing "didn't boost score". Pseudo-labelled 2019 test ("huge boost") twice, plus pseudo-labelled 2015 test. — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- "For pretrain, we validated on all current train. For current train, we validated on 5% of the current train and public LB." Learning rates were not stated. — [same, author reply](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)

**4th, Qishen Ha (private 0.932-0.933; local 5-fold CV about 0.933): pre-train then fine-tune, the most complete hyperparameters**
- "Pretrain on 2015 full dataset (both train and test) for 25 epochs without validation. Then do 5-fold CV on 2019 dataset." — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- Author reply: "Simple cosine LR schedule, for pre-train I set `init_lr = 0.0005`, for finetune I set `init_lr = 0.00001`. I've try something like warmup but it seem to be not helpful on my local CV." Also: "it's not so different between using final epoch / best checkpoint" for pre-training. — [same, comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- Models were EfficientNet B2-B7, small models at high resolution and large models at low resolution: B7@224, B6@240, B5@256, B4@320, B3@352, B2@376. "all of these model have a similar local 5-fold CV around 0.933". With Ben's preprocessing, B7/B5 CV was "around 0.927", i.e. 0.006 lower. Crop-from-gray, plus normalising all images to one "image type" (border brightness) for 2015 and 2019 alike. — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)

**5th, Gary: combined data**
- "according to the better public LB, I did not pretrained on 2015-data and then finetune on 2019-data, I just combined 2015-data and 2019-data together", validating on 2015 private test plus 2019 valid. Adam, step lrs [5e-4, 1e-4, 1e-5, 1e-6] with decays at epochs [10, 16, 22], 25 epochs, 384x384, EfficientNet B0-B3, joint classification and regression loss. Confident pseudo labels on 2019 public test. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)

**7th, Eugene Khvedchenya (BloodAxe): pre-train, train, pseudo-label, then fine-tune the head**
- "Pre-train models on data from 2015, validate on this competition's data and Idrid (test)", then train on 2019 + IDRiD + Messidor, pseudo-label test / Messidor-2 / 2015 data, then fine-tune. Under "What did work": "Pre-training on 2015 data", "Pseudo-labeling AND fine-tuning of head ONLY", "RAdam". 512x512, no Ben preprocessing ("All of them failed to outperform using images as-is"). — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058)

**8th, DrHB: switched from two-step to combined**
- With "STEP 1: pretrain on 2015 data, STEP 2: finetune on 2019 data", "our local CV was around 0.931 but LB was 0.811"; "just pertaining on old data using 2019 as Validation results in LB 0.75". The team then combined 2015+2019, after removing duplicates and confusing labels, plus random-zoom augmentation. fastai one-cycle with progressive resizing. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- Combined-data results. B5: 5-fold CV 0.823 / LB 0.821 / private 0.926. B4: CV 0.831 / LB 0.816 / private 0.927. After 3 epochs on pseudo labels: private 0.930 and 0.929, ensemble 0.931. (CV here is on combined data, so it is not comparable to 2019-only CV.) — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)

**9th, Psi + dott: pre-train then a short fine-tune**
- "We did pretraining on full 2015 data which was very important to get better results on test data… We fitted for around 15-30 or so epochs depending on the model… either did some slight LR scheduling, or just kept it constant and either used Radam or some other Adam variation." They picked prefit epochs by best 2019 validation score: "Sometimes you can get 'lucky' prefits". — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- Fine-tune: "4-fold CV on 2019 data… Adam optimizer and did a once cycle policy for overall 4 fixed epochs with learning rate ranging from 0.00001 to 0.0005", no epoch selection. EffNet-B7 and SE-ResNeXt50 at 320. Private 0.931. — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)

**11th, 4uiiurz1: combined data**
- 1st level: SE-ResNeXt50/101 and SENet154, MSE, SGD momentum 0.9, CosineAnnealingLR 1e-3→1e-5, 30 epochs, 256 px, on "2019 train dataset (5-folds cv) + 2015 dataset". 2nd level: RAdam 1e-3→1e-5 cosine, 10 epochs, plus 2019 test pseudo labels. Public 0.826 / private 0.930. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107958); [repo](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)

**12th, JIANJIAN: pre-train then fine-tune, keeping the optimizer state**
- "pretrain on 2015 data and finetune on 2019 data… Finetune should preserve the optimizer's parameters from pretrain, it made a big difference to me." EffNet B5/B6, 320/288. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)

**13th, Camaro team: combined beat two-step**
- "When I pre-trained on 2015 dataset and fine-tuned on 2019 dataset, my single model(SeResNext)'s score was LB0.811, but after I changed to above method [train on 4/5 of 2019 + 2015 train, validate on 1/5 of 2019], my score was boost to LB0.827." The EfficientNet-B5 variant pre-trained on 4/5 of 2015 at 256, then fine-tuned at 320 on **1/5 of 2015 + 4/5 of 2019**, i.e. it kept 2015 data in the fine-tune. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- Pseudo-labelling on a single SeResNeXt101: private 0.922 → 0.925 (own labels) → 0.931 (labels from SeResNeXt + EffNet). — [same](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)

**17th, Xuan Cao: pre-train then fine-tune at your resolution**
- EfficientNet b0/b3/b4/b5 at 380 px. "Pretrain Data: 2015 train data, valid on 2015 public test and test on 2015 private test. Finetune Data: 2019 train data (5-fold CV)." Local CV 0.93456 against private 0.930. Epochs and lr not stated. — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)

**Lower-ranked but informative**
- 76th (MamatShamshiev): "2015 competition data was used for pretraining all our models. Without it out models performed much worse." "starting finetuning with freezing all layers and training only last FC layer gave us more stable results." Adam + OneCycle; "WarmUp helped to get more stable results". — [repo](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- 83rd (khornlund): "I pretrained on the train + test data from 2015, before fine-tuning on the 2019 training set". Ranger, flat-cosine annealing, 256 px, label noise N(0, 0.1) on regression targets. — [repo](https://github.com/khornlund/aptos2019-blindness-detection)
- 37th (tahsin314): about 5+10 epochs on 2015 with 2019 as validation, then 15 epochs of fine-tuning on an 80:20 2019 split. — [repo](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection) (via search snippet; not read in full)

### Inferences
- Use of the 2015 test labels: confirmed for 1st, Qishen Ha 4th, Mykhailo 2nd-after-cleaning (as a labelled set), 9th ("full 2015 data"), khornlund, and 1st's early experiments. **So the standard was about 88k labelled 2015 images, not 35k.** Your 35,108-image pre-train uses about 40% of the available labelled EyePACS data.
- Resolution: the 2015 data was used at the same resolution as the final model (300-512). Your 380 px is in range.
- Ben preprocessing is not part of any top-5 recipe. Three top teams tested it explicitly and found no gain or a small loss (Qishen Ha CV 0.933 vs 0.927; Mykhailo "didn't boost"; BloodAxe "failed to outperform"). Inferred: this is not the cause of your negative transfer, but it is not helping.

### Gaps
- No write-ups exist for the original (removed) 2nd/3rd place teams.
- Learning rates for Mykhailo (2nd after cleaning), Xuan Cao (17th) and DrHB were not stated in the posts or replies.

## Q2. Fine-tuning learning rate after in-domain pre-training, and evidence of forgetting

### Takeaway
The best-documented APTOS recipe pre-trains at 5e-4 and fine-tunes at **1e-5 with cosine decay and no warm-up**. Guanshuo Xu's second phase drops 10x from 2e-4 to 2e-5. Your 3e-4 peak with a fresh warm-up is 30x above Qishen Ha's fine-tune lr and above what most top teams used even for pre-training. The epoch 5-7 dip you saw falls just after warm-up, at peak lr. That pattern fits the "fine-tuning distorts pretrained features" mechanism, but you have not tested it yet.

### Cited Findings
- Qishen Ha (4th): pre-train `init_lr = 0.0005`, fine-tune `init_lr = 0.00001`, both "Simple cosine", warm-up "not helpful on my local CV". — [Kaggle 107987 comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- Guanshuo Xu (1st): Adam 2e-4, then 2e-5 for 5-10 epochs (a 10x drop) when LB stopped improving. — [Kaggle 108065 comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Psi/dott (9th): fine-tune one-cycle 1e-5 to 5e-4, only **4 epochs**, fixed. — [Kaggle 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- Gary (5th, combined data, not a fine-tune): Adam 5e-4 → 1e-4 → 1e-5 → 1e-6, steps at epochs 10/16/22. — [Kaggle 107960](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
- JIANJIAN (12th): keeping the pre-train optimizer state (Adam moments) for the fine-tune "made a big difference". — [Kaggle 107995](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)
- BloodAxe (7th): after pseudo-labelling, fine-tuning the "head ONLY" worked. MamatShamshiev (76th): starting the fine-tune with all layers frozen except the last FC gave "more stable results". — [Kaggle 108058](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058); [MamatShamshiev repo](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- A visitor's comment on the 4th-place thread (Firas Baba, not a top finisher) reports that Adam lr=1e-4 with ReduceLROnPlateau converged in 15-20 epochs. — [Kaggle 107926 comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- Li et al. (ICLR 2020): optimal fine-tuning hyperparameters, "in particular, the effective learning rate, are not only dataset dependent but also sensitive to the similarity between the source domain and target domain". Effective lr there is lr/(1-momentum), so Adam beta1 and lr interact. Reference-based regularisation (staying close to the initial weights) "does not necessarily apply" for dissimilar datasets. — [arXiv 2002.11770](https://arxiv.org/abs/2002.11770)
- Kumar et al. (ICLR 2022): during full fine-tuning, while the head is learned "the lower layers of the neural network change simultaneously and distort the pretrained features". Fine-tuning gave on average 2% higher ID but 7% lower OOD accuracy than linear probing. LP-FT (linear probe first, then full fine-tune) was "1% better ID, 10% better OOD than full fine-tuning". — [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)

### Inferences
- (Inferred) Check whether your fine-tune keeps or resets the regression head. If the head was kept from EyePACS pre-training, a fresh 1-epoch warm-up to 3e-4 re-randomises little but takes large steps on all layers. That would explain an OOF dip at epochs 5-7, right after peak lr, followed by partial recovery as cosine decays. This matches the Kumar et al. mechanism and the APTOS practice of 1e-5 fine-tunes. If the head was re-initialised, that is a second reason to linear-probe first.
- (Inferred) Concrete lr grid for one fold, from the sources: {1e-5 (Qishen Ha), 3e-5 (≈ Guanshuo's second-phase 2e-5), 1e-4}, cosine, no warm-up (or warm-up to that low peak), 8-15 epochs. Psi used only 4 epochs, and Qishen Ha's epoch count for fine-tuning is not stated.
- (Inferred) EMA 0.998 with a 20-epoch run of about 92 steps/epoch (3,662×0.8/32, if batch 32) averages over about 500 steps, so EMA cannot shield against a mid-run distortion. It will mostly track the late, recovered weights.
- (Inferred) Layer-wise lr decay is common in ViT fine-tuning but appears in none of the APTOS write-ups. Freezing or head-only (LP-FT style) is the variant that APTOS practitioners actually used.

### Gaps
- No APTOS source gives a controlled ablation of fine-tune lr (e.g. 3e-4 vs 1e-5) after 2015 pre-training. The 1e-5 recommendation is practice, not a measured delta.
- Qishen Ha's number of fine-tuning epochs was not stated.

## Q3. Label and domain shift between EyePACS and APTOS, and how winners handled it

### Takeaway
The grade distributions differ a lot: EyePACS train is about 73% grade 0, APTOS 49%. The datasets also differ in country, cameras and grading, and the APTOS 2019 images carry a strong image-size/crop "leak". A 2015-only model scored only about 0.73-0.75 public LB on APTOS. Winners handled the shift with combined training, pseudo-labels, and label smoothing of external data with model predictions. None of them reported a dataset flag or input. Several kept 2015 data in the final training stage, so it was never fully "forgotten".

### Cited Findings
- EyePACS 2015 train: 35,126 images; grade 0 = 25,810 (73.48%), grade 1 = 2,443, grade 2 = 5,292, grade 3 = 2.48% (873 by subtraction), grade 4 = 708. Test: 53,576 images. — [search-result table from DR papers, e.g. arXiv 2110.14160 / 1812.10595](https://arxiv.org/pdf/2110.14160) (I did not determine which listed paper holds the exact table; the counts match the commonly cited Kaggle numbers.)
- APTOS 2019 train (from this repo's `dr-repro/runs/s2_faithful/split.csv`, 3,662 rows): grade 0 = 1,805 (49.3%), 1 = 370 (10.1%), 2 = 999 (27.3%), 3 = 193 (5.3%), 4 = 295 (8.1%). — local file /Users/subhammahapatra/Downloads/dr-grading-reproduction/dr-repro/runs/s2_faithful/split.csv
- EyePACS and APTOS were collected in different countries (USA vs India), in separate Kaggle competitions. — [arXiv 2201.09042](https://arxiv.org/pdf/2201.09042)
- A 2025 review separates covariate shift (cameras, protocols, demographics) from label shift ("different grading rubrics"). It notes that a model tuned to EyePACS can underperform on other datasets with the same architecture. — [arXiv 2511.11065](https://arxiv.org/pdf/2511.11065)
- 2015-only models on APTOS: public LB "~0.73-0.75" (Mykhailo) and "LB 0.75" (DrHB). The comparable two-step models reached about 0.811-0.82 public. — [Kaggle 107926](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926); [Kaggle 108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- 1st place could not get 2015-train → 2019-validation to correlate with the LB. He gave up separate validation and trained on combined data. — [Kaggle 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- In the APTOS training data, image size and black-border crop strongly indicate diagnosis. Meta features alone gave 0.70+ local kappa but a negative public LB. — [Kaggle 100815 (Tom Aindow)](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815). Xuan Cao (17th): "~99% of 1:1 images are Class 1" in 2019 after border removal. He therefore centre-square-cropped the 2015 data and random-zoom-cropped 2019 1:1 images to the 4:3 look. — [Kaggle 107916](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- Qishen Ha classified every 2015 and 2019 image into three border-brightness "types" and transformed them all to one type before training. — [Kaggle 107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- DrHB removed duplicates and "confusing labels" before combining 2015+2019, to prevent leaks. 2019 images "looked zoomed to the center", which the team handled with random-zoom augmentation up to 1.35x. — [Kaggle 108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- Label-bias handling for external data (1st place): IDRiD labels were averaged with stage-1 predictions "to mitigate the labeling bias". Messidor labels were bounded to within ±0.5 of each ground-truth group's mean prediction. — [Kaggle 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Thresholds: 1st place moved the 0/1 threshold to 0.7 because the public test had fewer normals than train. 9th used [0.5, 1.5, 2.43, 3.32] from CV simulation. — [Kaggle 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065); [Kaggle 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- Grader variability is large even within one rubric: inter-grader kappa about 0.40-0.65 in a Google study. — [arXiv 1710.01711](https://ar5iv.arxiv.org/html/1710.01711)

### Inferences
- (Inferred) A 5-epoch MSE model trained on 73% grade-0 data acquires a prior biased toward 0/low grades. On APTOS (49% grade 0, 13% grades 3-4) that bias must be unlearned, and a high lr does this by moving the whole backbone. A lower lr plus your existing OOF threshold optimisation should absorb the prior shift. Alternatively, rebalance or subsample EyePACS grade 0 in pre-training (inferred; not in any write-up).
- (Inferred) Your OOF evaluation uses only APTOS, so it stays clean. Dedupe any APTOS↔EyePACS overlap if you combine data (DrHB did). Your repo already has `dr-repro/results/duplicate_rows.csv` for within-APTOS duplicates.
- No winner reported a dataset-ID flag or embedding. The "flag" approach is therefore unsourced for this competition.

### Gaps
- I found no quantitative study of EyePACS-vs-APTOS grading-rubric differences (e.g. kappa between the two label sets on common images).
- No source measures how much the 2019 size/crop leak affects transfer from 2015.

## Q4. How many pre-training epochs are needed? Is 5 enough?

### Takeaway
Top solutions pre-trained for **15-80 epochs** on 2015 data (25 for Qishen Ha on 88k images; about 15 for B5 and about 80 for B3 at Mykhailo's team; 15-30 for 9th; 30-50 + 5-10 for 1st on combined data). None used as few as 5. On 35k images, 5 epochs is about 5.5k steps at batch 32. That is likely under-trained for strong transfer (inferred), although the evidence is indirect.

### Cited Findings
- Qishen Ha: 25 epochs on 2015 train+test (about 88.7k images), cosine from 5e-4, no validation, and final vs best epoch made little difference. — [Kaggle 107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- Mykhailo/aispiriants: "~80 epochs on b3, ~15 epochs on b5" pre-train on 2015 train (full-size images). — [Kaggle 107926](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- Psi/dott: "around 15-30 or so epochs depending on the model". Prefit quality was noisy ("lucky" prefits). — [Kaggle 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- Guanshuo Xu: 30-50 epochs at 2e-4 on combined 2015+2019, then 5-10 at 2e-5. — [Kaggle 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- khornlund: with ">80,000 images in the 2015 dataset and heavy augmentation I could train for many epochs" (count not given). — [repo](https://github.com/khornlund/aptos2019-blindness-detection)
- tahsin314 (37th): about 5+10 epochs on 2015. — [repo](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection)
- jionie (forum comment): a model pre-trained on resized 2015 data reached about 0.78 validation kappa on the 2015 task; on APTOS, CV 0.9189 / LB 0.782. — [Kaggle 97860 comments](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/97860)

### Inferences
- (Inferred) With your timing (11 min/epoch on 35k at 380 px), 25 epochs on 88k images would take about 11 × 2.5 × 25 ≈ 690 min ≈ 11.5 GPU-h, which exceeds your budget. A feasible compromise: about 10 more epochs continuing from your 5-epoch checkpoint on 35k images (about 1.8 h + 43 min load). Alternatively, about 4 epochs on 88k (train+test, if the resized test set is available) ≈ 1.8 h. Both stay under about 2.5 h, leaving about 1.5 h for the fine-tune.
- (Inferred) Before spending on more pre-training, test the cheap lever (low-lr fine-tune of the existing 5-epoch checkpoint) first. The fine-tune lr is the larger documented difference from successful recipes (30x), while pre-training length is a 3-5x difference.
- (Inferred) A cheap diagnostic of pre-training quality: score the 5-epoch EyePACS checkpoint directly on APTOS OOF/test with no fine-tuning, after threshold fitting. Winners' 2015-only models were around 0.73-0.75 public LB, which is on the public-LB scale, not directly comparable.

### Gaps
- No source reports an epoch-count ablation of 2015 pre-training measured on APTOS.

## Q5. Papers quantifying gains from EyePACS pre-training on APTOS

### Takeaway
Clean, published supervised EyePACS→APTOS vs ImageNet→APTOS QWK ablations are scarce. The best-documented number is SSiT: self-supervised EyePACS pre-training gives APTOS fine-tuning kappa 92.97 (ViT-S, 384 px) vs 70.87 from random init. SSiT gives no ImageNet baseline, so it does not isolate the EyePACS gain over ImageNet. Kaggle evidence suggests a gain of about +0.01 on private QWK from 2015 data, but no team reports a clean with/without ablation.

### Cited Findings
- SSiT (Huang et al., IEEE JBHI 2024): ViT-S pre-trained self-supervised on EyePACS images (no labels). APTOS2019 fine-tuning kappa 92.97 ± 0.29 vs random init 70.87 ± 0.59 (ViT-S), 63.79 (ResNet-50), 65.87 (VGG16). Best other ViT SSL method (EsViT) 91.03. Fine-tune resolution 384x384. No ImageNet-pretrained baseline and no supervised EyePACS comparison. Downstream fine-tune lr/epochs not stated in the parsed text. — [arXiv 2210.10969](https://arxiv.org/html/2210.10969v5)
- Kaggle (indirect): 2015-only → APTOS public LB 0.73-0.75; two-step public LB about 0.811-0.82; combined public LB 0.821-0.827 (on the public-LB scale, where top private ≈ 0.93). — [Kaggle 107926](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926); [Kaggle 108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030); [Kaggle 107944](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- MamatShamshiev: without 2015 data "our models performed much worse" (not quantified). — [repo](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- Psi: 2015 pre-training "was very important to get better results on test data" (not quantified). — [Kaggle 107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- Reference CV levels on APTOS 2019 train 5-fold after 2015 data: Qishen Ha about 0.933 (no Ben) / 0.927 (Ben); Xuan Cao 0.93456 at 380 px EfficientNet; DrHB two-step about 0.931. — [Kaggle 107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987); [Kaggle 107916](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916); [Kaggle 108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- PRISM-DR (MDPI Sensors 2026), trained on EyePACS + DDR, reports APTOS QWK 0.8652 cross-domain (no APTOS fine-tune). This is a different protocol, so it is not a transfer gain. — [MDPI Sensors 26(14):4515](https://www.mdpi.com/1424-8220/26/14/4515) (search snippet only)

### Inferences
- (Inferred) The Kaggle CVs of 0.927-0.935 after 2015 data, set against your 0.920 without it, suggest a realistic target of about +0.005 to +0.015 OOF QWK from correctly used EyePACS data. This is not a controlled number: the teams' CV protocols differ (fold-mean vs pooled OOF, dedup or not, resolution, model).
- (Inferred) Private-LB single models with 2015 data but before pseudo-labelling were about 0.922-0.931 (Camaro SeResNeXt 0.922; Guanshuo singles 0.923-0.931).

### Gaps
- RETFound and other foundation-model papers mainly report APTOS AUROC, not QWK under supervised EyePACS pre-training, and I did not retrieve their tables. A clean supervised "ImageNet→APTOS vs ImageNet→EyePACS→APTOS" QWK ablation in a peer-reviewed paper was not found in this search.

## Q6. Joint (mixed) training vs sequential pre-training

### Takeaway
In APTOS 2019, the evidence leans toward **combined training, or keeping some 2015 data in the final stage**, over pure sequential pre-train→fine-tune. 1st, 5th, 8th, 11th and 13th all combined, and the 13th team reported +0.016 public LB on a single model. But the 4th (Qishen Ha), 2nd-after-cleaning, 7th, 9th and 12th pre-trained sequentially and also reached gold. With a low fine-tune lr, sequential works.

### Cited Findings
- 13th (Camaro): two-step SeResNeXt public LB 0.811 → combined (2015 train + 4/5 of 2019) 0.827. Their B5 variant kept 1/5 of 2015 in the fine-tune stage. — [Kaggle 107944](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- 8th (DrHB): the two-step setup had CV 0.931 vs public LB 0.811, a gap the team distrusted. Combining made "Local CV matches very well public scores". Final private 0.931. — [Kaggle 108030](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- 5th (Gary): chose combined "according to the better public LB". Pseudo-labelling lets the model see "three different distribution data". — [Kaggle 107960](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
- 1st (Guanshuo): combined 2015 train+test + 2019; stage-1 ensemble private 0.934. — [Kaggle 108065](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- 11th (4uiiurz1): combined at level 1, then a fine-tune with pseudo labels at level 2. — [Kaggle 107958](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107958)
- Sequential and successful: Qishen Ha 4th (private 0.932-0.933; 25 ep at 5e-4 → fine-tune at 1e-5), Psi 9th (0.931; 4-epoch fine-tune), Mykhailo 2nd-after-cleaning, BloodAxe 7th, JIANJIAN 12th. — [107987](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987); [107990](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990); [107926](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926); [108058](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058); [107995](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)
- MamatShamshiev tried both "first train on old data, then finetuning on the new train" and "train on both data, the finetune on new train data", with no comparison given. — [repo](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)

### Inferences
Recommended ~4 GPU-hour experiment (all inferred from the sources above):
- **Arm A (direct test of your hypothesis, cheapest):** start from your existing 5-epoch EyePACS checkpoint (if saved) and fine-tune on APTOS with peak lr **1e-5 to 3e-5**, cosine, **no warm-up restart**, about 10-15 epochs, EMA kept. Optionally 1 epoch head-only first (LP-FT; MamatShamshiev and BloodAxe practice). Run a 1-fold lr probe {1e-5, 3e-5, 1e-4} first (about 3 × 15 epochs × about 1 min ≈ 45 min, assuming APTOS epochs ≈ 11 min × 2,930/35,108 ≈ 0.9 min plus eval). Then 5 folds at the best lr (about 75-90 min). Success criterion: OOF > 0.920 with no epoch 5-7 dip. If even 1e-5 does not beat ImageNet, the 5-epoch pre-train itself is the weak link.
- **Arm B (if A fails, or as the second experiment):** extend pre-training (continue the checkpoint about 8-10 epochs at about 2e-4 → cosine, ideally adding 2015 test labels) and repeat the low-lr fine-tune. This roughly matches the Qishen Ha / Psi recipe within budget.
- **Arm C (alternative to A+B, combined training):** per fold, train from ImageNet on 4/5 of APTOS + a subsample of EyePACS (e.g. 10-15k, grade-0 downsampled so APTOS is not swamped), validate on the APTOS fold. This follows Camaro and Guanshuo. It costs about 5 folds × (epoch time on about 15k images ≈ 5 min) × 10-15 epochs ≈ 4-6 h, likely over budget unless you run fewer folds or epochs.
- **Diagnostic to log:** zero-shot QWK of the EyePACS checkpoint on APTOS OOF before fine-tuning, and per-epoch OOF to see whether the dip disappears at low lr. If it disappears, the forgetting hypothesis is confirmed.
- Also consider keeping the Adam/AdamW moments from pre-training (JIANJIAN) instead of a fresh optimizer with warm-up.

### Gaps
- No controlled, same-architecture CV comparison of combined vs sequential exists in these write-ups; the 13th place's +0.016 is a public-LB number on one model. The direction is mixed: sequential with a low lr also won gold.
- No source reports the optimal mix ratio of 2015:2019 in combined training, or whether to downsample EyePACS grade 0.
