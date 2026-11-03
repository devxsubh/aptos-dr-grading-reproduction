# What the top APTOS 2019 solutions did (Kaggle write-ups, repos, papers)

Retrieval note: the Kaggle discussion pages need JavaScript to render, so WebFetch only returned the page title. The text below was read from Kaggle's own discussion JSON endpoint (`discussions.DiscussionsService/GetForumTopicById`), which returns each post's markdown, and the final ranks and scores came from Kaggle's leaderboard endpoint (`competitions.LeaderboardService/GetLeaderboard`, competitionId 14774). Each claim is cited to the human-readable URL of that post. Some write-up titles give a rank from before Kaggle cleaned the leaderboard. In those cases the final rank comes from the cleaned leaderboard and the gold-medal index thread ([Vopani, "Gold Medal Solutions"](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108307)).

How to read the numbers. The public LB was about 15% of the test set (1,928 images) and the private LB the other 85% (about 13,000 images) ([Inoichan slides](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)). Public LB scores (0.80–0.85) are much lower than private ones (0.92–0.936) because the two splits come from different distributions. Only "CV on 2019 train" numbers are comparable to our OOF of 0.925.

## Q0. Final leaderboard and who wrote what

### Takeaway
The top 20 private scores are packed between 0.930 and 0.936, so 1st and 17th are only 0.006 apart. Every gold-medal team used the 2015 EyePACS data in some form. A write-up exists for every top-12 team except 3rd ([ka.kr] Save our eyes) and 6th ([ods.ai] topcoders), for which I found none.

### Cited Findings
- Final private LB (QWK), top 17: 1 Guanshuo Xu 0.936129; 2 [ods.ai] Eye of Private LB (Mykhailo Matviiv, Artur Ispiriants) 0.934310; 3 [ka.kr] Save our eyes (pudae, Youhan Lee, Wonho Song, Limerobot, Youngsoo Lee) 0.933720; 4 Qishen Ha 0.933693; 5 Gary 0.933646; 6 [ods.ai] topcoders (Victor Durnov, Selim Seferbekov) 0.933495; 7 Eugene Khvedchenya 0.933155; 8 Eye of Stars (DrHB et al.) 0.932106; 9 The Zoo (Psi, dott, Matt Motoki, ivovkanych, Thomas Yokota) 0.931033; 10 [kaggler-ja] CMY (mhiro2, Camaro, Y.Nakama) 0.930927; 11 4ui_iurz1 0.930744; 12 "The luckiest one!" (incl. JIANJIAN) 0.930685; 17 Xuan Cao 0.930451; 30th place 0.927641 — [Kaggle leaderboard](https://www.kaggle.com/competitions/aptos2019-blindness-detection/leaderboard)
- Kaggle "cleaned the leaderboard and disqualified some teams that have violated the rules." The competition had 2,944 teams and 60,711 submissions — [Kaggle staff, "Competition is Finalized"](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108460). According to a participant, the teams originally in 2nd and 3rd, and one of the teams in 5th/6th, were removed — [Homoalways](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108689)
- The 2nd-place write-up is titled "4th place solution (2nd after LB cleaning)" — [Matviiv](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)

### Inferences
- The top 20 are within 0.006 of each other on the private LB, and that spread is smaller than the effect of label noise that the 9th-place team reports (Q5). Beyond about 0.930 private, rank depended a great deal on submission choice and luck. Two teams said they had unselected submissions that scored higher (7th, 2nd; see Q3).

### Gaps
- I found no write-up for 3rd place ([ka.kr] Save our eyes) or 6th place ([ods.ai] topcoders) in the forum. A search of the forum's topic list for "3rd", "6th", "Save our eyes" and "topcoders" returned nothing relevant.

## Q1. Resolutions, backbones, losses, pre-processing, augmentation, epochs, optimisers

### Takeaway
The winning pipelines were ordinary. They used ImageNet backbones of medium size (SE-ResNeXt50/101, Inception-v4, Inception-ResNet-v2, EfficientNet B2–B7) at 224–512 px, nearly always with a single regression output (MSE or SmoothL1), heavy augmentation, and little or no pre-processing. The winner used no pre-processing at all, and several top teams found that Ben Graham's pre-processing did not help or made things worse. No team credited its rank to architecture, resolution or loss. They credited data: 2015 pre-training and pseudo-labels (Q2).

### Cited Findings
**1st, Guanshuo Xu (private 0.936)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Models: 2× Inception-ResNet-v2 at 512, 2× Inception-v4 at 512, 2× SE-ResNeXt50 at 512, 2× SE-ResNeXt101 at 384. Each model was trained twice with different seeds. "If I could have two more weeks I would definitely add some EfficientNets."
- Resolution: "I did not find a lot beneficial to go beyond 384 based on the public LB feedback". He still pushed to 512 "because the private set might benefit".
- Pre-processing: "no special preprocessing, just plain resizing". In comments he said he had "not tried any preprocessing", did no cropping, and "did nothing to deal with the class imbalance".
- Loss: only `nn.SmoothL1Loss()` (regression).
- Pooling: generalised-mean (GeM) pooling, initial p=3, "better than the original average pooling".
- Augmentation: contrast 0.2, brightness 20, hue 10, saturation 20, blur/sharpen, rotation 180°, scale 0.2, shear 0.2, shift 0.2, mirror.
- Optimiser (from comments): Adam lr 2e-4 for 30–50 epochs, then 2e-5 for 5–10 more epochs. The epoch count was chosen by probing the public LB in steps of 5.

**2nd, [ods.ai] Eye of Private LB (private 0.9343)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- "Only NN architecture that performed well was efficient-net". SE-ResNeXt50/101 and DenseNet "were slow to train and performed poorly". The final set was B3 at 300, B4 at 460 and B5 at 456.
- Pre-processing: crop the black background and resize. The teammate only resized. "Ben's color preprocessing ... didn't boost score."
- Augmentation (albumentations): Blur, Flip, RandomBrightnessContrast, ShiftScaleRotate, ElasticTransform, Transpose, GridDistortion, HueSaturationValue, CLAHE, CoarseDropout.
- Epochs: pre-training about 80 (B3) or 15 (B5) on 2015 data, then fine-tuning about 50 (B3) or 15 (B5) on 2019 data. In comments: "All models are regression", "Loss function is MSE", no training-data cleaning ("just trained on all data").

**4th, Qishen Ha (private 0.9337)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- EfficientNets only, with resolution scaled inversely to model size: B7@224, B6@240, B5@256, B4@320, B3@352, B2@376. "Large model training on high resolution images is much more likely to be overfitting, due to the small size of training set."
- Pre-processing: `crop_image_from_gray`. For training only, each image was also mapped to a single "type" of field-of-view crop. In comments he reported that applying this type normalisation at test time as well lowered CV.
- Augmentation: dihedral, RandomCrop, rotation, contrast, brightness, Cutout, perspective, CLAHE.
- Optimiser (from comments): cosine schedule, initial lr 5e-4 for pre-training and 1e-5 for fine-tuning. Warm-up "seem to be not helpful on my local CV." Regression with plain `round()`.

**5th, Gary (private 0.9336)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
- EfficientNet B0–B3 at 384×384. The loss combined classification and regression ("multi loss maybe not easy to overfitting"), and only the regression output was used at test time.
- Pre-processing: only cropping of the black background. Augmentation: flips, rotation 0–360, zoom 1.0–1.35, and a self-designed perspective transform.
- Adam, lr steps [5e-4, 1e-4, 1e-5, 1e-6] at epochs [10, 16, 22], 25 epochs, 5-fold CV, 2× TTA, default thresholds.

**7th, Eugene Khvedchenya / BloodAxe (private 0.9332)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058), [code](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- SE-ResNeXt50, SE-ResNeXt101, Inception-v4, all at 512×512.
- Pre-processing: crop, pad to square, resize. Ben's, CLAHE-based, red-channel dropping and spherical unwarping "All of them failed to outperform using images as-is".
- Target: "4 elements with sigmoid activation followed by summation" (bounded 0–4), trained with MSE, then Huber, and finally Cauchy loss. "This problem formulation worked best."
- Heads: global average pooling. GWAP, avg+max, FPN and LSTM pooling were "equally or worse".
- RAdam ("by default"), fp16. Heavy albumentations pipeline: ShiftScaleRotate/OpticalDistortion, RandomSizedCrop, brightness/contrast/gamma, FancyPCA/RGBShift/HSV, CLAHE, flips.
- Did not work: UDA, synthetic lesion augmentation, Mixup/MixMatch, EfficientNet trained from scratch, heavy models (SENet154, ResNeXt152, DenseNet201 "over-fitting badly"), and stacking.

**8th, Eye of Stars / DrHB (private 0.9321)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- EfficientNet-B5 and B4, MSE, fastai one-cycle with progressive resizing. Augmentation: 360° rotation, flips, zoom up to 1.35×, lighting.
- Removed duplicates and conflicting labels before merging the 2015 and 2019 data. Used crop-from-gray pre-processing with the aspect ratio preserved.

**9th, The Zoo / Psi (private 0.9310)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- EfficientNet-B7 and SE-ResNeXt50 at 320 px. "The type of model is not that important in this competition."
- Pre-processing close to Ben's: crop the retina, scale by radius, pad to 640×640, "remove blur", circle crop (90% in training, 95% at inference), resize to 320.
- Fine-tuning used 4-fold CV × 3 bags, Adam with one-cycle from 1e-5 to 5e-4 for a fixed 4 epochs, and no epoch selection. About half the models used per-image normalisation (subtract the image's own mean, divide by its own std). Most used MSE, a few MSE+BCE. "I don't think image augmentations are that helpful apart from flipping."

**10th, [kaggler-ja] CMY / Camaro (private 0.9309)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- 2× SE-ResNeXt101 at 320 (RMSE, Ben's crop) and 2× EfficientNet-B5 (256→320, MSE, resize only).

**11th, 4ui_iurz1 (private 0.9307)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107958)
- SE-ResNeXt50/101 and SENet154 at 256×256, MSE. SGD with momentum 0.9 and cosine from 1e-3 to 1e-5 for 30 epochs, then RAdam for 10 epochs on pseudo-labels. Ben's crop (scale radius) only.

**12th, JIANJIAN team (private 0.9307)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)
- EfficientNet B5/B6 at 320 and 288. The final blend was 80% regression and 20% classification models. "Finetune should preserve the optimizer's parameters from pretrain, it made a big difference to me."

**17th, Xuan Cao (private 0.9305)** — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- Regression, linear stacking of EfficientNet B0/B3/B4/B5 at 380, 8× TTA. Crop trick: center-square crop on all 2015 images, and random zoom crops to 4:3 on the 1:1 images in the 2019 data. His reason: "~99% of 1:1 images are Class 1".

**Lower-ranked but informative**
- 46th, Miroslav Valan (private 0.925–0.926): classification with CE plus label smoothing, EfficientNet B3 (448–456) and B5 (456–512) plus SE-ResNeXt101 (416), fastai, one-cycle with max lr 6e-4, "barely using external data", no pseudo-labels — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119)
- 58th, Tymchenko et al. (private 0.925466): B4@380, B5@456, SE-ResNeXt50 @512 and @380. "No improvement after 512x512." Small models (ResNet34, DenseNet121) and very large ones (B6/B7, ResNeXt101, DenseNet201) did not work. Multi-head training (classification + ordinal + regression) combined by a linear layer. Random label noise of ±0.3 on the regression target "greatly reduced overfitting." Switching from circle crop / Ben's to no pre-processing plus hard augmentation helped on both CV and LB — [write-up](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108007); [paper arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- Public-LB experience with pre-processing: "Heavy preprocessing may reduce the performance of EfficientNet (0.811 -> 0.732)" at 256 px — [Quan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104818)

### Inferences
- Our backbone/resolution/loss choices (B3/B4 at 380–384, MSE regression) are the same as the gold-medal recipes. Of those teams, 4 of 11 (2nd, 4th, 7th, 58th) found Ben's or other heavy colour pre-processing neutral or harmful. Our use of Ben Graham pre-processing is therefore a candidate for an ablation, though not an obvious source of a 0.03 gap. The 4th-place CV fell from about 0.933 to about 0.927 with Ben's (Q5).
- Two cheap components appear in several top write-ups and are absent from ours: GeM pooling (1st) and bounded or robust regression losses (SmoothL1 for 1st; Cauchy/Huber on a sigmoid-sum head for 7th; label-noise injection for 58th).

### Gaps
- None of the top write-ups reports batch size and weight decay together. The 1st-place post gives Adam lr only (in a comment).

## Q2. External data (2015 EyePACS, IDRiD, Messidor) and pseudo-labelling: what each added

### Takeaway
Every gold-medal team used the 2015 EyePACS data (about 35k train images, plus the 2015 test set for several teams), either as a pre-training stage or merged into training. Most also pseudo-labelled the APTOS test set. The numbers the teams report for pseudo-labelling on the private LB are small to moderate: +0.001 (1st, together with IDRiD/Messidor), +0.003 to +0.009 for single models (10th, 8th, Neuron Engineer), and "huge" on the public LB (2nd). The 9th-place team reached 0.931 private with no pseudo-labels at all, which places 2015 data as the common factor.

### Cited Findings
- 1st place, stage 1 (2015 train+test merged with 2019 train, validated only on the public LB): singles Inception-ResNet-v2 public 0.831 / private 0.927, Inception-v4 0.826 / 0.924, SE-ResNeXt50 0.826 / 0.931, SE-ResNeXt101 0.819 / 0.923; 8-model ensemble 0.844 / 0.934 — [Guanshuo Xu](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- 1st place, stage 2: added soft pseudo-labelled public test images, plus IDRiD (labels smoothed by averaging with stage-1 predictions) and Messidor (4-level labels mapped through stage-1 soft predictions, outliers bounded to the group mean ± 0.5). Each model was trained 10 more epochs. The result was public 0.850 and private 0.935, so private gained +0.001 over stage 1. Changing the thresholds then gave 0.936 — [Guanshuo Xu](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- 2nd place: pre-trained on full-size 2015 train images. "Models trained on old data only gave us ~0.73-0.75 on public LB." The "secret ingredient" was pseudo-labelling the current test data and fine-tuning on train plus pseudo-labelled test: "It gave us a huge boost", and a second round "boosted a little bit more". They also pseudo-labelled the 2015 test data. All pseudo-labels were used, with no confidence filtering. No private-LB delta is given — [Matviiv](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- 4th place: pre-trained on the 2015 full dataset (train and test) for 25 epochs without validation, then 5-fold CV on 2019. No pseudo-labels: "I've tried pseudo once but didn't get good result so I gave up" (comment) — [Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- 5th place: trained jointly on 2015 and 2019 data rather than pre-training then fine-tuning ("according to the better public LB"), validating on the 2015 private test plus 2019 validation data. Confidence-filtered pseudo-labels from the public test set. "I think the pseudo labels is the most import key to win in this competition" — [Gary](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960)
- 7th place: pre-trained on 2015, then trained on 2019 + IDRiD train + Messidor, then pseudo-labelled the APTOS test set, Messidor-2 and the 2015 data, and fine-tuned the head only. What worked: "Pre-training on 2015 data", "Heavy augmentations", "Pseudo-labeling AND fine-tuning of head ONLY", RAdam — [Khvedchenya](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058)
- 7th-place repo ablation, SE-ResNeXt50 regression, 4-fold CV, each extra dataset added to APTOS 2019 on its own. These are CV / public LB pairs: +APTOS2015 0.9244 / 0.752; +IDRiD 0.8737 / 0.668; +Messidor 0.9006 / 0.554; +all three 0.9134 / 0.779. The baseline without extra data was 0.9225 / 0.787. The CV in this table moves with the composition of the validation set, so read it with care — [BloodAxe README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- 8th place: merging 2015 and 2019 into one training set gave B5 5-fold CV 0.823 / public 0.821 / private 0.926 and B4 0.831 / 0.816 / 0.927. Retraining for 3 epochs with test pseudo-labels from the two-model average gave B5 private 0.930 (+0.004) and B4 0.929 (+0.002); the ensemble reached 0.931 — [DrHB](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030). The write-up lists B5's post-pseudo public LB as "0.93", which looks like a typo, since the public LB was otherwise 0.80–0.85.
- 9th place, with no pseudo-labels: "We did pretraining on full 2015 data which was very important to get better results on test data." Pseudo-labelling the out-of-fold data inside CV "either did not improve much or only a tiny bit", and pseudo-labelling the full test set "did not change public LB." Their private score was 0.931 — [Psi](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- 10th place: training on the 2015 data merged with 4/5 of 2019, rather than pre-train then fine-tune, raised the SE-ResNeXt public LB from 0.811 to 0.827. Single SE-ResNeXt101 private: 0.922 before pseudo-labels, 0.925 with its own pseudo-labels, 0.931 with pseudo-labels from the SE-ResNeXt + EfficientNet ensemble. That last single model beat their final ensemble. Two kinds of pseudo-label were used: soft labels with retraining from ImageNet, and hard labels with fine-tuning. Messidor and "IEEE" (IDRiD) data were used sometimes: "I'm not sure it's necessary or not" — [Camaro](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- 11th place: 2019 train plus 2015 data in the first-level models, then second-level fine-tuning on 2019 train plus pseudo-labelled 2019 test (public and private), split into 5 parts with one per fold. Final public 0.826 / private 0.930 — [4ui_iurz1](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107958)
- 12th place: pseudo-labels moved the public LB from 0.840 to 0.842. They gave the pseudo-label model a low weight, and "As it turned out, I was wrong" — [JIANJIAN](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)
- Neuron Engineer (team not in gold): three rounds of iterative pseudo-labelling raised the public LB from 0.82x to 0.850, but private only from 0.926 to 0.929 — [comment in the 4th-place thread](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987); [own post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107864)
- Lex Toumbourou (37–50 range, ImageNet init, 2015 + 2019 merged): pseudo-labelling gave "+1-2%" on the public LB but "doesn't appear to have had a significant effect on the private results" — [post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947)
- Inoichan (silver, 66th): EfficientNet B2/B5 at 224, pre-trained on 2015 and fine-tuned with 5 folds on 2019, private 0.924 — [slides](https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome)
- Tymchenko et al. (58th): 2015 pre-training (20 epochs, SGD), then 2019 + IDRiD (413 images) + Messidor (1,200, relabelled) for 5-fold training (75 epochs, RAdam). Private 0.925466 — [arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)

**Results with no or minimal external data**
- Miroslav Valan (46th), classification with official data only, public LB: B3 0.813 (448 px), B5 0.817 (456 px), SE-ResNeXt101 0.799 (416 px). With external train data clipped at 2,000 per class: B3 0.822 (456 px), and "CV improved to 0.94+" for B5 (512 px). Final ensemble private 0.925–0.926 — [post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119)
- BloodAxe repo, 2019 data only, 4-fold CV / public LB: ResNet18 classification 0.9077 / 0.625, ResNet18 regression 0.9093 / 0.646, SE-ResNeXt50 classification 0.9213 / 0.772, SE-ResNeXt50 regression 0.9225 / 0.787 (2 folds), SE-ResNeXt101 regression 0.9176 / 0.763, ResNeXt50-rms regression 0.9234 / 0.791 — [README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- ilovescience / agscin (78th): EfficientNet-B3 trained "purely on 2019 train data" with polar unrolling (300×768), public 0.788 / 0.782. This team's ResNet50 runs started from the author's own 2015-trained model, so they are not ImageNet-only. Final private 0.923 — [post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108209)

### Inferences
- For the private LB, the reported ingredient that separates about 0.925 from about 0.930–0.934 is 2015 EyePACS data. Pseudo-labelling of APTOS test images adds roughly +0.001 to +0.009 on top. Neither is available in our strict ImageNet-only, APTOS-only protocol, and neither moves our OOF metric on APTOS train in a way we can measure. Pseudo-labelling targets the test distribution, not train CV.
- Strong teams that used no or minimal external data report around 0.92–0.925 CV on 2019 train (BloodAxe 0.921–0.923 with 4 folds and no extra data) and around 0.925 on the private LB (Valan). These numbers match our 0.925 OOF.

### Gaps
- No team reports a clean ablation of 2015 pre-training on private LB versus 2019-only training with everything else held fixed. The nearest are the 2nd-place figure of 0.73–0.75 public for 2015-only training and the BloodAxe 2019-only rows, which are public LB only.
- Private-LB deltas for pseudo-labelling are reported only by the 1st, 8th and 10th teams and Neuron Engineer. The 2nd-place team says "huge" without giving numbers.

## Q3. Ensembling strategy, diversity and threshold optimisation

### Takeaway
Plain averaging of regression outputs won. Stacking (LR, KNN, trees, extra trees) was tried by the 7th and 9th teams and lost or tied. Diversity came from different architecture families (Inception + ResNet at 1st), different resolutions per EfficientNet size (4th), different pre-processing (4th, 9th, 10th) and different 2015 pre-training checkpoints (9th). Ensembling added about +0.002 to +0.004 private over the best single model. Most teams kept the thresholds at or near [0.5, 1.5, 2.5, 3.5], and threshold changes were worth at most about 0.001 on the private LB.

### Cited Findings
- 1st: simple average of 8 models. "Inceptions and ResNets usually blend well." Best stage-1 single private 0.931 (SE-ResNeXt50) against ensemble 0.934. Thresholds changed from [0.5, 1.5, 2.5, 3.5] to [0.7, 1.5, 2.5, 3.5] hours before the deadline took private from 0.935 to 0.936. His reason: the public test set had fewer normal cases than train, so moving the class-0/1 cut-off up lets the model predict more 0s if private looks like train. "0.5 is a dispersion value I manually set" — [Guanshuo Xu](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- 2nd: "Pick top-performing epochs ... blend them by a simple mean", with flip TTA. Selection was by QWK or by loss, with close results. They had an unselected submission that would have placed 1st or 2nd — [Matviiv](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926)
- 4th: average of 6 EfficientNets (B2–B7 at different resolutions), 5 folds × 8 dihedral TTA each. Best single private 0.929 (B7@224). Ensemble v1 private 0.932; v2, which swapped in two models trained on Ben's pre-processing "with more variance", private 0.933. No threshold tuning: optimised thresholds came out at about [0.503, 1.497, 2.508, 3.501], so he used `round()` — [Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- 7th: average of SE-ResNeXt50, SE-ResNeXt101, Inception-v4 and a pseudo-label-fine-tuned Inception-v4. "Stacking. None of KNN, LogisticRegression, Trees didn't work." An unselected submission with optimised thresholds "scored at Top 4" — [Khvedchenya](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058). The repo records optimised OOF thresholds [0.527, 1.477, 2.428, 3.394] — [README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- 9th: 12 model types × 4 folds × 4 TTA, 192 predictions averaged. Thresholds adjusted "based on CV simulations" to [0.5, 1.5, 2.43, 3.32]. An extra-trees stacker "worked well on both CV as well as public LB, but a tiny bit worse on private" — [Psi](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- 10th: pseudo-labelling combined with ensembling raised single models to public 0.846. One single SE-ResNeXt101 trained on ensemble pseudo-labels (private 0.931) beat the final ensemble — [Camaro](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- 5th and 12th: default thresholds [0.5, 1.5, 2.5, 3.5]. The 12th team voted among classification models — [Gary](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107960); [JIANJIAN](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107995)
- 17th: linear stacking of 4 EfficientNets — [Xuan Cao](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- 46th: ensembled only models with prediction correlation below 0.9 — [Valan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119)
- 58th: 20 models (4 configs × 5 folds) × 10 TTA combined with a 0.25-trimmed mean. On the holdout the trimmed mean scored 0.971 against 0.968 for the plain mean, without TTA — [arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- 37–50 range: "Ensembling models with slightly different preprocessing had big impact on our LB score in the early days. However, as our individual models improved, the improvement of this technique decreased" — [Toumbourou](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947)

### Inferences
- Our two models are both EfficientNets with the same pre-processing and correlate at 0.984, so a +0.001 blend gain is what these reports would predict. The only teams that report ensemble gains (+0.003 at 1st, +0.003–0.004 at 4th) mixed families (Inception/ResNet) or pre-processing variants (raw vs Ben's), and even those gains are small. Adding a GeM-pooled SE-ResNeXt or Inception model trained on un-pre-processed images is the closest thing to the winner's diversity recipe. Going by these reports, a realistic gain is a few thousandths, not 0.03.
- QWK-tuned cut-offs barely move CV. The 4th-place optimum was within 0.01 of the defaults. Threshold tuning will not close a 0.03 gap.

### Gaps
- None of the write-ups reports a CV-based (OOF) ensemble gain on 2019 train. Ensemble deltas are all on the public or private LB.

## Q4. CV QWK on the APTOS 2019 train set (the number most comparable to our 0.925 OOF)

### Takeaway
The top teams that report CV on 2019 train give about 0.92–0.935, and all of those runs used 2015 pre-training. The 2019-only runs (BloodAxe, 4 folds) give 0.909–0.923. No top team reports CV at or above 0.954 on APTOS 2019 train with a k-fold or OOF protocol. The only QWK at 0.96–0.97 is the 58th-place team's single 736-image holdout. That model used 2015 + IDRiD + Messidor data and still scored only 0.925 on the private LB.

### Cited Findings
- 4th place: "all of these model have a similar local 5-fold CV around 0.933" (EfficientNet B2–B7, 2015 pre-trained, raw images). The two Ben's-pre-processed models were "around 0.927". Final private 0.933 — [Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- 17th place: "For my best solution (QWK 0.930), the local CV (QWK) is 0.93456." Setup: 5-fold CV on 2019 after 2015 pre-training, B0–B5 at 380, linear stacking — [Xuan Cao](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916)
- 37th place: one 20% stratified holdout of 2019 after 2015 pre-training. Single EfficientNet B0–B5 (256–300 px) scored 0.919–0.926 on validation, with private 0.914–0.921. The ensemble scored 0.932 on validation and 0.926 private — [Tahsin Mostafiz](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108017)
- DrHB, before the competition ended, B3 at 224 pre-trained on 2015, 5-fold CV on 2019: no augmentation 0.93 (public 0.78); flips 0.934 (0.783); rotate-360 0.92 (0.79); zoom-to-center 0.90 (0.793); flip + zoom + rotate + contrast 0.89 (0.809) — [comment in "Discrepancies between CV/LB"](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815)
- DrHB, early, ResNet50 at 256 with a 20% holdout on 2019 only: CV 0.932 (public 0.755). Balanced classes: 0.934 (0.751). Progressive resizing 128→256→512: 0.919 (0.734) — [DrHB "Some Tips"](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/99353)
- DrHB, 8th-place final: "local CV was around 0.931 but LB was 0.811" with pre-train then fine-tune. After merging 2015 and 2019, 5-fold CV fell to 0.823–0.831. Those folds evidently include 2015 images, so they are not comparable to a 2019-only OOF — [DrHB](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030)
- BloodAxe, 2019 only, 4-fold CV (mean ± std over folds): ResNet18 0.908–0.909, SE-ResNeXt50 0.921–0.923, SE-ResNeXt101 0.918, ResNeXt50-rms 0.9234. With APTOS2015 added, SE-ResNeXt50 scored 0.9244 — [README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- Neuron Engineer: "CV >> LB .92x >> 0.77x, .93x >> 0.79x, .94x >> 0.80x" — [comment](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815)
- Valan: B5 with external data reached "CV improved to 0.94+". This was 5-fold stratified on official data, with external data added to training only. Its private-LB contribution is not reported separately; the ensemble scored 0.925–0.926 — [post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119)
- Tymchenko et al. on a 736-image APTOS-train holdout: single models 0.960–0.971 QWK, trimmed-mean ensemble 0.971 without TTA and 0.969 with TTA. The same system scored 0.818/0.827 on the Kaggle validation (public) split and 0.925 on test (private) — [arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261)
- Training-set leak: "using just meta features you can score 0.70+ validation kappa locally, but get a negative LB score." Image size and black-space cropping "are strongly indicative of diagnosis in the training data, but not in the public test data" — [Tom Aindow](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815). After cropping the black region, 2019 images come in two aspect ratios (1:1 and 4:3), and "~99% of 1:1 images are Class 1" — [Xuan Cao](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107916). Over 70% of the test set is 480×640 — [Camaro](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)

### Inferences
- Our 0.925 OOF, with ImageNet weights and APTOS data only, matches the 2019-only CVs from the BloodAxe repo (0.921–0.923 for SE-ResNeXt50) and sits only about 0.008 below what top teams reach with 2015 pre-training (0.933–0.935 on 5-fold 2019 CV). From these reports, 2015 pre-training is worth roughly +0.01 on 2019 CV. It is not available under our protocol.
- A 2019-train QWK of 0.954 lies above every k-fold CV a top team reported. A number that high appears in these sources only for a single random holdout (0.96–0.97, 58th place), on a system that scored 0.925 private. Two inferred explanations follow. First, single random splits of 2019 train can score much higher than 5-fold OOF, especially when duplicate images (the 8th and 9th teams discuss duplicates and conflicting labels) or the size/aspect-ratio leak cross the split. Second, the paper's 0.954 is unlikely to come from a better model alone. It more likely reflects its split, duplicate handling or leak exposure. This is an inference, not a reported fact.
- DrHB's augmentation ablation shows that heavier augmentation lowers APTOS-train CV (0.934 → 0.89) while raising LB, because CV rewards the leaky meta-features. Our strong augmentation (and the field-of-view normalisation in Ben pre-processing) may be costing CV in the same way. A light-augmentation run would test this; it would probably gain CV while making the model less robust.

### Gaps
- I found no top-team CV on 2019 train with ImageNet-only weights and no 2015 data, apart from the BloodAxe repo rows. The 1st- and 2nd-place teams did not report 2019 CV: the 1st validated only on the public LB, and the 2nd on 5% of train plus the public LB.
- None of the write-ups states whether duplicates were removed before computing CV, except the 8th, which removed duplicates and conflicting labels via a public kernel.

## Q5. Ablations that quantify individual components

### Takeaway
The component ablations are scattered and mostly measured on the public LB. The quantified effects are: Ben's pre-processing cost about 0.006 CV (4th); regression beat classification by about 0.001–0.002 CV (BloodAxe); resolution above 384–512 gave no gain (1st, 58th, BloodAxe 768); a small backbone at low resolution matched a large one at high resolution (4th); heavier augmentation lowered train CV but raised LB (DrHB); and pseudo-labels added +0.002 to +0.009 private on single models (8th, 10th).

### Cited Findings
- **Resolution × model size**: EfficientNet B7@224 public 0.840, B6@240 0.832, B5@256 0.831, B4@320 0.826, B2@376 0.828, all with similar 5-fold CV of about 0.933. Best single private was B7@224 at 0.929 — [Qishen Ha](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987). A commenter: "My efficientnet-b5 performed best with 256x images instead of 456 (the default)" — [Rishabh Agrahari, same thread](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987)
- **Resolution, 384 vs 512**: "I did not find a lot beneficial to go beyond 384 based on the public LB" — [1st](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065). "We got no improvement after 512x512 images" — [58th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108007). SE-ResNeXt50 on the 58th-place holdout: 512 px 0.969 QWK vs 380 px 0.960 without TTA, and 0.971 vs 0.962 with TTA — [arXiv 2003.02261](https://ar5iv.labs.arxiv.org/html/2003.02261). ResNeXt50 with extra data: 768 px CV 0.9200 / public 0.803 vs default size 0.9129 / 0.804 — [BloodAxe README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection). One participant: sizes 384, 512, 768 and 1024 tried, "higher image sizes didn't give me a higher score" — [Stanislav Malorodov](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/99603)
- **Regression vs classification** (2019-only, 4-fold CV / public LB): ResNet18 classification 0.9077 / 0.625 vs regression 0.9093 / 0.646; SE-ResNeXt50 classification 0.9213 / 0.772 vs regression 0.9225 / 0.787 — [BloodAxe README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection). ResNet50, 2019 only: classification then regression CV 0.930 vs regression 0.932 — [DrHB](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/99353). A classification-only ensemble still reached private 0.925–0.926 — [Valan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108119). ResNet50 at 78th place: classification private 0.903–0.905 vs regression 0.903 — [ilovescience](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108209)
- **Pre-processing**: raw-image models CV about 0.933 vs Ben's about 0.927. The Ben's models still added ensemble diversity (private 0.932 → 0.933) — [4th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987). Crop-from-gray took ResNet50 from public 0.749 / private 0.893 to 0.784 / 0.903 — [78th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108209). Heavy pre-processing on EfficientNet at 256: public 0.811 → 0.732 — [Quan](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104818)
- **Augmentation** (B3@224, 2015 pre-trained, 5-fold 2019 CV / public): none 0.93 / 0.78 → flips 0.934 / 0.783 → all four 0.89 / 0.809 — [DrHB](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/100815)
- **Label smoothing / mixup** (ResNet50, 78th): mixup private 0.906 vs 0.903 baseline; label smoothing 0.905; longer training raised public but left private unchanged (0.903) — [ilovescience](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108209). Label smoothing "helped a lot with overfitting and having a stable LB score" — [Toumbourou](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947)
- **Pooling**: GeM better than average pooling (1st, no number) — [Guanshuo Xu](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065). GWAP vs GAP on SE-ResNeXt50-d at 512: CV 0.9231 / public 0.813 vs 0.9216 / 0.809 — [BloodAxe README](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- **Joint vs staged use of 2015 data**: public 0.811 → 0.827 for SE-ResNeXt when training on 2015 + 4/5 of 2019 together — [10th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944)
- **Pseudo-labels on private LB**: SE-ResNeXt101 0.922 → 0.925 (own labels) → 0.931 (ensemble labels) — [10th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107944). B5 0.926 → 0.930, B4 0.927 → 0.929 — [8th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108030). 0.926 → 0.929 after 3 iterations — [Neuron Engineer](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107987). Pseudo-labels + IDRiD + Messidor on an 8-model ensemble: 0.934 → 0.935 — [1st](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- **Label noise ceiling**: setting about 30 test labels from their duplicate images in train "dropped our score by nearly 0.01 on LB", which the 9th-place team attributes to "a lot of labels are inconsistent" — [Psi](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107990)
- **Model size**: heavy models (SENet154, ResNeXt152, DenseNet201) "over-fitting badly" — [7th](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108058). Small (ResNet34, DenseNet121) and very large (B6/B7, ResNeXt101, DenseNet201) models did not work at 58th — [Tymchenko](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108007)

### Inferences
- In these ablations, no single architecture, resolution or loss change moves 2019-train CV by more than about 0.01, except augmentation strength (−0.04 CV with heavy augmentation, DrHB) and 2015 pre-training (about +0.01). Ranked by expected CV effect under our strict protocol, the components worth trying are:
  1. Lighter or leak-friendlier augmentation: raises CV, possibly at the expense of generalisation.
  2. Dropping Ben pre-processing for raw cropped images: about +0.006 in the 4th-place report.
  3. GeM pooling, and SmoothL1 or bounded sigmoid-sum regression heads.
  4. A non-EfficientNet family (SE-ResNeXt50 at 512, Inception-v4) for ensemble diversity.

  These are inferences from scattered public reports. Together they might add 0.005–0.015, which is short of 0.03.
- Two kinds of evidence point the same way. Top-team 2019 CV saturates around 0.933–0.935 even with 2015 data. Label inconsistencies are large enough to move the LB by 0.01 on 30 images. On this reading, the remaining gap to 0.954 is better explained by evaluation protocol (split, duplicates, leak) than by missing modelling components.

### Gaps
- None of the gold-medal write-ups ablates 5-fold OOF QWK on 2019 train against resolution or loss with everything else fixed. Most deltas are on the public LB, which has a different distribution.
- I found no write-up for the 3rd-place ([ka.kr]) or 6th-place ([ods.ai] topcoders) teams, so their methods are unknown.
- The 12th-place write-up attributed to Thomas Yokota (The Zoo, 9th) returned no body from the API.
