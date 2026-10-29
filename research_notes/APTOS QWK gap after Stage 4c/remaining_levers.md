# Remaining training levers to lift APTOS 2019 QWK above ~0.93 within ~9 Kaggle T4 GPU-hours

Scope and baseline (from the assignment, not a source). Strict protocol: 5-fold CV on 3,112 images plus a 550-image hold-out. The current best is a blend of three 5-fold ensembles: B4-NS 380, B3 384 and B4-NS 512. It scores OOF 0.928 and test 0.918 (95% CI 0.900-0.935). Members correlate at 0.98-0.99.

Measured T4 costs (5 folds x 20 epochs): B4-NS 380 about 87 min, B3 384 about 64 min, B4-NS 512 about 138 min, SE-ResNeXt50 512 about 118 min. The budget of about 9 GPU-hours is about 540 min.

This file builds on the earlier notes in `research_notes/Closing the APTOS 2019 QWK gap/` (backbones_resolution.md, ordinal_training_techniques.md). It does not repeat their evidence on ordinal losses, ConvNeXt and SE-ResNeXt.

Evidence grades used below:
- **A:** controlled ablation on APTOS.
- **B:** controlled result on another DR dataset (a transfer).
- **C:** competition use without an ablation.
- **D:** inference or general machine learning only.

## 1. Pseudo-labelling / semi-supervised use of unlabelled images (APTOS 2019 Kaggle test, other fundus images)

### Takeaway
All top-4 APTOS 2019 private-LB solutions (0.934-0.936) used pseudo-labelling of the competition test images, and one summary says "almost" every team did. None of them published a before/after number, so the expected gain is a prior, not a measurement.

On our protocol, pseudo-labelling the 1,928 unlabelled APTOS test images uses no external labels. It does use images outside the 3,662 labelled set, so report it as "strict + unlabelled in-competition images". It is the only lever left that adds new information rather than another correlated view of the same 3,112 labels. It needs fold-safe teachers to keep the OOF score honest.

### Cited Findings
- **1st place (Guanshuo Xu, private 0.936, public 0.850), grade C.** Stage 1 was 8 models: SE-ResNeXt50, SE-ResNeXt101, Inception-v4 and Inception-ResNet-v2, with 2 seeds each, Huber loss and GeM pooling, at 384/512 px. Stage 2 retrained the same 8 models with soft pseudo-labels on the public test set, plus IDRiD and Messidor (external labels). Cut-offs were [0.7, 1.5, 2.5, 3.5]. The gain from pseudo-labelling was not reported. — [U-Tokyo Med slide summary](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med); [Kaggle 1st place post](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- **2nd place (private 0.934, public 0.848), grade C.** EfficientNet B3/B4/B5 at three input sizes (456, 460, 300 px) "for feature-map diversity". Used DRD-2015 data and "iterative loops" of pseudo-labelling on the test data, with flip TTA, simple blending and epoch ensembling. No gain reported. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **4th place (private 0.934), grade C.** Pre-trained 25 epochs on DRD-2015, then 5-fold CV on APTOS, with pseudo-labelling and 8-view dihedral TTA. No gain reported. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **"Top solutions" deck, grade C.** States that "almost [all teams] used pseudo labeling" in this competition. It lists a 7th place with pseudo-labelling (repeated), but its two 7th-place descriptions conflict, so treat that entry as unverified. — [Slideshare "Aptos Top Solutions"](https://fr.slideshare.net/ssuser21af5b/aptos-top-solutions)
- **11th place (private 0.930), grade C.** Pseudo-labels were a weighted average of the first-level models on the full public and private test set, split into 5 parts with a different part used per fold. Second-level SE-ResNeXt50/101 models started from the first-level weights and trained 10 epochs (MSE, RAdam, cosine 1e-3 to 1e-5) on 2019 train plus pseudo-labelled test. Images were 256 px. No before/after or CV scores. — [4uiiurz1 GitHub](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)
- **Grade B (OCTA DR grading, DRAC22, transfer).** VUNO's EfficientNet-B2 used confidence-filtered "reliable" pseudo-labels: per-class top-N%, growing over 5 rounds, on the 611 unlabelled test images vs 386 labelled ones. Combined with ensembling, TTA and post-processing, QWK rose from 0.827 to 0.891. The flattened tables do not let the pseudo-labelling share be isolated, and the authors say post-processing was "most effective" for DR grading. — [Bag of Tricks, arXiv 2210.09558](https://ar5iv.labs.arxiv.org/html/2210.09558)
- **Negative result (grade B, ultra-wide-field fundus, domain adaptation).** "Both pseudo-labeling and mixup compromise the DR grading performance." Cited in the earlier notes. — [arXiv 2309.10619](https://arxiv.org/pdf/2309.10619)
- **Domain shift of the APTOS test images.** The top teams' public-LB QWK (0.83-0.85) was far below their private LB (0.93-0.94) and local CV. For example, one 28th-place team had local CV 0.936, public 0.832 and private 0.927. So the 1,928 public-test images (and their label behaviour) differ from the training distribution. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **No APTOS semi-supervised paper reporting QWK.** The academic semi-supervised DR papers I found report other metrics or other datasets. Examples: ACCN on Messidor/APTOS with fewer annotations, and knowledge distillation with unlabelled EyePACS/Messidor, which claims "improved significantly" in the abstract without numbers. — [ACCN, PMC8979701](https://pmc.ncbi.nlm.nih.gov/articles/PMC8979701/); [arXiv 2009.00982](https://arxiv.org/abs/2009.00982)

### Inferences
- **Strictness.** Pseudo-labelling the APTOS Kaggle test images uses no labels other than our own model's predictions. It is strict with respect to labels but transductive with respect to images. Our 550-image hold-out comes from the labelled train set, so it is never seen as long as the teachers never saw it, which is already the case. Report it as a separate row, "strict + unlabelled APTOS test images", not as plain strict.
- **Fold safety (important).** If one 5-fold teacher ensemble labels the test images, those labels carry knowledge of every fold's labels. A fold-k student would then be scored on images whose labels indirectly shaped its targets, so the OOF score would be optimistic. Fix: label the test images for student fold k only with teacher models for fold k (trained without fold k), averaged across members and TTA. The hold-out test score is valid either way.
- **Recipe that fits the budget.**
  1. Inference: label the 1,928 images with the fold-k members of the existing 3-member blend, using 2-4 TTA views. This is 15 fold-models x 1,928 images. At roughly 50-100 img/s per view on a T4 with AMP (an estimate, not measured), it takes about 15-40 min.
  2. Use soft (continuous regression) targets, as the 1st place did. Optionally drop the images whose ensemble prediction sits near a cut-off, or down-weight them (for example, loss weight 0.5).
  3. Retrain B4-NS 380 and B3 384 students per fold on about 2,490 labelled + 1,928 pseudo-labelled images (×1.77 data). Either train from scratch for 20 epochs or, like the 11th place, start from the teacher weights and train 8-10 epochs.
- **Cost.**
  - Students from scratch: B4-NS 380 ≈ 87 × 1.77 ≈ 155 min, B3 384 ≈ 113 min, plus about 30 min inference. Total about 5 h.
  - Warm-started 10-epoch students: about half that, about 2.5 h.
  - A B4-NS 512 student is ≈ 245 min from scratch, so the full 3-member student blend from scratch does not fit alongside anything else.
- **Expected gain (grade D, prior only).** Every top team used it, but none measured it, and the 1,928 images come from a shifted distribution. I would expect +0.002 to +0.008 OOF on the blend, centred around +0.004, with a real chance of about 0. Over 0.93 OOF is plausible (0.928 + 0.004). A test gain of that size cannot be detected on 550 images, where the paired-bootstrap width is about ±0.006-0.009.
- **Larger unlabelled pools.** The 35k EyePACS-2015 train and 53k test images could be pseudo-labelled without their labels. The images are external, though, which makes this a non-strict row. Even 10k images would multiply the epoch cost by about 4-5, which breaks the budget.

### Gaps
- No public before/after QWK for pseudo-labelling exists from any APTOS team. The Kaggle write-ups (discussions 108065, 107926) did not render for fetching, so details such as soft-label weighting, the number of rounds and confidence filtering come only from slide summaries.
- The 1st-place slide's description of how soft labels were formed is garbled ("averaged between provided labels and stage-1 predictions"). Whether that applied to train labels as well (a form of self-distillation) is unclear.

## 2. Bigger EfficientNets (B5 456, B6 528, B7 600) and other backbones that worked on APTOS

### Takeaway
No source gives a controlled APTOS comparison of B5/B6/B7. Top teams used B5 (2nd place, at about 456 px) and B5-B7 only at *low* resolution (4th place: B7 at 224, B6 at 240, B5 at 256) "to limit overfitting". Our own data say resolution gains are small (+0.004 for B4 380→512). B5-NS 456 fits on a T4 (about 3 h for 5 folds). B6 528 only just fits (about 5 h), and B7 600 does not.

Transformers on APTOS sit at QWK 0.89-0.92 in published work, and SE-ResNeXt101 / Inception-ResNet-v2 were members of the winning ensemble (single Inception-ResNet-v2 0.927 private). Expect at most +0.001 to +0.004 to the blend from any new CNN backbone, given the 0.98+ correlation.

### Cited Findings
- **2nd place (0.934).** EfficientNet B3/B4/B5 at 300/456/460 px. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **4th place (0.934).** "More complex models use lower resolution to limit overfitting": B7@224, B6@240, B5@256, B4@320, plus Ben-processed B7 (224) and B5 (256). — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **1st place (0.936).** Inception-ResNet-v2 and Inception-v4 at 512, SE-ResNeXt50 at 512, SE-ResNeXt101 at 384, 2 seeds each. A single Inception-ResNet-v2 from this ensemble scored 0.927 private, so the 8-model ensemble added about +0.009. — [Fang et al. 2023, arXiv 2301.04644](https://arxiv.org/pdf/2301.04644); earlier notes
- **Fang, Kornblith and Schmidt (NeurIPS 2023).** On APTOS, architectures after VGG show "no significant gains" in QWK. The best of 19 ImageNet models on a local 80/20 split was ConvNeXt-Tiny at 0.930 (256 px). — [arXiv 2301.04644](https://arxiv.org/pdf/2301.04644)
- **Swin on APTOS.**
  - Swin-Tiny 0.898 (5 seeds, 224 px).
  - A standalone Swin at 0.912 and an EfficientNet-B4 + Swin hybrid at 0.925 on an APTOS test split.
  - Swin 0.914 with 5-fold CV.
  - All at different splits from ours. — [preprints.org 202608.0899](https://www.preprints.org/manuscript/202608.0899); [IJPS hybrid paper](https://www.ijpsjournal.com/article/a-hybrid-deep-learning-framework-with-explainable-ai-for-robust-detection-and-grading-of-diabetic-retinopathy); [DergiPark](https://dergipark.org.tr/tr/download/article-cite-file/1535708/type/6)
- **Kaggle 54th place (Tymchenko et al.).** A multistage-transfer ensemble reported QWK 0.9255 on the APTOS private test. — [arXiv 2003.02261](https://arxiv.org/abs/2003.02261)
- **T4 throughput estimates.** From the earlier notes (timm RTX-3090 AMP benchmark scaled to our measured B4 rate), per epoch on 2,562 images:

  | Model | Resolution | s/epoch |
  |---|---|---|
  | B4-NS (measured) | 380 | ~50 |
  | B5-NS | 456 | ~100 |
  | B5-NS | 512 | ~130 |
  | B6-NS | 528 | ~175 |
  | B7-NS | 600 | ~300 |

  On a 3090, timm ran B5@456 at batch 32, B6@528 at 16 and B7@600 at 8, so a T4 should fit roughly 2/3 of those batches. — [timm benchmark CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv) (via backbones_resolution.md)

### Inferences
- **Our-protocol cost (5 folds x 20 epochs, scaled from the measured 87 min at B4-NS 380).**

  | Model | Resolution | Minutes | Batch on T4 |
  |---|---|---|---|
  | B5-NS | 456 | ≈ 175 | 12-16 |
  | B5-NS | 512 | ≈ 225 | |
  | B6-NS | 528 | ≈ 300 | 8 (BatchNorm noisy, use accumulation) |
  | B7-NS | 600 | ≈ 520 | 4-6 |

  B7 alone uses the whole budget, so it is not feasible.
- **Expected gain.** The B4-NS 512 arm (OOF 0.9237, correlation 0.989) added +0.0035 to the blend. A B5-NS 456 arm is likely of the same kind, maybe slightly more diverse (a different width/depth): about +0.001 to +0.004 to the blend. Low risk, about 3 h. B6 adds cost without evidence of more gain.
- **Inception-ResNet-v2 / Inception-v4 at 512 (Huber + GeM).** These are the winner's family and differ structurally from EfficientNet. But our SE-ResNeXt50 512 + GeM arm (0.916, correlation 0.957) added only +0.0025. Expect a similar result (about +0.001 to +0.003, about 2-2.5 h).
- **Swin/ViT/BiT.** Published APTOS numbers are below our B4-NS, and windowed attention is awkward at 512. Low priority. I found no APTOS evidence for BiT.

### Gaps
- No APTOS QWK for B5/B6/B7 Noisy Student under a CV protocol at matched resolution. No measured T4 numbers for B5/B6 in our pipeline: the figures above are scaled estimates, so a 1-epoch timing check is needed.
- No APTOS QWK found for BiT-M/L or SE-ResNeXt101 alone.

## 3. Retinal foundation models (RETFound, VisionFM, FLAIR) and DINOv2

### Takeaway
The only APTOS 5-class QWK I found for a retinal foundation model is RETFound-DINOv2 ViT-L/14 at **0.910**. That was on a 733-image test split at 392 px, with only the last two blocks fine-tuned and MSE regression. It is below our single B4-NS (about 0.92 OOF).

Other APTOS results are AUROC only: RETFound 0.943-0.944, DINOv2-L 0.952. RETFound weights are gated on HuggingFace (request form) under CC BY-NC 4.0. Pre-training used about 90% Moorfields and about 10% EyePACS images (self-supervised, no labels).

FLAIR's pre-training includes APTOS itself, so it must be avoided. Full ViT-L fine-tuning on a T4 costs about 70 min per 1,000 images at 224 px for 50 epochs, so 5-fold full fine-tuning is about 6 h at 224 and more at higher resolution. Its likely value is diversity (a different architecture and pre-training) rather than single-model accuracy.

### Cited Findings
- **RETFound-DINOv2 ViT-L/14 on APTOS (grade A-ish, different split).**
  - Recipe: stratified 64/16/20 split (733-image test), retina bounding-box crop, 392×392 input, one-logit MSE regression head. Last two transformer blocks + fc_norm + head trainable (about 25.2M parameters). AdamW for 25 epochs, lr 1e-5 backbone / 1e-3 head, weight decay 0.01, fp16, RTX 4080 Super.
  - 5-class test QWK 0.9101, accuracy 79.54%.
  - No CNN baseline on the same split. — [arXiv 2605.14108](https://arxiv.org/html/2605.14108v1)
- **RETFound in its original paper.** APTOS AUROC 0.943, significantly better than SL-ImageNet. 224 px, ViT-L, batch 16, 50 epochs. Fine-tuning on one T4 took about 70 min per 1,000 images. QWK not reported. — [Zhou et al. 2023, Nature](https://www.nature.com/articles/s41586-023-06555-x)
- **RET-CLIP table (fine-tuning, APTOS2019).** RETFound AUROC 0.943 / AUPR 0.726. DINOv2 0.920 / 0.675. FLAIR 0.932. — [RET-CLIP, arXiv 2405.14137](https://arxiv.org/pdf/2405.14137)
- **DINOv2 vs RETFound.** Fine-tuned DINOv2-large beat RETFound for DR detection across three datasets (AUROC 0.850-0.952 vs 0.823-0.944, all P ≤ 0.007). On APTOS, 0.952 vs 0.944. Kappa values are in a supplement I could not fetch (403). — [Ophthalmology Science 2025](https://www.ophthalmologyscience.org/article/S2666-9145(25)00221-0/fulltext); [preprint summary](https://themoonlight.io/review/is-an-ultra-large-natural-image-based-foundation-model-superior-to-a-retina-specific-model-for-detecting-ocular-and-systemic-diseases)
- **Weights and licence.**
  - RETFound weights (RETFound_mae: natureCFP/natureOCT/meh/shanghai; RETFound_dinov2: meh/shanghai; all ViT-L, about 300M parameters) are on HuggingFace behind an access form. The README fine-tune example is 224 px, batch 24, 50 epochs.
  - The repo also supports DINOv2 ViT-S/B and DINOv3 ViT-S/B as smaller backbones.
  - Licence CC BY-NC 4.0 (earlier notes). — [rmaphoh/RETFound](https://github.com/rmaphoh/RETFound)
- **Contamination.** FLAIR pre-training includes 15_APTOS and 01_EYEPACS (Apache-2.0 weights). VisionFM uses about 3.4M images including large private sets, and its exact sources were not verified. — earlier notes; [jusiro/FLAIR](https://github.com/jusiro/FLAIR)
- **Distillation evidence.** On APTOS-2019 (balanced accuracy), a ConvNeXt-Base teacher gave a stronger student than a RETFound teacher in all four settings. "RETFound distillation can even fall below the no-distillation baseline." — [Informatica article 15303](https://www.informatica.si/index.php/informatica/article/view/15303)

### Inferences
- **Strictness.** RETFound pre-training is self-supervised (MAE/DINOv2) on external fundus images that include EyePACS, with no DR labels used. So it is "external unlabelled fundus images", not external labels. Still report it as a separate non-strict track. FLAIR is excluded because it may have seen our hold-out images.
- **T4 feasibility.**
  - Full ViT-L fine-tuning at 224, 5 folds × about 2,490 images × 20 epochs: scaling 70 min/1,000 images/50 epochs gives ≈ 70 × 2.49 × 0.4 ≈ 70 min per fold, ≈ 6 h for 5 folds.
  - At 392 px (784 tokens vs 196) attention cost grows by more than 4×, so full fine-tuning is infeasible.
  - The partial fine-tune of arXiv 2605.14108 (last 2 blocks) avoids backward passes through the frozen blocks. Its forward cost at 392 px is still about 4× the 224 cost. A rough guess is 2-4 h for 5 folds × 20 epochs. Not measured.
- **Expected gain.** As a single model it is likely 0.90-0.92, below B4-NS. Its use is diversity. Our least-correlated arms (correlation 0.957-0.959) added +0.0009 to +0.0025. A ViT-L with different pre-training might correlate at about 0.93-0.95 and add about +0.002 to +0.005, if its OOF is at least about 0.91. Medium-high risk: gated access delay, CC BY-NC, ViT-L memory at batch 16 on 16 GB.
- **DINOv2-L (Apache-2.0, ImageNet-like natural images).** Matched or beat RETFound on APTOS AUROC and stays inside the "no external fundus data" rule. It is the stricter foundation-model option at the same cost.

### Gaps
- No APTOS QWK for RETFound-MAE, VisionFM, DINOv2-L or FLAIR under a CV protocol. No measured T4 throughput for ViT-L at 392-518 px.
- The VisionFM licence and its inclusion of APTOS/EyePACS images were not verified.
- A search snippet said an independent evaluation found foundation models "do not consistently outperform fine-tuned CNNs" (EfficientNet-B0 first or second on IDRiD, MESSIDOR-2 and OEFI). The source paper was not identified or fetched, so this claim is unverified.

## 4. Loss/target variants, SWA, distillation, seeds, progressive resizing

### Takeaway
None of these has a controlled APTOS result showing more than about +0.005 over a tuned MSE-regression baseline. The top teams used MSE/Huber regression, not fancier ordinal losses.

The cheapest lever with some support is progressive resizing / FixRes-style fine-tuning: short fine-tunes of the existing 380 px checkpoints at 512-576 px. It costs about 35-60 min and should recover most of the +0.004 that full 512 training gave.

Soft-label self-distillation fits naturally with pseudo-labelling (Section 1). SWA on top of EMA 0.998, label smoothing, and CORN/SORD are untested on APTOS and likely ≤ +0.002.

### Cited Findings
- **Losses used by top solutions.** 1st place Huber + GeM. 11th MSE. 28th BCE + clipped MSE (local CV 0.936). — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med); [4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection)
- **Ordinal losses.** No clean held-out gain for CORAL, CORN or SORD over MSE on APTOS was found. The strongest ordinal-loss evidence is CLM + QWK loss vs softmax CE on EyePACS at 128 px (+0.084), not vs MSE. — earlier notes, ordinal_training_techniques.md
- **Progressive resizing on APTOS (grade C, confounded).** A gold-medal team (DrHB) trained 224→352→448 on 2015+2019 data to CV 0.8216 / LB 0.818. Resolution was not isolated. — [DrHB](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION) (via earlier notes)
- **Multi-resolution diversity.** The 2nd place deliberately trained at 300/456/460 px "for feature-map diversity". — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **Train-test resolution discrepancy (FixRes, general ImageNet, grade D transfer).** Augmentation at train time makes apparent object size differ from test time. A short fine-tune at the test resolution fixes this. — [Touvron et al. 2019, arXiv 1906.06423](https://arxiv.org/abs/1906.06423) (not re-fetched in this session)
- **Knowledge distillation on APTOS.** A ConvNeXt-Base teacher gave stronger students than RETFound, and RETFound distillation sometimes fell below the no-distillation baseline (balanced accuracy, small students). — [Informatica 15303](https://www.informatica.si/index.php/informatica/article/view/15303)
  - A 2025 conference abstract claims an ensemble teacher with APTOS QWK 0.975 and a MobileNetV2 student at 0.938. The unusually high teacher number suggests a non-strict split. — [BRAC University repository](https://dspace.bracu.ac.bd/items/87e00522-386b-42a3-ba25-0b7523df3119)
  - In a 2020 VGG study, the student stayed well below the teacher. — [arXiv 2002.03321](https://arxiv.org/pdf/2002.03321)
- **SWA / EMA / label smoothing.** No DR- or APTOS-specific ablation found (earlier notes reached the same conclusion).
- **Seeds and ensembles.**
  - The 1st place used 2 seeds per architecture.
  - The 2nd place used epoch (snapshot) ensembling. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
  - Random-initialisation diversity is "sufficient in practice" for deep ensembles, and correlated members "produce an aggregate close to one model's output". — [Diversity-promoting ensemble, arXiv 2210.12388](https://arxiv.org/pdf/2210.12388); [Deep Ensembles: A Loss Landscape Perspective, 1912.02757](https://alphaxiv.org/abs/1912.02757)
  - Gains flatten past about 16 members in a non-medical study. — [arXiv 2508.09888](https://arxiv.org/pdf/2508.09888)

### Inferences
- **Expected gain from more seeds or members at correlation 0.98-0.99 (grade D).** For an average of M members with pairwise correlation ρ, the variance part of the error scales as ρ + (1−ρ)/M.
  - Each of our blend components is already a 5-fold average. At ρ ≈ 0.985 between components, the remaining reducible variance is only about 1.5% of the member variance.
  - Empirically, adding B4-NS 512 (ρ 0.989, and itself the best single arm) gave +0.0035, and the SE-ResNeXt50 arm (ρ 0.957, weaker) gave +0.0025.
  - So a second seed of an existing arm is likely to add +0.000 to +0.002 (B4-NS 380 seed ≈ 87 min, B4-NS 512 seed ≈ 138 min). Low risk, but it is unlikely to cross 0.93 alone.
- **FixRes-style fine-tune (grade D, cheap).** Take each fold's B4-NS 380 (or B3 384) checkpoint and fine-tune 4-5 epochs at 512-576 px with low lr (for example 3e-5, cosine, EMA kept), with lighter scale augmentation. Then TTA at that size.
  - Cost: 138 min × 5/20 ≈ 35 min for 512, about 50 min for 576.
  - Expected: matches or beats the full 512 run (+0.004 single, +0.001 to +0.003 to the blend). A 576-640 px variant adds some resolution diversity.
  - Requires the per-fold checkpoints to have been saved. That is not verified here.
- **Soft-label self-distillation on train images (1st-place-style label averaging).** Replace each training label with α·label + (1−α)·OOF ensemble prediction (α ≈ 0.5-0.7). The OOF prediction must come from fold-k teachers so that the fold-k validation score stays clean. This acts as label-noise smoothing, which matters because APTOS labels are noisy. Cost is one more training run (about 87 min for B4-NS 380). Expected gain: +0.000 to +0.004, no APTOS evidence. It combines well with pseudo-labelling at no extra cost.
- **Ordinal losses / label smoothing / SWA.** Our auxiliary 5-class head gave only +0.001. Expect ≤ +0.002 from CORN/SORD or label smoothing on the regression target. SWA over the last 5 epochs mostly duplicates EMA 0.998. Cheap to try (no extra training if snapshots exist), but expected gain about 0.
- **Distillation** compresses an ensemble into a single model. It does not raise the ensemble's QWK, so it is irrelevant for the goal unless it is used as the pseudo-label student.

### Gaps
- No APTOS ablation of SWA, label smoothing, CORN/SORD vs MSE, or FixRes fine-tuning.
- Whether Kaggle counts a T4 ×2 session as one or two GPU-hours against the weekly quota was not verified. If it counts as one, running two folds in parallel would roughly double the effective budget.

## 5. Synthesis: ranked list of untried levers (expected OOF gain to the 0.928 blend, T4 cost, risk)

### Takeaway
The best single bet for crossing 0.93 OOF within 9 GPU-hours is **fold-safe soft pseudo-labelling of the 1,928 APTOS test images**: +0.002 to +0.008 expected, about 2.5-5 h, medium risk. Every top-4 APTOS team used it, and it is the only lever that adds new information.

The best cheap companion is a **FixRes-style 512-576 px fine-tune** of existing checkpoints (or **B5-NS 456** if checkpoints are gone): +0.001 to +0.004, about 0.6-3 h, low risk.

On a 550-image test set, none of these gains will be statistically detectable. Judge them on the 3,112-image OOF score.

### Cited Findings
- Top-4 private LB (0.934-0.936) all used test-set pseudo-labelling plus multi-resolution EfficientNet/Inception ensembles. — [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- The only APTOS retinal-foundation-model QWK found is 0.910 (RETFound-DINOv2, 392 px, partial fine-tune). — [arXiv 2605.14108](https://arxiv.org/html/2605.14108v1)
- Ensemble gain of the winner over its best single model is about +0.009. — [Fang et al., arXiv 2301.04644](https://arxiv.org/pdf/2301.04644)

### Inferences
Ranked table (all expected gains are grade-D priors unless noted; costs are 5-fold × 20 epochs on one T4, scaled from our measurements):

| Rank | Lever | Protocol status | Expected Δ OOF (blend) | T4 cost | Risk / notes |
|---|---|---|---|---|---|
| 1 | Soft pseudo-labels on 1,928 APTOS test images, fold-k teachers only, B4-NS 380 + B3 384 students (warm-start 10 epochs, or from scratch 20) | Strict labels + unlabelled competition images (separate row) | +0.002 to +0.008 | ~0.5 h inference + 2.5 h (warm) / 4.5 h (scratch) | Medium: no published ablation, domain-shifted images, OOF leak if one ensemble labels all folds |
| 2 | FixRes-style fine-tune of existing 380/384 fold checkpoints at 512-576 px, 4-5 epochs, low lr | Strict | +0.001 to +0.004 | ~0.6-1 h for two members | Low: needs saved checkpoints |
| 3 | EfficientNet-B5-NS 456 (batch 12-16) as a new blend member | Strict | +0.001 to +0.004 | ~3 h (estimate; time 1 epoch first) | Low-medium: like B4-NS 512 (+0.0035), likely ρ ≈ 0.985 |
| 4 | Soft-label self-distillation of train labels (α·label + (1−α)·fold-safe OOF prediction), combined with #1 | Strict | +0.000 to +0.004 | ~0 extra if folded into #1; else ~1.5 h | Medium: untested on APTOS; may just reproduce the teacher |
| 5 | Second seed of B4-NS 512 (or of B3 384) | Strict | +0.000 to +0.002 | 1.1-2.3 h | Low risk, low ceiling (ρ ≈ 0.99) |
| 6 | Inception-ResNet-v2 / Inception-v4 512, Huber + GeM (1st-place family) | Strict | +0.001 to +0.003 | ~2-2.5 h | Low-medium: our SE-ResNeXt50 analogue gave +0.0025 |
| 7 | DINOv2-L (natural-image, Apache-2.0) or RETFound-DINOv2 (gated, CC BY-NC, external fundus) partial fine-tune at 392 px | DINOv2 strict; RETFound non-strict (external unlabelled fundus) | +0.002 to +0.005 via diversity; single ≈ 0.90-0.92 | ~2-4 h partial; ~6 h+ full at 224 | Medium-high: memory/time unmeasured, access delay for RETFound |
| 8 | EyePACS-2015 pre-train with a corrected low-lr APTOS fine-tune (4th-place style, 25 epochs pre-training) | Non-strict (external labels) | unknown (our 5-epoch version: −0.006) | 3-5 h incl. pre-training | Medium-high; separate track |
| 9 | CORN/SORD/label smoothing/SWA on top of EMA | Strict | ≤ +0.002 | 1-1.5 h each (SWA ~0 if snapshots exist) | Low gain; evidence weak |
| — | Not recommended | | | | B7 600 (~8.7 h alone), B6 528 (~5 h, no evidence beyond B5), Swin/ViT-B at 224-384 (APTOS 0.89-0.92), FLAIR (pre-trained on APTOS, likely leak), distillation to smaller models (cannot beat the teacher ensemble) |

- **Suggested 9-hour plan.**
  1. Lever 1 with warm-started students plus lever 4's soft train labels: about 3-3.5 h.
  2. Lever 2 (or lever 3 if no checkpoints): about 1-3 h.
  3. Re-blend with the existing greedy rule (≥ +0.002).
  4. Keep about 1.5-2 h in reserve for an extra seed of whichever member gains most.

  If both levers hit their median, the blend moves to about 0.928 + 0.004 + 0.002 ≈ 0.934 OOF. The test QWK would probably still be in the 0.915-0.93 range.
- **Context (grade D).** The local CVs reported by Kaggle teams (for example 0.936 for the 28th place) and the winner's 0.936 private LB suggest that about 0.93-0.94 is near the practical ceiling for APTOS-label QWK. Label noise is a plausible cause, and our earlier notes describe conflicting duplicates. A gap to 0.954 that is still open after these levers points more to protocol or split differences than to missing training tricks.

### Gaps
- All expected gains here are priors from competition practice and our own measured deltas. No source isolates the effect of pseudo-labelling, B5, FixRes fine-tuning, seeds or soft labels on APTOS QWK.
- T4 times for B5/B6, ViT-L at 392 px, and pseudo-label inference are estimates, not measurements.
