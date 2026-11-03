# Backbones, pre-training sources and input resolution for APTOS 2019 DR grading (and T4 cost)

Project context (from the assignment, not a source): strict-protocol validation QWK, 1 seed: EfficientNet-B3 300 px 0.916; B3 384 px 0.922; EfficientNet-B4 Noisy Student 380 px 0.927; EfficientNetV2-S IN-21k 384 px 0.910. 5-fold OOF QWK of B4-NS 380 + B3 384 is about 0.925. Measured T4 cost: B3 300 px about 25 s/epoch, B4-NS 380 px about 50 s/epoch (2,562 images, batch 16, AMP). Target paper reports 0.954.

Note on comparability: results below use different splits. "Local validation" means a random split of the 3,662 labelled APTOS training images (like ours). Kaggle private-LB QWK is on the hidden test set, which behaves very differently: CV of about 0.92 went with public-LB scores of about 0.77-0.84 in one top-10 solution, and the winning private score was 0.936.

## 1. Effect of input resolution (224 / 300 / 384 / 512 / 768+), especially for Mild DR

### Takeaway
Larger inputs reliably help DR grading, but with diminishing returns past about 512 px. In the one clean ablation found (Inception-v3, private Finnish data, 2019), 5-class macro-AUC rose 0.901 (256) -> 0.919 (299) -> 0.951 (512) -> 0.961 (1024) -> 0.962 (2095). On APTOS, every top Kaggle solution trained at 384-512 px. No source isolates resolution for Mild DR or microaneurysms on APTOS.

### Cited Findings
- Sahlsten et al. (2019) trained Inception-v3 (ImageNet-pretrained) at 256, 299, 512, 1024 and 2095 px on 41,122 graded fundus images (Digifundus, Finland; 70/10/20 split by patient). Referable-DR AUC: 0.961 / 0.970 / 0.979 / 0.984 / 0.987. 5-class macro-AUC: 0.901 / 0.919 / 0.951 / 0.961 / 0.962. A 6x512 ensemble scored 0.984 / 0.958, matching the 1024 single model. — [Sahlsten et al., PMC6656880](https://pmc.ncbi.nlm.nih.gov/articles/PMC6656880/)
- Cost in the same study: at 2095 px, per-epoch training time was about 20x the 512 baseline. It needed batch size 1, instance norm in place of batch norm, and gradient accumulation over 15 steps. Total training was 383 min for one 2095 model vs about 102 min for six 512 models. The resolution effect was less clear on external Messidor. — [Sahlsten et al., PMC6656880](https://pmc.ncbi.nlm.nih.gov/articles/PMC6656880/)
- Sahlsten et al. also say that larger inputs sharply increase training and inference time and memory, with "minimal performance improvements in most classification tasks" at the very largest size. Under a fixed compute budget, a smaller input with a broader hyperparameter search may be better. — [arXiv 1904.08764 via search summary](https://arxiv.org/pdf/1904.08764)
- APTOS 2019 1st place (Guanshuo Xu, private QWK 0.936) averaged 8 models: 2x inception_resnet_v2 at 512, 2x inception_v4 at 512, 2x seresnext50 at 512 and 2x seresnext101 at 384. He chose large inputs because larger sizes did better in the 2015 competition. Training merged 2015 and 2019 data, with plain resizing and no special preprocessing. — [Kaggle 1st place summary (search snippet; page not fetchable)](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065)
- Fang, Kornblith and Schmidt (NeurIPS 2023) note the 1st-place recipe also used L1 loss, GeM pooling, extra augmentation and multiple input sizes. In their own Kaggle submissions, 512 px plus extra augmentation plus external data took ResNet-50 and Inception-ResNet-v2 to 0.896 private QWK, but resolution was not isolated from the other changes. — [Fang et al. 2023, ar5iv](https://ar5iv.arxiv.org/html/2301.04644)
- A top-10 APTOS 2019 solution (BloodAxe) trained mostly at 512 px (one run at 768). Its best single-model public-LB scores came from SE-ResNeXt50d at 512 px (CV 0.9231 +/- 0.0043, LB 0.813; GAP variant CV 0.9216, LB 0.809). — [BloodAxe/Kaggle-2019-Blindness-Detection](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- Counter-example: a top-3% APTOS write-up reported that its best single model was EfficientNet-B5 at only 224 px. — [MamatShamshiev/Kaggle-APTOS-2019 (search snippet)](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- Microaneurysms and new vessels cover very few pixels, which is why papers argue for high-resolution retinal images. — [Pre-segmenting DR model, PMC11810534](https://pmc.ncbi.nlm.nih.gov/articles/PMC11810534/)
- Mild DR is the weak class even in the target paper's own (EyePACS) results. From its Table 4 (EfficientNet-B3 ensemble, reduced EyePACS), Mild recall is about 23.5% (80/341). Most errors are Mild predicted as No DR (169) or Moderate (91). — [Chilukoti et al. 2024, PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- At 224 px on APTOS (2026), Mild-DR recall was 0.553 +/- 0.004 for EfficientNet-B0 and 0.506 +/- 0.125 for Swin-Tiny. — [Diagnostics 2026, PMC13565497](https://pmc.ncbi.nlm.nih.gov/articles/PMC13565497)

### Inferences
- Our own runs follow the same trend: B3 gains about +0.006 QWK going from 300 to 384. On Sahlsten's curve the biggest step is 299 -> 512 (+0.032 macro-AUC), then 512 -> 1024 (+0.010). So 512 px is the most likely resolution step to pay off. 768+ is probably not worth it on a T4 for APTOS, whose images vary in native resolution and camera.
- APTOS is small (3,662 images). Resolution gains reported on a 41k-image private set may shrink here, because a higher-resolution model has more nuisance detail to overfit to.
- Gains in Mild-DR recall are where resolution should matter most. Track per-class Mild recall and Mild-vs-0 confusion at 384 vs 512, not only QWK. QWK barely penalises 0<->1 confusions.

### Gaps
- I found no controlled resolution ablation on APTOS itself and no per-class Mild-DR or microaneurysm sensitivity vs resolution. The Sahlsten study is from 2019, so slightly outside the preferred window.
- I could not open the Kaggle 1st-place write-up (JS-rendered). Its details come from a search snippet and from Fang et al.'s description.

## 2. Comparative results by architecture (EfficientNet B3-B7, NS, V2, ConvNeXt, SE-ResNeXt, RegNet, Swin/ViT)

### Takeaway
On APTOS, ImageNet accuracy barely predicts downstream QWK. In 19 ImageNet models (NeurIPS 2023) the slope was 0.01 and the correlation 0.06, with nearly everything after VGG in the 0.91-0.93 local-validation band. ConvNeXt-Tiny was best (0.930). EfficientNets were slightly worse than average. Augmentation (+0.03 QWK) and ensembling diverse models matter more than picking the "best" backbone. Our 0.910-0.927 single-model numbers sit exactly in this band.

### Cited Findings
- Fang, Kornblith and Schmidt (NeurIPS 2023): 19 ImageNet-pretrained architectures, 80/20 class-balanced local split of APTOS train (2,930/732), random-resized-crop 256 (224 for DeiT), 30 epochs, batch 128. Validation QWK: ConvNeXt-Tiny 0.9297 (best), VGG-16-BN 0.9287, DenseNet-121 0.9287, PNASNet-5-large 0.9271, VGG-13-BN 0.9268, ViT-B/16 0.9262, EfficientNet-B0 0.9195, Inception-ResNet-v2 0.9168, DeiT-tiny 0.9130, AlexNet 0.8835. Slope vs ImageNet top-1 (models above 70%) was 0.01 and correlation 0.06. "DeiT and EfficientNets performed slightly worse," and deeper models in the same family helped slightly. — [Fang et al. 2023, ar5iv](https://ar5iv.arxiv.org/html/2301.04644)
- Same paper: stronger colour and affine augmentation alone gave about +0.03 QWK on the Kaggle leaderboard. ResNet-50 fine-tuning with RandAugment went 0.9210 -> 0.9250, and to 0.9302 when the ImageNet pre-training also used RandAugment. DeiT showed no consistent augmentation gain. A single Inception-ResNet-v2 trained with the 1st-place interventions reached 0.927 private QWK. — [Fang et al. 2023, ar5iv](https://ar5iv.arxiv.org/html/2301.04644)
- Same paper, CLIP backbones (LP-FT) on APTOS validation: CLIP-RN50 0.9135, CLIP-B/16 0.9213, CLIP-L/14 0.9330, CLIP-L/14@336 0.9337. Like-for-like, CLIP trailed ImageNet (RN50 0.9135 vs 0.9215; ViT-B/16 0.9213 vs 0.9262). Only the largest CLIP models stood out. — [Fang et al. 2023, ar5iv](https://ar5iv.arxiv.org/html/2301.04644)
- Kaggle 2019 top solutions used Inception/SE-ResNeXt (1st), SE-ResNeXt and SENet154 with MSE loss (11th) and SE-ResNeXt50/101, ResNeXt50, DenseNet201 and InceptionV4 (top-10, BloodAxe). Another top-3% team found EfficientNets best. — [Kaggle 1st place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065); [11th place, 4uiiurz1](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection); [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection); [MamatShamshiev](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)
- BloodAxe (top-10) 5-fold CV QWK: SE-ResNeXt50 classification 0.9213, SE-ResNeXt50 regression 0.9225, ResNeXt50 regression 0.9234, SE-ResNeXt50d GWAP 512 px 0.9231, DenseNet201 regression 0.9018, InceptionV4 regression 0.9053. Regression beat classification slightly. — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- Diagnostics (2026), ImageNet weights, 224 px, 5 seeds, APTOS test-split QWK: Swin-Tiny 0.898 +/- 0.010, Swin-Base 0.897 +/- 0.015, ViT 0.876 +/- 0.010, EfficientNet-B0 0.850 +/- 0.032, MobileNetV2 0.817 +/- 0.015, ResNet50 0.801 +/- 0.013. Short training (max 20 epochs, weighted CE, AdamW 1e-4, A100). — [PMC13565497](https://pmc.ncbi.nlm.nih.gov/articles/PMC13565497)
- 2026 CNN-transformer ensemble paper: 5-fold stratified CV on APTOS with ResNet-50, ConvNeXt-Tiny, Swin, DeiT-Small, EfficientNet-B3 and DenseNet-121, selected by QWK. The best ensemble reached QWK of about 0.934. Single-model values were not retrievable (PDF not parseable). — [arXiv 2604.23079 / Expert Systems with Applications (search snippet)](https://www.sciencedirect.com/science/article/pii/S0957417426027569)
- Survey (2022) of an APTOS experiment: ImageNet init gave ResNet50 0.893 and DeiT-S 0.896 kappa. Random init dropped to 0.849 and 0.687, so ViTs depend far more on pre-training. — [Transformers in Medical Imaging survey, arXiv 2201.09873 (search snippet)](https://arxiv.org/pdf/2201.09873)
- Retina-wide comparison (Nov 2025, 5-class ICDR on a 143,669-image "EAM" pool, possibly EyePACS+APTOS+Messidor): SwinV2-tiny (27.6M) reached 69.61% acc / AUROC 0.9202, within about 1.5 points of RETFound-DINOv2 (303M, 71.15% / 0.9323). ConvNeXtV2-tiny reached 67.46% / 0.9176 and Swin-tiny 67.19% / 0.9188. — [Isztl et al., arXiv 2511.22001](https://arxiv.org/html/2511.22001v1)
- Target paper (Chilukoti et al. 2024) compared VGG, ResNet-18 to 152 and EfficientNet-B0 to B6 only on reduced EyePACS, not APTOS. Its APTOS number (0.954) is for EfficientNet-B3 at 300 px with CLAHE + Gaussian blur. It does not report the APTOS split. — [PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)

### Inferences
- The literature does not support big gains from swapping EfficientNet-B3 for a newer CNN or ViT. Expect differences of about +/-0.01 QWK, which is within seed noise on a 732-image validation set. The best reason to add architectures is ensemble diversity (CNN + ConvNeXt/Swin, different input sizes). That is what the 1st place and the 2026 ensemble (about 0.934) did.
- ConvNeXt-Tiny/Small and SE-ResNeXt50 at 448-512 are the most evidence-backed "diverse" additions. Swin/SwinV2-Tiny is the best-supported transformer, but its fixed window sizes make non-native resolutions (such as 512) more awkward than for CNNs.
- No source gives APTOS results for EfficientNet-B5 to B7 Noisy Student, EfficientNetV2-M or RegNet under a comparable protocol. Our own B4-NS > B3 > V2-S ordering is as good as any published evidence.

### Gaps
- No APTOS-specific numbers found for RegNetY, EfficientNetV2-M/L, or B5-B7 (ImageNet or NS) under a controlled protocol.
- Papers with Code's APTOS 2019 leaderboard was not checked (not fetched). Many reported APTOS QWKs above 0.95 come from non-comparable random splits.

## 3. Retinal foundation models (RETFound, VisionFM, FLAIR, others) and DINOv2-style weights

### Takeaway
Retinal FMs are all pre-trained on fundus data that includes Kaggle EyePACS, and FLAIR's pre-training also includes APTOS itself. Using any of them counts as external fundus data, and FLAIR would leak the test domain. They run at 224 px (ViT-L, 300M params). On APTOS they are reported only as AUROC (RETFound 0.943). Generalist DINOv2-L matched or beat RETFound on APTOS (AUROC 0.952 vs 0.944). Under a strict ImageNet-only protocol, DINOv2 (Apache-2.0, natural images only) is the only clean option.

### Cited Findings
- RETFound (Nature, Sept 2023): CFP model pre-trained with MAE on 904,170 colour fundus photos. 815,468 (90.2%) come from Moorfields MEH-MIDAS and 88,702 (9.8%) from Kaggle EyePACS. APTOS-2019 is used only downstream. APTOS AUROC was 0.943 (95% CI 0.941-0.944), significantly better than SL-ImageNet. QWK is not reported. — [Zhou et al. 2023, PMC10550819](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550819/)
- RETFound fine-tuning setup: images resized to 256, random crops to 224, ViT-L/16, batch 16, 50 epochs (10 warm-up + cosine). It was fine-tuned on one NVIDIA Tesla T4 (16 GB) at "about 70 minutes per 1,000 images". Pre-training took 8x A100-40GB for about 14 days (800 epochs, batch 1,792). — [PMC10550819](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550819/)
- The RETFound repo now offers RETFound-MAE (natureCFP, natureOCT, meh, shanghai) and RETFound-DINOv2 (meh, shanghai) ViT-L weights (about 300M params), described as pre-trained on 1.6M retinal images. It also supports fine-tuning DINOv2/DINOv3. The example config uses input 224, batch 24, 50 epochs, 1 GPU. — [rmaphoh/RETFound](https://github.com/rmaphoh/RETFound)
- RETFound licence: the repo LICENSE is Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0); GitHub reports "Other". — [rmaphoh/RETFound LICENSE](https://github.com/rmaphoh/RETFound)
- DINOv2 vs RETFound (Ophthalmology Science, 2025): after fine-tuning on APTOS-2019, DINOv2-large reached AUROC 0.952 vs RETFound 0.944. Accuracy, kappa and F1 "consistently showed that DINOv2 models outperformed RETFound in most scenarios" (kappa values are in the supplement, not retrieved). Trained on IDRiD and tested on APTOS: DINOv2-L 0.783, DINOv2-S 0.808, RETFound 0.752. — [Ophthalmology Science 2025 (search snippet; page 403)](https://www.ophthalmologyscience.org/article/S2666-9145(25)00221-0/fulltext)
- Contrary view (arXiv 2509.03421, 2025): RETFound-DINOv2 beat DINOv2 and DINOv3 on ocular disease detection and oculomics. The authors conclude specialist models remain strongest, while the gap narrows with scale. — [arXiv 2509.03421 (search snippet)](https://arxiv.org/abs/2509.03421v1)
- Isztl et al. (Nov 2025): RETFound-DINOv2-CFP was the top DR model (71.15% acc, AUROC 0.9323). DR grading was the only task where retina-specific pre-training clearly beat ImageNet init, and SwinV2-tiny came within about 1.5 points at about 1/11 the parameters. Recipe: AdamW, effective batch 256 via gradient accumulation, 100 epochs, bf16. — [arXiv 2511.22001](https://arxiv.org/html/2511.22001v1)
- FLAIR (MedIA 2025): language-image model pre-trained on an assembly of 37-38 open fundus datasets. The list includes 01_EYEPACS and 15_APTOS, plus IDRID, DeepDRiD, OIA-DDR, MESSIDOR, BRSET and others. Code and weights are Apache-2.0. — [jusiro/FLAIR](https://github.com/jusiro/FLAIR); [FLAIR paper, arXiv 2308.07898](https://arxiv.org/pdf/2308.07898)
- UrFound (2024) built its pre-training set from FLAIR's collection (187,270 public CFP+OCT images), which lists EyePACS. APTOS (3,662 images) appears as a DR dataset. — [UrFound, arXiv 2408.05618 (search snippet)](https://arxiv.org/pdf/2408.05618)
- VisionFM (NEJM AI, 2024; preprint arXiv 2310.04992): pre-trained on about 3.4M images from more than 500,000 individuals across 8 ophthalmic modalities. Evaluated on 53 public and 12 private datasets with average AUROC 0.950 across 8 disease categories. It reportedly graded DR at above 90% accuracy on an unseen modality (news report, not the primary paper). The repo licence shows as "Other" on GitHub (not inspected further). — [CUHK news](https://www.erg.cuhk.edu.hk/erg/node/2801); [arXiv 2310.04992](https://arxiv.org/abs/2310.04992v1); [ABILab-CUHK/VisionFM](https://github.com/ABILab-CUHK/VisionFM)
- DINOv2 is Apache-2.0 and pre-trained on natural images (LVD-142M; no fundus). The DINOv3 repo licence shows as "Other" on GitHub. — [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2); [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- A 2026 paper cites RETFound's 0.943 APTOS AUROC. Its own DINOv2-LoRA scored 0.907 AUROC on external APTOS, and it warns that cross-paper comparisons are unreliable because protocols differ. — [arXiv 2608.28207](https://arxiv.org/html/2608.28207)

### Inferences
- External fundus data, by model: RETFound includes EyePACS 2015, so it counts as external fundus data. FLAIR (and FLAIR-derived UrFound) includes EyePACS and APTOS, so it counts as external data and is a likely leak of APTOS images and labels via text prompts. Avoid FLAIR for any APTOS evaluation. VisionFM includes large private fundus collections, so it counts as external data. DINOv2/DINOv3 and CLIP are not fundus-pretrained, but they are not "ImageNet-only" either.
- RETFound's reported numbers are AUROC at 224 px, so they cannot be compared directly with QWK 0.954. The best-supported claim is that RETFound/DINOv2-L are competitive with, not dramatically better than, strong ImageNet CNNs on APTOS. Since 224 px is low for microaneurysms, the ViT-L FMs trade resolution for pre-training.
- For a non-commercial reproduction, CC BY-NC 4.0 (RETFound) is usable. It should still be reported as a separate "external fundus pre-training" track, not mixed with the ImageNet-only protocol.

### Gaps
- No source gives APTOS QWK for RETFound, VisionFM or FLAIR under a protocol like ours. The DINOv2-vs-RETFound kappa values are in a supplement I could not access (403).
- The VisionFM licence text and its exact pre-training sources (whether they include EyePACS/APTOS) were not verified.
- DINOv2/DINOv3 fine-tuning at more than 224 px on fundus (for example 448 or 518 px with patch-14) was not found in a DR context.

## 4. EyePACS (Kaggle DR 2015) pre-training: reported gains on APTOS

### Takeaway
EyePACS is the standard external source. The 2019 winner and most top-10 teams trained on 2015+2019 data. But the controlled gain reported by the target paper itself is tiny: 0.954 -> 0.956 QWK for EfficientNet-B3 (0.967 only with an added ensemble). Top-10 CV evidence is mixed, with lower CV but higher LB.

### Cited Findings
- Chilukoti et al. (BMC Med Inform Decis Mak, 2024), Table 6 on APTOS: 0.954 (no DR pre-training, no ensemble), 0.956 (EyePACS-pretrained, no ensemble), 0.967 (EyePACS-pretrained + snapshot ensemble). No "ImageNet + ensemble" APTOS baseline is reported, and the APTOS split is not reported. Hardware: Colab Pro, GPU unspecified. — [PMC11323616](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323616/)
- An independent reproduction of that paper reports that the paper's recipe gave QWK 0.82 vs the reported 0.954, raised to 0.908 with fixes. — [devxsubh/aptos-dr-grading-reproduction](https://github.com/devxsubh/aptos-dr-grading-reproduction)
- APTOS 1st place: validating on 2019 train after training on 2015 data did not track the public LB, so Xu merged 2015+2019 data into one training set and validated on the public LB. Fang et al. describe the external data as 88,702 images from the 2015 competition. — [Kaggle 1st place (snippet)](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065); [Fang et al. 2023](https://ar5iv.arxiv.org/html/2301.04644)
- BloodAxe top-10: adding Aptos2015 + IDRiD + Messidor lowered ResNeXt50 CV (0.9234 -> 0.9129) but raised public LB (0.791 -> 0.804). — [BloodAxe](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)
- A 10th-place solution's run with Aptos2015 extra data had CV 0.9244 and public LB 0.752, below its best variant. — [search summary of 10th place code](https://github.com/abhuse/aptos-retinopathy-detection)
- Fang et al.: combining 512 px, extra augmentation and external 2015 data took their submissions up by at least 0.03 QWK on the private LB. The effect of external data alone was not isolated. — [Fang et al. 2023](https://ar5iv.arxiv.org/html/2301.04644)

### Inferences
- On a held-out split of APTOS train (our protocol), EyePACS pre-training is likely to add only about 0.00-0.01 QWK, judging by the paper's own +0.002 and BloodAxe's lower CV. Its main documented benefit is robustness on the shifted Kaggle test set. It should not be counted on to close a 0.03 gap, and it violates a strict "ImageNet-only, no external fundus data" protocol.
- EyePACS labels are noisy and its image style differs from APTOS. Two-stage use (pre-train on EyePACS, then fine-tune on APTOS) is safer than naive merging when validation is on APTOS.

### Gaps
- I found no controlled same-architecture, same-split ImageNet-vs-EyePACS-init ablation on APTOS other than Chilukoti's Table 6.
- The SSiT (2022) and lesion-based contrastive self-supervised EyePACS pre-training results on APTOS were not retrieved.

## 5. Practical compute on a T4 (16 GB) at 384-512 px

### Takeaway
Our measured T4 rates match timm's RTX-3090 AMP training benchmark divided by about 3.7. Scaling by pixel count, a 512 px EfficientNet-B3 is about 75 s/epoch and B4 about 90 s/epoch on 2,562 images. B5 at 456-512 is about 100-130 s, ConvNeXt-Small or SE-ResNeXt101 at 512 is about 100-130 s, and B7 is 3-5 min/epoch. A 5-fold x 30-epoch run of B4 at 512 is therefore about 4 GPU-hours, within one Kaggle session. Batch 8-16 with AMP fits for B3-B5 at 512, and gradient checkpointing or accumulation covers larger models.

### Cited Findings
- timm training benchmark (RTX 3090, AMP, NCHW, PyTorch 1.12; columns = train samples/s at the given batch and image size): efficientnet_b3 456.8 (b128, 288); tf_efficientnet_b3(_ns) 370.1 (b128, 300); tf_efficientnet_b4(_ns) 194.2 (b64, 380); tf_efficientnet_b5(_ns) 92.1 (b32, 456); tf_efficientnet_b6(_ns) 54.2 (b16, 528); tf_efficientnet_b7(_ns) 31.4 (b8, 600); tf_efficientnetv2_s 389.1 (b128, 300); tf_efficientnetv2_m 142.4 (b48, 384); convnext_tiny 813.2 (b256, 224); convnext_small 489.8 (b192, 224); convnext_base 337.0 (b128, 224); seresnext50_32x4d 613.6 (b256, 224); seresnext101_32x4d 374.7 (b192, 224); regnety_032 696.6, regnety_040 640.2, regnety_080 385.8 (224); swin_tiny_224 646.8; swin_base_224 279.7; swin_base_window12_384 74.0 (b32, 384); vit_base_patch16_224 390.9; vit_base_patch16_384 97.0 (b32, 384); inception_v4 316.9 (299); inception_resnet_v2 261.4 (299). — [timm results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv)
- The timm benchmark batch sizes show the memory scaling. On a 24 GB 3090 with AMP, B5@456 ran at batch 32, B6@528 at 16, B7@600 at 8, Swin-B@384 and ViT-B@384 at 32. — [timm benchmark CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv)
- RETFound (ViT-L, 224 px) fine-tuning on a single T4 was about 70 minutes per 1,000 images (for its 50-epoch schedule, batch 16). — [PMC10550819](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550819/)
- Gradient checkpointing trades extra compute for memory. It can cut activation memory by 50% or more, at roughly 20% extra time, and gradient accumulation preserves the effective batch. For EfficientNet-B5-class models, activations (not weights) dominate memory. — [HF single-GPU training guide](https://huggingface.co/docs/transformers/en/perf_train_gpu_one); [aman.ai primer](https://aman.ai/primers/ai/grad-accum-checkpoint/)
- Sahlsten et al. needed batch size 1, instance norm and 15-step accumulation at 2095 px for Inception-v3. — [PMC6656880](https://pmc.ncbi.nlm.nih.gov/articles/PMC6656880/)

### Inferences
- Calibration check: our B3@300 rate (2,562/25 s = about 102 img/s) vs timm 370 img/s gives about 3.6x. Our B4-NS@380 rate (about 51 img/s) vs 194 gives about 3.8x. So T4 ≈ 3090/3.7 for these models (data loading included).
- Estimated T4 epoch times for 2,562 images, assuming cost scales with pixel count. These are rough and unvalidated, so measure before planning around them:

  | Model | Resolution | Estimated s/epoch |
  |---|---|---|
  | B3 | 512 | ~73 |
  | B4-NS | 448 | ~70 |
  | B4-NS | 512 | ~90 |
  | B5-NS | 456 | ~100 |
  | B5-NS | 512 | ~130 |
  | B6-NS | 528 | ~175 |
  | B7-NS | 512 | ~220 |
  | B7-NS | 600 | ~300 |
  | V2-M | 480 | ~105 |
  | ConvNeXt-Tiny | 512 | ~60 |
  | ConvNeXt-Small | 512 | ~100 |
  | SE-ResNeXt50 | 512 | ~80 |
  | SE-ResNeXt101 | 512 | ~130 |
  | Swin-B | 384 | ~130 |
  | ViT-B/16 | 384 | ~100 |

  Transformers scale worse than linearly in pixels, so their numbers at 512 are optimistic. RETFound's own T4 figure suggests ViT-L@224 is about 200 s/epoch for 2,562 images.
- Memory (16 GB T4 vs 24 GB 3090): roughly 2/3 of the timm batch sizes should fit. That means B5@456 about batch 16-20, B5@512 about 12-16, B6@528 about 8-10, B7@600 about 4-6 (use accumulation or checkpointing). B3/B4@512 at batch 16 should fit with AMP. BatchNorm statistics get noisy below about 8 per batch, so consider freezing BN stats or using GroupNorm/accumulation for B6/B7.
- Budget: Kaggle offers T4 x2 sessions, but our notebooks use one GPU. 5 folds x 25 epochs x 90 s ≈ 3.1 h for B4@512, and ≈ 4.5 h for B5@512. Testing 2-3 backbones at 512 in 5-fold CV is feasible within weekly Kaggle quotas. B7 at 600 in 5-fold is not practical.
- Priority suggested by the evidence: (1) B4-NS and B3 at 448-512 with stronger augmentation (the best-documented single change, +0.03 QWK in Fang et al.). (2) Add 1-2 architecturally different members (ConvNeXt-Small or SE-ResNeXt50 at 512, maybe Swin-T) for ensemble diversity. (3) Only then consider external-data tracks (EyePACS, RETFound), reported separately from the ImageNet-only protocol.

### Gaps
- I found no published T4-specific per-epoch or memory measurements at 384-512 px for these backbones. The table above is extrapolated from timm 3090 numbers using our two measured calibration points.
- Kaggle's GPU quota and session limits were not verified from an official page in this research pass.
