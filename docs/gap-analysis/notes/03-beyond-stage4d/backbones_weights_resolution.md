# Backbones, pre-trained weights and input resolution beyond Stage 4d (APTOS 2019, strict protocol)

Scope note (from the assignment, not a source): the strict best is test QWK 0.918 [0.900, 0.935] and OOF 0.928. It comes from a 5-fold blend of B4-NS 380 + B3 384 + B4-NS 512 (regression head, Ben Graham preprocessing, lr 3e-4, 20 epochs, EMA, 8-view TTA). These already failed: EfficientNetV2-S IN-21k 384 (val 0.910), ConvNeXt-Tiny IN-1k 448 (OOF 0.897) and SE-ResNeXt50 512 (OOF 0.916).

This file adds to `APTOS QWK gap after Stage 4c/remaining_levers.md` and `Closing the APTOS 2019 QWK gap/backbones_resolution.md` and does not repeat them. Already covered there:
- Fang et al.'s headline numbers.
- Top Kaggle solutions.
- RETFound and DINOv2 AUROC.
- RETFound-DINOv2 at 0.910.
- The Sahlsten resolution curve.
- timm T4 throughput estimates.
- B5/B6/B7 cost estimates.

New here:
- Fang et al.'s per-model grid and lr sensitivity, including EfficientNet-B4 itself.
- SSiT's APTOS table, which uses the same 70/15/15 split shape as ours.
- nnMobileNet's 5-fold APTOS result and its recipe ablation.
- DINOv3 / DINOv2-g benchmarking.
- A conflicting RETFound-DINOv2 QWK.
- Further resolution evidence above 512 px.
- Protocol-status flags for each weight family.

Comparability legend:
- **[local split]**: random split of the 3,662 labelled images.
- **[5-fold CV]**: cross-validation on the labelled images.
- **[private LB]**: Kaggle hidden test set, which is distribution-shifted.
- **[other dataset]**: not APTOS.

## 1. Natural-image backbones with reported APTOS QWK, and best value per T4 GPU-hour

### Takeaway
Most of the named newer backbones have no published APTOS QWK under a stated protocol. I found none for EfficientNet-B5/B6/B7-NS, EfficientNetV2-M/L, ConvNeXt V2, ConvNeXt-S/B IN-22k, Swin V2, MaxViT, EVA-02, BEiT, DeiT III, CoAtNet, RegNetY or NFNet. The literature that does exist is one controlled sweep, Fang et al. (19 ImageNet models at 256 px, per-model tuned grid, [local split] 80/20). It ranks ConvNeXt-Tiny (0.9297) above EfficientNet-B4 (0.9211) by +0.009, and puts everything after VGG inside a band of about 0.92-0.93.

So the expected single-model gain from swapping B4-NS for any newer supervised backbone is about 0 to +0.01. The blend gain is smaller still (+0.001 to +0.004, as the earlier notes estimated). Per T4 GPU-hour, the best untested strict options are:
1. EfficientNet-B5-NS 456: same family, cheapest risk.
2. ConvNeXt-Small/Base IN-22k with a re-tuned recipe (see Section 5).
3. EfficientNetV2-M IN-21k 384-448.

### Cited Findings
- **Fang, Kornblith and Schmidt (NeurIPS 2023), APTOS [local split] 80/20 class-balanced, 256 px random-resized-crop (min area 0.65), horizontal flip only, batch 128, 30 epochs.** Each architecture was tuned over a full grid:
  - Optimizers: AdamW-cosine, SGD-cosine and SGD-step.
  - lr: SGD 1e-1 to 1e-4; AdamW 1e-2 to 1e-5.
  - Weight decay: {1e-3, 1e-4, 1e-5, 1e-6, 0}.

  Best QWK per model: ConvNeXt-Tiny 0.9297, VGG-16-BN 0.9287, DenseNet-121 0.9287, PNASNet-5-large 0.9271, VGG-13-BN 0.9268, ViT-B/16 0.9262, MobileNetV3-small 0.9230, ResNet-152 0.9228, ShuffleNetV2x0.5 0.9227, ResNet-50 0.9215, ResNeXt-50 0.9212, **EfficientNet-B4 0.9211**, DeiT-small 0.9205, EfficientNet-B0 0.9195, Inception-ResNet-v2 0.9168, DeiT-tiny 0.9130, AlexNet 0.8835. Swin was not evaluated. The per-model chosen lr/wd sit in an external spreadsheet, not in the paper. — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Same paper, augmentation × pre-training ([local split]).**
  - ResNet-50: base aug 0.9210, RandAug fine-tune 0.9250. With RandAug-pre-trained ImageNet weights plus RandAug fine-tune: 0.9302.
  - DeiT-small: base 0.9293, RandAug 0.9277 (no gain). — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Inference-relevant point.** The pre-training recipe of the *ImageNet* weights (RandAug-trained ResNet-50, 78.8% top-1) gave +0.009 over the standard torchvision ResNet-50 at fixed downstream augmentation. That is a protocol-compliant "better ImageNet weights" lever. — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **nnMobileNet (CVPRW 2024), APTOS [5-fold CV] following the MIL-VT split.**
  - Kappa: nnMobileNet 0.934, MIL-VT 0.920, GREEN-SE-ResNeXt50 0.912, GREEN-ResNet50 0.908, CANet 0.900.
  - Recipe: cross-entropy, 1000 epochs, batch 32, lr 1e-3 cosine with 20 warm-up epochs, weight decay 0.05, AdamP, heavy augmentation, RTX 3090.
  - Input resolution is not stated on the HTML page, and the pre-training source of the weights was not confirmed. — [Zhu et al., arXiv 2306.01289v4](https://arxiv.org/html/2306.01289v4)
- **Hybrid CNN-ViT evidence paper (2025), APTOS [local split] 7:1:2.** Kappa: ResNet-50 0.870, PVT v2-B2 0.896, PVT v2-B3 0.889, STViT-Base 0.883, SMT-Large 0.873, HiFuse-Tiny 0.789, the authors' hybrid 0.911. Recipe: SGD lr 1e-3, polynomial decay, 500 epochs, augmentation only random crop + noise. These baselines are far below Fang's tuned numbers (ResNet-50 0.870 vs 0.9215), which shows how much recipe dominates backbone in published APTOS tables. — [Qiu et al., arXiv 2510.26315](https://arxiv.org/pdf/2510.26315)
- **No APTOS QWK found for Swin V2, MaxViT or ConvNeXt-Base.** A targeted search found nothing. One 2026 paper (PRISM-DR) uses a ConvNeXtV2-Base backbone on APTOS 2019, but the page returned 403 and its numbers and protocol could not be read. — [Sensors 26(14):4515](https://doi.org/10.3390/s26144515)
- **Low-data fine-tuning benchmark (non-medical-specific).** ConvNeXt, RegNet and EfficientNet "perform well compared to others on a diverse set of domains consistently". Attention-based architectures "tend to perform poorly under low data finetuning tasks compared to CNNs." — [Which Backbone to Use, arXiv 2406.05612](https://arxiv.org/abs/2406.05612)
- **Trachoma fundus-adjacent comparison [other dataset].** DINOv2 ViT-B/14 beat EVA02-Base, ConvNeXt-Tiny and ConvNeXtV2-Base on accuracy (91.66%) and AUC (96.06%). No QWK. — [arXiv 2609.23832](https://arxiv.org/pdf/2609.23832) (search summary only; not fetched)
- **Throughput (from the earlier notes, timm RTX-3090 AMP train benchmark).**

  | Model | Resolution | Samples/s |
  |---|---|---|
  | tf_efficientnetv2_m | 384 | 142 |
  | tf_efficientnet_b4_ns | 380 | 194 |
  | tf_efficientnet_b5_ns | 456 | 92 |
  | convnext_small | 224 | 490 |
  | convnext_base | 224 | 337 |
  | regnety_080 | 224 | 386 |
  | swin_base_window12 | 384 | 74 |

  — [timm benchmark CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv)

### Inferences
- **Expected single-model Δ over B4-NS (grade D unless noted).**
  - ConvNeXt (properly tuned) about +0.00 to +0.01. This is grade B: Fang's +0.009 at 256 px for ConvNeXt-T vs B4, both IN-1k-supervised and grid-tuned. Note that Fang's B4 was plain IN-1k, not Noisy Student.
  - EfficientNet-B5-NS 456: about +0.000 to +0.005.
  - EfficientNetV2-M IN-21k: about −0.01 to +0.005, because V2-S already underperformed for us (0.910).
  - Swin/SwinV2/MaxViT/DeiT III/BEiT/EVA-02 at ≤ 384 px: about −0.02 to +0.005, given that published APTOS transformer numbers cluster at 0.89-0.92 and "attention models do worse in low data".
  - RegNetY/NFNet: no evidence at all.
- **Per GPU-hour on a T4, scaled from our measured B4-NS 380 cost of about 87 min per 5 folds × 20 epochs:**
  - EfficientNetV2-M 384: about 1.4× → about 2 h.
  - B5-NS 456: about 2.1× → about 3 h (as in the earlier note).
  - ConvNeXt-Small at 384: 490 samples/s at 224 ÷ (384/224)² ≈ 167 samples/s, vs 194 for B4-NS 380. That is about 1.2× B4 → about 1.7 h. ConvNeXt-Base 384 is about 1.7× → about 2.5 h. These are pixel-scaled estimates; time 1 epoch first.
  - Swin-B 384 is about 2.6× B4 → about 4 h, for the weakest evidence.
- **Best value.** Re-running ConvNeXt with a corrected recipe (Section 5) is likely the best value. Our ConvNeXt-T (0.897 OOF) underperformed Fang's tuned ConvNeXt-T relative to B4 by about 0.04 in relative terms (−0.031 vs B4-NS for us, +0.009 vs B4 for Fang). That gap is larger than any expected backbone effect and points to a recipe problem, not an architecture limit.
- **Better ImageNet weights.** Weights trained with stronger ImageNet recipes are protocol-compliant by construction. Examples: timm "a1/a2" ResNet-50 recipes, RandAug-trained checkpoints, ConvNeXt IN-22k→1k. Fang's +0.009 for RandAug-pre-trained ResNet-50 suggests this is a real and cheap lever.

### Gaps
- No APTOS QWK under any stated protocol for B5/B6/B7-NS, EfficientNetV2-M/L, ConvNeXt V2, ConvNeXt-S/B IN-22k, Swin V2, MaxViT, EVA-02, BEiT, DeiT III, CoAtNet, RegNetY, NFNet or Inception-ResNet-v2 at ≥ 384 px. The only Inception-ResNet-v2 numbers are 0.9168 at 256 [local split] and 0.927 [private LB] with the 1st-place recipe.
- The arXiv 2604.23079 5-fold CNN+transformer ensemble paper (about 0.934 ensemble) returned 404 in HTML. Its single-model CV values remain unretrieved.
- nnMobileNet's resolution and pre-training source are not confirmed.
- Measured T4 times for ConvNeXt-S/B and EfficientNetV2-M at 384-512 are not available.

## 2. Self-supervised / web-scale natural-image weights (DINOv2, DINOv3, MAE, CLIP/OpenCLIP, SigLIP) vs supervised ImageNet on APTOS

### Takeaway
The only APTOS **QWK** comparison of web-scale weights against supervised ImageNet under one protocol is Fang et al.'s CLIP table ([local split], LP-FT). CLIP-L/14@336 reached **0.9337** and CLIP-L/14 0.9330, the two highest numbers in that paper. That is +0.0126 over EfficientNet-B4 (0.9211) and +0.004 over ConvNeXt-T. Like-for-like smaller CLIP models trailed their ImageNet counterparts.

For DINOv2/DINOv3 there are only AUROC results:
- DINOv2-L beats RETFound on APTOS (0.952 vs 0.944).
- DINOv3-ViT-L and DINOv2-g are the strongest generalists in a 10-task retinal benchmark.

No APTOS QWK was found for DINOv2/v3, natural-image MAE, OpenCLIP or SigLIP.

Protocol status:
- None of these are trained on ImageNet alone.
- Noisy Student weights (already in our blend) also use unlabelled non-ImageNet images.
- The rule "ImageNet-type" therefore needs an explicit decision. The flags are below; this note does not decide.

### Cited Findings
- **Fang et al., CLIP LP-FT on APTOS ([local split], same 256-ish pipeline):** CLIP-RN50 0.9135, CLIP-RN101 0.9235, CLIP-B/32 0.9137, CLIP-B/16 0.9213, CLIP-L/14 0.9330, CLIP-L/14@336 0.9337. The ImageNet equivalents were ResNet-50 0.9215 and ViT-B/16 0.9262. — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Hou et al. (2025; Ophthalmology Science), DINOv2-S/B/L vs RETFound.**
  - Recipe: MLP head on DINOv2 features, full fine-tune, batch 32, 100 epochs (10 warm-up to lr 5e-3, cosine to 1e-6), AdamW with weight decay 0.05. Augmentation: random crop, horizontal/vertical flips, rotation, brightness/contrast. Checkpoint chosen by validation AUROC.
  - APTOS internal AUROC: DINOv2-L 0.952 (0.950-0.954) vs RETFound 0.944 (0.941-0.946), P < 0.001.
  - Accuracy, kappa and F1 "consistently largely showed that DINOv2 models outperformed RETFound", but the kappa values are only in the supplement.
  - Label efficiency: on APTOS, DINOv2 and RETFound were "comparable" across 10-100% training data. — [Hou et al., arXiv 2502.06289](https://arxiv.org/pdf/2502.06289)
- **Zhou et al. (2025), generalist vs specialist, includes DINOv3.**
  - Recipe: 224 px (resize 256, random-resized-crop 20-100%), batch 24, 50 epochs, lr warm-up to 5e-4 over 10 epochs then cosine to 1e-6. AutoMorph preprocessing, 55:15:30 split. Metrics are AUROC/AUPRC/F1 only (no kappa).
  - Average AUROC across 10 tasks: DINOv2-g 0.800, DINOv3-ViT-L 0.816, RETFound-DINOv2 0.830, RETFound-MAE 0.809.
  - Within DINOv2 scales, ViT-g (0.800) was clearly above S/B/L (0.746/0.727/0.731).
  - On APTOS-2019 adaptation, DINOv2-g needed about 3.5× the GPU memory and 2× the run time of the ViT-L models. — [Zhou et al., arXiv 2509.03421](https://arxiv.org/pdf/2509.03421)
- **SSiT paper, APTOS [local split] 70/15/15, 384 px fine-tune (pre-training 224 px).** All SSL methods were pre-trained on EyePACS without labels, so these are *in-domain* SSL numbers, not natural-image SSL.
  - Kappa: random-init ViT-S 0.709, MAE ViT-S 0.910, DINO ViT-S 0.905, MoCo-v3 ViT-S 0.902, SimCLR ViT-S 0.903, EsViT 0.910, SSiT ViT-S 0.930 ± 0.003.
  - No supervised-ImageNet ViT baseline is in the table. — [Huang et al., SSiT, arXiv 2210.10969v5](https://arxiv.org/html/2210.10969v5)
- **Protocol facts behind the weights (background; primary papers not re-fetched in this session).**
  - Noisy Student EfficientNets are trained with ImageNet labels plus pseudo-labelled extra unlabelled images, with "equal-or-larger students" and noise. — [Xie et al., arXiv 1911.04252](https://arxiv.org/abs/1911.04252v4)
  - Noisy Student's unlabelled pool is JFT-300M, per the original paper.
  - DINOv2 uses the curated LVD-142M web set, per its paper. — [Oquab et al., arXiv 2304.07193](https://arxiv.org/abs/2304.07193) (not fetched)
  - CLIP uses about 400M web image-text pairs. — [Radford et al., arXiv 2103.00020](https://arxiv.org/abs/2103.00020) (not fetched)

### Inferences
- **Protocol status flags (for the user to decide).**

  | Status | Weights | Note |
  |---|---|---|
  | Clearly in protocol | ImageNet-1k supervised (any recipe), ImageNet-21k/22k supervised | |
  | Grey zone already accepted | Noisy Student (ImageNet labels + unlabelled JFT images) | B4-NS is in our blend |
  | Grey zone, same kind as NS | DINOv2/DINOv3 (unlabelled curated web images, no labels), natural-image MAE (IN-1k only → clearly in) | |
  | Further out | CLIP/OpenCLIP/SigLIP (web image-text pairs, i.e. weak text labels), EVA-02 (MIM distilled from EVA-CLIP, then IN-21k fine-tune) | |
  | Off-protocol | Anything pre-trained on fundus images (Section 3) | |

- **Expected gain.** On Fang's evidence, a large CLIP-L/14@336 single model could sit about +0.01 above a plain IN-1k B4 at a low-res [local split], or about +0.005 above our B4-NS. As a blend member it would be architecturally and data-wise diverse, so it might be worth +0.003 to +0.008 to the blend (grade D). It is ViT-L at 336 px on a T4, which is likely 4-6 h for 5 folds with LP-FT (unmeasured).
- **DINOv2/v3.**
  - The scale effect is steep: g ≫ L ≈ B ≈ S in Zhou et al.
  - ViT-L/14 at 518 px is too costly on a T4. DINOv2-B/14 or DINOv3-B at 392-448 px is the feasible size.
  - The AUROC lead over RETFound is about 0.008. That does not translate into a known QWK lead over a tuned EfficientNet-NS.
  - Expected single-model QWK about 0.90-0.93. Its value lies in blend diversity.
- **MAE (natural images, IN-1k).** No APTOS numbers on natural-image MAE weights. SSiT's EyePACS-MAE ViT-S at 0.910 suggests plain MAE is not a big win even in-domain.

### Gaps
- No APTOS QWK for DINOv2/DINOv3, natural-image MAE, OpenCLIP, SigLIP or EVA-02 under any split. The Hou et al. kappa values are in a supplement not retrieved.
- CLIP results are LP-FT at low resolution. No 384-512 px CLIP or DINOv2 APTOS result was found.

## 3. Retinal foundation models: APTOS numbers as an off-protocol reference ceiling

### Takeaway
All are **off-protocol**: they are pre-trained on external fundus images, which include EyePACS, and FLAIR's pre-training includes APTOS itself. The best APTOS QWK from an in-domain pre-trained model is **SSiT ViT-S at 0.930 ± 0.003** (EyePACS SSL, 384 px, [local split] 70/15/15). That split is the same shape as ours and is only about 0.01 above our strict 0.918 test score.

RETFound-DINOv2 ViT-L has two conflicting APTOS QWKs in the same 2026 arXiv paper's search record: 0.910 (as fetched for the earlier note) and 0.818 (cloud-only baseline in a search snippet). RETFound/VisionFM/FLAIR are otherwise reported as AUROC only.

The reference ceiling is therefore modest: about 0.93 QWK on a local split. That is roughly where our strict blend sits on OOF (0.928).

### Cited Findings
- **SSiT (IEEE JBHI 2024), APTOS [local split] 70/15/15, 384 px, EyePACS-unlabelled pre-training.** Fine-tune kappa:
  - SSiT ViT-S 0.9297 ± 0.0029, ViT-B 0.9259 ± 0.0069, ViT-Ti 0.9253 ± 0.0040.
  - DiRA RN50 0.9234, DenseCL RN50 0.9115, MoCo-v2 RN50 0.9020.
  - Linear-probe SSiT ViT-S 0.8965.
  - Fine-tune hyperparameters are not given (grid search was used for baselines). — [SSiT, arXiv 2210.10969v5](https://arxiv.org/html/2210.10969v5)
- **RETFound-DINOv2 ViT-L/14 on APTOS 5-class, edge-cloud cascade paper.**
  - The earlier note recorded test QWK 0.9101 (64/16/20 split, 392 px, last 2 blocks tuned). — [arXiv 2605.14108v1](https://arxiv.org/html/2605.14108v1)
  - A search summary of the same paper reports cascade QWK 0.8167 vs a cloud-only baseline of 0.8184. These conflict. The difference may be the binary-referable vs 5-class task, a different version, or a different subset; it was not resolved. — [arXiv 2605.14108 (search summary)](https://arxiv.org/pdf/2605.14108)
- **Nature RETFound paper.** APTOS AUROC 0.943, 224 px, ViT-L, MAE on 904k fundus images (about 10% EyePACS). No QWK. — earlier notes; [Zhou et al. 2023](https://www.nature.com/articles/s41586-023-06555-x)
- **DRStageNet.** RETFound was fine-tuned as a baseline and scored below the authors' model. The search summary says the re-run's APTOS AUC differed from the original report. The numbers were not verified. — [arXiv 2312.14891](https://arxiv.org/html/2312.14891v1) (search summary only)
- **Generalist vs specialist.** RETFound-DINOv2 had the best 10-task average AUROC (0.830 vs DINOv3-L 0.816). No QWK. — [arXiv 2509.03421](https://arxiv.org/pdf/2509.03421)
- **CNN vs FM in-distribution [other dataset; APTOS removed from the pool].** EfficientNet-B3 achieved the strongest in-distribution performance, beating RETFound ViT-L with about 25× fewer parameters. — [Diagnostics 16(19):3109](https://doi.org/10.3390/diagnostics16193109) (search summary only)
- **MIL-VT (ViT pre-trained on a large private fundus set, per nnMobileNet's comparison).** APTOS 5-fold kappa 0.920, below nnMobileNet's 0.934. — [nnMobileNet, arXiv 2306.01289v4](https://arxiv.org/html/2306.01289v4)
- **VisionFM.** Fine-tuned weights were released for 8 public datasets. No APTOS QWK was found. — [ABILab-CUHK/VisionFM](https://github.com/ABILab-CUHK/VisionFM)

### Inferences
- **Ceiling.** Retinal-domain pre-training buys about 0.93 QWK on a 15% local APTOS test split (SSiT). The best published strict-style natural-image results on local splits are already 0.92-0.934:
  - Fang CLIP-L 0.934 and ConvNeXt-T 0.930 (80/20 split).
  - nnMobileNet 0.934 (5-fold).

  So domain pre-training does not visibly raise the APTOS ceiling. That agrees with the earlier note's view that about 0.93-0.94 is the label-noise ceiling. Our OOF 0.928 is already at that level, and the remaining gap to 0.954 is not explained by the choice of pre-trained weights.
- **Option for a reference row.** If the user wants an off-protocol reference row, SSiT ViT-S is the cheapest: it is small and its weights are released by the authors (released code/weights not verified in this session). It is directly comparable in split shape.

### Gaps
- No APTOS QWK for RETFound-MAE, VisionFM, FLAIR, RETFound-DINOv2 (full fine-tune), RETFound v2 or newer FMs under any CV protocol.
- The 0.910 vs 0.818 RETFound-DINOv2 conflict is unresolved.

## 4. Resolution above 512 px for DR grading (Mild DR / microaneurysms)

### Takeaway
Resolution evidence above 512 px is all **[other dataset]**, and gains are small and plateau:
- Sahlsten 5-class macro-AUC: 0.951 (512) → 0.961 (1024), from the earlier note.
- ResNeXt-50 referable-DR AUC: 0.963 / 0.970 / 0.968 / 0.971 at 256 / 512 / 768 / 1024 px.
- An EfficientNetV2-M-based encoder swept 128→768 on EyePACS, with 768 px giving kappa 0.853 (per a search summary).

Segmentation work shows 1024 px is needed for microaneurysms but hurts larger lesions. No APTOS study isolates >512 px or reports Mild-DR recall vs resolution. Our own +0.004 for B4-NS 380→512 fits a curve of +0.00 to +0.004 for 512→640/768. That would cost about 1.6-2.3× the 512 run.

### Cited Findings
- **ADRPPA study (Sci Rep 2024), ResNeXt-50 at 256/512/768/1024 px.** Referable-DR AUROC 0.963 / 0.970 / 0.968 / 0.971. For progression prediction, the 768 model ranked best and 1024 second. — [Nature Sci Rep s41598-024-82884-9](https://www.nature.com/articles/s41598-024-82884-9) (search summary; not fetched)
- **CAM-EfficientNetV2-M encoder, EyePACS sweep 128/256/512/768 px.** 768 px gave test kappa 0.8528. "The resolution of input images plays a crucial role in the performance of DR grading." Intermediate values were not retrieved, and the page fetch timed out, so the attribution to this exact URL is from a search summary. — [ScienceDirect S0010482524003305](https://www.sciencedirect.com/science/article/pii/S0010482524003305)
- **Multi-resolution lesion segmentation (2026).** "High-resolution inputs (1024×1024) are essential for resolving small, critical lesions like Microaneurysms", but "can simultaneously degrade the segmentation performance for larger lesions like Hemorrhages." — [arXiv 2607.08679](https://arxiv.org/html/2607.08679v1) (search summary)
- **Lesion-anchored multi-view transformer (2026).** Uses 1024 px input with 768 px crops around lesion-attention peaks for cross-dataset grading. This is a design reference, not an ablation. — [ScienceDirect S2405959526001098](https://www.sciencedirect.com/science/article/pii/S2405959526001098) (search summary)
- **APTOS practice.** The 4th place used *lower* resolution for larger EfficientNets (B7@224, B6@240, B5@256) "to limit overfitting". The 2nd place mixed 300/456/460 px. — earlier notes; [U-Tokyo Med slides](https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med)
- **SSiT.** Fine-tuned at 384 px. 1024 px appears only in attention-map visualisation. — [SSiT, arXiv 2210.10969v5](https://arxiv.org/html/2210.10969v5)

### Inferences
- **Why small gains on APTOS.** APTOS images vary widely in native resolution and camera. Many are downscaled or blurry (an assumption carried from the earlier notes, not re-verified). So the information gain above 512 px is smaller than on EyePACS/Digifundus. QWK also barely penalises 0↔1 confusion, the place where microaneurysm resolution matters most.
- **Expected gain.** Expect +0.000 to +0.004 single-model from 512→640, and less to the blend.
- **Cheapest test.** A FixRes-style fine-tune of existing B4-NS 512 checkpoints at 640 px for 3-5 epochs, at low lr. Monitor Mild-DR recall and 0↔1 confusion on OOF, not only QWK.
- **Cost.** B4-NS at 640 is about (640/512)² ≈ 1.56× the 138-min 512 run, about 215 min for full training, or about 55 min as a 5-epoch fine-tune. Batch size on a T4 will drop to about 8, so use gradient accumulation and keep BatchNorm in train mode carefully.
- **Not recommended.** 768-1024 px on a T4 is about 2.3-4× the 512 cost per epoch, with no APTOS evidence of benefit.

### Gaps
- No APTOS ablation above 512 px. No per-class (Mild DR) recall vs resolution on any public DR grading dataset found.
- Intermediate values of the EfficientNetV2-M resolution sweep were not retrieved.

## 5. Why ConvNeXt / SE-ResNeXt may have underperformed: recipe sensitivity evidence

### Takeaway
There is direct APTOS evidence that the best lr is **architecture-specific** and that the effect is as large as our ConvNeXt shortfall.

In Fang et al.'s Kaggle runs at 512 px, the same augmentation gave:
- ResNet-50: private QWK 0.861 at lr 1e-3 vs 0.895 at 1e-4, a 0.034 swing.
- Inception-ResNet-v2: 0.893 vs 0.890, flat.

Stronger augmentation *hurt* at the high lr (0.844 vs less-aug 0.867) but *helped* at the low lr (0.895-0.900 vs 0.853). Fang's per-model grid also made ConvNeXt-T the best APTOS model, ahead of B4 by +0.009.

nnMobileNet's ablation attributes about +2 kappa points each to heavy augmentation and to the optimizer (AdamP).

Our single shared recipe (AdamW lr 3e-4, 20 epochs, EMA, strong augmentation), tuned for EfficientNet, is therefore a plausible cause of the ConvNeXt −0.031 and SE-ResNeXt −0.012 gaps vs B4-NS. No APTOS study tested layer-wise lr decay or the interaction with Ben Graham preprocessing.

### Cited Findings
- **Fang et al., Table 5 (Kaggle [private LB] / public, 512 px RRC, AdamW-cosine, rotation, colour jitter, + 2015 data; private / public QWK):**

  | Model | lr | wd 1e-4 | wd 1e-5 | wd 1e-6 |
  |---|---|---|---|---|
  | ResNet-50 | 1e-3 | 0.861 / 0.632 | 0.857 | 0.855 |
  | ResNet-50 | 1e-4 | 0.895 / 0.753 | 0.892 | 0.896 |
  | ResNet-152 | 1e-3 | 0.866 | 0.869 | 0.864 |
  | ResNet-152 | 1e-4 | 0.890 | 0.884 | 0.886 |
  | Inception-ResNet-v2 | 1e-3 | 0.893 | 0.891 | 0.896 |
  | Inception-ResNet-v2 | 1e-4 | 0.890 | 0.893 | 0.894 |

  — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Fang et al., Table 6 (ResNet-50, private LB, wd 1e-4/1e-5/1e-6):**

  | Augmentation | lr | wd 1e-4 | wd 1e-5 | wd 1e-6 |
  |---|---|---|---|---|
  | Less | 1e-3 | 0.867 | 0.852 | 0.861 |
  | Less | 1e-4 | 0.853 | 0.857 | 0.848 |
  | Less | 1e-5 | 0.819 | 0.829 | 0.829 |
  | More | 1e-3 | 0.844 | 0.855 | 0.852 |
  | More | 1e-4 | 0.895 | 0.897 | 0.900 |
  | More | 1e-5 | 0.872 | 0.869 | 0.862 |

  The best lr shifts by 10× depending on augmentation strength. — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Fang et al., [local split].** Every architecture was searched over 3 optimizers × lr × wd. ConvNeXt-T 0.9297 vs EfficientNet-B4 0.9211 under per-model best settings. "DeiT and EfficientNets performed slightly worse" (from the earlier notes). — [Fang et al. 2023, ar5iv](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **nnMobileNet ablation (Messidor-2, kappa gains per step).** Channel configuration +2.03, heavy augmentation +2.36, spatial dropout +0.06, AdamP optimizer +2.2, ReLU6 +0.65. Long schedule: 1000 epochs, lr 1e-3, weight decay 0.05. — [arXiv 2306.01289v4](https://arxiv.org/html/2306.01289v4)
- **Published-baseline spread.** ResNet-50 on APTOS ranges from kappa 0.870 (SGD 1e-3, 500 epochs, weak augmentation, 7:1:2 split) to 0.9215 (Fang, tuned grid) for the same architecture. — [arXiv 2510.26315](https://arxiv.org/pdf/2510.26315); [Fang et al.](https://ar5iv.labs.arxiv.org/html/2301.04644)
- **Foundation-model fine-tune recipes** use very different lr regimes from ours. Examples:
  - lr 5e-3 peak with 10-epoch warm-up, 100 epochs, weight decay 0.05 (DINOv2). — [Hou et al.](https://arxiv.org/pdf/2502.06289)
  - 5e-4 peak, 50 epochs (RETFound/DINOv3). — [Zhou et al.](https://arxiv.org/pdf/2509.03421)
- **Layer-wise lr decay.** Used in a 2026 ultrasound benchmark for ConvNeXt and CNNs: 4 depth-ordered groups, decay 0.9. Its EfficientNet ablation was not retrieved. — [UltraBench 2, arXiv 2609.28610](https://arxiv.org/pdf/2609.28610) (search summary)
  - A 2025 review notes discriminative fine-tuning is "sensitive to depth and hyperparameters". — [arXiv 2511.01284](https://arxiv.org/pdf/2511.01284) (search summary)
- **BloodAxe (top-10).** SE-ResNeXt50 regression 5-fold CV 0.9225 and SE-ResNeXt50d-GWAP 512 0.9231. Both are about +0.007 above our SE-ResNeXt50 512 (OOF 0.916) on a similar CV protocol. — earlier notes; [BloodAxe repo](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection)

### Inferences
- **Likely causes for ConvNeXt-T 448 at 0.897 (grade D, ranked).**
  1. **lr / schedule mismatch.** AdamW 3e-4 for 20 epochs is high for ConvNeXt fine-tuning without layer-wise decay. Typical ConvNeXt fine-tune recipes use about 5e-5 to 1e-4 with LLRD 0.7-0.8, warm-up, and drop-path 0.1-0.2. That recipe is general practice and is not sourced for APTOS. Fang's 10× lr sensitivity under strong augmentation is the closest APTOS evidence.
  2. **IN-1k 224 weights run at 448 px (4× tokens/area), with no resolution warm-up.** ConvNeXt has no BatchNorm to re-calibrate. A short 224→448 progressive schedule, or IN-22k 384 weights (`convnext_small.fb_in22k_ft_in1k_384`), would match the pre-training statistics better.
  3. **Ben Graham preprocessing interaction.** The local-mean subtraction shifts the input distribution far from ImageNet's mean/std. BatchNorm models (EfficientNet, SE-ResNeXt) adapt their running statistics. ConvNeXt's LayerNorm does too, but its stem patchify (4×4, stride 4) may be more sensitive to the high-frequency enhanced input. No source tests this, so it remains an untested hypothesis. A cheap check is one fold with plain crop+resize (no Ben Graham) at the same lr.
  4. **EMA decay tuned for EfficientNet's step count.** Minor.
- **SE-ResNeXt50 512 at 0.916 vs BloodAxe's 0.9225-0.9231.** That is a 0.007 gap. It could be lr: Fang shows ResNet-family models are the most lr-sensitive, preferring 1e-4 over 1e-3 with strong augmentation. It could also be GeM/GWAP pooling or a longer schedule. Expected recovery from a re-tune is +0.003 to +0.008 single-model. Because SE-ResNeXt correlates least with the blend (ρ 0.957), a stronger version could add more than the observed +0.0025.
- **Suggested 1-fold screen before any new 5-fold run (about 20-30 min per config on a T4).**
  - ConvNeXt-T/S: lr ∈ {5e-5, 1e-4} × LLRD ∈ {0.75, 1.0} × {Ben Graham, plain} at 384 px, with IN-22k-ft-384 weights for ConvNeXt-S.
  - Promote to 5 folds only if fold-0 val is ≥ B4-NS fold-0 − 0.005.
  - Choices must use validation folds only, never the test set.

### Gaps
- No APTOS (or DR) ablation of layer-wise lr decay, drop-path, or Ben Graham vs plain preprocessing for ConvNeXt/transformers.
- Fang's per-model chosen lr/wd values sit in an external spreadsheet that was not retrieved, so the ConvNeXt-T optimum on APTOS is unknown.
- The ConvNeXt lr range quoted in Inference 1 is general fine-tuning practice, not an APTOS-sourced number.
