# Stage 7 (M3): Hypothesis and Prediction. A retina-pretrained backbone

*Written on 2 Nov 2026, **before** any Stage 7 code was run. Commit this file before running the Stage 7 notebook, and copy the thresholds in section 4 into its `PREDICTIONS` cell. Stages 2–6 stay in the repository untouched. Stage 7 is a post-test experiment (the test set was opened in Stage 4c); its numbers drive no earlier choice.*

## 1. Starting point

Every backbone we tried so far (EfficientNet-B3/B4, ConvNeXt, SE-ResNeXt) started from ImageNet weights. The Stage 5 baseline (arm H: soft labels, σ = 0.5, tuned cut-offs, EfficientNet-B3) scores QWK 0.916 ± 0.005, Severe recall 0.39, PDR recall 0.62, 14.7 errors of two or more grades (3 seeds, 550 test images). Stage 4c showed that more ImageNet architectures plateau near 0.92. Stage 4c/4d also tried 5 epochs of EyePACS-2015 pre-training on top of ImageNet weights and found it a *worse* starting point, but that was a short, CNN-based, supervised pre-training, not a foundation model.

## 2. Evidence that motivates the change

1. **The remaining errors are in the rare, lesion-defined grades.** Severe recall is 0.39 and Mild recall 0.27 (Stage 5). Those grades depend on small lesions that ImageNet features were never trained to detect.
2. **Retinal foundation models are pre-trained on the target domain.** RETFound (Zhou et al., *Nature* 2023) is a ViT-Large pre-trained with masked-autoencoder reconstruction on about 1.6 million unlabelled retinal images, and is reported to need less labelled data than ImageNet initialisation for retinal tasks. We did not verify its APTOS numbers and make no claim about them. Our literature search on 2 Nov 2026 found no controlled comparison on APTOS under a fixed protocol, which is what Stage 7 provides.
3. **Our APTOS training set is small** (about 2,560 images, 29 Severe and 44 PDR test images), the setting in which a domain-specific starting point should matter most.

## 3. Hypothesis (one change, with its control)

> **Initialising the backbone from retina-pretrained RETFound weights (ViT-Large, masked-autoencoder pre-training on colour fundus photographs) instead of ImageNet weights will raise QWK and the recall of the rare grades, because its features have already been learned on retinal lesions and vessels, which the 2,560 APTOS training images are too few to teach.**

**The control that isolates the pre-training domain.** Changing from EfficientNet-B3 to a ViT-L changes architecture, resolution (224 px, the ViT's fixed input), optimiser recipe and pre-training all at once. To attribute any gain to *retina* pre-training we run **the same ViT-Large with the same recipe, initialised from ImageNet** (`timm vit_large_patch16_224.augreg_in21k_ft_in1k`). The contrast RETFound − ImageNet-ViT-L is the hypothesis test. The contrast RETFound − EfficientNet-B3 (arm H) is the practical comparison.

**What stays fixed:** data and split (stratified 70/15/15, seed 42), CLAHE + Gaussian-blur pre-processing, flip/rotation augmentation, soft-label loss (σ = 0.5), cut-offs tuned on validation only, seeds. **What must change for a ViT-L** (and is identical for both ViT arms): input 224 px, AdamW (peak lr 3e-4 for the head with layer-wise decay 0.65, i.e. the RETFound fine-tuning defaults scaled to batch 16; weight decay 0.05; 3 warm-up epochs then cosine), batch 16, 25 epochs, drop-path 0.2, gradient clipping 1.0, a linear head on the class token instead of the paper's 3-layer head, and 5 snapshots (every 5th epoch) instead of 10, because the schedule is 25 epochs.

**Weights.** The official RETFound weights are gated behind a Hugging Face login. We use the ungated, `timm`-compatible copy `bitfount/RETFound_MAE` (a copy of the colour-fundus (CFP) checkpoint of `open-eye/RETFound_MAE`, licence CC-BY-NC-4.0, non-commercial research use). It is a third-party mirror, so the notebook loads the file with `torch.load(weights_only=True)` and checks that its patch-embedding weights differ from the ImageNet ones, but we cannot verify it byte for byte against the official file.

## 4. Predictions (written before running)

R_arm = RETFound ViT-L with the soft-label pipeline (vote and tuned-cut-off rules); I_arm = ImageNet ViT-L, same recipe (seed 42 only); H = Stage 5 arm H (EfficientNet-B3). Snapshot ensemble (5 snapshots for the ViTs, 10 for H), mean over seeds unless stated.

| # | Prediction | Counts as confirmed if |
|---|---|---|
| S1 | **Retina pre-training beats ImageNet at the same architecture** | QWK of R (tuned) − I (tuned) ≥ +0.01 on seed 42 |
| S2 | **The rare grades gain** | mean of Severe and PDR recall of R (tuned) − I (tuned) ≥ +0.05 (seed 42) |
| S3 | **RETFound is competitive with the tuned CNN, not clearly better** | abs(R − H) < 0.015 in QWK, mean over 3 seeds |
| S4 | **Large errors do not increase against the CNN** | errors of ≥ 2 grades of R ≤ 1.10 × H (14.7 → ≤ 16.2) |
| S5 | **Seed-to-seed spread is larger for the ViT** | std over 3 seeds of R (tuned) QWK > 0.005 (H's std) |
| S6 | **Ranking quality is at least as good** | referable AUC of R ≥ H − 0.005 |
| S7 | *Sanity, not a hypothesis.* The mirror holds retina weights, not ImageNet weights | patch-embedding weights of the mirror differ from the ImageNet ViT-L (relative difference > 1e-3) |

Being wrong is allowed, and S3 is the likeliest outcome: a gain from 0.916 to a clearly higher value would be surprising given that Stage 4c's stronger ImageNet backbones, ensembles and cleaning plateaued at 0.918. If S1 fails and I_arm ≈ R_arm, the honest reading is that pre-training domain is not the bottleneck on APTOS, and the remaining gap to the paper lies in the labels or the protocol. ViT-L has 300 M parameters; training with only 2,560 images may also be unstable, which S5 anticipates.
