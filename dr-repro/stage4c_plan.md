# Stage 4c: Diversity and External Data (plan and predictions)

*Written on 23 Oct 2026, after Stage 4b Blocks 1-3 finished and **before** any Stage 4c run and before the test set was opened. Evidence behind the choices: [`../reports/Closing the APTOS 2019 QWK gap.md`](../reports/Closing%20the%20APTOS%202019%20QWK%20gap.md). Notebook: [`stage4c_push.ipynb`](stage4c_push.ipynb) (Stage 4b plus sections 12-13); Kaggle kernels from [`kaggle/stage4b/build_kernels.py`](kaggle/stage4b/build_kernels.py).*

## 1. Where Stage 4b left us (validation and out-of-fold only; test still locked)

- Adopted levers: peak lr 3e-4, strong augmentation, label cleaning, Ben Graham pre-processing (all four beat their predictions). 30 epochs did not help.
- Block 2: B4 Noisy Student 380 px 0.927, B3 384 px 0.922, B3 300 px 0.916, EfficientNetV2-S 384 px 0.910 (cross-fitted val QWK, TTA).
- Block 3 (5-fold, 3,112 OOF images): B4-NS 380 0.920, B3 384 0.924, **blend 0.925**. The two members correlate at 0.984, so blending added +0.001 against a predicted +0.01 to +0.02.

The research report concludes that 0.925 is already at the level of careful ImageNet-only work on APTOS (0.92-0.93), that the remaining strict-protocol levers are worth a few thousandths each, and that most of the gap to 0.954 is likely protocol. Stage 4c spends the remaining compute on the levers with the best support, and measures the protocol effect on our own model.

## 2. Rules (fixed now)

- **Same data, folds and recipe as Stage 4b Block 3:** the same 5 folds of train + val (identical images kept together), the adopted levers (lr 3e-4, strong augmentation, label cleaning), 20 epochs with warm-up, cosine decay and EMA, final EMA weights, 8-view TTA. Only the changes listed per arm differ.
- **The test set stays locked** until all Stage 4c runs have finished, and is opened once, in the last kernel, together with Stage 4b.
- **Blend rule (headline):** start from the Stage 4b blend (B4-NS 380 + B3 384). Greedy forward selection over the strict arms: add the arm that raises the cross-fitted OOF QWK (TTA) of the simple-mean blend the most, if the gain is at least **+0.002**; repeat until no arm qualifies. Cut-offs are then tuned on all 3,112 OOF predictions of that blend.
- **External data is never in the headline.** The EyePACS arm gets its own rows, and a second blend that applies the same greedy rule to strict arms plus the EyePACS arm.
- **Optimistic rows are reported and labelled as not results:** headline with cut-offs tuned on the test set; best single model chosen on the test set; and the best blend subset with cut-offs, both chosen on the test set. They show how far paper-style choices move our own model.
- **Budget:** at most 16 Kaggle T4 GPU-hours for Stage 4c (20 h available), strict arms at most 9 h. A time guard stops starting new runs before Kaggle's 12-hour session limit.

## 3. Arms

| Arm | Change from the Stage 4b recipe | Why | Est. T4 time (5 folds) |
|---|---|---|---|
| **A** `A_b4ns380_crop_aux` | B4-NS 380; **retina crop** (cut to the retina, pad square, resize; no CLAHE or Graham); **auxiliary 5-class head** (cross-entropy x 0.3); grade still read from the regression output | raw images beat Ben Graham for the 4th-place team (~0.933 vs ~0.927 CV); a different input should make the member less correlated | ~1.5 h |
| **B** `B_convnextT448` | ConvNeXt-Tiny (ImageNet-1k), 448 px, retina crop, SmoothL1 loss | best of 19 ImageNet models on APTOS in Fang et al. 2023; a different architecture family | ~1.7 h |
| **B2** `B2_seresnext50_512` | SE-ResNeXt50 (ImageNet-1k), 512 px, retina crop, GeM pooling, SmoothL1 | the 2019 winner's family, pooling and loss; ResNets "blend well" with EfficientNets/Inceptions | ~2.2 h |
| **C** `C_b4ns512` | B4-NS at 512 px, otherwise the Stage 4b recipe | resolution: Mild DR depends on microaneurysms a few pixels wide | ~2.5 h |
| **X** `X_b4ns380_eyepacs` (external) | B4-NS 380 initialised from 5 epochs of training on the 35,108 EyePACS-2015 training images (same recipe, regression); then the Stage 4b fine-tune | every gold-medal team in 2019 used the 2015 data; the paper's own EyePACS row is 0.956 | ~2.5 h |

## 4. Predictions (cross-fitted OOF QWK with TTA, unless stated)

| # | Prediction |
|---|---|
| Q1 | A alone: 0.918-0.928 (vs B4-NS 380 0.920); its correlation with the Stage 4b members is lower than 0.984 |
| Q2 | B and B2 alone: 0.912-0.928 each; at least one of them is selected by the blend rule |
| Q3 | C alone: 0 to +0.006 over B4-NS 380 |
| Q4 | X alone: +0.002 to +0.010 over B4-NS 380 |
| Q5 | Stage 4c headline blend OOF: 0.927-0.935 (+0.002 to +0.010 over the Stage 4b blend) |
| Q6 | Test QWK of the Stage 4c headline: 0.915-0.930 (plausible range 0.905-0.945). Probability of reaching 0.954 under this protocol: below 5% |
| Q7 | Optimistic (test-tuned) rows: 0.935-0.950, i.e. paper-style choices on the test set recover most of the remaining gap |

If Q6 holds and Q7 holds, the reproduction conclusion is that the paper's 0.954 is consistent with our model plus an optimistic protocol, and not reachable under a strict one with ImageNet weights and APTOS data.

## 5. Kernels

| Kernel | Work | Inputs |
|---|---|---|
| `dr-s4c-c0-quick` | quick test of every Stage 4c code path (toy sizes), plus a memory and speed probe of every arm at full size | Stage 4b k3 outputs, EyePACS dataset |
| `dr-s4c-c1-ab` | arms A and B | Stage 4b k3 outputs |
| `dr-s4c-c1-b2c` | arms B2 and C | Stage 4b k3 outputs |
| `dr-s4c-c1-x` | EyePACS pre-training, arm X | Stage 4b k3 outputs, EyePACS dataset |
| `dr-s4c-c2-final` | no training; opens the test set once: Part 0, Stage 4b final table, Stage 4c final table | the three c1 kernels, Stage 4 outputs dataset |
