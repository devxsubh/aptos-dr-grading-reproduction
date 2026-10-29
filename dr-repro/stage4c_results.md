# Stage 4b and 4c: Results and Prediction Check

*Results of the Kaggle kernels built from [`stage4b_push.ipynb`](stage4b_push.ipynb) and [`stage4c_push.ipynb`](stage4c_push.ipynb) ([`kaggle/stage4b/`](kaggle/stage4b/)), run on Kaggle T4s on 22-23 Oct 2026. The test set was opened once, in `dr-s4c-c2-final`, after all training had finished. The predictions being checked are in [`stage4b_plan.md`](stage4b_plan.md) and [`stage4c_plan.md`](stage4c_plan.md), both written before the runs they predict. Raw numbers are in [`results/stage4b/`](results/stage4b/).*

## 1. Validation and out-of-fold results (before the test set was opened)

Cross-fitted QWK with 8-view TTA on the 3,112 train + val images (5 folds, identical images kept in the same fold).

| Member | OOF QWK | Correlation with B4-NS 380 | Gain when added to the 4b blend | Minutes (5 folds) |
|---|---|---|---|---|
| B4-NS 380 (4b) | 0.9195 | 1.000 | | 87 |
| B3 384 (4b) | 0.9220 | 0.984 | | 64 |
| A: B4-NS 380, retina crop, auxiliary 5-class head | 0.9227 | 0.981 | +0.0014 | 81 |
| B: ConvNeXt-Tiny 448, retina crop, SmoothL1 | 0.8967 | 0.959 | +0.0009 | 84 |
| B2: SE-ResNeXt50 512, retina crop, GeM, SmoothL1 | 0.9162 | 0.957 | +0.0025 | 118 |
| C: B4-NS 512 | 0.9237 | 0.989 | +0.0035 | 138 |
| X: B4-NS 380 after 5 epochs on EyePACS-2015 (external) | 0.9131 | 0.975 | +0.0007 | 82 (+99 pre-training and loading) |

Greedy blend selection (rule: add the arm with the largest gain, if ≥ +0.002): start 0.9248 (B4-NS 380 + B3 384), add C → **0.9283**. After C, no arm gained ≥ +0.002. The external greedy picked the same blend, so X is not used.

## 2. Test results (550 images, opened once)

| Model | Test QWK [95% CI] | Accuracy | Errors ≥ 2 grades | QWK without the 36 twins |
|---|---|---|---|---|
| Stage 4 A: cross-entropy, vote (seed 42) | 0.898 [0.868, 0.921] | 0.829 | 22 | 0.900 |
| Stage 4 D: ordinal, tuned cut-offs (seed 42) | 0.910 [0.887, 0.929] | 0.802 | 17 | 0.912 |
| 4b single model B4-NS 380 (val cut-offs) | 0.906 [0.883, 0.924] | 0.787 | 18 | 0.904 |
| 4b CV ensemble B4-NS 380 | 0.909 | 0.786 | 16 | 0.909 |
| 4b CV ensemble B3 384 | 0.913 | 0.804 | 15 | 0.913 |
| **4b headline: blend B4-NS 380 + B3 384** | **0.916 [0.896, 0.932]** | 0.802 | 14 | 0.916 |
| 4c arm A | 0.908 | 0.804 | 17 | 0.909 |
| 4c arm B | 0.871 | 0.751 | 27 | 0.874 |
| 4c arm B2 | 0.900 | 0.776 | 17 | 0.902 |
| 4c arm C | 0.917 | 0.818 | 14 | 0.916 |
| 4c arm X (external data) | 0.889 | 0.753 | 17 | 0.896 |
| **4c headline: blend B4-NS 380 + B3 384 + C** | **0.918 [0.900, 0.935]** | 0.807 | 14 | 0.917 |
| *Optimistic, not a result:* 4c headline, cut-offs tuned on test | 0.924 | 0.827 | 14 | 0.924 |
| *Optimistic, not a result:* best single model chosen on test (C) | 0.917 | 0.818 | 14 | 0.916 |
| *Optimistic, not a result:* best of 63 subsets + cut-offs, both chosen on test (B3 384 alone) | 0.928 | 0.791 | 6 | 0.928 |
| Paper (Chilukoti et al. 2024, ImageNet B3) | 0.954 | 0.942 | | |

Paired bootstrap of the 4c headline against: Stage 4 A +0.020 [+0.001, +0.042]; Stage 4 D +0.008 [−0.005, +0.024]; 4b headline +0.003 [−0.003, +0.009].

Protocol effect on the Stage 4 models (Part 0, [`protocol_gap.csv`](results/stage4b/protocol_gap.csv)): tuning cut-offs on the test set adds +0.008 to +0.024; picking the snapshot on the test set adds −0.007 to +0.010.

## 3. Were the predictions right?

| # | Prediction | Observed | Verdict |
|---|---|---|---|
| Q1 | A: 0.918-0.928; correlation < 0.984 | 0.9227; 0.981 | Confirmed |
| Q2 | B and B2: 0.912-0.928 each; at least one selected | B 0.897, B2 0.916; neither selected | **Not confirmed** |
| Q3 | C: 0 to +0.006 over B4-NS 380 | +0.004 | Confirmed |
| Q4 | X: +0.002 to +0.010 over B4-NS 380 | −0.006 | **Not confirmed** |
| Q5 | 4c headline OOF 0.927-0.935 | 0.928 | Confirmed (lower end) |
| Q6 | 4c headline test 0.915-0.930; P(≥ 0.954) < 5% | 0.918; CI upper bound 0.935 | Confirmed |
| Q7 | Optimistic rows 0.935-0.950 | 0.917-0.928 | **Not confirmed** |

## 4. What this means

- Stages 4b and 4c moved the test QWK from 0.898 (Stage 4 baseline) to 0.918. The gap to the paper's 0.954 went from 0.056 to **0.036**. The paper's value lies outside the 95% CI of our best strict model.
- Most of the 4b/4c gain came from levers measured on validation: learning rate, augmentation, label cleaning, Ben Graham pre-processing, Noisy Student B4, higher resolution and 5-fold ensembles. Architecture diversity (ConvNeXt, SE-ResNeXt) did not help here.
- The EyePACS arm was worse, not better. The likely cause is that fine-tuning restarted at the full peak learning rate (3e-4 after a 1-epoch warm-up), which wipes out most of 5 short pre-training epochs; the OOF QWK falls in epochs 5-7 of every fold. This is a guess, not a tested result.
- Q7 failed: test-set choices of cut-offs, members and subsets recover only +0.006 to +0.010 on our model, reaching at most 0.928. So the protocol choices we can measure do not explain the paper's 0.954. The remaining gap of about 0.026 needs another explanation, such as a different split, external data, or an error in the paper. The paper's reported accuracy of 94.2% is far above any model here (best 0.83).
- The 36 test images with a twin in train + val do not inflate our scores (QWK with and without them differs by ≤ 0.007).
