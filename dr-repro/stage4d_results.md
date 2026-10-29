# Stage 4d: Results and Prediction Check (post-test)

*Results of `dr-s4d-d1-x2` (Kaggle T4, 23 Oct 2026, about 2.1 h), built from [`stage4d_push.ipynb`](stage4d_push.ipynb). Predictions: [`stage4d_plan.md`](stage4d_plan.md), written before the run. The test set had already been opened in Stage 4c, so every test number here is **post-test**. Raw numbers: [`results/stage4b/stage4d_*.csv`](results/stage4b/).*

## 1. Results (OOF = cross-fitted QWK with TTA)

**Zero-shot:** the EyePACS weights with no APTOS training score OOF **0.877** (test 0.860).

**lr probe, fold 0** (held-out QWK):

| Fine-tune | Fold-0 QWK |
|---|---|
| EyePACS init, lr 1e-5, no warm-up | 0.911 |
| **EyePACS init, lr 3e-5, no warm-up (chosen)** | **0.915** |
| EyePACS init, lr 1e-4, no warm-up | 0.908 |
| EyePACS init, lr 3e-4, 1-epoch warm-up (arm X) | 0.914 |
| ImageNet (Noisy Student) init, lr 3e-4 (B4-NS 380) | 0.931 |

**Five folds:**

| Member | OOF QWK, all folds | OOF QWK, folds 1-4 | Correlation with B4-NS 380 | Gain added to 4b blend |
|---|---|---|---|---|
| B4-NS 380 (ImageNet) | 0.9195 | 0.9167 | 1.000 | |
| X (EyePACS, lr 3e-4 + warm-up) | 0.9131 | 0.9132 | 0.975 | +0.0007 |
| **X2 (EyePACS, lr 3e-5, no warm-up)** | **0.9061** | **0.9050** | 0.970 | −0.0000 |

The greedy blend did not select X2, so the external-data blend stays B4-NS 380 + B3 384 + C (OOF 0.928), the same as the Stage 4c headline.

**Post-test rows:** zero-shot 0.860; X2 alone 0.884 [0.858, 0.904], against arm X 0.889 and the B4-NS 380 CV ensemble 0.909. Paired difference X2 − X on test: −0.006 [−0.019, +0.007].

## 2. Were the predictions right?

| # | Prediction | Observed | Verdict |
|---|---|---|---|
| D1 | Zero-shot OOF 0.70-0.85 | 0.877 | **Not confirmed** (higher) |
| D2 | Chosen lr 3e-5 or 1e-4 | 3e-5 | Confirmed |
| D3 | X2 OOF 0.918-0.930 | 0.906 | **Not confirmed** |
| D4 | Folds 1-4: X2 ≥ X + 0.005 and ≥ B4-NS 380 − 0.002 | X2 0.905, X 0.913, B4-NS 0.917 | **Not confirmed** |
| D5 | No dip in epochs 5-7 | The rounded per-epoch QWK of X2 is smooth (0.86-0.89 in epochs 5-7, against arm X's drops to 0.81-0.83) | Confirmed, but it did not lead to a better final model |
| D6 | X2 selected by the blend (about 50%) | not selected (gain 0.000) | Not selected |
| D7 | Test difference to the 4c headline < 0.01 | the external blend is identical to the headline (difference 0) | Confirmed, trivially |

## 3. What this means

- **The learning-rate hypothesis is rejected.** A low fine-tune lr removed the dip but ended lower: 0.905 against 0.913 for arm X on folds 1-4. On fold 0 the order is 3e-5 ≈ 3e-4 > 1e-5 > 1e-4, all within 0.007 of each other, and all at least 0.016 below the ImageNet Noisy Student start (0.931). Fold-to-fold noise is about ±0.01, so the probe cannot rank the lrs reliably, but every lr was worse than the ImageNet start.
- **The problem is the EyePACS starting point, not the fine-tune.** On its own, the EyePACS model already grades APTOS reasonably well (zero-shot 0.877). But as a starting point, it is worse than the Noisy Student ImageNet weights at every fine-tune lr tried. The more of the EyePACS features the fine-tune keeps (lower lr), the worse the final model.
- **Caveats.** The pre-training was only 5 epochs on the 35k EyePACS training images. The top 2019 teams used 15-80 epochs, about 88k images (with the 2015 test labels), no Ben Graham pre-processing, and often trained on combined data instead of pre-train then fine-tune. Our result says that *this* cheap form of EyePACS transfer does not help B4-NS on APTOS. It does not say that EyePACS cannot help.
- **The headline does not change.** The strict Stage 4c headline (test QWK 0.918) remains the best result, and the external-data blend is identical to it.
