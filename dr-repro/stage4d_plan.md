# Stage 4d: Corrected EyePACS Fine-tune (plan and predictions)

*Written on 23 Oct 2026, **after** the test set was opened once in `dr-s4c-c2-final` ([`stage4c_results.md`](stage4c_results.md)) and **before** any Stage 4d run. Everything here is therefore a post-test experiment. It never replaces the Stage 4b or 4c headlines, and its test rows are labelled "post-test". Evidence behind the design: [`../reports/APTOS QWK gap after Stage 4c.md`](../reports/APTOS%20QWK%20gap%20after%20Stage%204c.md) and its EyePACS notes. Notebook: [`stage4d_push.ipynb`](stage4d_push.ipynb) (Stage 4c plus section 16); kernels from [`kaggle/stage4b/build_kernels.py`](kaggle/stage4b/build_kernels.py) with chain `4d`.*

## 1. Question

Arm X (B4-NS 380, 5 epochs of EyePACS-2015 pre-training, then the Stage 4b fine-tune at peak lr 3e-4 after a 1-epoch warm-up) scored OOF QWK 0.913, below the same network from ImageNet (0.920). The top APTOS 2019 teams fine-tuned after EyePACS at about 1e-5 to 2e-5, with no warm-up restart. **Is arm X's failure caused by the fine-tune learning rate?**

## 2. Design (fixed now)

- **Same EyePACS weights as arm X** (`eyepacs_pretrain/tf_efficientnet_b4_380_graham_e5.pt`, reused, not retrained), same folds, same recipe (Graham, strong augmentation, label cleaning, 20 epochs, EMA 0.998, Adam, batch 16, 8-view TTA). **Only two things change:** no warm-up (`warmup_epochs=0`, cosine decay from step 0) and a lower peak lr.
- **Zero-shot score:** the EyePACS weights with no APTOS training, scored on the 3,112 OOF images (cross-fitted cut-offs, TTA).
- **lr probe on fold 0:** peak lr 1e-5, 3e-5 and 1e-4. The lr with the best fold-0 held-out QWK (cross-fitted, TTA) is chosen; a tie within 0.001 goes to the lower lr.
- **Arm X2:** the chosen lr on all 5 folds. Its fold-0 run is the probe run.
- Because fold 0 was used to choose the lr, X2 is also compared with B4-NS 380 (ImageNet) and arm X **on folds 1-4 only**. That comparison is the clean one.
- **Blend:** the Stage 4c external greedy rule (start from B4-NS 380 + B3 384, add the arm with the largest OOF gain while it is at least +0.002), with X2 added to the pool. This blend uses external data, so it can never be a headline.
- **Budget:** about 2.5 Kaggle T4 GPU-hours, which leaves about 6 h of the 20.

## 3. Predictions (OOF = cross-fitted QWK with TTA)

| # | Prediction |
|---|---|
| D1 | Zero-shot OOF QWK: 0.70-0.85 |
| D2 | Chosen lr: 3e-5 or 1e-4, not 1e-5 |
| D3 | X2 OOF QWK on all 5 folds: 0.918-0.930 |
| D4 | On folds 1-4, X2 beats arm X by at least +0.005 and is at least level with B4-NS 380 from ImageNet (−0.002 or better) |
| D5 | The OOF QWK dip in fine-tune epochs 5-7 seen in arm X does not appear in X2 |
| D6 | X2 is selected by the external greedy blend (gain ≥ +0.002): probability about 50% |
| D7 | Test (post-test): any difference to the 4c headline is below 0.01 and its 95% CI includes 0 |

If D4 holds, arm X's failure was caused by the fine-tune recipe, not by EyePACS. If D4 fails, 5 EyePACS epochs with this recipe do not help B4-NS on APTOS, whatever the fine-tune lr.
