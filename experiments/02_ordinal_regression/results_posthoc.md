# Stage 4 post-hoc check: votes vs. averages (exploratory)

Added after Stage 4 results were known (notebook section 12, check 1 and 2). Not pre-registered, so it does not change the verdicts in [`experiments/02_ordinal_regression/results.md`](results.md). No model was retrained.

**How it was run:** locally, from the saved `experiments/02_ordinal_regression/runs/*/outputs.npz` files, with [`experiments/02_ordinal_regression/posthoc/stage4_soft_vote_local.py`](posthoc/stage4_soft_vote_local.py). This script re-implements the notebook's `qwk`, `fit_cutoffs`, `majority_vote` and `pooled_bootstrap`. As a check, it re-derives arms A–D from the same files, and every QWK matches `experiments/02_ordinal_regression/results/results_per_seed.csv` to four decimals (`experiments/02_ordinal_regression/results/posthoc_qwk_check.csv`).

```
uvx --from numpy python -I experiments/02_ordinal_regression/posthoc/stage4_soft_vote_local.py experiments/02_ordinal_regression/runs experiments/02_ordinal_regression/results
```

## Question

In the ensemble, arm A uses the paper's majority vote, while B, C and D average the 10 snapshots. As single models, D − A is only +0.002. So the ensemble gain of D over A could come from **averaging instead of voting**, not from the ordinal loss. `A_soft` is the cross-entropy ensemble with averaged probabilities and arg-max. **D − A_soft compares the two losses with the same kind of ensemble.**

## Result

Ensemble, main split (`experiments/02_ordinal_regression/results/posthoc_soft_vote.csv`, `experiments/02_ordinal_regression/results/posthoc_bootstrap.csv`):

| Arm | QWK seed 42 / 43 / 44 | Mean | Severe recall (mean) | PDR recall (mean) | Errors ≥ 2 (mean) |
|---|---|---|---|---|---|
| A: CE, majority vote | 0.898 / 0.894 / 0.901 | 0.897 | 0.28 | 0.61 | 21.7 |
| A_soft: CE, averaged probabilities | 0.898 / 0.899 / 0.900 | 0.899 | 0.25 | 0.61 | 21.3 |
| D: ordinal, tuned cut-offs | 0.910 / 0.917 / 0.914 | 0.914 | 0.40 | 0.58 | 16.0 |

Paired bootstrap of the mean ΔQWK over 3 seeds (2,000 resamples of the 550 test images; test-set noise only):

| Contrast | ΔQWK | 95% CI | Excludes 0 |
|---|---|---|---|
| A_soft − A (averaging vs. voting) | +0.001 | [−0.002, +0.006] | no |
| **D − A_soft (loss, same ensemble type)** | **+0.015** | **[+0.002, +0.030]** | **yes** |
| D − A (as in Stage 4) | +0.016 | [+0.003, +0.032] | yes |
| D − C (tuned cut-offs vs. rounding) | +0.018 | [+0.007, +0.029] | yes |
| D − A, single models | +0.002 | [−0.011, +0.016] | no |

Per seed, D − A_soft is +0.012 / +0.018 / +0.015, and its interval excludes 0 only for seed 43. On the de-duplicated split it is +0.012 [−0.002, +0.027].

## Reading

- **Averaging vs. voting explains almost none of the gain.** A_soft ties A (+0.001).
- **The ensemble gain of ordinal regression holds against a matched soft-vote baseline.** It is +0.015 when averaged over seeds, and the pooled interval excludes 0. Per seed it is significant for 1 of 3, the same pattern as the pre-registered P1.
- **The gain still appears only in the ensemble.** As single models the arms tie. The more likely explanation is that averaging 10 regression outputs, then applying tuned cut-offs, uses the snapshot spread better than averaging 10 softmax outputs. This is an interpretation and was not tested separately.
- This is post-hoc and covers test-set noise only, with 3 seeds. It strengthens the Stage 4 reading but does not turn P1 into a confirmed prediction.

## Still pending (needs the images, so a Colab re-run of `experiments/02_ordinal_regression/02_ordinal_regression.ipynb` sections 11–12)

- Mann-Whitney test of sharpness and brightness for misgraded vs. correct images (check 3).
- Grad-CAM restricted to the retina (check 4). The weights are in `experiments/02_ordinal_regression/runs/*/best_val_weights.pt`.
