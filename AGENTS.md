# dr-grading-reproduction

Research reproduction of Chilukoti et al. (2024), diabetic retinopathy grading on APTOS 2019 with EfficientNet-B3 and a snapshot ensemble. Read `README.md` first; it holds the results table and the paper protocol (300 px, 60 epochs, batch 16, seed 42).

## Layout
- `dr-repro/`: notebooks (`colab_dr_grading.ipynb`, `stage4*_push.ipynb`), stage plans and results (`stage4*_plan.md`, `stage4*_results.md`), `runs/`, `results/`, `kaggle/`
- `reports/`, `research_notes/`, `stage4_runs/`, and the Stage 2/3 PDFs and slides at the top level

## Rules
- This is a reproduction study. Make no clinical claims.
- The test set (550 images, stratified 70/15/15, seed 42) must never be used for any decision.
- Do not overwrite earlier stage results; add new files for new runs.
- Run/environment commands: TODO (work is notebook-based on Colab/Kaggle; no scripts or dependency file at the top level).
