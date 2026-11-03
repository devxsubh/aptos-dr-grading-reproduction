# dr-grading-reproduction

Research reproduction of Chilukoti et al. (2024), diabetic retinopathy grading on APTOS 2019 with EfficientNet-B3 and a snapshot ensemble, plus nine hypotheses tested on top of it. Read `README.md` first; it holds the results table and the paper protocol (300 px, 60 epochs, batch 16, seed 42).

## Layout
- `experiments/NN_name/`: one folder per experiment, numbered in run order (01 baseline, 02 ordinal, 03 to 05 gap closing, 06 imbalance losses, 07 soft labels, 08 blur-aware, 09 RETFound, 10 stacked recipe). Each has `hypothesis.md` (written before the run), the notebook `NN_name.ipynb`, `results.md`, `results/` and, for Kaggle runs, `kaggle/` (kernel builder, generated kernels, `executed.ipynb`). See `experiments/README.md`.
- `docs/`: `reports/` (Full_Project_Report.pdf, earlier reports, `latex/` sources, `full_report.html`), `slides/`, `literature/`, `gap-analysis/`.
- `tools/attach_kaggle_outputs.py`: rebuilds `executed.ipynb` (cells plus outputs) from a Kaggle kernel log.

## Rules
- This is a reproduction study. Make no clinical claims.
- The test set (550 images, stratified 70/15/15, seed 42) must never be used for any decision.
- Do not overwrite earlier experiments; add a new numbered folder for a new experiment.
- Write `hypothesis.md` with numeric predictions and commit it before running the experiment.
- Keep `.env` out of git (it holds the Kaggle key).

## Run and environment commands
- Work is notebook-based on Colab or Kaggle (T4). Local environment: `pip install -r requirements.txt`.
- Colab: upload `experiments/NN_name/NN_name.ipynb`, set a T4 GPU, Run all. Set `QUICK_TEST = True` first.
- Kaggle: `python experiments/NN_name/kaggle/build_kernels.py <kaggle-username>`, then `kaggle kernels push -p experiments/NN_name/kaggle/<kernel>` (Kaggle CLI 2.x; token in `KAGGLE_API_TOKEN`).
- Experiments 03 to 05 share one builder: `python experiments/03_levers_and_backbones/kaggle/build_kernels.py <user> 4b|4c|4d`.
- Rebuild a kernel's executed notebook: `python tools/attach_kaggle_outputs.py <kernel notebook.ipynb> <kernel log> <results dir>`.
- Report PDF: render `docs/reports/full_report.html` with headless Chrome (`--print-to-pdf`). LaTeX sources are in `docs/reports/latex/`.
