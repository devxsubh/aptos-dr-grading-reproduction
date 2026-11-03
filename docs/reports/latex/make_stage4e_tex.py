"""Writes the Stage 4e LaTeX tables for final_report.tex from the CSVs of the dr-s4e-e2-final kernel.

Usage: python make_stage4e_tex.py [results dir, default experiments/06_imbalance_losses/results] [output dir, default generated]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "experiments" / "06_imbalance_losses" / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "generated"
OUT.mkdir(exist_ok=True)
read = lambda name: list(csv.DictReader(open(RES / name)))

LABEL = {"A": "A: CE, vote", "A_soft": r"A$_\text{soft}$: CE, averaged", "B": "B: CE, tuned",
         "E_vote": r"E$_\text{vote}$: focal, vote", "E_soft": r"E$_\text{soft}$: focal, averaged",
         "E_tuned": r"E$_\text{tuned}$: focal, tuned", "C": "C: MSE, rounding", "D": "D: MSE, tuned",
         "F_round": r"F$_\text{round}$: CB-MSE, rounding", "F_tuned": r"F$_\text{tuned}$: CB-MSE, tuned"}
ORDER = ["A", "A_soft", "B", "E_vote", "E_soft", "E_tuned", "C", "D", "F_round", "F_tuned"]

rows = [r for r in read("results_per_seed.csv") if r["group"] == "main" and r["mode"] == "ensemble"]
by_arm = defaultdict(list)
for r in rows:
    by_arm[r["arm"]].append(r)
f = lambda arm, col: [float(r[col]) for r in by_arm[arm]]

lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Stage 4e (post-test) with the Stage 4 arms, 10-snapshot ensemble, mean over seeds. "
         r"CB-MSE: class-balanced MSE. Columns as in Table~\ref{tab:stage4}.}",
         r"\label{tab:stage4e}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Arm & QWK & Sev. & PDR & $\ge$2 & Sens./Spec. \\", r"\midrule"]
for arm in ORDER:
    if arm not in by_arm:
        continue
    q = f(arm, "qwk")
    sd = stdev(q) if len(q) > 1 else 0.0
    lines.append(f"{LABEL[arm]} & ${mean(q):.3f}\\pm.{round(sd * 1000):03d}$ & {mean(f(arm, 'recall_Severe')):.2f} & "
                 f"{mean(f(arm, 'recall_PDR')):.2f} & {mean(f(arm, 'off_by_2_or_more')):.1f} & "
                 f".{round(mean(f(arm, 'ref_sensitivity')) * 1000):03d}/.{round(mean(f(arm, 'ref_specificity')) * 1000):03d} \\\\")
    if arm in ("B", "E_tuned", "D"):
        lines.append(r"\midrule")
if lines[-1] == r"\midrule":
    lines.pop()
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "stage4e_table.tex").write_text("\n".join(lines) + "\n")

# D - A_soft is the Stage 4 soft-vote check; the paper quotes it from experiments/02_ordinal_regression/results_posthoc.md instead.
boot = [r for r in read("paired_bootstrap.csv") if r["mode"] == "ensemble" and r["seed"].startswith("pooled")
        and r["contrast"] != "D - A_soft"]
verdict = read("prediction_check.csv")
SUB = {"A_soft": r"A$_\text{soft}$", "E_vote": r"E$_\text{vote}$", "E_soft": r"E$_\text{soft}$",
       "E_tuned": r"E$_\text{tuned}$", "F_round": r"F$_\text{round}$", "F_tuned": r"F$_\text{tuned}$"}


def subscripts(s):
    for k, v in SUB.items():
        s = s.replace(k, v)
    return s


tex_escape = lambda s: subscripts(s.replace(" - ", " $-$ ").replace(" -0.", " $-$0.").replace("&", r"\&").replace("%", r"\%").replace(">=", r"$\ge$").replace("<=", r"$\le$")
                        .replace("|", "$|$").replace("->", r"$\to$").replace("<", "$<$")
                        .replace(">", "$>$"))
# Caveats from experiments/06_imbalance_losses/results.md: Q3 holds in sign only, Q4 rests on one seed (per seed 0 / +0.17 / 0).
NOTE = {"Q3": r"$^a$", "Q4": r"$^b$"}
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Stage 4e predictions (fixed before the runs) and pooled paired-bootstrap contrasts "
         r"(ensemble, mean over seeds, 95\% CI).}", r"\label{tab:stage4e_pred}", r"\footnotesize",
         r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}p{0.07\columnwidth}p{0.5\columnwidth}p{0.3\columnwidth}c@{}}",
         r"\toprule", r"\# & Prediction & Observed & OK? \\", r"\midrule"]
for v in verdict:
    ok = {"True": "yes", "False": r"\textbf{no}"}.get(v["confirmed"], "n/a") + NOTE.get(v["id"], "")
    lines.append(f"{v['id']} & {tex_escape(v['prediction'])} & {tex_escape(v['observed'])} & {ok} \\\\")
lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{Contrast: $\Delta$QWK [95\% CI]}} \\"]
for b in boot:
    name = subscripts(b["contrast"].replace(" - ", " $-$ "))
    lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{{name}: ${float(b['delta_qwk']):+.3f}$ "
                 f"[${float(b['ci_low']):+.3f}$, ${float(b['ci_high']):+.3f}$]}} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\par\vspace{2pt}\parbox{\columnwidth}{\footnotesize $^a$In sign only; the CI includes 0. "
          r"$^b$On the mean only; per seed the gain is 0, +0.17 and 0 (five images in one seed).}", r"\end{table}"]
(OUT / "stage4e_predictions.tex").write_text("\n".join(lines) + "\n")

gc = RES / "posthoc_gradcam_corner_share.csv"
if gc.exists():
    shares = [float(r["top_left_share"]) for r in csv.DictReader(open(gc))]
    (OUT / "gradcam_share.tex").write_text(
        f"For the {len(shares)} images we inspected, that single cell holds {min(shares):.0%}--{max(shares):.0%} "
        f"of the raw map (median {sorted(shares)[len(shares) // 2]:.0%}).".replace("%", r"\%") + "\n")
print("wrote", *sorted(p.name for p in OUT.glob("*stage4e*")), "gradcam_share.tex" if gc.exists() else "")
