"""Stage 4 post-hoc check 1+2 (notebook section 12), run locally from the saved outputs.npz files.

Replicates the notebook's qwk / fit_cutoffs / majority_vote / pooled_bootstrap exactly. No retraining.
Usage: python posthoc_soft_vote.py <runs dir> <output dir>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

K = 5
ROUNDING = np.array([0.5, 1.5, 2.5, 3.5])


def qwk(y, pred, k=K):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    e = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    den = (w * e).sum()
    return float(1 - (w * cm).sum() / den) if den > 0 else 0.0


def severity(out):
    return out @ np.arange(K) if out.shape[-1] == K else out[..., 0]


def apply_cutoffs(s, c):
    return np.digitize(s, c).astype(int)


def fit_cutoffs(s, y, step=0.01, passes=3):
    c = ROUNDING.copy()
    best = qwk(y, apply_cutoffs(s, c))
    for _ in range(passes):
        for i in range(4):
            lo = c[i - 1] + step if i > 0 else -0.5
            hi = c[i + 1] - step if i < 3 else K - 0.5
            for v in np.arange(lo, hi + 1e-9, step):
                t = c.copy(); t[i] = v
                q = qwk(y, apply_cutoffs(s, t))
                if q > best:
                    best, c = q, t
    return c


def majority_vote(sp):
    votes = sp.argmax(-1)
    tally = np.stack([(votes == c).sum(0) for c in range(K)], 1).astype(float)
    tally += 1e-3 * sp.mean(0)
    return tally.argmax(1)


def pooled_bootstrap(triples, n_boot=2000, seed=0):
    y0 = triples[0][0]
    assert all(np.array_equal(y, y0) for y, _, _ in triples)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y0), len(y0))
        diffs[i] = np.mean([qwk(y[idx], n[idx]) - qwk(y[idx], o[idx]) for y, o, n in triples])
    return np.percentile(diffs, [2.5, 97.5])


def recall(y, p, c):
    m = y == c
    return float((p[m] == c).mean()) if m.any() else float("nan")


def main(runs_dir, out_dir):
    runs = {}
    for d in sorted(Path(runs_dir).iterdir()):
        if (d / "outputs.npz").exists():
            cfg = json.loads((d / "config.json").read_text())
            runs[d.name] = (cfg, dict(np.load(d / "outputs.npz")))

    decisions, metric_rows = {}, []
    for name, (cfg, r) in runs.items():
        if cfg["loss"] != "ce" or name.replace("ce_", "mse_", 1) not in runs:
            continue
        key_base = (cfg["dedupe"], cfg["seed"])
        mcfg, m = runs[name.replace("ce_", "mse_", 1)]
        for mode in ("single", "ensemble"):
            for loss, run in (("ce", r), ("mse", m)):
                vo, to = (run["best_val"], run["best_test"]) if mode == "single" else (run["snap_val"].mean(0), run["snap_test"].mean(0))
                if loss == "ce":
                    default = to.argmax(1) if mode == "single" else majority_vote(run["snap_test"])
                    arms = ("A", "B")
                else:
                    default = apply_cutoffs(severity(to), ROUNDING)
                    arms = ("C", "D")
                tuned = apply_cutoffs(severity(to), fit_cutoffs(severity(vo), run["y_val"]))
                decisions[key_base + (mode, arms[0])] = (run["y_test"], default)
                decisions[key_base + (mode, arms[1])] = (run["y_test"], tuned)
        probs = r["snap_test"].mean(0)
        pred = probs.argmax(1)
        y = r["y_test"]
        decisions[key_base + ("ensemble", "A_soft")] = (y, pred)
        metric_rows.append({"data": "dedupe" if cfg["dedupe"] else "main", "seed": cfg["seed"], "mode": "ensemble",
                            "arm": "A_soft", "qwk": qwk(y, pred), "accuracy": float((y == pred).mean()),
                            "recall_Severe": recall(y, pred, 3), "recall_PDR": recall(y, pred, 4),
                            "off_by_2_or_more": int((np.abs(y - pred) >= 2).sum())})

    # sanity check against the committed results_per_seed.csv (arms A-D must reproduce exactly)
    pairs = sorted({k[:2] for k in decisions})
    posthoc = [("ensemble", "D", "A"), ("ensemble", "D", "A_soft"), ("ensemble", "A_soft", "A"), ("ensemble", "B", "A"),
               ("ensemble", "C", "A"), ("ensemble", "D", "C"),
               ("single", "D", "A"), ("single", "B", "A"), ("single", "C", "A"), ("single", "D", "C")]
    rows = []
    for mode, new, old in posthoc:
        triples = {p: (decisions[p + (mode, old)][0], decisions[p + (mode, old)][1], decisions[p + (mode, new)][1]) for p in pairs}
        groups = [("dedupe" if p[0] else "main", str(p[1]), [t]) for p, t in triples.items() if "A_soft" in (new, old)]
        main_t = [t for p, t in triples.items() if not p[0]]
        groups.append(("main", f"mean of {len(main_t)}", main_t))
        for data, seed, ts in groups:
            lo, hi = pooled_bootstrap(ts)
            delta = float(np.mean([qwk(y, n) - qwk(y, o) for y, o, n in ts]))
            rows.append({"data": data, "seed": seed, "mode": mode, "contrast": f"{new} - {old}", "delta_qwk": round(delta, 4),
                         "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "excludes_0": bool(lo > 0 or hi < 0)})

    per_arm = []
    for (dd, s, mode, arm), (y, p) in sorted(decisions.items(), key=str):
        per_arm.append({"data": "dedupe" if dd else "main", "seed": s, "mode": mode, "arm": arm, "qwk": round(qwk(y, p), 4)})

    out = Path(out_dir)
    for fname, data in (("posthoc_soft_vote.csv", metric_rows), ("posthoc_bootstrap.csv", rows), ("posthoc_qwk_check.csv", per_arm)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
        print(f"== {fname}")
        for row in data:
            print(row)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
