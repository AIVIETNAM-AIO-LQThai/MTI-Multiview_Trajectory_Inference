"""Figure from saved E4 stage-1 units only: mean regret / VOI_clean against n for J-MLE, EX-MLE (known eta) and ETA- (eta fixed at 0.03),
L in {3, 4, 8}, truths T0 and T3, both physics, with the margin Delta/VOI. Replicate-mean with 90% t-interval band.

    python experiments/plot_e4.py   ->   docs/reports/fig_e4_regret.png
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = (300, 1000, 3000, 10000, 30000, 100000)
ARMS = (("EX-MLE", "tab:green", "known eta (EX-MLE)"), ("J-MLE", "tab:blue", "joint (J-MLE)"), ("ETA--MLE", "tab:red", "eta fixed 0.03"))
FLOOR = 3e-6


def main():
    cells = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", "e4", "stage1", "units", "*_L*_T*_r*.json")):
        u = json.load(open(f))
        if u["kind"] == "main" and u["truth"] in ("T0", "T3"):
            cells[(u["pname"], u["truth"], u["L"])].append(u)
    rows = [("lead", "T0"), ("lead", "T3"), ("sec", "T0"), ("sec", "T3")]
    fig, axes = plt.subplots(4, 3, figsize=(11, 11), sharex=True)
    for ri, (pn, tn) in enumerate(rows):
        for ci, L in enumerate((3, 4, 8)):
            ax = axes[ri, ci]
            us = cells[(pn, tn, L)]
            voi, delta = us[0]["voi"], us[0]["delta"]
            for arm, col, lab in ARMS:
                xs, mu, lo, hi = [], [], [], []
                for n in NS:
                    y = np.array([u["arms"][arm][str(n)]["regret"] for u in us if str(n) in u["arms"][arm]]) / voi
                    if len(y) < 2:
                        continue
                    h = stats.t.ppf(0.95, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
                    xs.append(n), mu.append(max(y.mean(), FLOOR)), lo.append(max(y.mean() - h, FLOOR)), hi.append(max(y.mean() + h, FLOOR))
                ax.plot(xs, mu, "-o", ms=3, color=col, label=lab)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.18)
            ax.axhline(delta / voi, color="k", ls=":", lw=1, label="margin")
            ax.set_xscale("log"), ax.set_yscale("log")
            ax.set_title(f"{pn}, {tn}, L={L}", fontsize=9)
            if ci == 0:
                ax.set_ylabel("regret / VOI_clean")
            if ri == 3:
                ax.set_xlabel("corrupted prefixes n")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("E4: regret vs the true-law oracle (regret 0). R=20 for n<=3e4, R=10 at n=1e5; band: 90% t-interval over replicates. "
                 "Values floored at 3e-6.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(ROOT, "docs", "reports", "fig_e4_regret.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
