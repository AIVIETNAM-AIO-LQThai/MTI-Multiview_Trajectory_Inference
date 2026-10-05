"""E4 analysis (protocol docs/e4_protocol.md section 5 and amendment A1). python experiments/analyze_e4.py [--units results/e4/stage1/units]

Margins: Delta_t per cell (stored). Replicate-paired 90% t-intervals. Classification of a difference d = E(a) - E(b) of regrets:
  equivalent  : CI within [-Delta, Delta];  a worse: CI_lo > Delta;  a better: CI_hi < -Delta;  otherwise inconclusive.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_SE = {}                                                   # (physics, L, truth) -> predicted SE(eta_hat) at n = 1e4 (stage 0)


def load(units):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(units, "*.json")))]


def ci90(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def classify(d, delta):
    m, lo, hi = ci90(d)
    if np.isnan(lo):
        return "n/a"
    if lo > delta:
        return "a worse"
    if hi < -delta:
        return "a better"
    if lo >= -delta and hi <= delta:
        return "equivalent"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=os.path.join(ROOT, "results", "e4", "stage1", "units"))
    args = ap.parse_args()
    for f in glob.glob(os.path.join(ROOT, "results", "e4", "stage0", "units", "*.json")):
        r = json.load(open(f))
        PRED_SE[(r["physics"], r["L"], r["truth"])] = r["se_pred_n1e4"]
    U = load(args.units)
    main_u = [u for u in U if u["kind"] == "main"]
    cells = defaultdict(list)
    for u in main_u:
        cells[(u["pname"], u["L"], u["truth"])].append(u)

    def col(units, arm, n, key):
        return np.array([u["arms"][arm][str(n)][key] for u in units if str(n) in u["arms"][arm]])

    def paired(units, a, b, n):
        sel = [u for u in units if str(n) in u["arms"][a]]
        return np.array([u["arms"][a][str(n)]["regret"] - u["arms"][b][str(n)]["regret"] for u in sel]), (sel[0]["delta"] if sel else np.nan)

    out = [f"# E4 stage-1 summary (units: {len(U)}; main {len(main_u)}, F1 {len(U) - len(main_u)})", ""]
    NS = (300, 1000, 3000, 10000, 30000, 100000)
    out += ["## P1: primary contrasts at n = 1e4 (regret difference / VOI_clean, 90% CI; class in brackets)", "",
            "| physics | L | truth | R | Delta/VOI | J-MLE - EX-MLE | ETA--MLE - J-MLE | ETA+-MLE - J-MLE | J-MLE regret/VOI | eta_hat mean (sd) | beta_hat mean |", "|" + "---|" * 11]
    for key in sorted(cells):
        us = cells[key]
        voi = us[0]["voi"]
        d1, dl = paired(us, "J-MLE", "EX-MLE", 10000)
        d2, _ = paired(us, "ETA--MLE", "J-MLE", 10000)
        d3, _ = paired(us, "ETA+-MLE", "J-MLE", 10000)
        f = lambda d: (lambda m, lo, hi: f"{m / voi:+.4f} [{lo / voi:+.4f},{hi / voi:+.4f}]")(*ci90(d))
        # class of (a=EX or ETA) vs J: report "J better/worse" by flipping sign for the first contrast
        c1 = classify(d1, dl)
        c1 = {"a worse": "J worse", "a better": "J better"}.get(c1, c1)
        c2 = {"a worse": "J better", "a better": "J worse"}.get(classify(d2, dl), classify(d2, dl))
        c3 = {"a worse": "J better", "a better": "J worse"}.get(classify(d3, dl), classify(d3, dl))
        eh, bh = col(us, "J-MLE", 10000, "eta"), col(us, "J-MLE", 10000, "beta")
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(eh)} | {dl / voi:.4f} | {f(d1)} ({c1}) | {f(d2)} ({c2}) | {f(d3)} ({c3}) | "
                   f"{col(us, 'J-MLE', 10000, 'regret').mean() / voi:.4f} | {eh.mean():.4f} ({eh.std(ddof=1):.4f}) | {bh.mean():.3f} |")

    out += ["", "## P2: crossover n (smallest n at which J-MLE is equivalent to EX-MLE, and stays so at larger n)", "", "| physics | L | truth | crossover n | classes by n (300 … 1e5) |", "|---|---|---|---|---|"]
    for key in sorted(cells):
        us = cells[key]
        cls = []
        for n in NS:
            d, dl = paired(us, "J-MLE", "EX-MLE", n)
            c = classify(d, dl) if len(d) else "—"
            cls.append({"a worse": "worse", "a better": "better", "equivalent": "eq", "inconclusive": "inc"}.get(c, c))
        cross = None
        for i, n in enumerate(NS):
            if all(c in ("eq", "better") for c in cls[i:] if c != "—") and cls[i] != "—":
                cross = n
                break
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {cross} | {' '.join(cls)} |")

    out += ["", "## P3: L = 3 negative control (T3, T6): spread of eta_hat and regret gap (no convergence expected)", "",
            "| physics | truth | n | SD(eta_hat J-MLE) | mean beta_hat | J-MLE regret/VOI | Delta/VOI |", "|---|---|---|---|---|---|---|"]
    for pn in ("lead", "sec"):
        for tn in ("T3", "T6"):
            us = cells.get((pn, 3, tn), [])
            for n in (1000, 10000, 100000):
                if not us:
                    continue
                eh = col(us, "J-MLE", n, "eta")
                if len(eh) > 1:
                    out.append(f"| {pn} | {tn} | {n} | {eh.std(ddof=1):.4f} | {col(us, 'J-MLE', n, 'beta').mean():.3f} | {col(us, 'J-MLE', n, 'regret').mean() / us[0]['voi']:.4f} | {us[0]['delta'] / us[0]['voi']:.4f} |")

    out += ["", "## P4: CV-n_c vs J-MLE at n = 1e4 (class of CV relative to J-MLE), and CV eta_hat sd", "",
            "| physics | L | truth | CV-30 | CV-100 | CV-1000 | sd eta_cv (30/100/1000) |", "|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        us = cells[key]
        cs, sds = [], []
        for nc in (30, 100, 1000):
            d, dl = paired(us, f"CV-{nc}", "J-MLE", 10000)
            m, lo, hi = ci90(d)
            cs.append(f"{classify(d, dl)} ({m / us[0]['voi']:+.4f})")
            sds.append(f"{np.std([u['eta_cv'][str(nc)] for u in us], ddof=1):.4f}")
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {cs[0]} | {cs[1]} | {cs[2]} | {'/'.join(sds)} |")

    out += ["", "## P5: J-MOM vs J-MLE regret ratio (mean regret ratio) and eta_hat sd at n = 1e4", "", "| physics | L | truth | n=1e3 | n=1e4 | n=1e5 | sd eta J-MOM 1e4 |", "|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        us = cells[key]
        rr = []
        for n in (1000, 10000, 100000):
            a, b = col(us, "J-MOM", n, "regret"), col(us, "J-MLE", n, "regret")
            rr.append(f"{a.mean() / max(b.mean(), 1e-12):.2f}" if len(a) else "—")
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {rr[0]} | {rr[1]} | {rr[2]} | {col(us, 'J-MOM', 10000, 'eta').std(ddof=1):.4f} |")

    out += ["", "## P7: measured SD(eta_hat) of J-MLE vs stage-0 predicted SE (interior truths T3, T6 only, amendment A1); band [0.7, 1.4]", "",
            "| physics | L | truth | n | SD measured | SE predicted | ratio |", "|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        if key[2] not in ("T3", "T6") or key[1] < 4:
            continue
        us = cells[key]
        for n in (10000, 30000, 100000):
            eh = col(us, "J-MLE", n, "eta")
            if len(eh) > 2:
                pred = PRED_SE[key] * np.sqrt(1e4 / n)
                out.append(f"| {key[0]} | {key[1]} | {key[2]} | {n} | {eh.std(ddof=1):.5f} | {pred:.5f} | {eh.std(ddof=1) / pred:.2f} |")

    f1 = [u for u in U if u["kind"] == "f1"]
    if f1:
        out += ["", "## P6 (descriptive): F1, record-heterogeneous persistence eta in {0.02, 0.08} (marginal 0.05)", "",
                "| truth | n | arm | eta_hat mean | beta_hat mean | regret/VOI | harm rate vs naive (B < -Delta) |", "|---|---|---|---|---|---|---|"]
        for tn in sorted({u["truth"] for u in f1}):
            us = [u for u in f1 if u["truth"] == tn]
            voi, dl = us[0]["voi"], us[0]["delta"]
            for n in sorted({int(k) for u in us for k in u["arms"]["J-MLE"]}):
                for arm in ("EX-MLE", "ETA--MLE", "ETA+-MLE", "J-MLE", "CV-1000"):
                    rows = [u["arms"][arm][str(n)] for u in us if str(n) in u["arms"][arm]]
                    harm = np.mean([(r["regret_naive_pipe"] - r["regret"]) < -dl for r in rows])
                    out.append(f"| {tn} | {n} | {arm} | {np.mean([r['eta'] for r in rows]):.4f} | {np.mean([r['beta'] for r in rows]):.3f} | "
                               f"{np.mean([r['regret'] for r in rows]) / voi:.4f} | {harm:.2f} |")
    txt = "\n".join(out) + "\n"
    open(os.path.join(os.path.dirname(args.units), "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
