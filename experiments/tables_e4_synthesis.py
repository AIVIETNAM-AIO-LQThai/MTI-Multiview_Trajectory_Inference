"""Markdown tables for docs/research_synthesis.md section 5.4-5.6, generated from saved E4 units only.  python experiments/tables_e4_synthesis.py"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_e4 import ci90, classify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage1", "units", "*.json")))]
main = [u for u in U if u["kind"] == "main"]
cells = defaultdict(list)
for u in main:
    cells[(u["pname"], u["L"], u["truth"])].append(u)


def reg(us, arm, n):
    return np.array([u["arms"][arm][str(n)]["regret"] for u in us if str(n) in u["arms"][arm]])


out = ["### Table A. Primary contrasts at n = 10⁴ (regret in units of VOI_clean; true-law oracle regret = 0; R = 20)", "",
       "| physics | L | truth | Δ | EX-MLE | J-MLE | ETA⁻ (η = 0.03) | J − EX [90% CI] | ETA⁻ − J [90% CI] | J vs EX | ETA⁻ vs J |", "|" + "---|" * 11]
for pn in ("lead", "sec"):
    for L in (8, 4):
        for tn in ("T0", "T3", "T4", "T6"):
            us = cells[(pn, L, tn)]
            voi, dl = us[0]["voi"], us[0]["delta"]
            j, e, m = reg(us, "J-MLE", 10000), reg(us, "EX-MLE", 10000), reg(us, "ETA--MLE", 10000)
            d1, d2 = j - e, m - j
            f = lambda d: (lambda a, lo, hi: f"{a / voi:+.4f} [{lo / voi:+.4f}, {hi / voi:+.4f}]")(*ci90(d))
            c1 = {"a worse": "J worse", "a better": "J better"}.get(classify(d1, dl), classify(d1, dl))
            c2 = {"a worse": "J better", "a better": "J worse"}.get(classify(d2, dl), classify(d2, dl))
            out.append(f"| {pn} | {L} | {tn} | {dl / voi:.4f} | {e.mean() / voi:.5f} | {j.mean() / voi:.5f} | {m.mean() / voi:.4f} | {f(d1)} | {f(d2)} | {c1} | {c2} |")

out += ["", "### Table B. Regret at small n relative to the margin (mean J-MLE and EX-MLE regret / Δ, n = 300 and 10⁴)", "",
        "| physics | L | truth | J-MLE n=300 | EX-MLE n=300 | J-MLE n=10⁴ | EX-MLE n=10⁴ |", "|---|---|---|---|---|---|---|"]
for pn in ("lead", "sec"):
    for L in (4, 8):
        for tn in ("T3", "T4", "T6"):
            us = cells[(pn, L, tn)]
            dl = us[0]["delta"]
            r = lambda a, n: reg(us, a, n).mean() / dl
            out.append(f"| {pn} | {L} | {tn} | {r('J-MLE', 300):.2f} | {r('EX-MLE', 300):.2f} | {r('J-MLE', 10000):.3f} | {r('EX-MLE', 10000):.3f} |")

out += ["", "### Table C. Clean-set comparison at n = 10⁴ (cells with L ∈ {4, 8}: 16 cells; class of CV relative to J-MLE)", "", "| clean prefixes n_c | equivalent | inconclusive | CV worse |", "|---|---|---|---|"]
for nc in (30, 100, 1000):
    c = defaultdict(int)
    for k, us in cells.items():
        if k[1] >= 4:
            d = reg(us, f"CV-{nc}", 10000) - reg(us, "J-MLE", 10000)
            c[classify(d, us[0]["delta"])] += 1
    out.append(f"| {nc} | {c['equivalent']} | {c['inconclusive']} | {c['a worse']} |")

f1 = [u for u in U if u["kind"] == "f1"]
out += ["", "### Table D. F1 (L = 8, lead physics, no corruption; R = 10) against the in-family control (R = 20): inferred β̂ and regret / VOI_clean", "",
        "| arm | n | F1 β̂ | in-family β̂ | F1 regret/VOI | in-family regret/VOI | Δ/VOI |", "|---|---|---|---|---|---|---|"]
fam = cells[("lead", 8, "T0")]
t0 = [u for u in f1 if u["truth"] == "T0"]
for arm in ("EX-MLE", "J-MLE"):
    for n in (10000, 30000):
        a = np.mean([u["arms"][arm][str(n)]["beta"] for u in t0])
        b = np.mean([u["arms"][arm][str(n)]["beta"] for u in fam if str(n) in u["arms"][arm]])
        ra = np.mean([u["arms"][arm][str(n)]["regret"] for u in t0]) / t0[0]["voi"]
        rb = np.mean([u["arms"][arm][str(n)]["regret"] for u in fam if str(n) in u["arms"][arm]]) / fam[0]["voi"]
        out.append(f"| {arm} | {n} | {a:.4f} | {b:.4f} | {ra:.4f} | {rb:.5f} | {t0[0]['delta'] / t0[0]['voi']:.4f} |")
cv = [(n, np.mean([u["arms"]["CV-1000"][str(n)]["beta"] for u in t0]), np.mean([u["arms"]["CV-1000"][str(n)]["regret"] for u in t0]) / t0[0]["voi"]) for n in (10000, 30000)]
out.append("")
out.append("CV-1000 on F1-T0 (clean prefixes from the same mixture): " + "; ".join(f"n={n}: β̂ {b:.4f}, regret/VOI {r:.4f}" for n, b, r in cv))
txt = "\n".join(out) + "\n"
open(os.path.join(ROOT, "results", "e4", "stage1", "synthesis_tables.md"), "w", encoding="utf-8").write(txt)
print(txt)
