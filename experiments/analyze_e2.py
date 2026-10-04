"""E2 analysis (protocol e2_protocol.md sections 5-7): screening curves, learned confirmation Q1-Q4, decomposition, strata, reproduction.

    python experiments/analyze_e2.py            # -> docs/reports/e2_tables.md, results/e2/analysis.json, docs/reports/fig_e2_*.png

Benefit-type quantity b (positive = good). Verdicts: material improvement (CI_lo > D), material harm (CI_hi < -D), equivalent (CI inside (-D, D)),
noninferior (CI_lo > -D otherwise), inconclusive. D = clip(0.10 V2_true, 0.002 VOI_clean, 0.02 VOI_clean) per cell on the confirmation set.
Learned intervals: 95% t over training seeds (10), conditional on the shared test set; the test-set standard error is reported separately.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE_COLORS = {0.0: "tab:blue", 0.5: "tab:green", 1.0: "tab:red"}


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    return float(x.mean()), (float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))


def dmin(v2, voi):
    return min(max(0.10 * v2, 0.002 * voi), 0.02 * voi)


def verdict(lo, hi, d):
    if lo > d:
        return "material improvement"
    if hi < -d:
        return "material harm"
    if lo > -d and hi < d:
        return "equivalent"
    if lo > -d:
        return "noninferior"
    return "inconclusive"


def f4(x):
    return f"{x:+.4f}"


def screening(md, out_json, figdir):
    md.append("## 1. Exact screening (exploratory; exact clean law; 100,000 prefixes per physics, master 8101)\n")
    md.append("B_exact = E_true(naive) − E_true(exact composition with q_assumed): positive means accounting for the channel with the (wrong) assumed prior still beats ignoring it. "
              "Paired prefix-level 95% intervals. Predicted-safe = location prior correct and (β_a ≤ β_t or odds(β_a) ≤ 2 odds(β_t)); there exact composition must not lose to naive on any record.\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False
    for p in (0, 1):
        path = os.path.join(ROOT, "results", "e2", "screen", f"phys{p}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rows = d["rows"]
        qw = d["q_w"]
        prov = d["provenance"]
        md.append(f"### Physics q_w = {qw} (code `{prov['git_commit'][:8]}`, dirty code: {prov['dirty'] or 'none'})\n")
        safe = [r for r in rows if r["predicted_safe"]]
        viol = sum(r["violations"] or 0 for r in safe)
        ctrl = max(r["E_R1"] for r in rows if r["alpha_t"] == r["alpha_a"] and r["beta_t"] == r["beta_a"])
        md.append(f"Per-record check: {len(safe)} predicted-safe (β_t, β_a, α) pairs × {d['n'] * 9:,} records, **{viol} violations**; exact-match controls (R1 = R0) max regret = {ctrl:.1e}; decomposition max error {max(r['decomp_err'] for r in rows):.1e}.\n")
        out_json[f"screen|{p}|violations_in_safe_region"] = viol
        md.append("**Rate axis** (location prior correct, α_a = α_t): B_exact ± 95% CI by β_a; `*` marks predicted-safe pairs; the break-even is the smallest β_a above the best value where B_exact ≤ 0 (CI upper bound ≤ 0 marked `<`).\n")
        betas_a = sorted({r["beta_a"] for r in rows if r["axis"] == "rate"})
        md.append("| α | β_t | V_2,true (naive regret) | " + " | ".join(f"{b:g}" for b in betas_a) + " | break-even |\n|---|---|---|" + "---|" * (len(betas_a) + 1))
        if have_plt:
            fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
        for a in (0.0, 0.5, 1.0):
            for bi, bt in enumerate((0.0, 0.05, 0.2, 0.5)):
                rr = sorted([r for r in rows if r["axis"] == "rate" and r["alpha_t"] == a and r["beta_t"] == bt], key=lambda r: r["beta_a"])
                cells = []
                be = "none ≤0.5"
                for r in rr:
                    h = 1.96 * r["B_exact_se"]
                    mark = "*" if r["predicted_safe"] else ""
                    lo_neg = (r["B_exact"] + h) < 0
                    cells.append(f"{r['B_exact']:+.3f}{mark}{'<' if lo_neg else ''}")
                    out_json[f"screen|{p}|rate|a{a}|bt{bt}|ba{r['beta_a']}"] = dict(B=r["B_exact"], half95=h, V2_true=r["V2_true"], safe=r["predicted_safe"])
                peak = max(rr, key=lambda r: r["B_exact"])
                for r in rr:
                    if r["beta_a"] > peak["beta_a"] and r["B_exact"] <= 0:
                        be = f"β_a ≈ {r['beta_a']:g}"
                        break
                md.append(f"| {a:g} | {bt:g} | {rr[0]['V2_true']:.4f} | " + " | ".join(cells) + f" | {be} |")
                if have_plt:
                    ax = axes[p if False else 0][bi] if False else axes[0][bi]
        if have_plt:
            for bi, bt in enumerate((0.0, 0.05, 0.2, 0.5)):
                for a in (0.0, 0.5, 1.0):
                    rr = sorted([r for r in rows if r["axis"] == "rate" and r["alpha_t"] == a and r["beta_t"] == bt], key=lambda r: r["beta_a"])
                    x = [r["beta_a"] for r in rr]
                    axes[0][bi].errorbar(x, [r["B_exact"] for r in rr], yerr=[1.96 * r["B_exact_se"] for r in rr], marker="o", ms=3, color=RATE_COLORS[a], label=f"α={a:g}")
                    axes[1][bi].plot(x, [r["D_none_R1"] for r in rr], marker="o", ms=3, color=RATE_COLORS[a])
                axes[0][bi].axhline(0, color="k", lw=0.6)
                axes[0][bi].axvline(bt, color="gray", ls=":", lw=0.8)
                axes[1][bi].axhline(0, color="k", lw=0.6)
                axes[0][bi].set_title(f"β_t = {bt:g}")
                axes[1][bi].set_xlabel("assumed β_a")
            axes[0][0].set_ylabel("B_exact (regret saved vs naive)")
            axes[1][0].set_ylabel("realised cost vs naive,\nclean records (D_none)")
            axes[0][0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"E2 exact screening, q_w = {qw}: benefit of exact composition with wrong β_a (dotted: β_a = β_t)")
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, f"fig_e2_screen_phys{p}.png"), dpi=110)
            plt.close(fig)
        md.append("\n**Location axis** (β = 0.2 both sides): rows α_true, columns α_assumed; B_exact (E_true(exact composition) in brackets).\n")
        la = sorted({r["alpha_a"] for r in rows if r["axis"] == "location"})
        md.append("| α_true \\ α_assumed | " + " | ".join(f"{x:g}" for x in la) + " |\n|---|" + "---|" * len(la))
        for at in la:
            cells = []
            for aa in la:
                r = next(r for r in rows if r["axis"] == "location" and r["alpha_t"] == at and r["alpha_a"] == aa)
                cells.append(f"{r['B_exact']:+.3f} [{r['E_R1']:.3f}]")
                out_json[f"screen|{p}|loc|at{at}|aa{aa}"] = dict(B=r["B_exact"], half95=1.96 * r["B_exact_se"], E_R1=r["E_R1"], V2_true=r["V2_true"])
            md.append(f"| {at:g} | " + " | ".join(cells) + " |")
        md.append("")
        md.append("**Strata for selected rate-axis pairs** (exact composition vs naive; realised hidden-mode costs, D = composed − naive, positive = costlier; mean ± 95% CI):\n")
        md.append("| α | β_t | β_a | B_exact | raw cost gain | D (all) | D clean records | D corrupted | D natural-switch |\n|---|---|---|---|---|---|---|---|---|")
        for a in (0.0, 1.0):
            for bt, ba in ((0.05, 0.05), (0.05, 0.1), (0.05, 0.2), (0.05, 0.5), (0.2, 0.05), (0.2, 0.2), (0.2, 0.5)):
                r = next(r for r in rows if r["axis"] == "rate" and r["alpha_t"] == a and r["beta_t"] == bt and r["beta_a"] == ba)
                pm = lambda k: f"{r[k]:+.3f} ± {1.96 * r[k + '_se']:.3f}"
                md.append(f"| {a:g} | {bt:g} | {ba:g} | {r['B_exact']:+.4f} | {r['cost_gain']:+.4f} | {pm('D_R1')} | {pm('D_none_R1')} | {pm('D_cor_R1')} | {pm('D_switch_R1')} |")
        md.append("")


def learned(md, out_json, figdir):
    base = os.path.join(ROOT, "results", "e2")
    ex_path = os.path.join(base, "exact_confirm.json")
    units = sorted(glob.glob(os.path.join(base, "units", "*.json")))
    if not os.path.exists(ex_path) or not units:
        return
    ex = json.load(open(ex_path))
    exrows = {r["cell"]: r for r in ex["rows"]}
    ex_arr = np.load(os.path.join(base, "arrays", "exact.npz"))
    U = [json.load(open(p)) for p in units]
    prov = U[0]["provenance"]
    md.append("## 2. Learned confirmation (lead physics q_w = 0.02; confirmation test set: 20,000 independent prefixes, master 8201)\n")
    md.append(f"Code `{prov['git_commit'][:8]}` + patch `{prov['patch_sha256']}` (dirty code: {prov['dirty_code'] or 'none'}). Seeds {sorted({u['seed'] for u in U})}. "
              "Arms receive the same q_assumed; naive versions are the same trained model with no channel (β=0 composition / zero conditioning).\n")
    byNS = defaultdict(list)
    for u in U:
        byNS[u["N"]].append(u)
    cells = list(exrows)
    E = {}           # (N, cell, arm) -> {seed: row}
    arr = {}
    for u in U:
        for r in u["rows"]:
            E.setdefault((u["N"], r["cell"], r["arm"]), {})[u["seed"]] = r
        z = np.load(os.path.join(base, "arrays", f"N{u['N']}_s{u['seed']}.npz"))
        for k in z.files:
            arr.setdefault((u["N"], k), {})[u["seed"]] = z[k]

    def seed_vec(N, cell, arm, key="E"):
        d = E[(N, cell, arm)]
        s = sorted(d)
        return s, np.array([d[x][key] for x in s])

    def test_se(N, cell, bx, by):
        """prefix-level SE of the seed-averaged per-prefix difference y_by - y_bx (b positive when bx has lower regret)."""
        sx, sy = arr[(N, f"{cell}|{bx}")], arr[(N, f"{cell}|{by}")]
        s = sorted(set(sx) & set(sy))
        dd = np.mean([sy[i] - sx[i] for i in s], axis=0)
        return float(dd.std(ddof=1) / np.sqrt(len(dd)))

    # --- reproduction check
    md.append("### Reproduction check (seeds 1–5, pilot test set, P1 and P3): retrained models vs the pilot/E1 values\n")
    rep_rows, maxd = [], 0.0
    for N in sorted(byNS):
        for arm, src in (("A4", "learned_pilot"), ("A6pd-broad", "e1")):
            for pr in ("P1", "P3"):
                diffs = []
                for u in byNS[N]:
                    if u["seed"] > 5:
                        continue
                    prior_u = json.load(open(os.path.join(ROOT, "results", src, "units", f"phys0_N{N}_s{u['seed']}.json")))
                    old = next(r["E"] for r in prior_u["rows"] if r["arm"] == arm and r["prior"] == pr)
                    diffs.append(u["repro"][f"{arm}|{pr}"] - old)
                maxd = max(maxd, max(abs(x) for x in diffs))
                rep_rows.append(f"| {N} | {arm} | {pr} | {np.mean(diffs):+.2e} | {max(abs(x) for x in diffs):.2e} |")
    md.append("| N | arm | prior | mean (new − old) | max abs over seeds |\n|---|---|---|---|---|\n" + "\n".join(rep_rows))
    md.append(f"\nMaximum absolute reproduction difference: {maxd:.2e} (bit-identical if 0).\n")
    out_json["repro_max_abs_diff"] = maxd

    # --- primary comparisons
    md.append("### Confirmatory comparisons (b positive = good; D = Δ margin; verdicts per protocol §7; half-width ≤ D/2 marks the precision target)\n")
    md.append("Q1 = B_pipe(A4) = E(A4 naive) − E(A4). Q2 = E(A6pd) − E(A4). Q3 = E(A6pd-cov) − E(A4). Q4 = B_exact(R1) on the exact reference (prefix-level).\n")
    tallies = defaultdict(lambda: defaultdict(int))
    for N in sorted(byNS):
        md.append(f"\n**N = {N}**\n\n| cell | axis | V_2,true | D | Q1 B_pipe(A4) | verdict | Q2 A6pd−A4 | verdict | Q3 A6pd-cov−A4 | verdict | Q4 B_exact(R1) | verdict | prec. |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cells:
            er = exrows[c]
            D = dmin(er["V2_true"], er["voi_clean"])
            cols = []
            prec = []
            for qid, (bx, by, mode) in {"Q1": ("A4", "A4n", "seed"), "Q2": ("A4", "A6pd", "seed"), "Q3": ("A4", "A6c", "seed")}.items():
                s1, ex_ = seed_vec(N, c, bx)
                s2, ey = seed_vec(N, c, by)
                b = ey - ex_
                m, h = tci(b)
                v = verdict(m - h, m + h, D)
                se_t = test_se(N, c, bx, by)
                cols.append(f"{m:+.4f} ± {h:.4f} (test SE {se_t:.4f})")
                cols.append(v)
                prec.append("y" if h <= D / 2 else "n")
                tallies[(qid, v)][N] += 1
                out_json[f"{qid}|{N}|{c}"] = dict(b=m, half95=h, test_se=se_t, D=D, verdict=v)
            y1, y2 = ex_arr[f"{c}|R1"], ex_arr[f"{c}|R2"]
            dd = y2 - y1
            m, se = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
            v = verdict(m - 1.96 * se, m + 1.96 * se, D)
            cols += [f"{m:+.4f} ± {1.96 * se:.4f}", v]
            prec.append("y" if 1.96 * se <= D / 2 else "n")
            tallies[("Q4", v)][N] += 1
            out_json[f"Q4|{N}|{c}"] = dict(b=m, half95=1.96 * se, D=D, verdict=v)
            md.append(f"| {c} | {er['axis']} | {er['V2_true']:.4f} | {D:.4f} | " + " | ".join(cols) + f" | {''.join(prec)} |")
    md.append("\n**Verdict tallies over the 14 confirmatory cells**\n\n| comparison | N | verdicts |\n|---|---|---|")
    for qid in ("Q1", "Q2", "Q3", "Q4"):
        for N in sorted(byNS):
            md.append(f"| {qid} | {N} | " + ", ".join(f"{v}: {tallies[(qid, v)][N]}" for v in ("material improvement", "equivalent", "noninferior", "inconclusive", "material harm") if tallies[(qid, v)][N]) + " |")

    # --- secondary: B_pipe all arms, B_exact all arms, decomposition, strata, cost
    md.append("\n### Secondary: B_pipe (own naive − own aware), B_exact (exact naive − arm) and the error decomposition (seed means; E_true = learning + channel + cross)\n")
    for N in sorted(byNS):
        md.append(f"\n**N = {N}** — columns: B_pipe / B_exact / [learn, channel, cross] for each arm\n")
        hdr = "| cell | R1: B_exact [channel] | " + " | ".join(f"{a}: B_pipe / B_exact [learn, channel, cross]" for a in ("A4", "A5", "A6pd", "A6c")) + " |\n|---|---|" + "---|" * 4
        md.append(hdr)
        for c in cells:
            er = exrows[c]
            v2 = er["V2_true"]
            line = [c, f"{er['V2_true'] - er['R1']['E']:+.4f} [{er['R1']['E']:.4f}]"]
            for a in ("A4", "A5", "A6pd", "A6c"):
                _, Ea = seed_vec(N, c, a)
                _, En = seed_vec(N, c, a + "n")
                _, Lr = seed_vec(N, c, a, "learn")
                _, Ch = seed_vec(N, c, a, "channel")
                _, Cr = seed_vec(N, c, a, "cross")
                line.append(f"{np.mean(En - Ea):+.4f} / {v2 - np.mean(Ea):+.4f} [{np.mean(Lr):.4f}, {np.mean(Ch):.4f}, {np.mean(Cr):+.4f}]")
                out_json[f"sec|{N}|{c}|{a}"] = dict(B_pipe=float(np.mean(En - Ea)), B_exact=float(v2 - np.mean(Ea)), learn=float(np.mean(Lr)), channel=float(np.mean(Ch)), cross=float(np.mean(Cr)))
            md.append("| " + " | ".join(line) + " |")
    md.append("\n### Strata and raw cost (A4 and exact composition; realised hidden-mode costs vs the same pipeline's naive output; seed means ± 95% t)\n")
    md.append("| N | cell | arm | raw cost gain (naive − aware) | D (all) | D clean records | D corrupted | D natural-switch |\n|---|---|---|---|---|---|---|---|")
    for N in sorted(byNS):
        for c in cells:
            for a in ("A4",):
                row = []
                s, ca = seed_vec(N, c, a, "cost")
                _, cn = seed_vec(N, c, a + "n", "cost")
                row.append("{:+.4f} ± {:.4f}".format(*tci(cn - ca)))
                for k in ("D", "D_none", "D_cor", "D_switch"):
                    _, v = seed_vec(N, c, a, k)
                    row.append("{:+.3f} ± {:.3f}".format(*tci(v)))
                r1 = exrows[c]["R1"]
                md.append(f"| {N} | {c} | {a} | " + " | ".join(row) + " |")
    out_json["tallies"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in tallies.items()}

    # --- figure: learned benefit curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ai, N in enumerate(sorted(byNS)):
            for a, col in ((0.0, "tab:blue"), (1.0, "tab:red")):
                cs = [c for c in cells if exrows[c]["axis"] == "rate" and exrows[c]["alpha_t"] == a]
                cs.sort(key=lambda c: exrows[c]["beta_a"])
                x = [exrows[c]["beta_a"] for c in cs]
                for arm, ls in (("R1", "-"), ("A4", "--"), ("A6pd", ":")):
                    if arm == "R1":
                        y = [exrows[c]["V2_true"] - exrows[c]["R1"]["E"] for c in cs]
                        axes[ai].plot(x, y, ls, color=col, marker="o", ms=3, label=f"exact R1 α={a:g}")
                    else:
                        y = [np.mean(seed_vec(N, c, arm + "n")[1] - seed_vec(N, c, arm)[1]) for c in cs]
                        axes[ai].plot(x, y, ls, color=col, marker="s", ms=3, label=f"{arm} B_pipe α={a:g}")
            axes[ai].axhline(0, color="k", lw=0.6)
            axes[ai].axvline(0.2, color="gray", ls=":", lw=0.8)
            axes[ai].set_title(f"β_t = 0.2, N = {N}")
            axes[ai].set_xlabel("assumed β_a")
        axes[0].set_ylabel("benefit over ignoring the channel")
        axes[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig_e2_learned.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        md.append(f"(figure skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "reports", "e2_tables.md"))
    args = ap.parse_args()
    figdir = os.path.dirname(args.out)
    md = ["# E2 tables (generated by `experiments/analyze_e2.py`; protocol: docs/e2_protocol.md)\n"]
    out_json = {}
    screening(md, out_json, figdir)
    learned(md, out_json, figdir)
    keys = list(out_json)
    assert len(keys) == len(set(keys))
    open(args.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out_json, open(os.path.join(ROOT, "results", "e2", "analysis.json"), "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
