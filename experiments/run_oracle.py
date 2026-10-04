"""Run the oracle information/opportunity experiment.

    python experiments/run_oracle.py --config configs/oracle_v1.toml --out results/oracle_v1
    python experiments/run_oracle.py --config configs/oracle_v1.toml --cells primary_L8_qw0.5_b0.2 --out results/scratch

Stages per cell: clean thresholds -> pilot (variance, sets N_main) -> recalibrator fit/validation
(independent seeds) -> main run (fixed N, Rao-Blackwellised over corruption views) -> independent check
run (single-draw estimator + simulated decision costs). Results are written as JSON (+ summary.csv).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import scipy

from mti.experiment import (CHECK_COLS, LAG_METRICS, MASS_STATS, PROF_STATS, PROF_TYPES, REL_EDGES, Cell,
                            check_chunk, clean_thresholds, column_names, fit_chunk, flag_names, merge_sums,
                            rb_chunk)
from mti.metrics import Z95, batch_means_se, mean_ci, ratio_ci
from mti.params import ChannelSpec, Physics
from mti.recal import BinnedRecal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _work(task):
    kind, cell, master, chunk_id, n, recal, thr, do_profile = task
    if kind == "rb":
        return rb_chunk(cell, master, chunk_id, n, recal, thr, do_profile)
    if kind == "check":
        return check_chunk(cell, master, chunk_id, n)
    if kind == "fit":
        return fit_chunk(cell, master, chunk_id, n)
    raise ValueError(kind)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def provenance(config_path):
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = [l for l in git("status", "--porcelain").splitlines()
              if not l[3:].startswith(("results/", "docs/reports/"))]
    return dict(
        git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        dirty_code_files=status, config_sha256=hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), command=" ".join(sys.argv),
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def make_channel(c):
    """Cell channel: pi = "uniform" (default), "recent" (geometric in lag), or "lag0" (all corruption mass on the last recorded action, j = L-1)."""
    pi = c.get("pi", "uniform")
    if pi == "uniform":
        return ChannelSpec.uniform(c["L"], c["beta"])
    if pi == "lag0":
        return ChannelSpec.point_mass(c["L"], c["beta"], c["L"] - 1)
    if pi == "recent":   # pi_j proportional to 2^-(L-1-j): lag 0 has mass ~0.50
        w = [2.0 ** -(c["L"] - 1 - j) for j in range(c["L"])]
        return ChannelSpec(beta=c["beta"], pi=tuple(x / sum(w) for x in w))
    raise ValueError(f"unknown pi spec {pi!r}")


def make_cells(cfg):
    ph = cfg["physics"]
    cells = []
    for i, c in enumerate(cfg["cells"]):
        phys = Physics(rho=ph["rho"], c=ph["c"], q_a=ph["q_a"], q_w=c["q_w"], lam=ph["lam"], eta=c["eta"], L=c["L"])
        cells.append(Cell(c.get("cell_id", i), c["name"], c["role"], phys, make_channel(c)))
    return cells


def run_chunks(pool, kind, cell, master, n_total, chunk_n, recal=None, thr=None, n_profile=0):
    n_chunks = math.ceil(n_total / chunk_n)
    n_prof_chunks = math.ceil(n_profile / chunk_n) if n_profile else 0
    tasks = [(kind, cell, master, k, chunk_n, recal or {}, thr, k < n_prof_chunks) for k in range(n_chunks)]
    return list(pool.map(_work, tasks)), n_chunks * chunk_n


def summarize_rb(cell, merged, names, n_total, chan):
    n = merged["n"]
    mean, sd, se, half = mean_ci(n, merged["S1"], merged["S2"])
    cols = {k: dict(mean=mean[i], sd=sd[i], se=se[i], half95=half[i]) for i, k in enumerate(names)}
    L = cell.phys.L
    fnames = flag_names(L)
    sm, sse = ratio_ci(merged["SF"][:, None], merged["SFY"], merged["SFY2"])
    strata = {f: dict(frac=merged["SF"][i] / n, mean={k: sm[i, j] for j, k in enumerate(names) if not k.startswith("rc:")},
                      se={k: sse[i, j] for j, k in enumerate(names) if not k.startswith("rc:")})
              for i, f in enumerate(fnames)}
    lm, lsd, lse, lh = mean_ci(n, merged["lag_S1"], merged["lag_S2"])
    lag = {m: dict(mean=lm[i], half95=lh[i], view_names=["none"] + [f"loc{j}(lag{L - 1 - j})" for j in range(L)])
           for i, m in enumerate(LAG_METRICS)}
    rel = merged["rel"] / n
    chunk_rel = merged["rel_chunks"] / (n / merged["rel_chunks"].shape[0])
    gap = chunk_rel[:, :, 2] - chunk_rel[:, :, 1]
    gm, gse = batch_means_se(gap)
    reliability = [dict(bin=f"[{REL_EDGES[b]:.2f},{min(REL_EDGES[b + 1], 1.0):.2f})", mass=rel[b, 0],
                        mean_pred=rel[b, 1] / rel[b, 0] if rel[b, 0] > 0 else None,
                        mean_obs=rel[b, 2] / rel[b, 0] if rel[b, 0] > 0 else None,
                        gap_x_mass=gm[b], gap_se=gse[b], z=(gm[b] / gse[b] if gse[b] > 0 else None))
                   for b in range(len(REL_EDGES) - 1)]
    w = chan.view_weights()
    wp = w[w > 0]
    prior = dict(logscore_cat=float(-(wp * np.log(wp)).sum()), brier_cat=float(1 - (w**2).sum()),
                 brier_any=float(chan.beta * (1 - chan.beta)),
                 logloss_any=float(-(chan.beta * np.log(chan.beta) + (1 - chan.beta) * np.log(1 - chan.beta))) if 0 < chan.beta < 1 else 0.0,
                 top1_cor=float(chan.pi_arr.max()))
    out = dict(n=n_total, columns=cols, strata=strata, lag=lag, reliability=reliability, prior_only=prior)
    if "prof" in merged:
        pn = merged["prof_n"]
        pm = merged["prof"] / pn
        per_chunk = merged["prof_chunks"] / (pn / merged["prof_chunks"].shape[0])
        _, pse = batch_means_se(per_chunk)
        mm = merged["mass"] / pn
        _, mse = batch_means_se(merged["mass_chunks"] / (pn / merged["mass_chunks"].shape[0]))
        out["profile"] = dict(n_prefixes=pn, types=PROF_TYPES, stats=PROF_STATS, mean=pm, se=pse,
                              strata=["no corruption (theta=none)", "attacked (pi-weighted over locations)"],
                              mass_stats=MASS_STATS, mass=mm, mass_se=mse)
        out["verify"] = merged["verify"]
    return out


def _z(num, a, b):
    d = math.hypot(a, b)
    return num / d if d > 0 else None


def summarize_check(merged, rb_cols, rb_names):
    n = merged["n"]
    mean, sd, se, half = mean_ci(n, merged["S1"], merged["S2"])
    cols = {k: dict(mean=mean[i], se=se[i], half95=half[i]) for i, k in enumerate(CHECK_COLS)}
    dm, dsd, dse, _ = mean_ci(n, merged["dS1"], merged["dS2"])
    kinds = ["zero", "clean", "naive", "aware", "mode"]
    paired = {k: dict(mean_diff=dm[i], se=dse[i], z=dm[i] / dse[i] if dse[i] > 0 else None) for i, k in enumerate(kinds)}
    z = {}
    for m in ("V2", "I", "X", "AW", "D"):
        a, b = rb_cols[m], cols[m]
        z[f"main_vs_check_{m}"] = _z(a["mean"] - b["mean"], a["se"], b["se"])
    for k in kinds:
        a, b = rb_cols[f"cost_{k}"], cols[f"sim_{k}"]
        z[f"analytic_RB_vs_simulated_{k}"] = _z(a["mean"] - b["mean"], a["se"], b["se"])
    z["any_corruption_rate_vs_posterior"] = _z(cols["any_cor"]["mean"] - cols["any_pred"]["mean"],
                                               cols["any_cor"]["se"], cols["any_pred"]["se"])
    z["mu2_calibration"] = _z(cols["cal_mu2"]["mean"], cols["cal_mu2"]["se"], 0.0)
    z["muH_calibration"] = _z(cols["cal_mH"]["mean"], cols["cal_mH"]["se"], 0.0)
    return dict(n=n, columns=cols, paired_sim_minus_analytic=paired, z=z)


def run_cell(pool, cfg, cell, args, prov):
    mc, seeds = cfg["mc"], cfg["seeds"]
    L = cell.phys.L
    chunk_n = cfg["chunk_prefixes"][f"L{L}"]
    scale = args.scale
    t0 = time.time()
    res = dict(cell=dict(name=cell.name, role=cell.role, cell_id=cell.cell_id, physics=cell.phys.__dict__,
                         beta=cell.chan.beta, pi=list(cell.chan.pi)), provenance=prov)
    thr = clean_thresholds(cell, seeds["pilot"], n=max(2000, int(mc["thr_n"] * scale)))
    res["thresholds"] = thr
    controls = cell.role.startswith("control")
    names_nr = column_names(cell, [])

    if controls:
        n_main = max(chunk_n, int(mc["control_n"] * scale) // chunk_n * chunk_n)
        res["pilot"] = None
        recal = {}
        res["sizing"] = dict(rule="control (exact-zero check, fixed N)", N_main=n_main)
    else:
        pn = max(chunk_n, int(mc["pilot_n"] * scale) // chunk_n * chunk_n)
        parts, pn = run_chunks(pool, "rb", cell, seeds["pilot"], pn, chunk_n, thr=thr)
        pm = merge_sums(parts)
        mean, sd, se, half = mean_ci(pm["n"], pm["S1"], pm["S2"])
        i_v2, i_voi = names_nr.index("V2"), names_nr.index("voi_clean")
        v2, sdv2, voi = mean[i_v2], sd[i_v2], mean[i_voi]
        if v2 >= mc["small_v2_frac"] * voi:
            rule, need = "relative", (Z95 * sdv2 / (mc["rel_target"] * v2)) ** 2
        else:
            rule, need = "absolute", (Z95 * sdv2 / (mc["abs_target_frac"] * voi)) ** 2
        n_main = int(min(max(mc["min_main"] * scale, math.ceil(mc["safety"] * need)), mc["cap_main"]))
        n_main = max(chunk_n, math.ceil(n_main / chunk_n) * chunk_n)
        res["pilot"] = dict(n=pn, seed=seeds["pilot"], V2=v2, sd_Y_V2=sdv2, half95_V2=half[i_v2], voi_clean=voi)
        res["sizing"] = dict(rule=rule, N_needed=need, safety=mc["safety"], N_main=n_main,
                             cap_binds=bool(n_main >= mc["cap_main"]))
        # recalibration diagnostic: independent fit and validation seeds
        rc = cfg["recal"]
        rcn = rc["chunk_n"]
        fit_total = max(rc["fit_sizes"])
        parts, _ = run_chunks(pool, "fit", cell, seeds["recal_fit"], math.ceil(fit_total * scale / rcn) * rcn, rcn)
        fit = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        parts, _ = run_chunks(pool, "fit", cell, seeds["recal_val"], math.ceil(rc["val_n"] * scale / rcn) * rcn, rcn)
        val = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        recal, table = {}, []
        for mode in rc["modes"]:
            for size in rc["fit_sizes"]:
                sz = min(int(size * scale) // 1, len(fit["mu"]))
                for ns, nm in rc["resolutions"]:
                    g = BinnedRecal(ns, nm, mode).fit(fit["abs_s"][:sz], fit["mu"][:sz], fit["mu2"][:sz], fit["kap"][:sz])
                    gv = g(val["abs_s"], val["mu"])
                    R_val = float(np.mean(val["kap"] * (gv - val["mu2"]) ** 2))
                    A_val = float(np.mean(val["kap"] * (val["mu"] - gv) ** 2))
                    v2_val = float(np.mean(val["kap"] * (val["mu"] - val["mu2"]) ** 2))
                    name = f"{mode}_fit{sz}_res{ns}x{nm}"
                    recal[name] = g
                    table.append(dict(name=name, mode=mode, fit_n=sz, resolution=[ns, nm],
                                      n_bins=[len(g.s_edges) + 1, len(g.m_edges) + 1],
                                      val_R=R_val, val_A=A_val, val_V2_single_draw=v2_val))
        big = [t for t in table if t["fit_n"] == max(t2["fit_n"] for t2 in table)]
        sel = min(big, key=lambda t: t["val_R"])["name"]
        sel_by_mode = {m: min([t for t in big if t["mode"] == m], key=lambda t: t["val_R"])["name"] for m in rc["modes"]}
        res["recal_fit"] = dict(table=table, selected=sel, selected_by_mode=sel_by_mode, val_n=len(val["mu"]),
                                fit_seed=seeds["recal_fit"], val_seed=seeds["recal_val"],
                                note="selected = lowest validation E[k(g-mu2)^2] at the largest fit size (validation data only)")
    names = column_names(cell, list(recal))
    parts, n_main = run_chunks(pool, "rb", cell, seeds["main"], n_main, chunk_n, recal=recal, thr=thr,
                               n_profile=int(mc["n_profile"] * scale))
    merged = merge_sums(parts)
    main = summarize_rb(cell, merged, names, n_main, cell.chan)
    main.update(seed=seeds["main"], chunk_n=chunk_n, n_chunks=n_main // chunk_n)
    res["main"] = main
    parts, n_check = run_chunks(pool, "check", cell, seeds["check"], n_main, chunk_n)
    res["check"] = summarize_check(merge_sums(parts), main["columns"], names)
    res["check"]["seed"] = seeds["check"]
    if controls:
        c = main["columns"]
        res["control_exact_zero"] = {k: (c[k]["mean"] == 0.0 and c[k]["sd"] == 0.0) for k in ("V2", "I", "X", "AW")}
    res["runtime_seconds"] = time.time() - t0
    return res


def write_summary(out_dir, results):
    rows = []
    for r in results:
        c = r["main"]["columns"]
        row = dict(cell=r["cell"]["name"], role=r["cell"]["role"], L=r["cell"]["physics"]["L"], q_w=r["cell"]["physics"]["q_w"],
                   beta=r["cell"]["beta"], eta=r["cell"]["physics"]["eta"], N=r["main"]["n"], seed=r["main"]["seed"])
        for k in ("V2", "I", "X", "AW", "D", "voi_clean", "cost_zero", "cost_mode", "cost_clean", "cost_naive", "cost_aware"):
            row[k] = c[k]["mean"]
            row[k + "_half95"] = c[k]["half95"]
        row["V2_over_X"] = c["V2"]["mean"] / c["X"]["mean"] if c["X"]["mean"] > 0 else ""
        row["V2_over_voi"] = c["V2"]["mean"] / c["voi_clean"]["mean"] if c["voi_clean"]["mean"] > 0 else ""
        row["runtime_s"] = r["runtime_seconds"]
        rows.append(row)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "oracle_v1.toml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "oracle_v1"))
    ap.add_argument("--cells", default="", help="comma-separated cell names (default: all)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink all sample sizes (smoke tests only)")
    args = ap.parse_args()
    cfg = tomllib.load(open(args.config, "rb"))
    cells = make_cells(cfg)
    if args.cells:
        want = args.cells.split(",")
        cells = [c for c in cells if c.name in want]
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or cfg["meta"]["workers"]
    prov = provenance(args.config)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell in cells:
            print(f"[{time.time() - t0:7.1f}s] cell {cell.name} ...", flush=True)
            r = run_cell(pool, cfg, cell, args, prov)
            with open(os.path.join(args.out, f"{cell.name}.json"), "w") as f:
                json.dump(jsonable(r), f, indent=1)
            c = r["main"]["columns"]["V2"]
            print(f"[{time.time() - t0:7.1f}s]   N={r['main']['n']}  V2={c['mean']:.5f} +- {c['half95']:.5f}  "
                  f"({r['runtime_seconds']:.0f}s)", flush=True)
            results.append(r)
    write_summary(args.out, results)
    json.dump(jsonable(dict(provenance=prov, total_seconds=time.time() - t0, config=cfg, workers=workers, scale=args.scale)),
              open(os.path.join(args.out, "run_manifest.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
