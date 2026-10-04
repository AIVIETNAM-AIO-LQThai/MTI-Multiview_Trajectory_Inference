"""Learned pilot (protocol v2): tuning on seed 0, then training seeds, all arms, all priors.

    python experiments/run_learned.py --stage all --out results/learned_pilot
    python experiments/run_learned.py --stage all --smoke --out results/learned_smoke      # fast pipeline check, not a result

Units are (physics, N, seed); each unit trains every arm at its tuned configuration on its own independent prefixes, then evaluates
all arms on one fixed 20,000-prefix test set per physics (exact mu_2 from the oracle is used for evaluation only). Single-thread
torch per worker process; units run in parallel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import torch

from mti.learned.data import MASTER_TRAIN, MASTER_VAL, make_probe_set
from mti.learned.evalset import (TestSet, candidate_from_density, compose_F, hmm_candidate, query_density, query_F, query_full)
from mti.learned.features import to_tensor
from mti.learned.hmm_fit import fit_eta
from mti.learned.train import (TrainCfg, train_A1, train_A1r, train_A2, train_A4, train_A6, train_A6p)
from mti.params import ChannelSpec, Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSICS = [Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.02, lam=0.1, eta=0.05, L=8),     # lead
           Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.05, L=8)]      # secondary
TUNED_ARMS = ["A1", "A2", "A4", "A6", "A6p-narrow", "A6p-broad"]
QUERIES_PER_RECORD = {"A1": 1, "A1r": 1, "A2-G": 9, "A2-M": 17, "A2-avg": 17, "A3-ens": 9, "A4": 9, "A5": "analytic O(L)", "A6": 1, "A6p-narrow": 1, "A6p-broad": 1}


def priors(L: int) -> dict:
    w = [2.0 ** -(L - 1 - j) for j in range(L)]
    return {"P1": ChannelSpec.uniform(L, 0.2), "P2": ChannelSpec.uniform(L, 0.5),
            "P3": ChannelSpec.point_mass(L, 0.2, L - 1), "P4": ChannelSpec.point_mass(L, 0.5, L - 1),
            "P5": ChannelSpec(beta=0.2, pi=tuple(x / sum(w) for x in w))}


def channel_priors(phys_idx: int) -> list:
    """Protocol v2: channel-trained arms are evaluated at P1/P2 on the secondary physics, at all priors on the lead physics."""
    return ["P1", "P2", "P3", "P4", "P5"] if phys_idx == 0 else ["P1", "P2"]


def grid(N: int, smoke: bool = False) -> list:
    ep = (6, 12) if N >= 100_000 else (60, 150)
    if smoke:
        ep = (1, 2)
    pat = 4 if N >= 100_000 else 15
    return [TrainCfg(lr=lr, wd=wd, epochs=e, patience=pat) for lr in (1e-3, 3e-3) for wd in (1e-5, 1e-3) for e in ep]


def train_arm(arm: str, tr, va, phys, cfg: TrainCfg):
    if arm == "A1":
        n = tr.n * 3 // 4                                           # protocol: A1 uses 3/4 of N, A1r calibrates on the last 1/4
        return train_A1(tr.take(0, n), va, phys, cfg)
    if arm == "A2":
        return train_A2(tr, va, phys, cfg)
    if arm == "A4":
        return train_A4(tr, va, phys, cfg)
    if arm == "A6":
        return train_A6(tr, va, phys, priors(phys.L)["P1"], cfg)
    if arm.startswith("A6p-"):
        return train_A6p(tr, va, phys, arm.split("-")[1], cfg)
    raise ValueError(arm)


def _init_worker():
    torch.set_num_threads(1)


def tune_task(args):
    phys_idx, N, arm, ci, cfg, smoke = args
    phys = PHYSICS[phys_idx]
    n = N if not smoke else N
    tr = make_probe_set(phys, n, MASTER_TRAIN, phys_idx, 0)
    va = make_probe_set(phys, n // 4, MASTER_VAL, phys_idx, 0)
    t = time.time()
    _, info = train_arm(arm, tr, va, phys, cfg)
    return dict(phys_idx=phys_idx, N=N, arm=arm, cfg_idx=ci, cfg=asdict(cfg), val=info.best_val, epochs_run=info.epochs_run, seconds=time.time() - t)


def run_unit(args):
    phys_idx, N, seed, cfgs, n_test, smoke = args
    phys = PHYSICS[phys_idx]
    L = phys.L
    t_unit = time.time()
    pri = priors(L)
    ts = TestSet(phys, n_test, cell_id=phys_idx)
    for k, ch in pri.items():
        ts.add_prior(k, ch)
    ts.add_exact_queries()
    n, V = ts.n, ts.V
    tr = make_probe_set(phys, N, MASTER_TRAIN, phys_idx, seed)
    va = make_probe_set(phys, N // 4, MASTER_VAL, phys_idx, seed)
    cfg = {a: TrainCfg(**cfgs[a], seed=seed) for a in TUNED_ARMS}
    models, infos = {}, {}
    for arm in TUNED_ARMS:
        models[arm], infos[arm] = train_arm(arm, tr, va, phys, cfg[arm])
    rows = []

    def add(arm, prior, mu, extra=None):
        r = dict(arm=arm, prior=prior, **ts.summarize(prior, mu))
        r.update(extra or {})
        rows.append(r)

    # exact references
    for k in pri:
        for arm, r in ts.exact_rows(k).items():
            rows.append(dict(arm=arm, prior=k, **r))
    # A5: learned persistence
    sL = tr.s[:, -1].numpy().astype(float)
    eta_hat, em_it, _ = fit_eta(tr.s.numpy().astype(float), tr.a.numpy().astype(float), (tr.d_pr.numpy() + phys.rho * sL).astype(float),
                                tr.a_pr.numpy().astype(float), phys)
    for k, ch in pri.items():
        aware, _ = hmm_candidate(ts, eta_hat, ch)
        add("A5", k, aware, dict(eta_hat=eta_hat))
    # A1 naive learned filter + A1r per prior (calibration = last quarter of the training prefixes)
    f_all = query_full(models["A1"], ts.s_rec, ts.a_rec, phys)
    f_t = torch.as_tensor(f_all, dtype=torch.float32)
    sabs = torch.as_tensor(np.repeat(ts.sL_abs, V) / math.sqrt(phys.var_s0), dtype=torch.float32)
    cal = tr.take(tr.n * 3 // 4, tr.n)
    rcfg = TrainCfg(lr=3e-3, wd=1e-4, batch=512, epochs=30 if not smoke else 2, patience=5, seed=seed)
    for k in pri:
        add("A1", k, f_all.reshape(n, V))
    for k in channel_priors(phys_idx):
        g, ginfo = train_A1r(models["A1"], cal, va, phys, pri[k], rcfg)
        with torch.no_grad():
            mu = g(sabs, f_t).double().numpy().reshape(n, V)
        add("A1r", k, mu, dict(refit_seconds=ginfo.seconds, refit_view_passes=ginfo.view_passes))
    # A2 / A3 shared fixed-view model: composition under each prior
    outF = query_F(models["A2"], ts.s_rec, ts.a_rec, phys)
    qdiag = ts.query_diagnostics(outF)
    for k, ch in pri.items():
        arms, comp = compose_F(outF, ch, ts.a_rec, n, V)
        gd = ts.gap_diagnostics(k, comp)
        for arm, mu in arms.items():
            add(arm, k, mu, gd if arm in ("A2-G", "A2-M", "A2-avg") else None)
    # A4 causal density scorer: candidate composition with learned density ratios
    ll, mu4 = query_density(models["A4"], ts.s_rec, ts.a_rec, phys)
    for k, ch in pri.items():
        add("A4", k, candidate_from_density(ll, mu4, ch, n, V))
    # channel-trained direct regressions
    mu6 = query_full(models["A6"], ts.s_rec, ts.a_rec, phys).reshape(n, V)
    for k in channel_priors(phys_idx):
        add("A6", k, mu6)
        for fam in ("narrow", "broad"):
            q = torch.as_tensor(np.tile(pri[k].q.astype(np.float32), (n * V, 1)))
            add(f"A6p-{fam}", k, query_full(models[f"A6p-{fam}"], ts.s_rec, ts.a_rec, phys, cond=q).reshape(n, V))
    kap_voi = float(np.mean(ts.kap * ts.oracle["P1"]["mu_t"][:, 0] ** 2))
    return dict(phys_idx=phys_idx, q_w=phys.q_w, N=N, seed=seed, n_test=n_test, voi_clean=kap_voi, eta_hat=eta_hat, em_iterations=em_it,
                cfgs=cfgs, train={a: dict(best_val=i.best_val, epochs_run=i.epochs_run, steps=i.steps, view_passes=i.view_passes,
                                          seconds=i.seconds, params=i.n_params) for a, i in infos.items()},
                query_diagnostics=qdiag, queries_per_record=QUERIES_PER_RECORD, rows=rows, unit_seconds=time.time() - t_unit)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def provenance():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.rstrip("\n")
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith(("results/", "docs/reports/"))]
    return dict(git_commit=git("rev-parse", "HEAD").strip(), dirty_code_files=dirty, python=sys.version.split()[0], torch=torch.__version__,
                numpy=np.__version__, command=" ".join(sys.argv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "learned_pilot"))
    ap.add_argument("--stage", choices=["tune", "main", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--physics", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    Ns = [1_000, 100_000] if not args.smoke else [800, 2_000]
    n_test = args.n_test if not args.smoke else 1_500
    os.makedirs(os.path.join(args.out, "units"), exist_ok=True)
    prov = provenance()
    t0 = time.time()
    tune_path = os.path.join(args.out, "tuning.json")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        if args.stage in ("tune", "all"):
            tasks = [(p, N, arm, ci, cfg, args.smoke) for p in args.physics for N in Ns for arm in TUNED_ARMS
                     for ci, cfg in enumerate(grid(N, args.smoke))]
            print(f"tuning: {len(tasks)} trainings", flush=True)
            res = list(pool.map(tune_task, tasks, chunksize=1))
            sel = {}
            for r in res:
                key = f"{r['phys_idx']}|{r['N']}|{r['arm']}"
                if key not in sel or r["val"] < sel[key]["val"]:
                    sel[key] = r
            json.dump(jsonable(dict(provenance=prov, results=res, selected=sel, seconds=time.time() - t0)), open(tune_path, "w"), indent=1)
            print(f"[{time.time() - t0:.0f}s] tuning done", flush=True)
        if args.stage in ("main", "all"):
            sel = json.load(open(tune_path))["selected"]
            tasks = []
            for p in args.physics:
                for N in Ns:
                    cfgs = {arm: {k: v for k, v in sel[f"{p}|{N}|{arm}"]["cfg"].items() if k not in ("seed",)} for arm in TUNED_ARMS}
                    for s in args.seeds:
                        tasks.append((p, N, s, cfgs, n_test, args.smoke))
            tasks.sort(key=lambda t: -t[1])                          # long units first
            print(f"main: {len(tasks)} units", flush=True)
            futs = [pool.submit(run_unit, t) for t in tasks]
            for t, f in zip(tasks, futs):
                r = f.result()
                r["provenance"] = prov
                with open(os.path.join(args.out, "units", f"phys{t[0]}_N{t[1]}_s{t[2]}.json"), "w") as fh:
                    json.dump(jsonable(r), fh)
                print(f"[{time.time() - t0:.0f}s] unit phys{t[0]} N{t[1]} seed{t[2]} done ({r['unit_seconds']:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
