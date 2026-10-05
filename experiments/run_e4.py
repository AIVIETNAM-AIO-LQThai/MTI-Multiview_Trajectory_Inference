"""E4 stage 1: joint identification of clean persistence and channel (protocol docs/e4_protocol.md sections 2, 4). No training.

    python experiments/run_e4.py --out results/e4/stage1            # all (physics, L, truth) x replicate jobs + F1
    python experiments/run_e4.py --smoke --out results/scratch/e4   # quick check

Job = (physics, L, truth, replicate). Adaptation records: simulate + exogenous channel (master 8501; cell = task id, chunk = replicate). Sets of n are the
first n of N_max records (nested). Evaluation: fixed 20,000 clean prefixes with all L+1 views (master 8601, cell 0) scored under the TRUE law.
Clean validation prefixes (CV arms): master 8701, same cell/chunk as the adaptation set. Estimators see records (s, a) and the declared physics
(rho, c, q_a, q_w, lambda, L) only; eta enters the EX/ETA arms as a stated constant and the J/CV arms as an estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mti.adapt import Precomputed
from mti.joint import (ETA_GRID, MixtureOracle, clean_eta_mle, fit_channel, interleave_persistence, jmom, profile_grid, refine_eta,
                       _queries)
from mti.learned.evalset import TestSet
from mti.inference import flip_batch
from mti.misspec import ExactRecords, TrueLaw
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_ADAPT, MASTER_EVAL, MASTER_CLEAN, MASTER_STAGE0, MASTER_F1B = 8501, 8601, 8701, 8801, 8901
N_EVAL = 20_000
NS = (300, 1_000, 3_000, 10_000, 30_000, 100_000)
R_BIG, R_MID = 10, 20          # replicates 0..R_BIG-1 run up to n = 1e5; the rest up to n = 3e4
LS = (3, 4, 8)
ETA_TRUE = 0.05
PHYS = {"lead": Physics(0.9, 1.0, 1.0, 0.02, 0.1, ETA_TRUE, 8), "sec": Physics(0.9, 1.0, 1.0, 0.5, 0.1, ETA_TRUE, 8)}
NCS = (30, 100, 1_000)
F1_TRUTHS, F1_NS, F1_R = ("T0", "T3"), (1_000, 10_000, 30_000), 10


def truths(L):
    w = [2.0 ** -(L - 1 - j) for j in range(L)]
    return [("T0", ChannelSpec.uniform(L, 0.0)), ("T3", ChannelSpec.interp(L, 0.2, 0.5)), ("T4", ChannelSpec.interp(L, 0.35, 1.0)),
            ("T6", ChannelSpec(0.2, tuple(x / sum(w) for x in w)))]


def task_id(pname, L, tidx):
    return 1000 * list(PHYS).index(pname) + 10 * L + tidx


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def provenance(out):
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    patch = git("diff", "HEAD", "--", "src", "experiments", "tests", "configs")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src", "experiments", "tests", "configs").splitlines()
    hashes = {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:16] for f in untracked if os.path.isfile(os.path.join(ROOT, f))}
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "code_patch.diff"), "w").write(patch)
    return dict(git_commit=git("rev-parse", "HEAD").strip(), branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                patch_sha256=hashlib.sha256(patch.encode()).hexdigest()[:16], untracked_code_hashes=hashes, python=sys.version.split()[0],
                numpy=np.__version__, command=" ".join(sys.argv))


def _arm(w, eta, regret_fn, pre_fn, **extra):
    pre = pre_fn(eta)
    return dict(eta=float(eta), beta=float(1.0 - w[0]), w=np.asarray(w), regret=regret_fn(pre, w), regret_naive_pipe=regret_fn(pre, np.eye(len(w))[0]), **extra)


def run_job(job):
    t0 = time.time()
    pname, L, tidx, r = job["pname"], job["L"], job["tidx"], job["r"]
    phys = PHYS[pname].with_(L=L)
    tname, truth = truths(L)[tidx]
    tid = task_id(pname, L, tidx)
    wt = truth.view_weights()
    ts = TestSet(phys, job.get("n_eval", N_EVAL), cell_id=0, master=MASTER_EVAL)
    n_ev, V = ts.n, ts.V
    ex = ExactRecords(ts)
    tl = TrueLaw(ts, ex)
    K = tl.K
    m2t = tl.reference(truth)
    V2, voi = float(((K * (ex.naive - m2t) ** 2) @ wt).mean()), tl.voi_clean()
    delta = float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi))
    cache = {round(ETA_TRUE, 9): Precomputed.build(ex.ll, ex.mu)}

    def pre_at(eta):
        key = round(float(eta), 9)
        if key not in cache:
            e = ExactRecords(ts, phys.with_(eta=float(eta)))
            cache[key] = Precomputed.build(e.ll, e.mu)
        return cache[key]

    def regret(pre, w):
        mu = pre.compose(w).reshape(n_ev, V)
        return float(((K * (mu - m2t) ** 2) @ wt).mean())

    nmax = job.get("nmax") or (NS[-1] if r < R_BIG else 30_000)
    ns = [n for n in job.get("ns", NS) if n <= nmax]
    smp = simulate(phys, nmax, MASTER_ADAPT, tid, r)
    S = apply_channel(smp.prefix, sample_theta(truth, nmax, MASTER_ADAPT, tid, r))
    H = simulate(phys, max(NCS), MASTER_CLEAN, tid, r).prefix                    # clean validation prefixes (CV arms)
    eta_cv = {nc: clean_eta_mle(Prefix(H.s[:nc], H.a[:nc]), phys) for nc in NCS}
    pg = profile_grid(S, phys, ns)
    gi = {e: int(np.argmin(np.abs(ETA_GRID - e))) for e in (0.03, ETA_TRUE, 0.08)}
    arms = {k: {} for k in ("EX-MLE", "ETA--MLE", "ETA+-MLE", "J-MLE", "J-MOM", "CV-30", "CV-100", "CV-1000")}
    prof = {}
    for n in ns:
        P = pg[n]
        for name, e in (("ETA--MLE", 0.03), ("ETA+-MLE", 0.08), ("EX-MLE", ETA_TRUE)):
            k = gi[e]
            arms[name][n] = _arm(P["w"][k], e, regret, pre_at, ll=float(P["ll"][k]))
        g = int(np.argmax(P["ll"]))
        eh, w, ll = refine_eta(S, phys, n, float(ETA_GRID[g]), P["w"][g])
        arms["J-MLE"][n] = _arm(w, eh, regret, pre_at, ll=ll, grid_best=float(ETA_GRID[g]), em_iter=int(P["it"].sum()))
        prof[n] = P["ll"]
        em, wm = jmom(S.s[:n], S.a[:n], phys)
        arms["J-MOM"][n] = _arm(wm, em, regret, pre_at)
        Sn = Prefix(S.s[:n], S.a[:n])
        for nc in NCS:
            llS, logell = _queries(Sn, phys, eta_cv[nc])
            wc, llc, _ = fit_channel(llS, logell)
            arms[f"CV-{nc}"][n] = _arm(wc, eta_cv[nc], regret, pre_at, ll=llc)
    return dict(kind="main", pname=pname, L=L, truth=tname, tidx=tidx, r=r, nmax=nmax, ns=ns, V2=V2, voi=voi, delta=delta,
                E_naive_exact=V2, truth_beta=truth.beta, truth_q=list(truth.q), eta_cv=eta_cv, arms=arms, profile_ll=prof,
                eta_grid=ETA_GRID, seconds=time.time() - t0)


def run_f1_job(job):
    """F1 (descriptive): record-heterogeneous persistence, eta_r in {0.02, 0.08} with equal probability (marginal edge rate 0.05)."""
    t0 = time.time()
    r, tname = job["r"], job["truth"]
    L = 8
    phys = PHYS["lead"].with_(L=L)
    tidx = [t for t, _ in truths(L)].index(tname)
    truth = truths(L)[tidx][1]
    tid = task_id("lead", L, tidx)
    wt = truth.view_weights()
    mo = MixtureOracle(phys)
    n_half = job.get("n_eval", N_EVAL) // 2
    tlo, thi = TestSet(phys.with_(eta=0.02), n_half, 0, master=MASTER_EVAL), TestSet(phys.with_(eta=0.08), n_half, 1, master=MASTER_EVAL)
    ev = Prefix(np.concatenate([tlo.s_rec, thi.s_rec]), np.concatenate([tlo.a_rec, thi.a_rec]))
    kap = np.concatenate([tlo.kap, thi.kap])
    n_ev, V = len(kap), L + 1
    K = kap[:, None]
    qs = mo.queries(ev)
    e0 = np.r_[1.0, np.zeros(L)]
    m2t = mo.beliefs(qs, wt)[0].reshape(n_ev, V)
    mu_naive = mo.beliefs(qs, e0)[0].reshape(n_ev, V)
    muH = mu_naive[:, 0]
    V2, voi = float(((K * (mu_naive - m2t) ** 2) @ wt).mean()), float(np.mean(kap * muH**2))
    delta = float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi))
    cache = {}

    def pre_at(eta):
        key = round(float(eta), 9)
        if key not in cache:
            ll, mu = flip_batch(ev, phys.with_(eta=float(eta)))
            cache[key] = Precomputed.build(ll, mu)
        return cache[key]

    def regret(pre, w):
        return float(((K * (pre.compose(w).reshape(n_ev, V) - m2t) ** 2) @ wt).mean())

    nmax = max(F1_NS) if not job.get("ns") else max(job["ns"])
    ns = [n for n in (job.get("ns") or F1_NS)]
    Hc, _, _ = interleave_persistence(phys, nmax, MASTER_ADAPT, MASTER_F1B, tid, r)
    S = apply_channel(Hc, sample_theta(truth, nmax, MASTER_ADAPT, tid, r))
    Hcv, _, _ = interleave_persistence(phys, max(NCS), MASTER_CLEAN, MASTER_F1B, tid + 500, r)
    eta_cv = clean_eta_mle(Hcv, phys)
    pg = profile_grid(S, phys, ns)
    gi = {e: int(np.argmin(np.abs(ETA_GRID - e))) for e in (0.03, ETA_TRUE, 0.08)}
    arms = {k: {} for k in ("EX-MLE", "ETA--MLE", "ETA+-MLE", "J-MLE", "CV-1000")}
    for n in ns:
        P = pg[n]
        for name, e in (("ETA--MLE", 0.03), ("ETA+-MLE", 0.08), ("EX-MLE", ETA_TRUE)):
            arms[name][n] = _arm(P["w"][gi[e]], e, regret, pre_at, ll=float(P["ll"][gi[e]]))
        g = int(np.argmax(P["ll"]))
        eh, w, ll = refine_eta(S, phys, n, float(ETA_GRID[g]), P["w"][g])
        arms["J-MLE"][n] = _arm(w, eh, regret, pre_at, ll=ll)
        llS, logell = _queries(Prefix(S.s[:n], S.a[:n]), phys, eta_cv)
        wc, llc, _ = fit_channel(llS, logell)
        arms["CV-1000"][n] = _arm(wc, eta_cv, regret, pre_at, ll=llc)
    return dict(kind="f1", L=L, truth=tname, r=r, ns=ns, V2=V2, voi=voi, delta=delta, truth_beta=truth.beta, eta_cv=eta_cv,
                arms=arms, seconds=time.time() - t0)


def run(job):
    return run_f1_job(job) if job["kind"] == "f1" else run_job(job)


def job_name(j):
    return f"f1_{j['truth']}_r{j['r']}" if j["kind"] == "f1" else f"{j['pname']}_L{j['L']}_{truths(j['L'])[j['tidx']][0]}_r{j['r']}"


def build_jobs(smoke=False):
    jobs = []
    for pname in PHYS:
        for L in LS:
            for tidx in range(4):
                for r in range(R_MID):
                    jobs.append(dict(kind="main", pname=pname, L=L, tidx=tidx, r=r))
    for tname in F1_TRUTHS:
        for r in range(F1_R):
            jobs.append(dict(kind="f1", truth=tname, r=r))
    if smoke:
        jobs = [j for j in jobs if j["kind"] == "main" and j["pname"] == "lead" and j["L"] in (3, 8) and j["tidx"] in (0, 1) and j["r"] == 0]
        jobs += [dict(kind="f1", truth="T3", r=0)]
        for j in jobs:
            j.update(ns=(300, 1_000) if j["kind"] == "main" else (1_000,), nmax=1_000, n_eval=2_000)
    return jobs


def cost(j):
    if j["kind"] == "f1":
        return 3.0
    return (10 if j["L"] == 8 else 3 if j["L"] == 4 else 1.5) * (3 if j["r"] < R_BIG else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e4", "stage1"))
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-L", type=int, nargs="*")
    ap.add_argument("--skip-f1", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    prov = provenance(out)
    json.dump(jsonable(prov), open(os.path.join(out, "provenance.json"), "w"), indent=1)
    jobs = build_jobs(args.smoke)
    if args.only_L:
        jobs = [j for j in jobs if j["kind"] == "f1" or j["L"] in args.only_L]
    if args.skip_f1:
        jobs = [j for j in jobs if j["kind"] != "f1"]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out, "units", job_name(j) + ".json"))]
    jobs.sort(key=cost, reverse=True)
    print(f"jobs: {len(jobs)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [(j, pool.submit(run, j)) for j in jobs]
        for j, f in futs:
            r = f.result()
            r["provenance"] = prov
            json.dump(jsonable(r), open(os.path.join(out, "units", job_name(j) + ".json"), "w"))
            print(f"[{time.time() - t0:.0f}s] {job_name(j)} ({r['seconds']:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
