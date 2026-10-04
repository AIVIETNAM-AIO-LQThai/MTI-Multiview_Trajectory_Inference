"""E2b (protocol e2_protocol.md section 10): robust declared priors under prior uncertainty. No training.

    python experiments/run_e2b.py --out results/e2b

Stage 1 (selection, master 8101, 100k prefixes, both physics): every candidate prior is scored under all 30 true channels (exact clean law);
selects P_K1 (Bayes grid optimum), P_K2 (no-harm), P_K3 (minimax regret) and compares them with the exact mean prior q_bar.
Stage 2 (confirmation, master 8201, 20k prefixes, independent of selection): exact scores of the selected priors and two references under all 30 truths,
and the same with the saved A4 checkpoints (lead physics, N in {1e3, 1e5}, seeds 1-10; no retraining).
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

import numpy as np
import torch

from mti.learned.evalset import TestSet, candidate_from_density, query_density
from mti.learned.models import CausalDensity
from mti.misspec import ExactRecords, TrueLaw, alpha_of, mean_prior, no_channel
from mti.params import ChannelSpec, Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 8
PHYSICS = [Physics(0.9, 1.0, 1.0, 0.02, 0.1, 0.05, L), Physics(0.9, 1.0, 1.0, 0.5, 0.1, 0.05, L)]
BETAS_T = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
ALPHAS_T = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS_A = (0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
ALPHAS_A = tuple(i / 8 for i in range(9))
MASTER_SEL, MASTER_CONF = 8101, 8201
N_SEL, N_CONF = 100_000, 20_000
TRUTHS = [(b, a) for b in BETAS_T for a in ALPHAS_T]
TRUTH_CH = [ChannelSpec.interp(L, b, a) for b, a in TRUTHS]
QBAR = mean_prior(TRUTH_CH)
PLAUSIBLE = ChannelSpec.interp(L, 0.2, 0.5)


def dmin(v2, voi):
    return min(max(0.10 * v2, 0.002 * voi), 0.02 * voi)


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


def provenance():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith(("results/", "docs/reports/"))]
    return dict(git_commit=git("rev-parse", "HEAD").strip(), branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(), dirty_code=dirty,
                python=sys.version.split()[0], torch=torch.__version__, command=" ".join(sys.argv))


def cand_name(c):
    return f"b{c.beta:.4f}_a{alpha_of(c):.4f}"


def selection(pi):
    """Score every candidate under every truth on the selection set. Returns arrays (cand x truth) of E, B, se and the truth-wise V2, Delta."""
    t0 = time.time()
    phys = PHYSICS[pi]
    ts = TestSet(phys, N_SEL, cell_id=pi, master=MASTER_SEL)
    ex = ExactRecords(ts)
    truth = TrueLaw(ts, ex)
    K, n = truth.K, ts.n
    refs = [truth.reference(c) for c in TRUTH_CH]
    yn = [(K * (ex.naive - r) ** 2) @ c.view_weights() for r, c in zip(refs, TRUTH_CH)]
    V2 = np.array([y.mean() for y in yn])
    voi = truth.voi_clean()
    dl = np.array([dmin(v, voi) for v in V2])
    cands = [ChannelSpec.interp(L, b, a) for b in BETAS_A for a in ALPHAS_A if not (b == 0.0 and a > 0)] + [QBAR]
    names = [cand_name(c) for c in cands]
    E = np.zeros((len(cands), len(TRUTHS)))
    B = np.zeros_like(E)
    SE = np.zeros_like(E)
    for i, c in enumerate(cands):
        m2a = ex.naive if c.beta == 0 else ex.mu2(c)
        for t, (r, ct) in enumerate(zip(refs, TRUTH_CH)):
            y = (K * (m2a - r) ** 2) @ ct.view_weights()
            d = yn[t] - y
            E[i, t], B[i, t], SE[i, t] = y.mean(), d.mean(), d.std(ddof=1) / np.sqrt(n)
    # descriptive hypothesis data: lag-0 declared / true corruption mass for every (candidate, truth) pair
    q0_a = np.array([c.q[-1] for c in cands])
    q0_t = np.array([c.q[-1] for c in TRUTH_CH])
    return dict(phys=pi, q_w=phys.q_w, n=n, voi=voi, V2=V2, delta=dl, cand=[dict(name=nm, beta=c.beta, alpha=alpha_of(c), q0=float(c.q[-1]), is_qbar=(c is QBAR)) for nm, c in zip(names, cands)],
                E=E, B=B, SE=SE, q0_a=q0_a, q0_t=q0_t, seconds=time.time() - t0)


def pick(sel):
    """P_K1: argmin mean_t E; P_K2: no-harm (best K1 among candidates with B_t >= -Delta_t for all t; else argmax min_t B); P_K3: argmin max_t E."""
    E, B, dl = sel["E"], sel["B"], sel["delta"]
    k1, k3 = E.mean(axis=1), E.max(axis=1)
    safe = np.where((B >= -dl[None, :]).all(axis=1))[0]
    i2 = int(safe[np.argmin(k1[safe])]) if len(safe) else int(np.argmax(B.min(axis=1)))
    return dict(K1=int(np.argmin(k1)), K2=i2, K3=int(np.argmin(k3)), n_no_harm=len(safe), qbar=len(k1) - 1)


def _init():
    torch.set_num_threads(1)


def score_priors(truth, ex, priors, ts_beliefs=None):
    """E, B (per-prefix paired against naive) of each prior under all truths. priors: dict name -> belief matrix (n,V)."""
    K, n = truth.K, truth.n
    refs = [truth.reference(c) for c in TRUTH_CH]
    out = {}
    base = [(K * (ts_beliefs["naive"] - r) ** 2) @ c.view_weights() for r, c in zip(refs, TRUTH_CH)]
    for name, m in priors.items():
        rows = []
        for t, (r, c) in enumerate(zip(refs, TRUTH_CH)):
            y = (K * (m - r) ** 2) @ c.view_weights()
            d = base[t] - y
            rows.append((float(y.mean()), float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))))
        out[name] = rows
    return out, [float(b.mean()) for b in base]


def confirm_exact(sel, picks):
    pi = sel["phys"]
    phys = PHYSICS[pi]
    ts = TestSet(phys, N_CONF, cell_id=pi, master=MASTER_CONF)
    ex = ExactRecords(ts)
    truth = TrueLaw(ts, ex)
    cs = sel["cand"]
    spec = {"qbar": QBAR, "K1": ChannelSpec.interp(L, cs[picks["K1"]]["beta"], cs[picks["K1"]]["alpha"]),
            "K2": ChannelSpec.interp(L, cs[picks["K2"]]["beta"], cs[picks["K2"]]["alpha"]),
            "K3": ChannelSpec.interp(L, cs[picks["K3"]]["beta"], cs[picks["K3"]]["alpha"]),
            "plausible": PLAUSIBLE}
    beliefs = {k: (ex.naive if c.beta == 0 else ex.mu2(c)) for k, c in spec.items()}
    beliefs["naive"] = ex.naive
    scores, base = score_priors(truth, ex, beliefs, dict(naive=ex.naive))
    V2 = np.array(base)
    dl = np.array([dmin(v, truth.voi_clean()) for v in V2])
    return dict(phys=pi, spec={k: dict(beta=c.beta, alpha=alpha_of(c)) for k, c in spec.items()}, scores=scores, V2=V2, delta=dl, voi=truth.voi_clean()), spec


def a4_unit(args):
    N, seed, spec_json, out = args
    spec = {k: ChannelSpec.interp(L, v["beta"], v["alpha"]) for k, v in spec_json.items()}
    phys = PHYSICS[0]
    ck = torch.load(os.path.join(ROOT, "results", "e2", "checkpoints", f"N{N}_s{seed}_A4.pt"), weights_only=False)
    model = CausalDensity(ck["cfg"].get("hidden", 64))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ts = TestSet(phys, N_CONF, cell_id=0, master=MASTER_CONF)
    ex = ExactRecords(ts)
    truth = TrueLaw(ts, ex)
    ll, mu = query_density(model, ts.s_rec, ts.a_rec, phys)
    n, V = ts.n, ts.V
    naive = mu[:, 0].reshape(n, V)
    beliefs = {k: (naive if c.beta == 0 else candidate_from_density(ll, mu, c, n, V)) for k, c in spec.items()}
    beliefs["naive"] = naive
    refs = [truth.reference(c) for c in TRUTH_CH]
    res = {k: [float(((truth.K * (m - r) ** 2) @ c.view_weights()).mean()) for r, c in zip(refs, TRUTH_CH)] for k, m in beliefs.items()}
    return dict(N=N, seed=seed, E=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e2b"))
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prov = provenance()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, initializer=_init) as pool:
        sels = list(pool.map(selection, [0, 1]))
    print(f"[{time.time() - t0:.0f}s] selection done", flush=True)
    result = dict(provenance=prov, truths=TRUTHS, qbar=dict(beta=QBAR.beta, alpha=alpha_of(QBAR)), plausible=dict(beta=0.2, alpha=0.5))
    specs0 = None
    for sel in sels:
        picks = pick(sel)
        conf, spec = confirm_exact(sel, picks)
        result[f"phys{sel['phys']}"] = dict(selection=sel, picks=picks, confirm=conf)
        if sel["phys"] == 0:
            specs0 = conf["spec"]
        print(f"phys{sel['phys']}: picks {picks} -> {conf['spec']}", flush=True)
    tasks = [(N, s, specs0, args.out) for N in (1000, 100000) for s in range(1, 11)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        result["a4"] = list(pool.map(a4_unit, tasks))
    json.dump(jsonable(result), open(os.path.join(args.out, "e2b.json"), "w"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
