"""E3: channel identification and adaptive composition (protocol docs/e3_protocol.md). No training.

    python experiments/run_e3.py --out results/e3

Task = (clean-law source, true channel). Adaptation records: simulate + exogenous channel (master 8301; truth id = cell, replicate = chunk; sets of n
are the first n of 10,000 records, so they are nested within a replicate). Evaluation: fixed 20,000 independent prefixes (master 8401), scored under
the TRUE channel with the exact clean law (TrueLaw). Adaptation uses (s, a) only.
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
import torch

from mti.adapt import Precomputed, em_channel, mom_channel, pp_channel
from mti.inference import flip_batch
from mti.learned.evalset import TestSet, query_density
from mti.learned.models import CausalDensity
from mti.misspec import ExactRecords, TrueLaw, mean_prior
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_ADAPT, MASTER_EVAL = 8301, 8401
N_EVAL, N_MAX = 20_000, 10_000
NS = (10, 30, 100, 300, 1_000, 3_000, 10_000)
L8 = 8
PHYS8 = Physics(0.9, 1.0, 1.0, 0.02, 0.1, 0.05, L8)
KAPPA0 = 30.0
FIXED = ["naive", "F-qbar", "F-noharm", "F-minimax", "ORACLE"]
ADAPT = ["MLE", "MAP", "PP", "MOM"]


def truths8():
    w = [2.0 ** -(L8 - 1 - j) for j in range(L8)]
    half = [0.0] * L8
    half[L8 - 1] = 0.5
    half[0] = 0.5
    return [
        ("T0", ChannelSpec.uniform(L8, 0.0)),
        ("T1", ChannelSpec.interp(L8, 0.02, 0.0)),
        ("T2", ChannelSpec.interp(L8, 0.05, 0.25)),
        ("T3", ChannelSpec.interp(L8, 0.2, 0.5)),
        ("T4", ChannelSpec.interp(L8, 0.35, 1.0)),
        ("T5", ChannelSpec.point_mass(L8, 0.125, L8 - 2)),
        ("T6", ChannelSpec(0.2, tuple(x / sum(w) for x in w))),
        ("T7", ChannelSpec(0.3, tuple(half))),
    ]


def qbar_L(L):
    cs = [ChannelSpec.interp(L, b, a) for b in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return mean_prior(cs)


def fixed_decl(L):
    return {"naive": ChannelSpec.uniform(L, 0.0), "F-qbar": qbar_L(L), "F-noharm": ChannelSpec.interp(L, 0.075, 0.0), "F-minimax": ChannelSpec.interp(L, 0.15, 0.875)}


def grid_w(L):
    cands = [ChannelSpec.interp(L, b, a) for b in (0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
             for a in tuple(i / 8 for i in range(9)) if not (b == 0.0 and a > 0)] + [qbar_L(L)]
    return np.stack([c.view_weights() for c in cands])


def eta_hat(N, seed):
    return float(json.load(open(os.path.join(ROOT, "results", "e2", "units", f"N{N}_s{seed}.json")))["eta_hat"])


def sources():
    s = [dict(name="EX", kind="EX", eta=0.05, eta_mom=0.05, ns=NS, reps=20),
         dict(name="ETA-", kind="ETA", eta=0.03, eta_mom=0.03, ns=NS, reps=20),
         dict(name="ETA+", kind="ETA", eta=0.08, eta_mom=0.08, ns=NS, reps=20),
         dict(name="HMM-1000-s1", kind="ETA", eta=eta_hat(1000, 1), eta_mom=eta_hat(1000, 1), ns=NS, reps=20),
         dict(name="HMM-100000-s1", kind="ETA", eta=eta_hat(100000, 1), eta_mom=eta_hat(100000, 1), ns=NS, reps=20)]
    small = (100, 1_000, 10_000)
    for N in (1000, 100000):
        for seed in range(2, 6):
            e = eta_hat(N, seed)
            s.append(dict(name=f"HMM-{N}-s{seed}", kind="ETA", eta=e, eta_mom=e, ns=small, reps=4))
        for seed in range(1, 6):
            s.append(dict(name=f"A4-{N}-s{seed}", kind="A4", eta=None, eta_mom=eta_hat(N, seed), ns=small, reps=4,
                          ckpt=os.path.join(ROOT, "results", "e2", "checkpoints", f"N{N}_s{seed}_A4.pt")))
    return s


def control_tasks():
    out = []
    for L, name, chans in ((1, "ctrlL1", [("b0.1", ChannelSpec(0.1, (1.0,))), ("b0.3", ChannelSpec(0.3, (1.0,)))]),
                           (2, "ctrlL2", [("a10", ChannelSpec(0.2, (1.0, 0.0))), ("a55", ChannelSpec(0.2, (0.5, 0.5)))])):
        for i, (tn, ch) in enumerate(chans):
            out.append(dict(source=dict(name=name, kind="EX", eta=0.05, eta_mom=0.05, ns=NS, reps=20), phys=Physics(0.9, 1.0, 1.0, 0.02, 0.1, 0.05, L),
                            truth_name=tn, truth=ch, tid=100 + 10 * L + i))
    return out


def wilson_free(x):
    return x


def run_task(task):
    t0 = time.time()
    src, phys, truth, tid, tname = task["source"], task["phys"], task["truth"], task["tid"], task["truth_name"]
    L = phys.L
    ns, reps = task.get("ns", src["ns"]), task.get("reps", src["reps"])
    ts = TestSet(phys, task.get("n_eval", N_EVAL), cell_id=0, master=MASTER_EVAL)
    n, V = ts.n, ts.V
    ex = ExactRecords(ts)
    tl = TrueLaw(ts, ex)
    K, wt = tl.K, truth.view_weights()
    m2t = tl.reference(truth)
    y_ex_naive = ((K * (ex.naive - m2t) ** 2) @ wt)
    V2, voi = float(y_ex_naive.mean()), tl.voi_clean()
    delta = float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi))
    model = None
    if src["kind"] == "EX":
        ll_e, mu_e = ex.ll, ex.mu
        src_phys = phys
    elif src["kind"] == "ETA":
        src_phys = phys.with_(eta=src["eta"])
        so = ExactRecords(ts, src_phys)
        ll_e, mu_e = so.ll, so.mu
    else:
        ck = torch.load(src["ckpt"], weights_only=False)
        model = CausalDensity(ck["cfg"].get("hidden", 64))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        ll_e, mu_e = query_density(model, ts.s_rec, ts.a_rec, phys)
        src_phys = phys
    pre = Precomputed.build(ll_e, mu_e)

    def regret(w):
        mu = pre.compose(w).reshape(n, V)
        return float(((K * (mu - m2t) ** 2) @ wt).mean())

    E_naive_pipe = regret(np.eye(L + 1)[0])
    fixed = {k: regret(c.view_weights()) for k, c in fixed_decl(L).items()}
    fixed["ORACLE"] = regret(wt)
    G = grid_w(L)
    prior = qbar_L(L).view_weights()
    nrep, nn = reps, len(ns)
    A = {k: np.full((nrep, nn, len(ADAPT)), np.nan) for k in ("E", "bhat", "tv", "r0", "raw")}
    pi_t, q_t = truth.pi_arr, truth.q
    for r in range(nrep):
        smp = simulate(phys, N_MAX, MASTER_ADAPT, cell_id=tid, chunk_id=r)
        S = apply_channel(smp.prefix, sample_theta(truth, N_MAX, MASTER_ADAPT, tid, r))
        if model is None:
            ll_a, _ = flip_batch(S, src_phys)
        else:
            ll_a, _ = query_density(model, S.s, S.a, phys)
        logell = ll_a[:, 1:] - ll_a[:, [0]]
        for ni, nsz in enumerate(ns):
            ws = {}
            ws["MLE"], _, _ = em_channel(logell[:nsz])
            ws["MAP"], _, _ = em_channel(logell[:nsz], prior=prior, kappa=KAPPA0)
            ws["PP"], _ = pp_channel(logell[:nsz], G)
            ws["MOM"], raw = mom_channel(S.s[:nsz], S.a[:nsz], phys, src["eta_mom"])
            for ei, name in enumerate(ADAPT):
                w = ws[name]
                bh = 1.0 - w[0]
                A["E"][r, ni, ei] = regret(w)
                A["bhat"][r, ni, ei] = bh
                A["tv"][r, ni, ei] = 0.5 * np.abs(w[1:] / max(bh, 1e-300) - pi_t).sum() if bh > 1e-12 else np.nan
                A["r0"][r, ni, ei] = w[L] / q_t[L - 1] if q_t[L - 1] > 0 else np.nan
                A["raw"][r, ni, ei] = raw if name == "MOM" else np.nan
    return dict(source=src["name"], kind=src["kind"], truth=tname, tid=tid, L=L, ns=list(ns), reps=nrep, eta_src=src.get("eta"), eta_mom=src["eta_mom"],
                V2=V2, voi=voi, delta=delta, E_naive_exact=V2, E_naive_pipe=E_naive_pipe, fixed=fixed, adapt=ADAPT,
                truth_beta=truth.beta, truth_q=list(truth.q), A={k: v.tolist() for k, v in A.items()}, seconds=time.time() - t0)


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
                torch=torch.__version__, numpy=np.__version__, command=" ".join(sys.argv))


def _init():
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e3"))
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    prov = provenance(out)
    json.dump(jsonable(prov), open(os.path.join(out, "provenance.json"), "w"), indent=1)
    tasks = []
    for s in sources():
        for tid, (tn, ch) in enumerate(truths8()):
            t = dict(source=s, phys=PHYS8, truth=ch, tid=tid, truth_name=tn)
            if args.smoke:
                t.update(ns=(10, 100), reps=2, n_eval=1500)
            tasks.append(t)
    tasks += control_tasks()
    if args.smoke:
        tasks = [t for t in tasks if t["source"]["name"] in ("EX", "A4-1000-s1", "ctrlL1", "ctrlL2")]
        for t in tasks:
            t.update(ns=(10, 100), reps=2, n_eval=1500)
    print(f"tasks: {len(tasks)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        futs = [pool.submit(run_task, t) for t in tasks]
        for t, f in zip(tasks, futs):
            r = f.result()
            r["provenance"] = prov
            json.dump(jsonable(r), open(os.path.join(out, "units", f"{t['source']['name']}__{t['truth_name']}.json"), "w"))
            print(f"[{time.time() - t0:.0f}s] {t['source']['name']} {t['truth_name']} done ({r['seconds']:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
