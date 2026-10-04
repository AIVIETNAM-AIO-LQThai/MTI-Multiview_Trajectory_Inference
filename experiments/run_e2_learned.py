"""E2 learned confirmation (protocol e2_protocol.md sections 4-7): lead physics (q_w=0.02), N in {1e3, 1e5}, seeds 1-10.

    python experiments/run_e2_learned.py --out results/e2

Per unit (N, seed): train A4 (clean density, pilot config), A6pd-broad (E1 config) and A6pd-cov (coverage control, same config), fit the HMM persistence by EM,
save checkpoints, then score every arm (with q_assumed and with its own 'ignore the channel' version) under q_true on an independent 20,000-prefix test set (master 8201).
Seeds 1-5 also re-evaluate A4 and A6pd-broad on the pilot's test set at P1/P3 for the reproduction check.
Training prefixes are those of the pilot (same masters, cell id 0, chunk = seed); seeds 6-10 are new.
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

from mti.inference import candidate_from_queries
from mti.learned.data import MASTER_TRAIN, MASTER_VAL, make_probe_set
from mti.learned.evalset import TestSet, candidate_from_density, query_density, query_density_belief
from mti.learned.hmm_fit import fit_eta
from mti.learned.train import TrainCfg, train_A4, train_A6pd
from mti.misspec import ExactRecords, TrueLaw, no_channel
from mti.params import ChannelSpec, Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS = Physics(0.9, 1.0, 1.0, 0.02, 0.1, 0.05, 8)
L = PHYS.L
MASTER_CONFIRM = 8201
N_TEST = 20_000
NS = (1_000, 100_000)
SEEDS = tuple(range(1, 11))
ARMS = ["R1", "A4", "A4n", "A5", "A5n", "A6pd", "A6pdn", "A6c", "A6cn"]


def make_cells():
    cells = {}
    for a in (0.0, 1.0):
        for ba in (0.05, 0.1, 0.2, 0.3, 0.5):
            cells[f"rate_a{a:g}_ba{ba:g}"] = dict(axis="rate", alpha_t=a, alpha_a=a, beta_t=0.2, beta_a=ba)
    for at, aa in ((0.0, 1.0), (1.0, 0.0), (0.5, 0.0), (0.5, 1.0)):
        cells[f"loc_at{at:g}_aa{aa:g}"] = dict(axis="location", alpha_t=at, alpha_a=aa, beta_t=0.2, beta_a=0.2)
    return cells


CELLS = make_cells()


def chans(c):
    return ChannelSpec.interp(L, c["beta_t"], c["alpha_t"]), ChannelSpec.interp(L, c["beta_a"], c["alpha_a"])


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


def provenance(out):
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    status = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith(("results/", "docs/reports/"))]
    patch = git("diff", "HEAD", "--", "src", "experiments", "tests", "configs")
    untracked = [f for f in git("ls-files", "--others", "--exclude-standard", "--", "src", "experiments", "tests", "configs").splitlines()]
    hashes = {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:16] for f in untracked if os.path.isfile(os.path.join(ROOT, f))}
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "code_patch.diff"), "w").write(patch)
    return dict(git_commit=git("rev-parse", "HEAD").strip(), branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(), dirty_code=status,
                patch_sha256=hashlib.sha256(patch.encode()).hexdigest()[:16], untracked_code_hashes=hashes,
                python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, command=" ".join(sys.argv))


def _init_worker():
    torch.set_num_threads(1)


def tuned(N):
    pil = json.load(open(os.path.join(ROOT, "results", "learned_pilot", "tuning.json")))["selected"]
    e1 = json.load(open(os.path.join(ROOT, "results", "e1", "tuning.json")))["selected"]
    strip = lambda c: {k: v for k, v in c.items() if k != "seed"}
    return dict(A4=strip(pil[f"0|{N}|A4"]["cfg"]), A6pd=strip(e1[f"0|{N}|A6pd-broad"]["cfg"]))


def run_unit(args):
    N, seed, out = args
    t0 = time.time()
    cfgs = tuned(N)
    tr = make_probe_set(PHYS, N, MASTER_TRAIN, 0, seed)
    va = make_probe_set(PHYS, N // 4, MASTER_VAL, 0, seed)
    m4, i4 = train_A4(tr, va, PHYS, TrainCfg(**cfgs["A4"], seed=seed))
    mb, ib = train_A6pd(tr, va, PHYS, "broad", TrainCfg(**cfgs["A6pd"], seed=seed))
    mc, ic = train_A6pd(tr, va, PHYS, "cov", TrainCfg(**cfgs["A6pd"], seed=seed))
    sL = tr.s[:, -1].numpy().astype(float)
    eta_hat, em_it, _ = fit_eta(tr.s.numpy().astype(float), tr.a.numpy().astype(float), (tr.d_pr.numpy() + PHYS.rho * sL).astype(float),
                                tr.a_pr.numpy().astype(float), PHYS)
    ck = os.path.join(out, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    for name, m in (("A4", m4), ("A6pd", mb), ("A6c", mc)):
        torch.save(dict(state_dict=m.state_dict(), cfg=cfgs["A4" if name == "A4" else "A6pd"], N=N, seed=seed, eta_hat=eta_hat,
                        cond_dim=getattr(m, "cond_dim", 0)), os.path.join(ck, f"N{N}_s{seed}_{name}.pt"))
    ts = TestSet(PHYS, N_TEST, cell_id=0, master=MASTER_CONFIRM)
    n, V = ts.n, ts.V
    ex = ExactRecords(ts)
    truth = TrueLaw(ts, ex)
    ex_h = ExactRecords(ts, PHYS.with_(eta=eta_hat))
    ll4, mu4 = query_density(m4, ts.s_rec, ts.a_rec, PHYS)
    cache = {}

    def direct(model, ca, tag):
        key = (tag, round(ca.beta, 9), tuple(np.round(ca.pi, 9)))
        if key not in cache:
            q = torch.as_tensor(np.tile(ca.q.astype(np.float32), (n * V, 1)))
            cache[key] = query_density_belief(model, ts.s_rec, ts.a_rec, PHYS, cond=q).reshape(n, V)
        return cache[key]

    zero = no_channel(L)
    rows, arrays = [], {}
    for cname, c in CELLS.items():
        ct, ca = chans(c)
        m2a = ex.mu2(ca)
        beliefs = {
            "R1": (m2a, ex.naive),
            "A4": (candidate_from_density(ll4, mu4, ca, n, V), mu4[:, 0].reshape(n, V)),
            "A5": (candidate_from_queries(ex_h.ll, ex_h.mu, ca).mu2.reshape(n, V), ex_h.naive),
            "A6pd": (direct(mb, ca, "b"), direct(mb, zero, "b")),
            "A6c": (direct(mc, ca, "c"), direct(mc, zero, "c")),
        }
        for arm, (aware, naive) in beliefs.items():
            for variant, mu in ((arm, aware), (arm + "n", naive)):
                r, y = truth.score(mu, ct, naive, mu2a=m2a)
                rows.append(dict(cell=cname, arm=variant, **r))
                arrays[f"{cname}|{variant}"] = y.astype(np.float32)
    repro = {}
    if seed <= 5:                                                      # reproduction check on the pilot's own test set at P1 and P3
        tp = TestSet(PHYS, N_TEST, cell_id=0)
        tp.add_prior("P1", ChannelSpec.uniform(L, 0.2))
        tp.add_prior("P3", ChannelSpec.interp(L, 0.2, 1.0))
        ll4p, mu4p = query_density(m4, tp.s_rec, tp.a_rec, PHYS)
        for pr, ch in (("P1", tp.chans["P1"]), ("P3", tp.chans["P3"])):
            repro[f"A4|{pr}"] = tp.summarize(pr, candidate_from_density(ll4p, mu4p, ch, tp.n, tp.V))["E"]
            q = torch.as_tensor(np.tile(ch.q.astype(np.float32), (tp.n * tp.V, 1)))
            repro[f"A6pd-broad|{pr}"] = tp.summarize(pr, query_density_belief(mb, tp.s_rec, tp.a_rec, PHYS, cond=q).reshape(tp.n, tp.V))["E"]
    os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "arrays", f"N{N}_s{seed}.npz"), **arrays)
    info = {nm: dict(best_val=i.best_val, epochs_run=i.epochs_run, steps=i.steps, seconds=i.seconds, params=i.n_params)
            for nm, i in (("A4", i4), ("A6pd", ib), ("A6c", ic))}
    return dict(N=N, seed=seed, eta_hat=eta_hat, em_iterations=em_it, cfgs=cfgs, train=info, rows=rows, repro=repro, unit_seconds=time.time() - t0)


def exact_stage(out):
    """Seed-independent exact references on the confirmation set: R1 and R2 per cell (arrays + scalars)."""
    ts = TestSet(PHYS, N_TEST, cell_id=0, master=MASTER_CONFIRM)
    ex = ExactRecords(ts)
    truth = TrueLaw(ts, ex)
    rows, arrays = [], {}
    for cname, c in CELLS.items():
        ct, ca = chans(c)
        m2a = ex.mu2(ca)
        r1, y1 = truth.score(m2a, ct, ex.naive, mu2a=m2a)
        r2, y2 = truth.score(ex.naive, ct, ex.naive, mu2a=m2a)
        rows.append(dict(cell=cname, **c, voi_clean=truth.voi_clean(), V2_true=r2["E"], R1=r1, R2=r2))
        arrays[f"{cname}|R1"], arrays[f"{cname}|R2"] = y1.astype(np.float32), y2.astype(np.float32)
    os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "arrays", "exact.npz"), **arrays)
    json.dump(jsonable(dict(cells=CELLS, rows=rows)), open(os.path.join(out, "exact_confirm.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e2"))
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--stage", choices=["exact", "units", "all"], default="all")
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    prov = provenance(out)
    json.dump(jsonable(prov), open(os.path.join(out, "provenance.json"), "w"), indent=1)
    t0 = time.time()
    if args.stage in ("exact", "all"):
        exact_stage(out)
        print(f"[{time.time() - t0:.0f}s] exact references done", flush=True)
    if args.stage in ("units", "all"):
        tasks = sorted([(N, s, out) for N in NS for s in args.seeds], key=lambda t: -t[0])
        print(f"units: {len(tasks)}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            futs = [pool.submit(run_unit, t) for t in tasks]
            for t, f in zip(tasks, futs):
                r = f.result()
                r["provenance"] = prov
                json.dump(jsonable(r), open(os.path.join(out, "units", f"N{t[0]}_s{t[1]}.json"), "w"))
                print(f"[{time.time() - t0:.0f}s] unit N{t[0]} seed{t[1]} done ({r['unit_seconds']:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
