"""E4 stage 0: exact-law diagnostic (protocol docs/e4_protocol.md section 3). No estimators beyond EM at fixed eta; no training.

    python experiments/run_e4_stage0.py --out results/e4/stage0

For every (physics, L, truth), on 2e5 corrupted records (master 8801):
  1. pattern distance D(eta') = min_q' TV of apparent-edge pattern laws,
  2. pseudo-true curve: q*(eta') = EM fit at eta', KL(eta') = mean log-likelihood deficit of (eta', q*) against the true law, regret*(eta') on the evaluation set,
  3. profile Fisher information of eta at the truth and the predicted SE(eta_hat) = 1/sqrt(n I_prof).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e4 import ETA_TRUE, LS, MASTER_EVAL, PHYS, jsonable, provenance, task_id, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.joint import fit_channel, pattern_tv, profile_fisher, _queries  # noqa: E402
from mti.learned.evalset import TestSet  # noqa: E402
from mti.misspec import ExactRecords, TrueLaw  # noqa: E402
from mti.simulate import apply_channel, sample_theta, simulate  # noqa: E402

MASTER_STAGE0 = 8801
N0 = 200_000
ETAS = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.12]


def run_one(task):
    t0 = time.time()
    pname, L, tname, truth, tid = task
    phys = PHYS[pname].with_(L=L)
    wt = truth.view_weights()
    S = apply_channel(simulate(phys, N0, MASTER_STAGE0, tid, 0).prefix, sample_theta(truth, N0, MASTER_STAGE0, tid, 0))
    ts = TestSet(phys, 20_000, cell_id=0, master=MASTER_EVAL)
    n, V = ts.n, ts.V
    ex = ExactRecords(ts)
    tl = TrueLaw(ts, ex)
    K = tl.K
    m2t = tl.reference(truth)
    V2, voi = float(((K * (ex.naive - m2t) ** 2) @ wt).mean()), tl.voi_clean()
    delta = float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi))
    llS0, logell0 = _queries(S, phys, ETA_TRUE)
    ll_true = llS0 + np.log(wt[0] + np.exp(logell0) @ wt[1:])                   # log p(S) under the true (eta, q)
    rows = []
    for e in ETAS:
        D, _ = pattern_tv(L, ETA_TRUE, wt, e)
        llS, logell = _queries(S, phys, e)
        w, ll, it = fit_channel(llS, logell)
        kl = float(np.mean(ll_true) - ll)
        exr = ex if abs(e - ETA_TRUE) < 1e-12 else ExactRecords(ts, phys.with_(eta=e))
        mu = Precomputed.build(exr.ll, exr.mu).compose(w).reshape(n, V)
        rows.append(dict(eta=e, pattern_tv=D, kl=kl, beta_star=float(1 - w[0]), q_star=list(w[1:]), regret=float(((K * (mu - m2t) ** 2) @ wt).mean()), em_iter=it))
    fi = profile_fisher(S, phys, truth)
    ip = fi["I_prof"]
    kl03 = [r for r in rows if abs(r["eta"] - 0.03) < 1e-12][0]["kl"]
    return dict(physics=pname, L=L, truth=tname, V2=V2, voi=voi, delta=delta, curve=rows, I_eta=fi["I_eta"], I_prof=ip,
                se_pred_n1e4=(1.0 / math.sqrt(1e4 * ip)) if ip > 0 else None, n_lr_reject_eta03=(1.92 / kl03) if kl03 > 0 else None,
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e4/stage0")
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "units"), exist_ok=True)
    prov = provenance(args.out)
    json.dump(jsonable(prov), open(os.path.join(args.out, "provenance.json"), "w"), indent=1)
    tasks = [(pn, L, tn, ch, task_id(pn, L, j)) for pn in PHYS for L in LS for j, (tn, ch) in enumerate(truths(L))]
    print(f"stage 0: {len(tasks)} tasks", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for t, r in zip(tasks, pool.map(run_one, tasks)):
            r["provenance"] = prov
            json.dump(jsonable(r), open(os.path.join(args.out, "units", f"{t[0]}_L{t[1]}_{t[2]}.json"), "w"))
            print(f"[{time.time() - t0:.0f}s] {t[0]} L={t[1]} {t[2]} I_prof/I_eta={r['I_prof'] / r['I_eta']:.3g} se@1e4={r['se_pred_n1e4']}", flush=True)


if __name__ == "__main__":
    main()
