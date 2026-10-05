"""Evaluation-set uncertainty for the E4 primary contrasts (protocol e4_protocol.md section 5: "evaluation-set SE is reported separately"; gap G1 in
docs/consolidation_review.md). Evaluation-only recomputation from saved estimates; no estimation, training or new data.

For P1 cells (L = 8, truth T0, n = 1e4, both physics, all replicates) it recomputes, on the fixed evaluation set (master 8601, cell 0, exactly as in
run_e4.run_job), the per-prefix regret of J-MLE, EX-MLE and ETA--MLE from the saved (eta_hat, w_hat), and
  1. checks that the recomputed mean regrets equal the saved ones (relative error),
  2. reports the prefix-level SE of the paired differences J-EX and ETA--J for each replicate, and the SE of their replicate mean treating replicates as
     independent (a lower bound, since all replicates share the same evaluation prefixes).

    python experiments/check_e4_eval_se.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e4 import MASTER_EVAL, N_EVAL, PHYS, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.learned.evalset import TestSet  # noqa: E402
from mti.misspec import ExactRecords, TrueLaw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_P1 = 10000
ARMS = ("J-MLE", "EX-MLE", "ETA--MLE")


def main():
    out = {}
    for pname in PHYS:
        phys = PHYS[pname]
        L = phys.L
        tname, truth = truths(L)[0]
        assert tname == "T0"
        wt = truth.view_weights()
        ts = TestSet(phys, N_EVAL, cell_id=0, master=MASTER_EVAL)
        n_ev, V = ts.n, ts.V
        ex = ExactRecords(ts)
        tl = TrueLaw(ts, ex)
        K = tl.K
        m2t = tl.reference(truth)
        cache = {}

        def pre_at(eta):
            key = round(float(eta), 9)
            if key not in cache:
                e = ex if abs(eta - phys.eta) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(eta)))
                cache[key] = Precomputed.build(e.ll, e.mu)
            return cache[key]

        def per_prefix(eta, w):
            mu = pre_at(eta).compose(np.asarray(w)).reshape(n_ev, V)
            return (K * (mu - m2t) ** 2) @ wt

        files = sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage1", "units", f"{pname}_L{L}_T0_r*.json")))
        rel_err, se_je, se_ej, d_je, d_ej = [], [], [], [], []
        voi = None
        for f in files:
            u = json.load(open(f))
            voi, delta = u["voi"], u["delta"]
            y = {}
            for arm in ARMS:
                a = u["arms"][arm][str(N_P1)]
                y[arm] = per_prefix(a["eta"], a["w"])
                rel_err.append(abs(y[arm].mean() - a["regret"]) / max(a["regret"], 1e-300))
            dje, dej = y["J-MLE"] - y["EX-MLE"], y["ETA--MLE"] - y["J-MLE"]
            d_je.append(dje.mean()), d_ej.append(dej.mean())
            se_je.append(dje.std(ddof=1) / np.sqrt(n_ev)), se_ej.append(dej.std(ddof=1) / np.sqrt(n_ev))
        R = len(files)
        out[pname] = dict(
            R=R, n_eval_prefixes=n_ev, voi=voi, delta_over_voi=delta / voi,
            max_rel_err_recomputed_vs_saved=float(max(rel_err)),
            J_minus_EX=dict(mean_over_voi=float(np.mean(d_je) / voi), prefix_se_per_replicate_over_voi=[float(min(se_je) / voi), float(max(se_je) / voi)],
                            se_of_replicate_mean_independent_over_voi=float(np.sqrt(np.sum(np.square(se_je))) / R / voi)),
            ETAm_minus_J=dict(mean_over_voi=float(np.mean(d_ej) / voi), prefix_se_per_replicate_over_voi=[float(min(se_ej) / voi), float(max(se_ej) / voi)],
                              se_of_replicate_mean_independent_over_voi=float(np.sqrt(np.sum(np.square(se_ej))) / R / voi)))
        print(pname, json.dumps(out[pname], indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "e4", "stage1", "eval_se_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
