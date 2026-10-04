"""E2 exact screening (protocol e2_protocol.md section 6): exact composition with q_assumed vs ignoring the channel, scored under q_true.

    python experiments/run_e2_screen.py --out results/e2/screen

Exploratory. Both physics, 100,000 fresh prefixes (master 8101). Exact clean law (true eta) for every arm, so the only error is the channel.
Arms: R0 = true-channel aware oracle (reference, E = 0), R1 = exact composition with q_assumed, R2 = exact naive (beta_a = 0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from mti.learned.evalset import TestSet
from mti.misspec import ExactRecords, TrueLaw, no_channel, predicted_safe, record_violations
from mti.params import ChannelSpec, Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSICS = [Physics(0.9, 1.0, 1.0, 0.02, 0.1, 0.05, 8), Physics(0.9, 1.0, 1.0, 0.5, 0.1, 0.05, 8)]
MASTER_SCREEN = 8101
ALPHAS = (0.0, 0.5, 1.0)
BETA_T = (0.0, 0.05, 0.2, 0.5)
BETA_A = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
LOC = (0.0, 0.25, 0.5, 0.75, 1.0)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def provenance():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.rstrip("\n")
    return dict(git_commit=git("rev-parse", "HEAD").strip(), branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                dirty=[l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith(("results/", "docs/reports/"))],
                python=sys.version.split()[0], numpy=np.__version__, command=" ".join(sys.argv))


def run_pair(truth, exact, axis, a_t, a_a, b_t, b_a, L):
    ct, ca = ChannelSpec.interp(L, b_t, a_t), ChannelSpec.interp(L, b_a, a_a)
    m2a = exact.mu2(ca)
    r1, y1 = truth.score(m2a, ct, exact.naive, mu2a=m2a)
    r2, y2 = truth.score(exact.naive, ct, exact.naive)
    b, b_se = truth.paired(y1, y2)                                   # B_exact = E(R2) - E(R1)
    return dict(axis=axis, alpha_t=a_t, alpha_a=a_a, beta_t=b_t, beta_a=b_a, V2_true=r2["E"], V2_true_se=r2["E_se"],
                E_R1=r1["E"], E_R1_se=r1["E_se"], B_exact=b, B_exact_se=b_se, cost_R1=r1["cost"], cost_R2=r2["cost"],
                cost_gain=r2["cost"] - r1["cost"], D_R1=r1["D"], D_R1_se=r1["D_se"], D_none_R1=r1["D_none"], D_none_R1_se=r1["D_none_se"],
                D_cor_R1=r1["D_cor"], D_cor_R1_se=r1["D_cor_se"], D_switch_R1=r1["D_switch"], D_switch_R1_se=r1["D_switch_se"],
                voi_clean=truth.voi_clean(), predicted_safe=predicted_safe(b_a, b_t, a_t == a_a),
                violations=record_violations(truth, ct, ca) if a_t == a_a else None,
                decomp_err=r1["decomp_max_abs_err"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e2", "screen"))
    ap.add_argument("--n", type=int, default=100_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prov = provenance()
    for pi_, phys in enumerate(PHYSICS):
        t0 = time.time()
        ts = TestSet(phys, args.n, cell_id=pi_, master=MASTER_SCREEN)
        exact = ExactRecords(ts)
        truth = TrueLaw(ts, exact)
        rows = []
        for a in ALPHAS:                                                   # rate axis: location prior correct
            for bt in BETA_T:
                for ba in BETA_A:
                    rows.append(run_pair(truth, exact, "rate", a, a, bt, ba, phys.L))
        for at in LOC:                                                     # location axis: beta = 0.2 on both sides
            for aa in LOC:
                rows.append(run_pair(truth, exact, "location", at, aa, 0.2, 0.2, phys.L))
        json.dump(jsonable(dict(provenance=prov, q_w=phys.q_w, n=args.n, master=MASTER_SCREEN, rows=rows, seconds=time.time() - t0)),
                  open(os.path.join(args.out, f"phys{pi_}.json"), "w"), indent=1)
        print(f"phys{pi_} (q_w={phys.q_w}): {len(rows)} pairs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
