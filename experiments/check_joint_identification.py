"""Derivation check for joint (eta, q) identification (docs/literature_review.md section 4.3).

Not an experiment: no training and no estimator runs. It verifies two analytic statements.

1. Apparent-edge reduction. Under S2 with the exogenous single-flip channel, the record law depends on
   (eta, q) only through the law of the apparent edge pattern E xor F_theta. Here E is iid Bern(eta) over
   the L-1 mode edges, and F_theta is the flip footprint: none -> 0, end flips -> one edge, interior flip
   j -> edges (j-1, j).
   - Pattern-level distance is an upper bound on record-level distinguishability: the record law is a
     garbling of the pattern law.
   - Zero pattern distance therefore means exact observational equivalence.
2. The distance
       D(eta') = min_q' TV(P_{eta,q}, P_{eta',q'})
   is computed by linear programming over the full channel simplex, for several L and truths.
   - D = 0 at some eta' != eta means non-identification.
   - D > 0 for every eta' != eta means the pattern law separates eta. Record-level separation then also
     needs the garbling to be injective; this is checked by the record-level likelihood on the L=3 ridge.

Usage: python experiments/check_joint_identification.py   (about 1 minute)
"""
import itertools
import json
import os

import numpy as np
from scipy.optimize import linprog

from mti.decision import kappa
from mti.inference import candidate_from_queries, flip_batch
from mti.params import ChannelSpec, Physics
from mti.simulate import apply_channel, sample_theta, simulate


def footprints(L):
    out = [np.zeros(L - 1, int)]
    for j in range(L):
        f = np.zeros(L - 1, int)
        if j >= 1:
            f[j - 1] = 1
        if j <= L - 2:
            f[j] = 1
        out.append(f)
    return out                                   # index 0 = none, 1 + j = flip at position j


def pattern_law(L, eta, w):
    pats = np.array(list(itertools.product([0, 1], repeat=L - 1)))
    K = np.zeros((len(pats), L + 1))
    for k, f in enumerate(footprints(L)):
        d = (pats + f) % 2
        K[:, k] = np.prod(np.where(d == 1, eta, 1 - eta), axis=1)
    return K @ w, K


def min_tv(L, eta, w, eta2):
    P, _ = pattern_law(L, eta, w)
    _, K2 = pattern_law(L, eta2, np.ones(L + 1))
    G, m = K2.shape                              # variables: q' (m), t (G)
    c = np.r_[np.zeros(m), 0.5 * np.ones(G)]
    A = np.block([[K2, -np.eye(G)], [-K2, -np.eye(G)]])
    b = np.r_[P, -P]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=np.r_[np.ones(m), np.zeros(G)][None], b_eq=[1.0],
                  bounds=[(0, None)] * (m + G), method="highs")
    return res.fun, res.x[:m]


def truth(L, beta, alpha):
    ch = ChannelSpec.interp(L, beta, alpha)
    return np.r_[1 - ch.beta, ch.q]


def main():
    eta = 0.05
    grid = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.12]
    out = {"eta": eta, "eta_grid": grid, "D": {}}
    print("D(eta') = min_q' TV of apparent-edge pattern laws (0 = observationally equivalent)")
    for L in (2, 3, 4, 5, 8):
        for name, (beta, alpha) in {"clean": (0.0, 0.0), "uniform_b0.2": (0.2, 0.0), "lag0_b0.2": (0.2, 1.0)}.items():
            w = truth(L, beta, alpha)
            D = [min_tv(L, eta, w, e2)[0] for e2 in grid]
            out["D"][f"L{L}_{name}"] = D
            print(f"L={L} {name:13s} " + " ".join(f"{d:8.1e}" for d in D))

    # Record-level check on the L=3 ridge: equal likelihood on every record, different beliefs
    L, eta2 = 3, 0.03
    p = Physics(0.9, 1.0, 1.0, 0.5, 0.1, eta, L)
    w = truth(L, 0.2, 0.5)
    d, w2 = min_tv(L, eta, w, eta2)
    ch = ChannelSpec(1 - w[0], tuple(w[1:] / w[1:].sum()))
    ch2 = ChannelSpec(1 - w2[0], tuple(w2[1:] / w2[1:].sum()))
    n = 20000
    S = apply_channel(simulate(p, n, 3).prefix, sample_theta(ch, n, 3))
    ll, mu = flip_batch(S, p)
    ll2, mu2 = flip_batch(S, p.with_(eta=eta2))
    a = candidate_from_queries(ll, mu, ch)
    b = candidate_from_queries(ll2, mu2, ch2)
    k = kappa(p, S.s[:, -1])
    rec = {
        "pattern_tv": d, "beta_true": float(1 - w[0]), "beta_ridge": float(1 - w2[0]),
        "max_abs_record_loglik_diff": float(np.abs((ll[:, 0] + a.log_Z) - (ll2[:, 0] + b.log_Z)).max()),
        "max_abs_belief_diff": float(np.abs(a.mu2 - b.mu2).max()),
        "mean_regret_of_ridge_belief": float(np.mean(k * (a.mu2 - b.mu2) ** 2)),
        "V2_true": float(np.mean(k * (a.mu2 - mu[:, 0]) ** 2)),
    }
    out["L3_ridge"] = rec
    print("L=3 ridge (eta 0.05 -> 0.03):", json.dumps(rec, indent=1))
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "derivation_checks")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "joint_identification.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
