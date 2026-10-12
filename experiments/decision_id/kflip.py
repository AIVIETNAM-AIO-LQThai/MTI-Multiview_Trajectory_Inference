"""At-most-k-flip channels: thresholds by brute force and LP (closeout appendix; discussion section 5.5). Exact-law only; no fitting.

Channel: sigma = (-1)^b with b in A_k = {|b| <= k} (b in {0,1}^L), exogenous law over A_k, footprint Db_j = b_j xor b_{j+1}. Predictions checked:
  injective            iff L >= 2k + 1               (channel identified with eta known)
  image = whole cube   iff L <= 2k + 1
  jointly identified   iff L >= 2k + 2               (eta unknown; common eta)
LP: minimum total variation over channels at alternative persistence values (an interior random channel, eta = 0.05); ambiguous iff it is ~0.
Scope: a common eta is needed; independent per-label flips are a different family. Seeds: default_rng(0), loop order k = 1..3, L = 2..9 (as received).

    python experiments/decision_id/kflip.py   ->  results/decision_id/kflip.json
"""
import itertools
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from did_common import save_json  # noqa: E402


def check(rng, L, k, eta=0.05, alts=(0.02, 0.035, 0.065, 0.08)):
    subsets = [b for b in itertools.product([0, 1], repeat=L) if sum(b) <= k]
    D = lambda b: tuple((b[j] ^ b[j + 1]) for j in range(L - 1))
    fp = [D(b) for b in subsets]
    inj = len(set(fp)) == len(fp)
    pats = list(itertools.product([0, 1], repeat=L - 1)) if L > 1 else [()]
    full = set(fp) == set(pats)
    comp_pairs = sum(1 for b in subsets if tuple(1 - x for x in b) in subsets and sum(b) < L - sum(b)
                     or (tuple(1 - x for x in b) in subsets and sum(b) == L - sum(b) and b < tuple(1 - x for x in b)))

    def K(e):
        M = np.zeros((len(pats), len(subsets)))
        for s, f in enumerate(fp):
            for p, x in enumerate(pats):
                d = [(x[i] + f[i]) % 2 for i in range(L - 1)]
                M[p, s] = np.prod([e if di else 1 - e for di in d]) if L > 1 else 1.0
        return M

    w = rng.dirichlet(np.ones(len(subsets)) * 2.0)
    P = K(eta) @ w
    tvs = []
    for e2 in alts:
        K2 = K(e2)
        G, m = K2.shape
        c = np.r_[np.zeros(m), 0.5 * np.ones(G)]
        A = np.block([[K2, -np.eye(G)], [-K2, -np.eye(G)]])
        r = linprog(c, A_ub=A, b_ub=np.r_[P, -P], A_eq=np.r_[np.ones(m), np.zeros(G)][None], b_eq=[1], bounds=[(0, None)] * (m + G), method="highs")
        tvs.append(r.fun)
    return dict(L=L, k=k, n_subsets=len(subsets), cube=len(pats), injective=inj, image_full=full, complement_pairs_allowed=comp_pairs,
                pred_known=(L >= 2 * k + 1), pred_joint=(L >= 2 * k + 2), min_tv_alt=[round(t, 8) for t in tvs])


def main():
    rng = np.random.default_rng(0)
    rows, bad = [], 0
    for k in (1, 2, 3):
        for L in range(2, 10):
            r = check(rng, L, k)
            ok_known = r["injective"] == r["pred_known"]
            ok_joint = ((not r["image_full"]) and r["injective"]) == r["pred_joint"]
            ambiguous = min(r["min_tv_alt"]) < 1e-9
            ok_lp = ambiguous == (not r["pred_joint"])
            ok = bool(ok_known and ok_joint and ok_lp)
            bad += not ok
            r["consistent_with_predictions"] = ok
            rows.append(r)
            print(r, "OK" if ok else "MISMATCH")
    print("mismatches", bad)
    print("saved", save_json("kflip.json", dict(rows=rows, mismatches=bad), __file__))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
