"""Exact L=3 identified set (supplied physics, full channel simplex): admissible eta' interval, ridge channels,
record-likelihood check, beliefs and decision distances on the E4 evaluation prefixes (master 8601, cell 0, 20,000)."""
import sys, json, itertools
import numpy as np

from e4common import PHYS, truths, MASTER_EVAL, N_EVAL, TestSet

from mti.misspec import ExactRecords, TrueLaw
from mti.adapt import Precomputed
from mti.joint import footprints

L = 3
pats = np.array(list(itertools.product([0, 1], repeat=L - 1)))           # pattern order (00,01,10,11)
fps = footprints(L)                                                         # state order: none, flip0, flip1, flip2
def pat_index(f): return int(np.where((pats == f).all(1))[0][0])
S_of_state = [pat_index(f) for f in fps]
H = np.array([[(-1) ** int(np.dot(A, x)) for x in pats] for A in pats])   # Walsh-Hadamard: chi_A(x)
absA = pats.sum(1)

def F_from_w(w):
    F = np.zeros(len(pats))
    for k, i in enumerate(S_of_state): F[i] += w[k]
    return F
def w_from_F(F):
    return np.array([F[i] for i in S_of_state])
def ridge_w(w, eta, eta2):
    t = (1 - 2 * eta) / (1 - 2 * eta2)
    Fh = H @ F_from_w(w)
    return w_from_F((H @ (Fh * t ** absA)) / len(pats))

def eta_hi(w, eta):
    # largest eta2 >= eta with ridge_w >= 0 (bisection on t; F'(x) is a quadratic in t)
    lo, hi = eta, 0.4999
    if ridge_w(w, eta, eta + 1e-9).min() < -1e-12: return eta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if ridge_w(w, eta, mid).min() >= 0 else (lo, mid)
    return lo

