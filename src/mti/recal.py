"""Privileged restricted-summary recalibration diagnostic (spec section 9).

g(|s_L|, mu~) approximates mu_B = E[mu_2 | s_L, mu~] by a kappa-weighted binned regression of the oracle target
mu_2 on quantile bins of (|s_L|, |mu~|), odd-symmetrised in mu~ (mu_B is even in s_L and odd in mu~, spec T13).
This is an information-analysis diagnostic fitted with oracle mu_2 targets; it is not the deployable recalibrator.
"""
from __future__ import annotations

import numpy as np


class BinnedRecal:
    def __init__(self, n_s: int, n_m: int):
        self.n_s, self.n_m = n_s, n_m
        self.s_edges = self.m_edges = self.table = None
        self.n_fit = 0

    def fit(self, abs_s, mu, mu2, kap):
        am = np.abs(mu)
        self.s_edges = np.unique(np.quantile(abs_s, np.linspace(0, 1, self.n_s + 1)[1:-1]))
        self.m_edges = np.unique(np.quantile(am, np.linspace(0, 1, self.n_m + 1)[1:-1]))
        ns, nm = len(self.s_edges) + 1, len(self.m_edges) + 1
        idx = np.searchsorted(self.s_edges, abs_s, side="right") * nm + np.searchsorted(self.m_edges, am, side="right")
        tgt = np.where(mu >= 0, 1.0, -1.0) * mu2
        num = np.bincount(idx, weights=kap * tgt, minlength=ns * nm)
        den = np.bincount(idx, weights=kap, minlength=ns * nm)
        overall = float(np.sum(kap * tgt) / np.sum(kap))
        self.table = np.where(den > 0, num / np.where(den > 0, den, 1.0), overall).reshape(ns, nm)
        self.n_fit = len(abs_s)
        return self

    def __call__(self, abs_s, mu):
        abs_s, mu = np.broadcast_arrays(np.asarray(abs_s), np.asarray(mu))
        i = np.searchsorted(self.s_edges, abs_s, side="right")
        j = np.searchsorted(self.m_edges, np.abs(mu), side="right")
        return np.where(mu >= 0, 1.0, -1.0) * self.table[i, j]
