"""Privileged restricted-summary recalibration diagnostic (spec section 9).

g(|s_L|, mu~) approximates mu_B = E[mu_2 | s_L, mu~] by a kappa-weighted binned regression on quantile bins of
(|s_L|, |mu~|), odd-symmetrised in mu~ (mu_B is even in s_L and odd in mu~, spec T13).
Two function classes (both fitted with oracle mu_2 targets; neither is the deployable recalibrator):
  mode="direct"  : g = sign(mu~) * bin-mean of sign(mu~) * mu_2
  mode="residual": g = mu~ + sign(mu~) * bin-mean of sign(mu~) * (mu_2 - mu~)
The residual class contains the identity map, so bin discretisation error is paid only on the (small)
correction; the direct class can be worse than the naive belief when V_2 is small.
"""
from __future__ import annotations

import numpy as np


class BinnedRecal:
    def __init__(self, n_s: int, n_m: int, mode: str = "direct"):
        if mode not in ("direct", "residual"):
            raise ValueError(mode)
        self.n_s, self.n_m, self.mode = n_s, n_m, mode
        self.s_edges = self.m_edges = self.table = None
        self.n_fit = 0

    def fit(self, abs_s, mu, mu2, kap):
        am = np.abs(mu)
        self.s_edges = np.unique(np.quantile(abs_s, np.linspace(0, 1, self.n_s + 1)[1:-1]))
        self.m_edges = np.unique(np.quantile(am, np.linspace(0, 1, self.n_m + 1)[1:-1]))
        ns, nm = len(self.s_edges) + 1, len(self.m_edges) + 1
        idx = np.searchsorted(self.s_edges, abs_s, side="right") * nm + np.searchsorted(self.m_edges, am, side="right")
        sign = np.where(mu >= 0, 1.0, -1.0)
        tgt = sign * (mu2 if self.mode == "direct" else mu2 - mu)
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
        corr = np.where(mu >= 0, 1.0, -1.0) * self.table[i, j]
        return corr if self.mode == "direct" else mu + corr
