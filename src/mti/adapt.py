"""Channel identification from unlabeled corrupted prefixes and adaptive composition (E3, protocol docs/e3_protocol.md).

All estimators work on a vector of channel weights w = [q_none, q_0, ..., q_{L-1}] (a point on the simplex) and use only the records (s, a),
a clean-law source (through the likelihood ratios l_j(S) = p(T_jS)/p(S)) and the declared physics. No probe, modes or corruption labels.

  em_channel  : EM for the mixture weights maximising sum_r log Z_w(S_r), Z_w = w_none + sum_j w_j l_j(S_r); optional Dirichlet prior (MAP)
  pp_channel  : posterior predictive over a candidate grid with a uniform prior (by linearity, composition with sum_g post_g q_g is the Bayes estimator)
  mom_channel : pair-sum moment estimator from E[z_i z_j] = c^2 q_a^2 (1-2 eta)^|i-j| (1 - 2(q_i + q_j)), z_k = recorded a_k * delta_k
  Precomputed : scaled matrices that compose any channel with a clean-law source in O(R L) without logs (identical to the log-domain route)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.special import logsumexp

from .params import ChannelSpec, Physics

W_FLOOR = 1e-12  # estimator-side floor on weights inside logs (never applied to oracle quantities)


def view_vector(chan: ChannelSpec) -> np.ndarray:
    return chan.view_weights()


def chan_from_w(w: np.ndarray) -> ChannelSpec:
    beta = float(1.0 - w[0])
    q = np.asarray(w[1:], float)
    if beta <= 1e-15:
        return ChannelSpec(beta=0.0, pi=tuple([1.0 / len(q)] * len(q)))
    return ChannelSpec(beta=beta, pi=tuple(q / q.sum()))


def start_weights(L: int) -> np.ndarray:
    w = np.full(L + 1, 0.5 / L)
    w[0] = 0.5
    return w


def em_channel(logell: np.ndarray, prior: np.ndarray | None = None, kappa: float = 0.0, w0: np.ndarray | None = None,
               iters: int = 5000, tol: float = 1e-10, trace: bool = False):
    """EM for w maximising the (optionally Dirichlet-penalised) log-likelihood. logell (n, L) = log l_j per record.

    Returns (w, mean_loglik, iterations[, loglik trace]). M-step: w = (sum_r gamma_r + kappa * prior) / (n + kappa).
    """
    n, L = logell.shape
    w = start_weights(L) if w0 is None else np.asarray(w0, float).copy()
    a0 = np.concatenate([np.zeros((n, 1)), logell], axis=1)
    prev, tr = -np.inf, []
    for it in range(iters):
        lw = np.log(np.maximum(w, W_FLOOR))
        a = a0 + lw[None, :]
        lz = logsumexp(a, axis=1)
        ll = float(lz.mean())
        if trace:
            tr.append(ll)
        gam = np.exp(a - lz[:, None])
        w = gam.sum(axis=0)
        if kappa > 0:
            w = w + kappa * prior
        w = w / w.sum()
        if it > 0 and ll - prev < tol:
            break
        prev = ll
    return (w, ll, it + 1, tr) if trace else (w, ll, it + 1)


def pp_channel(logell: np.ndarray, grid_w: np.ndarray):
    """Posterior predictive over candidate channels (rows of grid_w, view-weight vectors), uniform prior. Returns (predictive w, posterior)."""
    n, L = logell.shape
    a0 = np.concatenate([np.zeros((n, 1)), logell], axis=1)                       # (n, L+1)
    lg = np.log(np.maximum(grid_w, W_FLOOR))                                       # (G, L+1)
    logpost = np.empty(len(grid_w))
    step = max(1, 4_000_000 // (len(grid_w) * (L + 1)))
    acc = np.zeros(len(grid_w))
    for lo in range(0, n, step):
        a = a0[lo:lo + step, None, :] + lg[None, :, :]
        acc += logsumexp(a, axis=2).sum(axis=0)
    logpost = acc - logsumexp(acc)
    post = np.exp(logpost)
    return post @ grid_w, post


def mom_channel(s: np.ndarray, a: np.ndarray, phys: Physics, eta_hat: float):
    """Moment estimator. Returns (w projected onto q >= 0, sum q <= 1; raw unconstrained sum of q-hat before projection)."""
    n, L = a.shape
    if L < 2:                                                                       # no pairs: nothing identified from cross-moments
        w = np.zeros(L + 1)
        w[0] = 1.0
        return w, 0.0
    delta = s[:, 1:] - phys.rho * s[:, :-1]
    z = a * delta
    rows, t, f = [], [], []
    for i in range(L):
        for j in range(i + 1, L):
            m = float(np.mean(z[:, i] * z[:, j]))
            fij = phys.c**2 * phys.q_a**2 * (1.0 - 2.0 * eta_hat) ** (j - i)
            r = np.zeros(L)
            r[[i, j]] = 1.0
            rows.append(r)
            t.append(0.5 * (1.0 - m / fij))
            f.append(fij)
    X, t, f = np.array(rows), np.array(t), np.array(f)
    raw = np.linalg.lstsq(X * f[:, None], t * f, rcond=None)[0]
    q, _ = nnls(X * f[:, None], t * f)
    if q.sum() > 1.0:
        q = q / q.sum()
    w = np.concatenate([[1.0 - q.sum()], q])
    return w, float(raw.sum())


@dataclass
class Precomputed:
    """Clean-law source evaluated on a set of records, scaled so any channel composes with two matrix products.

    mu_hat = (w_none e mu_S + A q) / (w_none e + Lm q),  Lm = exp(logell - m), A = Lm mu_flip, e = exp(-m), m = max(0, max_j logell_j) per record.
    Identical to the log-domain composition (candidate_from_queries); the scaling only avoids overflow.
    """
    Lm: np.ndarray
    A: np.ndarray
    e: np.ndarray
    mu_S: np.ndarray

    @staticmethod
    def build(ll_all: np.ndarray, mu_all: np.ndarray) -> "Precomputed":
        logell = ll_all[:, 1:] - ll_all[:, [0]]
        m = np.maximum(0.0, logell.max(axis=1))
        Lm = np.exp(logell - m[:, None])
        return Precomputed(Lm, Lm * mu_all[:, 1:], np.exp(-m), mu_all[:, 0].copy())

    def compose(self, w: np.ndarray) -> np.ndarray:
        wn = w[0] * self.e
        return (wn * self.mu_S + self.A @ w[1:]) / (wn + self.Lm @ w[1:])
