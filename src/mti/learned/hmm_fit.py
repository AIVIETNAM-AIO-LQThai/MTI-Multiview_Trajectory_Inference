"""A5: symmetric two-state HMM with known physics; persistence eta learned from clean data by EM (probe transition included)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..inference import _log_trans, backward, forward, log_emissions
from ..params import Physics

ETA_MIN = 1e-4  # clamp: EM can return exactly 0 on small samples, which would make the filter deterministic


def fit_eta(s, a, s_probe_next, a_probe, phys: Physics, eta0: float = 0.1, iters: int = 300, tol: float = 1e-7):
    """EM for eta. Uses L+1 emissions per clean prefix (the L recorded transitions and the independent probe transition).

    Returns (eta_hat, n_iterations, log_likelihood_per_prefix). Only clean prefixes and their probe are used.
    """
    s_ext = np.concatenate([s, s_probe_next[:, None]], axis=1)
    a_ext = np.concatenate([a, a_probe[:, None]], axis=1)
    logE = log_emissions(s_ext, a_ext, phys)                                 # (n, L+1, 2): transitions 0..L
    n, T, _ = logE.shape
    eta = eta0
    ll = np.nan
    for it in range(iters):
        ls, lw = _log_trans(eta)
        pre, tilde = forward(logE, eta, keep_pre=True)
        beta = backward(logE, eta)
        ll_i = logsumexp(tilde, axis=-1)
        ll = float(ll_i.mean())
        # xi_k(m,m') for k = 0..T-2: alpha_k(m) e_k(m) P(m,m') e_{k+1}(m') beta_{k+1}(m')
        a_t = pre[:, :-1, :] + logE[:, :-1, :]                               # (n, T-1, 2)  over m
        nxt = logE[:, 1:, :] + beta[:, 1:, :]                                 # (n, T-1, 2)  over m'
        sw = np.logaddexp(a_t[..., 0] + lw + nxt[..., 1], a_t[..., 1] + lw + nxt[..., 0])
        p_switch = np.exp(sw - ll_i[:, None])
        new = float(p_switch.mean())
        new = min(max(new, ETA_MIN), 0.5)
        if abs(new - eta) < tol:
            eta = new
            break
        eta = new
    return eta, it + 1, ll
