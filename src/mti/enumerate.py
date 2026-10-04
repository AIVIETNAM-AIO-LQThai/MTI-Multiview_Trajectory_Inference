"""Brute-force references used only by the tests: enumerate every mode path (and every corruption hypothesis).

These share no HMM/forward-filter code with `inference.py`: they evaluate the full joint density
p_0(H, m_{0:L}) path by path.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from .params import ChannelSpec, Physics


def _paths(L: int) -> np.ndarray:
    return np.array(list(itertools.product([1.0, -1.0], repeat=L + 1)))  # (2^(L+1), L+1)


def log_joint_paths(s: np.ndarray, a: np.ndarray, phys: Physics):
    """Full-prefix log p_0(H, m_{0:L}) for every mode path; s (L+1,), a (L,). Returns (paths, logp)."""
    L = phys.L
    paths = _paths(L)
    delta = s[1:] - phys.rho * s[:-1]
    lq_s0 = -0.5 * np.log(2 * np.pi * phys.var_s0) - s[0] ** 2 / (2 * phys.var_s0)
    lq_a = np.sum(-0.5 * np.log(2 * np.pi * phys.q_a) - a**2 / (2 * phys.q_a))
    emis = -0.5 * np.log(2 * np.pi * phys.q_w) - (delta[None, :] - phys.c * paths[:, :L] * a[None, :]) ** 2 / (2 * phys.q_w)
    switch = paths[:, 1:] != paths[:, :-1]
    with np.errstate(divide="ignore"):
        lt = np.where(switch, np.log(phys.eta) if phys.eta > 0 else -np.inf, np.log1p(-phys.eta))
    logp = lq_s0 + lq_a + np.log(0.5) + lt.sum(axis=1) + emis.sum(axis=1)
    return paths, logp


def enum_loglik(s, a, phys: Physics) -> float:
    """Full-prefix log p_0(H) (modes marginalised)."""
    _, logp = log_joint_paths(s, a, phys)
    return float(logsumexp(logp))


def enum_transition_loglik(s, a, phys: Physics) -> float:
    """Transition log-likelihood log p(s_{1:L} | s_0, a) = full-prefix log p minus the sign-invariant factors."""
    L = phys.L
    lq_s0 = -0.5 * np.log(2 * np.pi * phys.var_s0) - s[0] ** 2 / (2 * phys.var_s0)
    lq_a = np.sum(-0.5 * np.log(2 * np.pi * phys.q_a) - a**2 / (2 * phys.q_a))
    return enum_loglik(s, a, phys) - lq_s0 - lq_a


def enum_belief(s, a, phys: Physics) -> float:
    """mu(H) = E_0[m_L | H] by direct enumeration over m_{0:L}."""
    paths, logp = log_joint_paths(s, a, phys)
    w = np.exp(logp - logsumexp(logp))
    return float(np.sum(w * paths[:, -1]))


def enum_marginal_mode(s, a, phys: Physics, k: int, observed_sign_removed_at: int | None = None):
    """E_0[m_k | H] by enumeration (used for the m_{L-1} mutation check)."""
    paths, logp = log_joint_paths(s, a, phys)
    w = np.exp(logp - logsumexp(logp))
    return float(np.sum(w * paths[:, k]))


def enum_folded(s, a, phys: Physics, j: int):
    """(psi_j, p_j^+, ll_fold_j) for the folded observer O_j = (S without a_j, |a_j|), by enumeration.

    The observer density is the sign-marginalised joint: sum over the two signs of a_j (each with prior 1/2
    given |a_j|) and over all mode paths; all other recorded values are held fixed.
    """
    sign = 1.0 if a[j] >= 0 else -1.0
    absa = abs(a[j])
    lps, ws = [], []
    paths = None
    for sg in (+1.0, -1.0):
        a2 = a.copy()
        a2[j] = sg * absa
        paths, lp = log_joint_paths(s, a2, phys)
        lps.append(lp)
    lp_plus, lp_minus = lps
    both = np.logaddexp(lp_plus, lp_minus)                  # path-wise joint of (O_j, either sign)
    w = np.exp(both - logsumexp(both))
    psi = float(np.sum(w * paths[:, -1]))
    p_plus = float(np.exp(logsumexp(lp_plus) - logsumexp(both)))
    # log-likelihood of the folded observer (transition ll): subtract sign-invariant factors, normalise sign mixture
    lq_s0 = -0.5 * np.log(2 * np.pi * phys.var_s0) - s[0] ** 2 / (2 * phys.var_s0)
    lq_a = np.sum(-0.5 * np.log(2 * np.pi * phys.q_a) - a**2 / (2 * phys.q_a))
    ll_fold = float(logsumexp(both) - np.log(2.0) - lq_s0 - lq_a)
    return psi, p_plus, ll_fold


def enum_corruption_posterior(s, a, phys: Physics, chan: ChannelSpec):
    """Posterior over (theta, m_{0:L}) given S by enumeration of every hypothesis.

    Returns dict with p_theta (L+1: none, 0..L-1), mu2 = E[m_L|S], mu_theta = mu(T_theta S) by enumeration,
    var = Var(mu_H|S) computed from enumerated quantities.
    """
    L = phys.L
    weights = chan.view_weights()                           # [1-beta, q_0..q_{L-1}]
    logw = []
    muth = []
    num = []
    for v in range(L + 1):
        a2 = a.copy()
        if v >= 1:
            a2[v - 1] = -a2[v - 1]
        paths, lp = log_joint_paths(s, a2, phys)
        with np.errstate(divide="ignore"):
            lwv = np.log(weights[v])
        logw.append(lwv + logsumexp(lp))
        num.append(lwv + lp)                                # joint log-weights over paths for this theta
        wp = np.exp(lp - logsumexp(lp))
        muth.append(float(np.sum(wp * paths[:, -1])))
    logw = np.array(logw)
    ptheta = np.exp(logw - logsumexp(logw))
    allj = np.concatenate(num)
    allm = np.tile(paths[:, -1], L + 1)
    wj = np.exp(allj - logsumexp(allj))
    mu2 = float(np.sum(wj * allm))
    muth = np.array(muth)
    var = float(np.sum(ptheta * (muth - mu2) ** 2))
    return dict(p_theta=ptheta, mu2=mu2, mu_theta=muth, var=var)
