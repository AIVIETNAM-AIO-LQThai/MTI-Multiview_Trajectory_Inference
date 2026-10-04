"""Exact clean-law, candidate-route and folded-route inference (spec sections 3-6).

Everything is float64 and log-domain. Public functions accept only a `Prefix`.
Mode index convention in the last axis: 0 -> m=+1, 1 -> m=-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .params import ChannelSpec, Physics
from .simulate import Prefix

LOG2 = float(np.log(2.0))
_MAX_ELEMS = 4_000_000  # cap on elements of the (B, queries, L, 2) emission tensor per internal batch


def _check_prefix(p) -> None:
    if not isinstance(p, Prefix):
        raise TypeError(f"inference accepts only Prefix, got {type(p).__name__}")


def _log_trans(eta: float):
    """(log stay, log switch); eta=0 gives switch=-inf, handled by logaddexp."""
    ls = float(np.log1p(-eta))
    with np.errstate(divide="ignore"):
        lw = float(np.log(eta)) if eta > 0 else -np.inf
    return ls, lw


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _log_q(chan: ChannelSpec) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(chan.q)


def log_emissions(s, a, phys: Physics) -> np.ndarray:
    """log N(delta_k; c m a_k, q_w) for m=+1,-1; shape (..., L, 2)."""
    delta = s[..., 1:] - phys.rho * s[..., :-1]
    ca = phys.c * a
    const = -0.5 * np.log(2.0 * np.pi * phys.q_w)
    lp = const - (delta - ca) ** 2 / (2.0 * phys.q_w)
    lm = const - (delta + ca) ** 2 / (2.0 * phys.q_w)
    return np.stack([lp, lm], axis=-1)


def forward(logE: np.ndarray, eta: float, keep_pre: bool = False):
    """Log-domain forward pass.

    Returns (pre, tilde): pre[..., k, m] = log P(m_k=m, e_0..e_{k-1}) (only if keep_pre),
    tilde[..., m] = log P(m_{L-1}=m, e_0..e_{L-1}).
    """
    ls, lw = _log_trans(eta)
    L = logE.shape[-2]
    alpha = np.full(logE.shape[:-2] + (2,), -LOG2)
    pre = np.empty_like(logE) if keep_pre else None
    for k in range(L):
        if keep_pre:
            pre[..., k, :] = alpha
        at = alpha + logE[..., k, :]
        if k == L - 1:
            return pre, at
        alpha = np.stack(
            [np.logaddexp(at[..., 0] + ls, at[..., 1] + lw), np.logaddexp(at[..., 1] + ls, at[..., 0] + lw)],
            axis=-1,
        )


def backward(logE: np.ndarray, eta: float) -> np.ndarray:
    """beta[..., k, m] = log p(e_{k+1..L-1} | m_k=m), beta[..., L-1, :] = 0."""
    ls, lw = _log_trans(eta)
    L = logE.shape[-2]
    beta = np.zeros_like(logE)
    for k in range(L - 2, -1, -1):
        nxt = logE[..., k + 1, :] + beta[..., k + 1, :]
        beta[..., k, 0] = np.logaddexp(ls + nxt[..., 0], lw + nxt[..., 1])
        beta[..., k, 1] = np.logaddexp(ls + nxt[..., 1], lw + nxt[..., 0])
    return beta


def belief_from_tilde(tilde: np.ndarray, eta: float):
    """(mu, ll): clean belief E_0[m_L|.] including the m_{L-1}->m_L step, and transition log-likelihood."""
    lam = tilde[..., 0] - tilde[..., 1]
    mu = (1.0 - 2.0 * eta) * np.tanh(lam / 2.0)
    ll = np.logaddexp(tilde[..., 0], tilde[..., 1])
    return mu, ll


def clean_belief(prefix: Prefix, phys: Physics):
    """mu(H) = E_0[m_L | H] and transition log-likelihood ll(H)."""
    _check_prefix(prefix)
    logE = log_emissions(prefix.s, prefix.a, phys)
    _, tilde = forward(logE, phys.eta)
    return belief_from_tilde(tilde, phys.eta)


def flip_batch(prefix: Prefix, phys: Physics):
    """Clean ll and mu for the record S itself (index 0) and each repair T_kS (index k+1).

    Flipping a_k swaps the two emission columns at k, so the base emission tensor is reused.
    Reference implementation: one full forward pass per query.
    """
    _check_prefix(prefix)
    L = phys.L
    n = prefix.n
    step = max(1, _MAX_ELEMS // ((L + 1) * L * 2))
    ll = np.empty((n, L + 1))
    mu = np.empty((n, L + 1))
    idx = np.arange(L)
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        logE = log_emissions(prefix.s[lo:hi], prefix.a[lo:hi], phys)
        q = np.repeat(logE[:, None, :, :], L + 1, axis=1)
        q[:, idx + 1, idx, :] = logE[:, idx, ::-1]
        _, tilde = forward(q, phys.eta)
        mu[lo:hi], ll[lo:hi] = belief_from_tilde(tilde, phys.eta)
    return ll, mu


@dataclass
class CandidateResult:
    mu: np.ndarray        # (N,)   naive belief mu(S)
    mu_flip: np.ndarray   # (N,L)  mu(T_jS)
    ll: np.ndarray        # (N,)   transition log-likelihood of S
    ll_flip: np.ndarray   # (N,L)  of T_jS
    log_ell: np.ndarray   # (N,L)
    log_Z: np.ndarray     # (N,)
    p_none: np.ndarray    # (N,)
    p: np.ndarray         # (N,L)
    mu2: np.ndarray       # (N,)   aware belief
    var_muH: np.ndarray   # (N,)   Var(mu_H | S)
    log_post: np.ndarray  # (N,L+1) exact log posterior over [none, 0..L-1]


def candidate_route(prefix: Prefix, phys: Physics, chan: ChannelSpec) -> CandidateResult:
    ll_all, mu_all = flip_batch(prefix, phys)
    return candidate_from_queries(ll_all, mu_all, chan)


def candidate_from_queries(ll_all, mu_all, chan: ChannelSpec) -> CandidateResult:
    ll_S, ll_flip = ll_all[:, 0], ll_all[:, 1:]
    mu_S, mu_flip = mu_all[:, 0], mu_all[:, 1:]
    log_ell = ll_flip - ll_S[:, None]
    with np.errstate(divide="ignore"):
        log_none = np.log1p(-chan.beta)
    terms = np.concatenate([np.full((ll_S.shape[0], 1), log_none), _log_q(chan)[None, :] + log_ell], axis=1)
    log_Z = logsumexp(terms, axis=1)
    log_post = terms - log_Z[:, None]
    p_all = np.exp(log_post)
    p_none, p = p_all[:, 0], p_all[:, 1:]
    mu2 = p_none * mu_S + np.sum(p * mu_flip, axis=1)
    # centred form of Var(mu_H|S); equals p_none mu^2 + sum p mu_flip^2 - mu2^2 but is non-negative by construction
    var = p_none * (mu_S - mu2) ** 2 + np.sum(p * (mu_flip - mu2[:, None]) ** 2, axis=1)
    return CandidateResult(mu_S, mu_flip, ll_S, ll_flip, log_ell, log_Z, p_none, p, mu2, var, log_post)


@dataclass
class FoldedResult:
    mu: np.ndarray        # (N,)  mu(S) (clean belief of the record, a folded-route input)
    psi: np.ndarray       # (N,L) E_0[m_L | O_j]
    log_r: np.ndarray     # (N,L) log Pr_0(opposite sign | O_j)
    log_1mr: np.ndarray   # (N,L) log Pr_0(recorded sign | O_j)
    log_ell: np.ndarray   # (N,L) log r - log(1-r)
    ll_fold: np.ndarray   # (N,L) log-likelihood of the sign-marginalised folded query O_j(S)
    log_Z: np.ndarray     # (N,)  folded normaliser Z_G
    w0: np.ndarray        # (N,)  weight on mu(S)
    w: np.ndarray         # (N,L) weight on psi_j
    G: np.ndarray         # (N,)  folded aware belief


def folded_route(prefix: Prefix, phys: Physics, chan: ChannelSpec) -> FoldedResult:
    """Folded route computed by marginalising the clean model, not by the candidate identity.

    A3: with the sign of a_j removed (|a_j| kept) the emission at j is
        e_bar_j = 0.5[N(delta_j; c|a_j|, q_w) + N(delta_j; -c|a_j|, q_w)],
    identical for both modes, so psi_j is the filter with that factor made neutral, and
    P(m_j | O_j) is the smoother marginal; sign probabilities follow from Bayes given m_j.
    """
    _check_prefix(prefix)
    L, eta = phys.L, phys.eta
    logE = log_emissions(prefix.s, prefix.a, phys)                      # (N,L,2)
    pre, tilde = forward(logE, eta, keep_pre=True)
    beta = backward(logE, eta)
    mu_S, _ = belief_from_tilde(tilde, eta)

    # smoother marginal at m_j given the observer O_j (the emission at j is neutral, so it cancels)
    lp = pre + beta
    lp = lp - logsumexp(lp, axis=-1, keepdims=True)                    # (N,L,2) log P(m_j=m | O_j)
    delta = prefix.s[:, 1:] - phys.rho * prefix.s[:, :-1]
    x = 2.0 * phys.c * np.abs(prefix.a) * delta / phys.q_w             # log odds of '+' sign given m=+1
    log_pplus = np.logaddexp(lp[..., 0] + log_sigmoid(x), lp[..., 1] + log_sigmoid(-x))
    log_pminus = np.logaddexp(lp[..., 0] + log_sigmoid(-x), lp[..., 1] + log_sigmoid(x))
    pos = prefix.a >= 0
    log_1mr = np.where(pos, log_pplus, log_pminus)
    log_r = np.where(pos, log_pminus, log_pplus)

    # folded queries: neutral emission at j
    lbar = np.logaddexp(logE[..., 0], logE[..., 1]) - LOG2              # (N,L)
    n = prefix.n
    step = max(1, _MAX_ELEMS // (L * L * 2))
    psi = np.empty((n, L))
    ll_fold = np.empty((n, L))
    idx = np.arange(L)
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        q = np.repeat(logE[lo:hi, None, :, :], L, axis=1)
        q[:, idx, idx, :] = lbar[lo:hi, :, None]
        _, t = forward(q, eta)
        psi[lo:hi], ll_fold[lo:hi] = belief_from_tilde(t, eta)

    with np.errstate(divide="ignore"):
        log_c0 = np.log(1.0 - 2.0 * chan.beta)
    terms = np.concatenate([np.full((n, 1), log_c0), _log_q(chan)[None, :] - log_1mr], axis=1)
    log_Z = logsumexp(terms, axis=1)
    wall = np.exp(terms - log_Z[:, None])
    w0, w = wall[:, 0], wall[:, 1:]
    G = w0 * mu_S + np.sum(w * psi, axis=1)
    return FoldedResult(mu_S, psi, log_r, log_1mr, log_r - log_1mr, ll_fold, log_Z, w0, w, G)


def compatibility_residual(fr: FoldedResult, cr: CandidateResult) -> np.ndarray:
    """C_j = psi_j - (1-r_j) mu(S) - r_j mu(T_j S); zero for exact conditionals."""
    return fr.psi - np.exp(fr.log_1mr) * cr.mu[:, None] - np.exp(fr.log_r) * cr.mu_flip


def folded_observer(prefix: Prefix, j: int):
    """O_j(S) as data: states, the action matrix with a_j zeroed (sign and magnitude removed), and |a_j|."""
    _check_prefix(prefix)
    a = prefix.a.copy()
    mag = np.abs(a[:, j]).copy()
    a[:, j] = 0.0
    return prefix.s.copy(), a, mag
