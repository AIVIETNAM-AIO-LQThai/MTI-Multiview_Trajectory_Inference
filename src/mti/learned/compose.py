"""Inference-time composition of learned (or exact) clean-law components under a declared prior (log domain).

Inputs per record: mu_S = mu^(S), psi_j = mu^(O_j(S)), log r_j / log(1-r_j) (opposite / recorded sign probability from the
sign head), mu_flip_j = mu^(T_j S). Folded route G, candidate route M (density ratio l_j = r_j/(1-r_j) from the SAME sign
head, same normaliser), their average, and the unweighted masked ensemble. Feeding exact conditionals reproduces mu_2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..params import ChannelSpec


@dataclass
class Composed:
    G: np.ndarray
    M: np.ndarray
    avg: np.ndarray
    ens: np.ndarray
    C: np.ndarray            # (N,L) compatibility residuals psi - (1-r) mu - r mu_flip
    gap: np.ndarray          # G - M
    gap_identity: np.ndarray  # Z^-1 sum_j q_j C_j / (1 - r_j): must equal gap algebraically


def compose(mu_S, psi, log_r, log_1mr, mu_flip, chan: ChannelSpec) -> Composed:
    n, L = psi.shape
    with np.errstate(divide="ignore"):
        logq = np.log(chan.q)
        log_c0 = np.log(1.0 - 2.0 * chan.beta)
        log_none = np.log1p(-chan.beta)
    tG = np.concatenate([np.full((n, 1), log_c0), logq[None, :] - log_1mr], axis=1)
    logZG = logsumexp(tG, axis=1)
    wG = np.exp(tG - logZG[:, None])
    G = wG[:, 0] * mu_S + np.sum(wG[:, 1:] * psi, axis=1)
    log_ell = log_r - log_1mr
    tM = np.concatenate([np.full((n, 1), log_none), logq[None, :] + log_ell], axis=1)
    logZ = logsumexp(tM, axis=1)
    wM = np.exp(tM - logZ[:, None])
    M = wM[:, 0] * mu_S + np.sum(wM[:, 1:] * mu_flip, axis=1)
    C = psi - np.exp(log_1mr) * mu_S[:, None] - np.exp(log_r) * mu_flip
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_id = np.sum(np.where(chan.q[None, :] > 0, np.exp(logq[None, :] - log_1mr - logZ[:, None]) * C, 0.0), axis=1)
    ens = (mu_S + psi.sum(axis=1)) / (L + 1)
    return Composed(G, M, 0.5 * (G + M), ens, C, G - M, gap_id)


def signed_sign_probs(sign_logit_at_slot: np.ndarray, a_rec: np.ndarray):
    """Sign-head logits (N,L) for '+' -> (log r, log(1-r)) given the recorded signs (a >= 0 counts as '+')."""
    lp_plus = -np.logaddexp(0.0, -sign_logit_at_slot)
    lp_minus = -np.logaddexp(0.0, sign_logit_at_slot)
    pos = a_rec >= 0
    return np.where(pos, lp_minus, lp_plus), np.where(pos, lp_plus, lp_minus)
