"""Token features for the sequence models.

Per transition k < L the token carries (s_k, a~_k, delta_k, asinh(z_k), mask_k) with
  delta_k = s_{k+1} - rho s_k,  a~_k = a_k (visible) or |a_k| (folded slot: the sign is removed, magnitude kept),
  z_k = c a~_k delta_k / q_w  (for a visible token this is half the log-likelihood ratio of the two modes;
  for a folded token it is sign-free evidence available to the sign head).
The last token carries s_L only. All quantities are functions of the prefix and the declared physics.
"""
from __future__ import annotations

import numpy as np
import torch

from ..params import Physics

FEAT_DIM = 6


def to_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def make_features(s: torch.Tensor, a: torch.Tensor, mask: torch.Tensor, phys: Physics) -> torch.Tensor:
    """s (B,L+1), a (B,L), mask (B,L) bool (True where the recorded sign is removed) -> (B, L+1, FEAT_DIM)."""
    sig_s = float(np.sqrt(phys.var_s0))
    sig_a = float(np.sqrt(phys.q_a))
    sig_d = float(np.sqrt(phys.c**2 * phys.q_a + phys.q_w))
    delta = s[:, 1:] - phys.rho * s[:, :-1]
    at = torch.where(mask, a.abs(), a)
    z = phys.c * at * delta / phys.q_w
    zero = torch.zeros_like(delta)
    f = torch.stack([s[:, :-1] / sig_s, at / sig_a, delta / sig_d, torch.asinh(z), mask.to(s.dtype), zero], dim=-1)
    last = torch.zeros(s.shape[0], 1, FEAT_DIM, dtype=s.dtype)
    last[:, 0, 0] = s[:, -1] / sig_s
    last[:, 0, 5] = 1.0
    return torch.cat([f, last], dim=1)


def onehot_mask(j: torch.Tensor, L: int) -> torch.Tensor:
    """(B,) integer slots -> (B,L) bool mask with exactly one True per row (j < 0 -> no mask)."""
    m = torch.zeros(j.shape[0], L, dtype=torch.bool)
    rows = torch.nonzero(j >= 0).squeeze(-1)
    m[rows, j[rows]] = True
    return m
