"""Training / validation data for the learned arms: clean prefixes with their logged probe. Hidden modes are dropped here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..params import Physics
from ..simulate import simulate
from .features import to_tensor

# Masters for the learned pilot (disjoint from oracle masters 1001-5001 and test masters 0-999).
MASTER_TRAIN, MASTER_VAL, MASTER_TEST, MASTER_CAL = 7001, 7002, 8001, 7003


@dataclass
class ProbeSet:
    """Clean prefixes + independent logged probe. No modes, no corruption labels."""
    s: torch.Tensor       # (n, L+1)
    a: torch.Tensor       # (n, L)
    a_pr: torch.Tensor    # (n,)
    d_pr: torch.Tensor    # (n,)  delta_L = s_{L+1} - rho s_L (training target ingredient)

    @property
    def n(self) -> int:
        return self.s.shape[0]

    def take(self, lo: int, hi: int) -> "ProbeSet":
        return ProbeSet(self.s[lo:hi], self.a[lo:hi], self.a_pr[lo:hi], self.d_pr[lo:hi])


def make_probe_set(phys: Physics, n: int, master: int, cell_id: int, chunk_id: int) -> ProbeSet:
    smp = simulate(phys, n, master, cell_id, chunk_id)
    sL = smp.prefix.s[:, -1]
    return ProbeSet(to_tensor(smp.prefix.s), to_tensor(smp.prefix.a), to_tensor(smp.probe.a),
                    to_tensor(smp.probe.s_next - phys.rho * sL))


def probe_norm(phys: Physics) -> float:
    return phys.q_w + phys.c**2 * phys.q_a


def probe_loss(belief: torch.Tensor, d_pr: torch.Tensor, a_pr: torch.Tensor, phys: Physics) -> torch.Tensor:
    """Self-supervised target (delta_L - c a_L h(I))^2, normalised; minimiser is E[m_L | I] when I excludes the probe."""
    return ((d_pr - phys.c * a_pr * belief) ** 2).mean() / probe_norm(phys)


def corrupt(a: torch.Tensor, probs: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply the single-flip channel: probs (L+1,) or (B,L+1) = [1-beta, q_0..q_{L-1}]; returns corrupted actions."""
    B = a.shape[0]
    p = probs.expand(B, -1) if probs.dim() == 1 else probs
    idx = torch.multinomial(p, 1, generator=gen).squeeze(-1)          # 0 = none, j+1 = location j
    rows = torch.nonzero(idx > 0).squeeze(-1)
    out = a.clone()
    out[rows, idx[rows] - 1] = -out[rows, idx[rows] - 1]
    return out


def channel_probs(chan) -> torch.Tensor:
    return torch.as_tensor(chan.view_weights(), dtype=torch.float32)


def sample_family(family: str, B: int, L: int, gen: torch.Generator):
    """Prior family for A6p training. Returns (view_probs (B,L+1), cond (B,L) = q).

    narrow: beta ~ U[0, 0.3], pi uniform.  broad: beta ~ U[0, 0.5], pi ~ Dirichlet(1_L).
    """
    if family == "narrow":
        beta = 0.3 * torch.rand(B, 1, generator=gen)
        pi = torch.full((B, L), 1.0 / L)
    elif family == "broad":
        beta = 0.5 * torch.rand(B, 1, generator=gen)
        e = -torch.log(torch.rand(B, L, generator=gen).clamp_min(1e-12))   # Exp(1) -> normalised = Dirichlet(1)
        pi = e / e.sum(dim=1, keepdim=True)
    else:
        raise ValueError(family)
    q = beta * pi
    return torch.cat([1.0 - beta, q], dim=1), q
