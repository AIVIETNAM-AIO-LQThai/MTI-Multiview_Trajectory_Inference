"""Declared physics and channel parameters (spec section 1-2)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Physics:
    rho: float
    c: float
    q_a: float
    q_w: float
    lam: float
    eta: float
    L: int

    def __post_init__(self):
        if not abs(self.rho) < 1:
            raise ValueError("need |rho| < 1")
        if self.c == 0:
            raise ValueError("need c != 0")
        if not (self.q_a > 0 and self.q_w > 0):
            raise ValueError("need q_a > 0 and q_w > 0 (q_w=0 is a separate reference case)")
        if self.lam < 0:
            raise ValueError("need lambda >= 0")
        if not (0.0 <= self.eta <= 0.5):
            raise ValueError("need eta in [0, 1/2]")
        if self.L < 1:
            raise ValueError("need L >= 1")

    @property
    def D(self) -> float:
        return self.c**2 + self.lam

    @property
    def var_s0(self) -> float:
        """Stationary variance of the state; also Var(s_k) for every k under the S2 initial law."""
        return (self.c**2 * self.q_a + self.q_w) / (1.0 - self.rho**2)

    def with_(self, **kw) -> "Physics":
        return replace(self, **kw)


@dataclass(frozen=True)
class ChannelSpec:
    """Single-flip action-sign channel: Pr(none)=1-beta, Pr(j)=q_j=beta*pi_j."""
    beta: float
    pi: tuple

    def __post_init__(self):
        pi = np.asarray(self.pi, dtype=float)
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("need beta in [0, 1)")
        if np.any(pi < 0) or abs(pi.sum() - 1.0) > 1e-12:
            raise ValueError("pi must be a probability vector")

    @staticmethod
    def uniform(L: int, beta: float) -> "ChannelSpec":
        return ChannelSpec(beta=beta, pi=tuple([1.0 / L] * L))

    @staticmethod
    def point_mass(L: int, beta: float, j: int) -> "ChannelSpec":
        pi = [0.0] * L
        pi[j] = 1.0
        return ChannelSpec(beta=beta, pi=tuple(pi))

    @property
    def pi_arr(self) -> np.ndarray:
        return np.asarray(self.pi, dtype=float)

    @property
    def q(self) -> np.ndarray:
        return self.beta * self.pi_arr

    @property
    def L(self) -> int:
        return len(self.pi)

    def view_weights(self) -> np.ndarray:
        """Length L+1: [1-beta, q_0..q_{L-1}] (view 0 = no corruption)."""
        return np.concatenate([[1.0 - self.beta], self.q])
