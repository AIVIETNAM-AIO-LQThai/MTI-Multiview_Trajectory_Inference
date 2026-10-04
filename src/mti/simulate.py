"""S2 simulator with documented random streams (spec sections 1, 2, 11).

Streams (SeedSequence(master, spawn_key=(cell_id, stream_id, chunk_id))):
  0 initial state, 1 modes, 2 actions, 3 transition noise, 4 corruption, 5 probe, 6 decision noise.

Inference code only ever sees a `Prefix`. Hidden modes, the logged probe and decision noise live in
separate containers inside `Sample` and are for evaluation / later self-supervision only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ChannelSpec, Physics

STREAM_S0, STREAM_MODES, STREAM_ACTIONS, STREAM_NOISE, STREAM_CORRUPT, STREAM_PROBE, STREAM_DECISION = range(7)


@dataclass(frozen=True)
class Prefix:
    """Permitted inference input: states s_0..s_L (N, L+1) and recorded actions a_0..a_{L-1} (N, L)."""
    s: np.ndarray
    a: np.ndarray

    def __post_init__(self):
        if self.s.shape[:-1] != self.a.shape[:-1] or self.s.shape[-1] != self.a.shape[-1] + 1:
            raise ValueError(f"shape mismatch s{self.s.shape} a{self.a.shape}")

    @property
    def L(self) -> int:
        return self.a.shape[-1]

    @property
    def n(self) -> int:
        return self.a.shape[0]


@dataclass(frozen=True)
class Probe:
    """Independent logged probe action a_L and its behaviour-branch transition (not an inference input)."""
    a: np.ndarray
    s_next: np.ndarray


@dataclass(frozen=True)
class Sample:
    prefix: Prefix
    modes: np.ndarray        # (N, L+1) hidden m_0..m_L in {-1,+1}; evaluation only
    probe: Probe
    decision_noise: np.ndarray  # (N,) physical noise w^dec for the evaluated decision branch


def stream_rng(master: int, cell_id: int, stream_id: int, chunk_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence(master, spawn_key=(int(cell_id), int(stream_id), int(chunk_id)))
    return np.random.default_rng(ss)


def simulate(phys: Physics, n: int, master: int, cell_id: int = 0, chunk_id: int = 0) -> Sample:
    L = phys.L
    sd_a, sd_w = np.sqrt(phys.q_a), np.sqrt(phys.q_w)
    s = np.empty((n, L + 1))
    s[:, 0] = np.sqrt(phys.var_s0) * stream_rng(master, cell_id, STREAM_S0, chunk_id).standard_normal(n)
    u = stream_rng(master, cell_id, STREAM_MODES, chunk_id).random((n, L + 1))
    m = np.empty((n, L + 1))
    m[:, 0] = np.where(u[:, 0] < 0.5, 1.0, -1.0)
    flips = u[:, 1:] < phys.eta
    for k in range(L):
        m[:, k + 1] = np.where(flips[:, k], -m[:, k], m[:, k])
    a = sd_a * stream_rng(master, cell_id, STREAM_ACTIONS, chunk_id).standard_normal((n, L))
    w = sd_w * stream_rng(master, cell_id, STREAM_NOISE, chunk_id).standard_normal((n, L))
    for k in range(L):
        s[:, k + 1] = phys.rho * s[:, k] + phys.c * m[:, k] * a[:, k] + w[:, k]
    z = stream_rng(master, cell_id, STREAM_PROBE, chunk_id).standard_normal((n, 2))
    a_pr = sd_a * z[:, 0]
    s_pr = phys.rho * s[:, L] + phys.c * m[:, L] * a_pr + sd_w * z[:, 1]
    w_dec = sd_w * stream_rng(master, cell_id, STREAM_DECISION, chunk_id).standard_normal(n)
    return Sample(Prefix(s, a), m, Probe(a_pr, s_pr), w_dec)


def sample_theta(chan: ChannelSpec, n: int, master: int, cell_id: int = 0, chunk_id: int = 0) -> np.ndarray:
    """Corruption indicator per prefix: -1 = none, j in 0..L-1 = flipped location. Exogenous stream 4."""
    u = stream_rng(master, cell_id, STREAM_CORRUPT, chunk_id).random(n)
    cum = np.cumsum(chan.view_weights())
    idx = np.searchsorted(cum, u, side="right")
    positive = np.flatnonzero(chan.view_weights() > 0)
    idx = np.minimum(idx, positive[-1])
    return idx - 1


def flip_location(prefix: Prefix, j) -> Prefix:
    """T_j: negate only the recorded action sign at location j (j scalar or per-row array)."""
    a = prefix.a.copy()
    rows = np.arange(prefix.n)
    j = np.broadcast_to(np.asarray(j), (prefix.n,))
    a[rows, j] = -a[rows, j]
    return Prefix(prefix.s.copy(), a)


def apply_channel(prefix: Prefix, theta: np.ndarray) -> Prefix:
    """S = T_theta H; theta == -1 leaves the record unchanged. States are never altered."""
    a = prefix.a.copy()
    rows = np.flatnonzero(theta >= 0)
    a[rows, theta[rows]] = -a[rows, theta[rows]]
    return Prefix(prefix.s.copy(), a)


def decision_state(phys: Physics, sample: Sample, action: np.ndarray) -> np.ndarray:
    """s_{L+1} for the evaluated decision branch using the physical mode m_L and independent noise."""
    sL = sample.prefix.s[:, -1]
    return phys.rho * sL + phys.c * sample.modes[:, -1] * action + sample.decision_noise
