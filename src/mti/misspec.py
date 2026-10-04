"""Channel misspecification (E2): evaluation of estimators that were given q_assumed under the TRUE law q_true.

Everything is evaluated on independent clean test prefixes; for each prefix the L+1 corruption views S_v = T_v H are averaged with the
weights of q_true (Rao-Blackwellised). The prefix is the sampling unit. The exact clean-law queries (ll, mu of every record and of each
repair T_jS) do not depend on any prior, so they are computed once and composed under any channel.

Estimand (protocol e2_protocol.md section 5):
    E_true(mu_hat; q_a) = E_{q_true}[ kappa(S) (mu_hat(S; q_a) - mu_2(S; q_true))^2 ].
Decomposition, with m2a = mu_2(S; q_a):  E_true = Lrn + Chn + Crs,
    Lrn = E[k (mu_hat - m2a)^2],  Chn = E[k (m2a - m2t)^2],  Crs = 2 E[k (mu_hat - m2a)(m2a - m2t)]   (exact per sample).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import base_cost
from .inference import candidate_from_queries, flip_batch
from .params import ChannelSpec
from .simulate import Prefix

NO_CHANNEL_BETA = 0.0


def no_channel(L: int) -> ChannelSpec:
    """beta = 0: composition with it returns the naive belief mu(S) exactly (p_none = 1)."""
    return ChannelSpec.uniform(L, NO_CHANNEL_BETA)


def odds(beta: float) -> float:
    return beta / (1.0 - beta)


def predicted_safe(beta_a: float, beta_t: float, same_pi: bool) -> bool:
    """Protocol section 3 (i)/(ii): with the location prior correct, composed regret <= naive regret on every record
    whenever beta_a <= beta_t or odds(beta_a) <= 2 odds(beta_t)."""
    return bool(same_pi and (beta_a <= beta_t or odds(beta_a) <= 2.0 * odds(beta_t) + 1e-12))


class ExactRecords:
    """Exact clean-law queries on every record of a TestSet (any physics eta may be passed for a fitted-HMM variant)."""

    def __init__(self, ts, phys=None):
        self.ts = ts
        self.phys = ts.phys if phys is None else phys
        self.ll, self.mu = flip_batch(Prefix(ts.s_rec, ts.a_rec), self.phys)         # (R, L+1) each
        self.n, self.V = ts.n, ts.V

    def mu2(self, chan: ChannelSpec) -> np.ndarray:
        """Composed belief mu_2(S; chan) on every record, shape (n, V)."""
        return candidate_from_queries(self.ll, self.mu, chan).mu2.reshape(self.n, self.V)

    @property
    def naive(self) -> np.ndarray:
        return self.mu[:, 0].reshape(self.n, self.V)


def _mean_se(x: np.ndarray):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


class TrueLaw:
    """Scores beliefs under q_true on a TestSet. `exact` must use the true clean law (it defines mu_2(S; q_true))."""

    def __init__(self, ts, exact: ExactRecords):
        self.ts, self.exact = ts, exact
        self.n, self.V, self.L = ts.n, ts.V, ts.L
        self.kap = ts.kap
        self.K = ts.kap[:, None]
        self.muH = exact.mu[:, 0].reshape(self.n, self.V)[:, 0]                      # clean belief of the prefix (view 0 record is H)
        self.base = base_cost(ts.phys, self.H_sL())
        self.mL = ts.mL
        self.sw = ts.sw
        self._ref = {}

    def H_sL(self):
        return self.ts.H.s[:, -1]

    def reference(self, chan_t: ChannelSpec) -> np.ndarray:
        key = (chan_t.beta, chan_t.pi)
        if key not in self._ref:
            self._ref[key] = self.exact.mu2(chan_t)
        return self._ref[key]

    def voi_clean(self) -> float:
        return float(np.mean(self.kap * self.muH**2))

    def score(self, muhat: np.ndarray, chan_t: ChannelSpec, muhat_naive: np.ndarray, mu2a: np.ndarray | None = None):
        """Returns (scalars, per-prefix regret array y). Hidden-mode strata use realised costs with m_L."""
        w, pi, K = chan_t.view_weights(), chan_t.pi_arr, self.K
        m2t = self.reference(chan_t)
        y = (K * (muhat - m2t) ** 2) @ w
        cost = self.base - self.kap * self.muH**2 + (K * (muhat - self.muH[:, None]) ** 2) @ w
        mL = self.mL[:, None]
        d = K * ((muhat - mL) ** 2 - (muhat_naive - mL) ** 2)                         # >0: costlier than the same pipeline's naive output
        yD, yDn, yDc = d @ w, d[:, 0], d[:, 1:] @ pi
        r = {}
        for k, v in (("E", y), ("cost", cost), ("D", yD), ("D_none", yDn), ("D_cor", yDc)):
            r[k], r[k + "_se"] = _mean_se(v)
        f = self.sw
        r["D_switch"], r["D_switch_se"] = _mean_se(yD[f]) if f.sum() > 30 else (float("nan"), float("nan"))
        r["switch_frac"] = float(f.mean())
        if mu2a is not None:
            lrn = (K * (muhat - mu2a) ** 2) @ w
            chn = (K * (mu2a - m2t) ** 2) @ w
            crs = 2.0 * (K * (muhat - mu2a) * (mu2a - m2t)) @ w
            r["learn"], r["channel"], r["cross"] = float(lrn.mean()), float(chn.mean()), float(crs.mean())
            r["decomp_max_abs_err"] = float(np.abs(lrn + chn + crs - y).max())
        return r, y

    def paired(self, y_a: np.ndarray, y_b: np.ndarray):
        """Benefit b = E(b) - E(a) (positive when arm a has lower regret), prefix-level paired mean and standard error."""
        d = y_b - y_a
        return _mean_se(d)


def record_violations(truth: TrueLaw, chan_t: ChannelSpec, chan_a: ChannelSpec, tol: float = 1e-12) -> int:
    """Number of records (any view) where exact composition with chan_a has larger regret than ignoring the channel."""
    m2t = truth.reference(chan_t)
    m2a = truth.exact.mu2(chan_a)
    ra = (truth.K * (m2a - m2t) ** 2)
    rn = (truth.K * (truth.exact.naive - m2t) ** 2)
    return int(np.sum(ra > rn + tol))


def convex_form(exact: ExactRecords, pi, beta: float):
    """Protocol section 3: mu_2(S; beta, pi) = (1 - w) mu(S) + w mbar(S),  w = o A/(1 + o A),  A = sum_j pi_j l_j,
    mbar = sum_j pi_j l_j mu(T_j S)/A. Returns (mu_S, mbar, A, w) on every record (R,)."""
    ell = np.exp(exact.ll[:, 1:] - exact.ll[:, [0]])
    pi = np.asarray(pi)
    A = ell @ pi
    mbar = (ell * exact.mu[:, 1:]) @ pi / A
    o = odds(beta)
    return exact.mu[:, 0], mbar, A, o * A / (1.0 + o * A)
