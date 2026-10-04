"""Test-set evaluation against the exact oracle (evaluation only: exact mu_2 is never a training signal).

The test prefixes are independent clean prefixes; for each, all L+1 corruption views S_v = T_v H are evaluated and averaged with
the prior weights (Rao-Blackwellised), exactly as in the oracle experiment. Network outputs on the views are prior-independent,
so clean-trained arms are queried once per model and composed under each prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..decision import kappa
from ..experiment import eval_views
from ..inference import candidate_from_queries, candidate_route, folded_route
from ..params import ChannelSpec, Physics
from ..simulate import Prefix, simulate
from .compose import compose, signed_sign_probs
from .data import MASTER_TEST
from .features import make_features, onehot_mask, to_tensor
from .train import mixture_logpdf, _tokens


@torch.no_grad()
def query_full(model, s_rec, a_rec, phys: Physics, cond=None, chunk: int = 20000) -> np.ndarray:
    """Belief of the full view for each record."""
    R, L = a_rec.shape
    out = np.empty(R)
    nomask = torch.zeros(1, L, dtype=torch.bool)
    for lo in range(0, R, chunk):
        hi = min(R, lo + chunk)
        s, a = to_tensor(s_rec[lo:hi]), to_tensor(a_rec[lo:hi])
        c = None if cond is None else cond[lo:hi]
        out[lo:hi] = model(make_features(s, a, nomask.expand(hi - lo, -1), phys), c)[0].double().numpy()
    return out


@torch.no_grad()
def query_F(model, s_rec, a_rec, phys: Physics, chunk: int = 2000):
    """All queries of the shared fixed-view model per record: full, folded O_j (belief + sign logit), flipped full T_j."""
    R, L = a_rec.shape
    Q = 2 * L + 1
    mu = np.empty(R)
    psi = np.empty((R, L))
    sl = np.empty((R, L))
    muf = np.empty((R, L))
    j = torch.arange(L)
    for lo in range(0, R, chunk):
        hi = min(R, lo + chunk)
        B = hi - lo
        s, a = to_tensor(s_rec[lo:hi]), to_tensor(a_rec[lo:hi])
        S_all = s.repeat(Q, 1)
        A_all = a.repeat(Q, 1)
        mask = torch.zeros(Q, B, L, dtype=torch.bool)
        mask[1 + j, :, j] = True                                      # queries 1..L: folded slot j
        A3 = A_all.view(Q, B, L)
        A3[L + 1 + j, :, j] = -A3[L + 1 + j, :, j]                    # queries L+1..2L: flipped sign at j
        bel, slg = model(make_features(S_all, A3.reshape(Q * B, L), mask.reshape(Q * B, L), phys))
        bel = bel.double().view(Q, B).numpy()
        slg = slg.double().view(Q, B, L + 1).numpy()
        mu[lo:hi] = bel[0]
        psi[lo:hi] = bel[1:L + 1].T
        sl[lo:hi] = np.stack([slg[1 + jj, :, jj] for jj in range(L)], axis=1)
        muf[lo:hi] = bel[L + 1:].T
    return mu, psi, sl, muf


@torch.no_grad()
def query_density(model, s_rec, a_rec, phys: Physics, chunk: int = 4000):
    """Causal density scorer: log-likelihood and belief of the record (index 0) and of each flipped repair T_k S (index k+1)."""
    R, L = a_rec.shape
    ll = np.empty((R, L + 1))
    mu = np.empty((R, L + 1))
    j = torch.arange(L)
    for lo in range(0, R, chunk):
        hi = min(R, lo + chunk)
        B = hi - lo
        s, a = to_tensor(s_rec[lo:hi]), to_tensor(a_rec[lo:hi])
        Q = L + 1
        S_all = s.repeat(Q, 1)
        A3 = a.repeat(Q, 1).view(Q, B, L)
        A3[1 + j, :, j] = -A3[1 + j, :, j]
        A_all = A3.reshape(Q * B, L)
        logits = model(_tokens(S_all, A_all, phys))                  # (QB, L+1)
        delta = S_all[:, 1:] - phys.rho * S_all[:, :-1]
        lp = mixture_logpdf(delta, A_all, logits[:, :L], phys).sum(dim=1)
        ll[lo:hi] = lp.double().view(Q, B).numpy().T
        mu[lo:hi] = torch.tanh(logits[:, L] / 2).double().view(Q, B).numpy().T
    return ll, mu


@torch.no_grad()
def query_density_belief(model, s_rec, a_rec, phys: Physics, cond=None, chunk: int = 20000) -> np.ndarray:
    """Direct belief read-out of a density-direct model: 2 w_L - 1 = tanh(logit_L / 2) on each record (one forward pass, no composition)."""
    R = a_rec.shape[0]
    out = np.empty(R)
    for lo in range(0, R, chunk):
        hi = min(R, lo + chunk)
        logits = model(_tokens(to_tensor(s_rec[lo:hi]), to_tensor(a_rec[lo:hi]), phys), None if cond is None else cond[lo:hi])
        out[lo:hi] = torch.tanh(logits[:, -1] / 2).double().numpy()
    return out


def mean_se(x: np.ndarray):
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


class TestSet:
    """Independent clean test prefixes with exact oracle references under any number of priors."""

    __test__ = False  # not a pytest class

    def __init__(self, phys: Physics, n: int, cell_id: int, chunk_id: int = 0, master: int = MASTER_TEST):
        self.phys, self.n, self.L, self.V = phys, n, phys.L, phys.L + 1
        smp = simulate(phys, n, master, cell_id, chunk_id)
        self.H = smp.prefix
        self.mL = smp.modes[:, -1]
        self.sw = smp.modes[:, -2] != smp.modes[:, -3] if phys.L >= 3 else np.zeros(n, bool)   # E_{L-1}: m_{L-2} -> m_{L-1}
        self.kap = kappa(phys, self.H.s[:, -1])
        s = np.repeat(self.H.s[:, None, :], self.V, axis=1)
        a = np.repeat(self.H.a[:, None, :], self.V, axis=1)
        jj = np.arange(self.L)
        a[:, jj + 1, jj] *= -1.0
        self.s_rec, self.a_rec = s.reshape(n * self.V, -1), a.reshape(n * self.V, -1)
        self.sL_abs = np.abs(self.H.s[:, -1])
        self.oracle: dict = {}
        self.chans: dict = {}

    def add_prior(self, name: str, chan: ChannelSpec):
        rec, cr = eval_views(self.H, self.phys, chan)
        n, V = self.n, self.V
        self.chans[name] = chan
        self.oracle[name] = dict(mu2=cr.mu2.reshape(n, V), mu_t=cr.mu.reshape(n, V), w=chan.view_weights(), pi=chan.pi_arr)

    def add_exact_queries(self):
        """Exact (prior-independent) conditionals on every record, for query-level diagnostics: mu(S), mu(T_jS), psi_j, log r_j, log(1-r_j)."""
        rec = Prefix(self.s_rec, self.a_rec)
        chan = ChannelSpec.uniform(self.L, 0.2)
        cr, fr = candidate_route(rec, self.phys, chan), folded_route(rec, self.phys, chan)
        self.ex = dict(mu=cr.mu, mu_flip=cr.mu_flip, psi=fr.psi, log_r=fr.log_r, log_1mr=fr.log_1mr)

    def query_diagnostics(self, out) -> dict:
        """Errors of the shared model's queries against the exact conditionals, by contamination count (unweighted over the L+1 views).

        Contamination count = number of action signs differing from the clean prefix H in the query:
        flipped repair T_jS: 0 if j is the attacked location else (1 if S is corrupted, else 0) + 1; folded O_j(S): 1 if S is corrupted elsewhere, else 0.
        """
        mu, psi, sl, muf = out
        L, V, n = self.L, self.V, self.n
        v = np.arange(n * V) % V
        theta = v - 1
        j = np.arange(L)[None, :]
        attacked = (v >= 1)[:, None]
        at_true = attacked & (theta[:, None] == j)
        cnt_T = np.where(at_true, 0, attacked.astype(int) + 1)
        cnt_F = (attacked & ~at_true).astype(int)
        kap = np.repeat(self.kap, V)[:, None]
        log_r, log_1mr = signed_sign_probs(sl, self.a_rec)
        ex = self.ex
        kl = np.exp(ex["log_r"]) * (ex["log_r"] - log_r) + np.exp(ex["log_1mr"]) * (ex["log_1mr"] - log_1mr)
        mae = np.abs(np.exp(log_r) - np.exp(ex["log_r"]))
        d = {}
        for c in (0, 1, 2):
            m = cnt_T == c
            d[f"flip_kmse_c{c}"] = float(np.mean(((muf - ex["mu_flip"]) ** 2 * kap)[m])) if m.any() else None
        for c in (0, 1):
            m = cnt_F == c
            d[f"folded_psi_kmse_c{c}"] = float(np.mean(((psi - ex["psi"]) ** 2 * kap)[m])) if m.any() else None
            d[f"sign_kl_c{c}"] = float(np.mean(kl[m])) if m.any() else None
            d[f"sign_mae_c{c}"] = float(np.mean(mae[m])) if m.any() else None
        full = kap[:, 0] * (mu - ex["mu"]) ** 2
        d["full_kmse_clean_records"] = float(full[v == 0].mean())
        d["full_kmse_corrupted_records"] = float(full[v >= 1].mean())
        return d

    def gap_diagnostics(self, prior: str, comp) -> dict:
        """Route-gap statistics under `prior` (RB weights): signed gap G-M, kappa-weighted squared gap, |C_j|, identity error."""
        o = self.oracle[prior]
        w, pi, n, V = o["w"], o["pi"], self.n, self.V
        gap = comp.gap.reshape(n, V)
        K = self.kap[:, None]
        absC = np.abs(comp.C).mean(axis=1).reshape(n, V)
        return dict(gap_signed=float((gap @ w).mean()), gap_signed_none=float(gap[:, 0].mean()), gap_signed_cor=float((gap[:, 1:] @ pi).mean()),
                    gap_ksq=float(((K * gap**2) @ w).mean()), gap_abs=float((np.abs(gap) @ w).mean()),
                    C_abs=float((absC @ w).mean()), gap_identity_max_err=float(np.abs(comp.gap - comp.gap_identity).max()))

    def summarize(self, prior: str, muhat: np.ndarray) -> dict:
        """Endpoint E = E[kappa (muhat - mu_2)^2] (RB over corruption views) and realised-cost diagnostics vs the exact naive action."""
        o = self.oracle[prior]
        w, pi, K = o["w"], o["pi"], self.kap[:, None]
        e = K * (muhat - o["mu2"]) ** 2
        d = K * ((muhat - self.mL[:, None]) ** 2 - (o["mu_t"] - self.mL[:, None]) ** 2)
        y, yn, yc = e @ w, e[:, 0], e[:, 1:] @ pi
        dy, dn, dc = d @ w, d[:, 0], d[:, 1:] @ pi
        r = {}
        for k, v in (("E", y), ("E_none", yn), ("E_cor", yc), ("D", dy), ("D_none", dn), ("D_cor", dc)):
            r[k], r[k + "_se"] = mean_se(v)
        f = self.sw
        r["D_switch"], r["D_switch_se"] = mean_se(dy[f]) if f.sum() > 30 else (float("nan"), float("nan"))
        r["E_switch"], r["E_switch_se"] = mean_se(y[f]) if f.sum() > 30 else (float("nan"), float("nan"))
        return r

    # --- reference rows
    def exact_rows(self, prior: str) -> dict:
        o = self.oracle[prior]
        return {"A0-oracle": self.summarize(prior, o["mu2"]), "exact-naive": self.summarize(prior, o["mu_t"])}


def compose_F(out, chan: ChannelSpec, a_rec: np.ndarray, n: int, V: int) -> dict:
    """Compose the shared-model outputs under `chan`; returns {arm: (n, V) beliefs} plus gap/identity diagnostics."""
    mu, psi, sl, muf = out
    log_r, log_1mr = signed_sign_probs(sl, a_rec)
    c = compose(mu, psi, log_r, log_1mr, muf, chan)
    rs = lambda x: x.reshape(n, V)
    return {"A2-G": rs(c.G), "A2-M": rs(c.M), "A2-avg": rs(c.avg), "A3-ens": rs(c.ens)}, c


def candidate_from_density(ll, mu, chan: ChannelSpec, n: int, V: int) -> np.ndarray:
    return candidate_from_queries(ll, mu, chan).mu2.reshape(n, V)


def hmm_candidate(ts: TestSet, eta_hat: float, chan: ChannelSpec) -> tuple[np.ndarray, np.ndarray]:
    """A5: exact oracle machinery with the learned persistence. Returns (aware, naive) beliefs on every record."""
    phys = ts.phys.with_(eta=eta_hat)
    cr = candidate_route(Prefix(ts.s_rec, ts.a_rec), phys, chan)
    return cr.mu2.reshape(ts.n, ts.V), cr.mu.reshape(ts.n, ts.V)
