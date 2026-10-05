"""Joint identification of clean persistence eta and the channel q from unlabeled corrupted prefixes (E4, docs/e4_protocol.md).

  pattern_tv / pattern_law : apparent-edge pattern laws and the exact minimum total variation over channels (LP), exact-law diagnostic
  profile_grid             : profile log-likelihood over an eta grid (EM over the full channel simplex at each eta, warm-started)
  refine_eta               : bounded 1-D refinement of the profile likelihood inside a bracket
  jmom                     : joint moment estimator of (eta, q) from E[z_i z_j] = c^2 q_a^2 r^|i-j| (1 - 2(q_i + q_j)), r = 1 - 2 eta
  clean_eta_mle            : eta MLE of the clean HMM from clean prefixes (the separate-validation baseline)
  profile_fisher           : profile Fisher information of eta (Schur complement over the free channel weights)
  interleave_persistence   : F1 generator, records with record-heterogeneous persistence (two Physics objects, unchanged simulator)
  MixtureOracle            : exact aware belief when the persistence class is unknown (F1 oracle)

Access: every estimator takes records (s, a) and the declared physics (rho, c, q_a, q_w, lambda, L); none receives eta, modes or theta.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog, minimize_scalar, nnls

from .adapt import em_channel
from .inference import clean_belief, flip_batch
from .params import ChannelSpec, Physics
from .simulate import Prefix, simulate

ETA_GRID = np.round(np.arange(1, 31) * 0.005, 6)           # 0.005 ... 0.15


# ----------------------------------------------------------------------------- exact-law pattern distance
def footprints(L: int):
    """Apparent-edge footprints of a flip: index 0 = none, 1 + j = flip at recorded action j (edges 0..L-2 between modes m_k, m_{k+1})."""
    out = [np.zeros(L - 1, int)]
    for j in range(L):
        f = np.zeros(L - 1, int)
        if j >= 1:
            f[j - 1] = 1
        if j <= L - 2:
            f[j] = 1
        out.append(f)
    return out


def pattern_kernel(L: int, eta: float):
    """K[pattern, state] = Pr(apparent edge pattern | channel state), E ~ iid Bern(eta) xor footprint."""
    pats = np.array(list(itertools.product([0, 1], repeat=L - 1)))
    K = np.zeros((len(pats), L + 1))
    for k, f in enumerate(footprints(L)):
        d = (pats + f) % 2
        K[:, k] = np.prod(np.where(d == 1, eta, 1 - eta), axis=1)
    return K


def pattern_tv(L: int, eta: float, w, eta2: float):
    """min over channels w' of TV(P_{eta,w}, P_{eta2,w'}) on the apparent-edge pattern law (LP). Returns (tv, argmin w')."""
    P = pattern_kernel(L, eta) @ np.asarray(w, float)
    K2 = pattern_kernel(L, eta2)
    G, m = K2.shape
    c = np.r_[np.zeros(m), 0.5 * np.ones(G)]
    A = np.block([[K2, -np.eye(G)], [-K2, -np.eye(G)]])
    res = linprog(c, A_ub=A, b_ub=np.r_[P, -P], A_eq=np.r_[np.ones(m), np.zeros(G)][None], b_eq=[1.0],
                  bounds=[(0, None)] * (m + G), method="highs")
    return float(res.fun), res.x[:m]


# ----------------------------------------------------------------------------- profile likelihood
def _queries(S: Prefix, phys: Physics, eta: float):
    ll_all, _ = flip_batch(S, phys.with_(eta=eta))
    return ll_all[:, 0], ll_all[:, 1:] - ll_all[:, [0]]


def mean_logZ(logell: np.ndarray, w: np.ndarray) -> float:
    a = np.concatenate([np.zeros((logell.shape[0], 1)), logell], axis=1) + np.log(np.maximum(w, 1e-300))[None, :]
    m = a.max(axis=1)
    return float(np.mean(m + np.log(np.exp(a - m[:, None]).sum(axis=1))))


def fit_channel(llS: np.ndarray, logell: np.ndarray, w0=None, tol: float = 1e-10, iters: int = 5000):
    """EM over the full channel simplex at fixed eta. Returns (w, mean log-likelihood of the records, iterations)."""
    if w0 is not None:
        w0 = 0.98 * np.asarray(w0, float) + 0.02 * np.full(logell.shape[1] + 1, 1.0 / (logell.shape[1] + 1))
    w, _, it = em_channel(logell, w0=w0, tol=tol, iters=iters)
    return w, float(llS.mean()) + mean_logZ(logell, w), it


def profile_grid(S: Prefix, phys: Physics, ns, etas=ETA_GRID, tol: float = 1e-10):
    """Profile over the eta grid for nested subsets S[:n], n in ns. The flip queries are computed once per eta on the full set.

    Returns dict n -> dict(ll=(G,) profile mean log-likelihood, w=(G, L+1) channel weights, it=(G,) EM iterations).
    """
    L = phys.L
    out = {n: dict(ll=np.zeros(len(etas)), w=np.zeros((len(etas), L + 1)), it=np.zeros(len(etas), int)) for n in ns}
    prev = {n: None for n in ns}
    for gi, eta in enumerate(etas):
        llS, logell = _queries(S, phys, float(eta))
        for n in ns:
            w, ll, it = fit_channel(llS[:n], logell[:n], prev[n], tol=tol)
            prev[n] = w
            out[n]["ll"][gi], out[n]["w"][gi], out[n]["it"][gi] = ll, w, it
    return out


def refine_eta(S: Prefix, phys: Physics, n: int, grid_eta: float, w0, xatol: float = 1e-4, step: float = 0.005):
    """Maximise the profile likelihood on S[:n] inside [grid_eta - step, grid_eta + step]. Returns (eta_hat, w, ll)."""
    Sn = Prefix(S.s[:n], S.a[:n])
    cache = {}

    def f(eta):
        llS, logell = _queries(Sn, phys, float(eta))
        w, ll, _ = fit_channel(llS, logell, w0)
        cache[round(float(eta), 12)] = (w, ll)
        return -ll

    lo, hi = max(grid_eta - step, 1e-3), min(grid_eta + step, 0.499)
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=xatol))
    w, ll = cache[round(float(r.x), 12)] if round(float(r.x), 12) in cache else (None, None)
    if w is None:
        f(r.x)
        w, ll = cache[round(float(r.x), 12)]
    return float(r.x), w, ll


# ----------------------------------------------------------------------------- moment estimator
def jmom(s: np.ndarray, a: np.ndarray, phys: Physics, r_grid=None):
    """Joint moment estimator. Returns (eta_hat, w) with w = [q_none, q_0..q_{L-1}] projected onto q >= 0, sum q <= 1.

    Uses only physics (rho, c, q_a) and the records; minimises sum_ij (m_ij - f_ij(r)(1 - 2(q_i + q_j)))^2 over r in (0,1) and q (NNLS inside),
    f_ij(r) = c^2 q_a^2 r^|i-j|, r = 1 - 2 eta.
    """
    n, L = a.shape
    if L < 3:
        return float("nan"), np.r_[1.0, np.zeros(L)]
    z = a * (s[:, 1:] - phys.rho * s[:, :-1])
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    m = np.array([np.mean(z[:, i] * z[:, j]) for i, j in pairs])
    d = np.array([j - i for i, j in pairs], float)
    X = np.zeros((len(pairs), L))
    for k, (i, j) in enumerate(pairs):
        X[k, [i, j]] = 1.0
    base = phys.c**2 * phys.q_a**2

    def fit(r):
        f = base * r**d
        t = 0.5 * (1.0 - m / f)
        q, _ = nnls(X * f[:, None], t * f)
        if q.sum() > 1.0:
            q = q / q.sum()
        return float(np.sum((m - f * (1.0 - 2.0 * (X @ q))) ** 2)), q

    rg = np.linspace(0.05, 0.999, 190) if r_grid is None else np.asarray(r_grid)
    obj = np.array([fit(r)[0] for r in rg])
    k = int(np.argmin(obj))
    lo, hi = rg[max(k - 1, 0)], rg[min(k + 1, len(rg) - 1)]
    r = minimize_scalar(lambda x: fit(x)[0], bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6)).x
    _, q = fit(r)
    return float((1.0 - r) / 2.0), np.r_[1.0 - q.sum(), q]


# ----------------------------------------------------------------------------- clean-law validation baseline
def clean_eta_mle(H: Prefix, phys: Physics, etas=ETA_GRID, xatol: float = 1e-4):
    """eta MLE of the clean HMM from clean prefixes (grid, then bounded refinement)."""
    def nll(eta):
        return -float(clean_belief(H, phys.with_(eta=float(eta)))[1].sum())

    obj = np.array([nll(e) for e in etas])
    g = float(etas[int(np.argmin(obj))])
    lo, hi = max(g - 0.005, 1e-3), min(g + 0.005, 0.499)
    return float(minimize_scalar(nll, bounds=(lo, hi), method="bounded", options=dict(xatol=xatol)).x)


# ----------------------------------------------------------------------------- profile Fisher information
def profile_fisher(S: Prefix, phys: Physics, chan: ChannelSpec, h: float = 1e-4):
    """Fisher information blocks at the truth (eta = phys.eta, channel chan), estimated as E[score score^T] over the records S.

    Free parameters: eta and q_0..q_{L-1} (q_none = 1 - sum q). Returns dict(I_eta, I_prof, I_full). Score in q analytic, in eta by central difference.
    """
    q = chan.q

    def loglik(eta):
        llS, logell = _queries(S, phys, eta)
        Z = (1.0 - q.sum()) + np.exp(logell) @ q
        return llS + np.log(Z), np.exp(logell), Z

    ll0, ell, Z = loglik(phys.eta)
    s_q = (ell - 1.0) / Z[:, None]
    s_eta = (loglik(phys.eta + h)[0] - loglik(phys.eta - h)[0]) / (2.0 * h)
    sc = np.column_stack([s_eta, s_q])
    I = sc.T @ sc / len(sc)
    Iqq = I[1:, 1:]
    Ieq = I[0, 1:]
    prof = float(I[0, 0] - Ieq @ np.linalg.solve(Iqq, Ieq))
    return dict(I_eta=float(I[0, 0]), I_prof=prof, I_full=I)


# ----------------------------------------------------------------------------- F1: record-heterogeneous persistence
def interleave_persistence(phys: Physics, n: int, master_a: int, master_b: int, cell_id: int, chunk_id: int, etas=(0.02, 0.08)):
    """Clean prefixes whose persistence alternates between etas[0] and etas[1] by record (even/odd rows). Uses the unchanged simulator with
    two Physics objects on disjoint masters. Returns (Prefix, class index per row, hidden modes for tests/evaluation only)."""
    lo = simulate(phys.with_(eta=etas[0]), (n + 1) // 2, master_a, cell_id, chunk_id)
    hi = simulate(phys.with_(eta=etas[1]), n // 2, master_b, cell_id, chunk_id)
    s = np.empty((n, phys.L + 1))
    a = np.empty((n, phys.L))
    m = np.empty((n, phys.L + 1))
    s[0::2], a[0::2], m[0::2] = lo.prefix.s, lo.prefix.a, lo.modes
    s[1::2], a[1::2], m[1::2] = hi.prefix.s, hi.prefix.a, hi.modes
    return Prefix(s, a), np.arange(n) % 2, m


class MixtureOracle:
    """Exact aware belief for records generated from a persistence mixture (classes with eta_c, equal weights) and channel w.

    log p(S, c, theta) = log pi_c + log w_theta + ll_c(T_theta S); mu2 = sum_{c,theta} post * mu_c(T_theta S).
    """

    def __init__(self, phys: Physics, etas=(0.02, 0.08), pi=(0.5, 0.5)):
        self.phys, self.etas, self.pi = phys, tuple(etas), np.asarray(pi, float)

    def queries(self, S: Prefix):
        return [flip_batch(S, self.phys.with_(eta=e)) for e in self.etas]

    def beliefs(self, qs, w: np.ndarray):
        """Returns (mu2 aware, mu naive-mixture (no channel), p_class_given_S for the no-channel case)."""
        lw = np.log(np.maximum(w, 1e-300))
        la = []
        for (ll, mu), lp in zip(qs, np.log(self.pi)):
            la.append(lp + lw[None, :] + ll)                                    # (N, L+1): [none, j...] log p(S|c,theta=.) up to class prior
        la = np.stack(la, axis=1)                                               # (N, C, L+1)
        m = la.max(axis=(1, 2), keepdims=True)
        p = np.exp(la - m)
        p /= p.sum(axis=(1, 2), keepdims=True)
        mu = np.stack([q[1] for q in qs], axis=1)                               # (N, C, L+1)
        mu2 = (p * mu).sum(axis=(1, 2))
        ln = np.stack([lp + q[0][:, 0] for q, lp in zip(qs, np.log(self.pi))], axis=1)
        pc = np.exp(ln - ln.max(axis=1, keepdims=True))
        pc /= pc.sum(axis=1, keepdims=True)
        mu_naive = (pc * np.stack([q[1][:, 0] for q in qs], axis=1)).sum(axis=1)
        return mu2, mu_naive, pc
