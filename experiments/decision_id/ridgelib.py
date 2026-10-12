"""L = 3 identified set, four-function belief representation and continuous certificate (closeout Part I.2, I.4).

Assumptions (Lemma 1 of docs/research_synthesis.md): S2, physics supplied, full single-flip channel simplex, L = 3, eta' in [0, 1/2).
Footprint order (mti.joint.footprints): none -> (0,0), flip0 -> (1,0), flip1 -> (1,1), flip2 -> (0,1); pattern order (00, 01, 10, 11).

Representation (derived and checked against the HMM in tests/test_decision_id.py):
    mu_r(S) = sum_a psi_a(S) J_a(r),   psi = (z2, z0 z1 z2, z1, z0) / den,   den = 1 + P0 z0 z1 + P1 z1 z2 + P01 z0 z2,
    J_A(r) = sum_B ghat(A xor B) Phat(B) r^(1+|A|-|B|),   r = 1 - 2 eta',   z_j = tanh(c a_j delta_j / q_w).
den and Phat are identified (functions of the record law); J_a(r) are Laurent polynomials with exponents -1..3.
"""
import itertools

import numpy as np
from numpy.polynomial import polynomial as Pl

from mti.joint import footprints

L = 3
PATS = np.array(list(itertools.product([0, 1], repeat=L - 1)))               # rows: 00, 01, 10, 11  (e0, e1)
FPS = footprints(L)
_idx = lambda f: int(np.where((PATS == f).all(1))[0][0])
S_OF_STATE = [_idx(f) for f in FPS]                                          # channel state k -> pattern index
H4 = np.array([[(-1) ** int(np.dot(A, x)) for x in PATS] for A in PATS])    # H4[A, x] = chi_A(x)
ABS_A = PATS.sum(1)
A_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]                                   # emptyset, {0}, {1}, {0,1}
A_IDX = [_idx(np.array(a)) for a in A_ORDER]
XOR_IDX = np.array([[_idx((PATS[i] + PATS[j]) % 2) for j in range(4)] for i in range(4)])
G_VEC = np.ones(4)
G_VEC[_idx(np.array([0, 1]))] = -1.0                                         # g(f) = -1 only at the final-action footprint
GHAT = (H4 @ G_VEC) / 4.0


def F_from_w(w):
    F = np.zeros(4)
    for k, i in enumerate(S_OF_STATE):
        F[i] += w[k]
    return F


def w_from_F(F):
    return np.array([F[i] for i in S_OF_STATE])


def ridge_w(w, eta, eta2):
    """Channel w' at persistence eta2 reproducing the record law of (eta, w): Fhat'(A) = t^|A| Fhat(A), t = (1-2eta)/(1-2eta2)."""
    t = (1 - 2 * eta) / (1 - 2 * eta2)
    Fh = H4 @ F_from_w(w)
    return w_from_F((H4 @ (Fh * t ** ABS_A)) / 4.0)


def is_uniform_F(w, tol=1e-12):
    F = F_from_w(w)
    return bool(np.abs(F - 0.25).max() <= tol)


def eta_hi(w, eta, tol=1e-12):
    """Largest eta' with (eta', w') admissible (w' >= 0). Returns (eta_hi, t_hi). np.inf for the uniform footprint law (non-compact: every eta' < 1/2 admissible).

    F'(x; t) is a quadratic in t for each x; the set {t >= 1: min_x F'(x; t) >= 0} is scanned exactly from the real roots of the four quadratics.
    """
    F = F_from_w(w)
    if is_uniform_F(w):
        return np.inf, np.inf
    Fh = H4 @ F
    quads = []
    for x in range(4):
        chi = H4[:, x]                                                        # chi_A(x) over A
        c = [0.0, 0.0, 0.0]
        for A in range(4):
            c[int(ABS_A[A])] += 0.25 * Fh[A] * chi[A]
        quads.append(c)                                                       # ascending in t: c0 + c1 t + c2 t^2

    def minval(t):
        return min(np.polyval(q[::-1], t) for q in quads)

    if minval(1 + 1e-9) < -1e-15:
        return eta, 1.0
    roots = sorted(r.real for q in quads for r in np.roots(np.trim_zeros(np.array(q[::-1]), "f")) if abs(r.imag) < 1e-12 and r.real > 1 + 1e-12)
    pts = [1.0] + roots + [(roots[-1] if roots else 1.0) + 1e6]
    for lo, hi in zip(pts[:-1], pts[1:]):
        if minval(0.5 * (lo + hi)) < -1e-15:
            return 0.5 * (1 - (1 - 2 * eta) / lo), lo
    return np.inf, np.inf


def eta_hi_bisect(w, eta):
    """Independent cross-check of eta_hi by bisection on admissibility (numerical cutoff 0.4999, not an identification endpoint)."""
    lo, hi = eta, 0.4999
    if ridge_w(w, eta, eta + 1e-9).min() < -1e-12:
        return eta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if ridge_w(w, eta, mid).min() >= 0 else (lo, mid)
    return lo


# ---------------------------------------------------------------- four-function representation
def phat_from(w, eta):
    """Identified Walsh coefficients of the apparent-edge law P = B_eta * F_w, in A_ORDER (emptyset, {0}, {1}, {0,1})."""
    r = 1 - 2 * eta
    full = (H4 @ F_from_w(w)) * r ** ABS_A
    return full[A_IDX]


def J_coef(Phat):
    """Coefficient array C[a, k+1], a over A_ORDER, k = -1..3, so that J_a(r) = sum_k C[a, k+1] r^k."""
    Pfull = np.zeros(4)
    for i, ai in enumerate(A_IDX):
        Pfull[ai] = Phat[i]
    C = np.zeros((4, 5))
    for a, ai in enumerate(A_IDX):
        for b in range(4):
            C[a, 1 + int(ABS_A[ai]) - int(ABS_A[b]) + 1] += GHAT[XOR_IDX[ai, b]] * Pfull[b]
    return C


def J_at(C, r):
    return C @ np.array([r ** k for k in range(-1, 4)])


def psi_features(s, a, c, rho, q_w, Phat):
    """psi (n, 4) and den (n,) from records (s (n, 4), a (n, 3)) under the identified Phat (order emptyset, {0}, {1}, {0,1})."""
    z = np.tanh(c * a * (s[:, 1:] - rho * s[:, :-1]) / q_w)
    den = 1 + Phat[1] * z[:, 0] * z[:, 1] + Phat[2] * z[:, 1] * z[:, 2] + Phat[3] * z[:, 0] * z[:, 2]
    psi = np.stack([z[:, 2], z[:, 0] * z[:, 1] * z[:, 2], z[:, 1], z[:, 0]], axis=1) / den[:, None]
    return psi, den


# ---------------------------------------------------------------- continuous worst case of a fixed rule over r-intervals
def _risk_numerator(C, M, c):
    """Polynomial (ascending) N(r) = r^2 * (J(r) - c)' M (J(r) - c)."""
    p = []
    for a in range(4):
        pa = np.zeros(5)
        pa[:5] = C[a]                                                         # r * J_a(r): exponents k+1 = 0..4
        pa[1] -= c[a]                                                         # - c_a r
        p.append(pa)
    N = np.zeros(9)
    for a in range(4):
        for b in range(4):
            prod = Pl.polymul(p[a], p[b])
            N[:len(prod)] += M[a, b] * prod
    return N


def risk_at(C, M, c, r):
    d = J_at(C, r) - c
    return float(d @ M @ d)


def sup_risk(C, M, c, intervals, grid_per_interval=2001):
    """Continuous sup over r in the union of intervals [(lo, hi), ...] (lo > 0) of rho_c(r) = (J(r)-c)'M(J(r)-c).

    Stationary points are the real roots of Q(r) = r N'(r) - 2 N(r) (degree <= 8) inside each interval; endpoints are added. A dense-grid maximum is
    computed as an independent sanity check. Returns dict(sup, argmax_r, max_root_residual_rel, dense_grid_max, dense_minus_sup_rel).
    """
    N = _risk_numerator(C, M, c)
    Q = Pl.polysub(Pl.polymul([0.0, 1.0], Pl.polyder(N)), 2 * N)
    Q = Pl.polytrim(Q, tol=1e-14 * max(1.0, np.abs(Q).max()))
    roots = Pl.polyroots(Q) if len(Q) > 1 else np.array([])
    best, arg, worst_res, dense = -np.inf, None, 0.0, -np.inf
    for lo, hi in intervals:
        cand = [lo, hi] + [float(z.real) for z in roots if abs(z.imag) <= 1e-8 * (1 + abs(z.real)) and lo - 1e-12 <= z.real <= hi + 1e-12]
        for r in cand:
            r = min(max(r, lo), hi)
            v = risk_at(C, M, c, r)
            if v > best:
                best, arg = v, r
            if r not in (lo, hi):
                scale = float(np.sum(np.abs(Q) * np.abs(r) ** np.arange(len(Q))))
                worst_res = max(worst_res, abs(float(Pl.polyval(r, Q))) / scale if scale > 0 else 0.0)
        g = np.linspace(lo, hi, grid_per_interval)
        dense = max(dense, max(risk_at(C, M, c, r) for r in g))
    return dict(sup=best, argmax_r=arg, max_root_residual_rel=worst_res, dense_grid_max=dense, dense_minus_sup_rel=(dense - best) / max(best, 1e-300))


# ---------------------------------------------------------------- pairwise Frank-Wolfe for max_pi  pi.d - pi'M pi  (squared Chebyshev radius)
def pairwise_fw(Mg, max_iter=200_000, tol_rel=1e-13):
    """Maximise f(pi) = pi.d - pi'Mg pi over the simplex (d = diag Mg) by pairwise Frank-Wolfe with exact line search.

    f(pi) is the Bayes value of the prior pi (a lower bound on the minimax value V* of the set). Returns dict with pi, value, fw_gap (standard FW duality gap
    max_s g_s - g.pi, which bounds f* - f(pi)), iterations and termination reason.
    """
    G = Mg.shape[0]
    d = np.diag(Mg).copy()
    pi = np.full(G, 1.0 / G)
    tol = tol_rel * max(d.max(), 1e-300)
    reason, gap = "max_iter", np.inf
    for it in range(1, max_iter + 1):
        g = d - 2 * Mg @ pi
        s = int(np.argmax(g))
        gap = float(g[s] - g @ pi)
        if gap <= tol:
            reason = "fw_gap<=tol"
            break
        supp = np.where(pi > 0)[0]
        v = supp[int(np.argmin(g[supp]))]
        dirv = Mg[s, s] + Mg[v, v] - 2 * Mg[s, v]
        gd = g[s] - g[v]
        if dirv <= 0 or gd <= 0:
            reason = "no_ascent_direction"
            break
        gamma = min(pi[v], gd / (2 * dirv))
        pi[s] += gamma
        pi[v] -= gamma
    value = float(pi @ d - pi @ Mg @ pi)
    return dict(pi=pi, value=value, fw_gap=gap, iterations=it, termination=reason, tol_abs=float(tol))


def support_report(pi, grid_eta, tol=1e-6):
    idx = np.where(pi > tol)[0]
    return dict(support_tol=tol, support={f"{grid_eta[i]:.5f}": float(pi[i]) for i in idx}, mass_below_tol=float(pi[pi <= tol].sum()),
                n_support=int(len(idx)))
