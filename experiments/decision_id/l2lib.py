"""L = 2 decision geometry (closeout Part I.3). Valid for supplied physics, eta in [0, 1/2), the full single-flip channel and an identified h in (0, 1).

Record belief:   mu_{u,v}(S) = phi . (u, v),   phi = (z1, z0) / (1 + h z0 z1),   z_j = tanh(c a_j (s_{j+1} - rho s_j) / q_w),
                 u = r (1 - 2 q1),  v = r^2 (1 - 2 q0),  r = 1 - 2 eta,  h = r (1 - 2 beta),  beta = q0 + q1.
Target set (eta unknown):  T_h = triangle with vertices A = (h, h^2), B = (h, 1), C = (1, h); beliefs are linear in (u, v).
Target set (eta known):    segment from (h, r^2) to (r, r h).
"""
import numpy as np


def h_of(eta, q0, q1):
    return (1 - 2 * eta) * (1 - 2 * (q0 + q1))


def uv_of(eta, q0, q1):
    r = 1 - 2 * eta
    return np.array([r * (1 - 2 * q1), r * r * (1 - 2 * q0)])


def vertex_worlds(h):
    """Three exactly observationally equivalent worlds (eta, q0, q1) at the triangle vertices A, B, C (valid for h in (0, 1))."""
    return {"A": (0.5 * (1 - h), 0.0, 0.0), "B": (0.0, 0.0, 0.5 * (1 - h)), "C": (0.0, 0.5 * (1 - h), 0.0)}


def vertices(h):
    return np.array([[h, h * h], [h, 1.0], [1.0, h]])


def world_from_uv(u, v, h):
    """Inverse map: (u, v) in T_h -> (eta, q0, q1), or None if outside. r = ? from v = r^2 - r (u - h) (positive root); q's from u, v."""
    # v = r^2 - r (u - h)  ->  r^2 - (u - h) r - v = 0
    disc = (u - h) ** 2 + 4 * v
    r = 0.5 * ((u - h) + np.sqrt(disc))
    if not (h - 1e-12 <= r <= 1 + 1e-12 and u - 1e-12 <= r + 1e-12):
        return None
    q1 = 0.5 * (1 - u / r)
    q0 = 0.5 * (1 - v / (r * r))
    beta = 0.5 * (1 - h / r)
    if q0 < -1e-12 or q1 < -1e-12 or abs(q0 + q1 - beta) > 1e-9:
        return None
    return 0.5 * (1 - r), max(q0, 0.0), max(q1, 0.0)


def in_triangle(uv, h, tol=1e-12):
    u, v = uv
    return bool(h - tol <= u <= 1 + tol and h * u - tol <= v <= 1 + h - u + tol)


def metric_d2(a, b, G):
    d = a - b
    return float(d @ G @ d)


def triangle_minimax(X, G):
    """Smallest enclosing ball of the triangle X (3 x 2) in the metric G. Returns dict(value, center, weights (barycentric LFP), kind, angle_inner, d2).

    kind is 'circumcentre' (acute in the G metric: three-point least-favourable prior) or 'longest_side' (obtuse/right: two-point prior).
    """
    d2 = {(i, j): metric_d2(X[i], X[j], G) for i in range(3) for j in range(i + 1, 3)}
    ang = {}
    for o in range(3):
        i, j = [k for k in range(3) if k != o]
        ang["ABC"[o]] = float((X[i] - X[o]) @ G @ (X[j] - X[o]))
    cands = []
    for (i, j), dd in d2.items():
        k = 3 - i - j
        m = 0.5 * (X[i] + X[j])
        if metric_d2(m, X[k], G) <= dd / 4 * (1 + 1e-12):
            w = np.zeros(3)
            w[i] = w[j] = 0.5
            cands.append((dd / 4, m, w, "longest_side"))
    A = np.stack([2 * (X[1] - X[0]) @ G, 2 * (X[2] - X[0]) @ G])
    rhs = np.array([X[1] @ G @ X[1] - X[0] @ G @ X[0], X[2] @ G @ X[2] - X[0] @ G @ X[0]])
    c = np.linalg.solve(A, rhs)
    lam = np.linalg.solve(np.vstack([X.T, np.ones(3)]), np.r_[c, 1.0])
    if lam.min() >= -1e-12:
        cands.append((metric_d2(c, X[0], G), c, lam, "circumcentre"))
    val, center, w, kind = min(cands, key=lambda t: t[0])
    return dict(value=float(val), center=center, weights=w, kind=kind, angle_inner=ang, max_side_sq=max(d2.values()), d2=d2)
