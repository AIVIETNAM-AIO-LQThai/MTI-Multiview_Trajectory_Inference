"""Exact L=3 identified set (supplied physics, full channel simplex): admissible eta' interval, ridge channels,
record-likelihood check, beliefs and decision distances on the E4 evaluation prefixes (master 8601, cell 0, 20,000)."""
import sys, json, itertools
import numpy as np

from e4common import PHYS, truths, MASTER_EVAL, N_EVAL, TestSet

from mti.misspec import ExactRecords, TrueLaw
from mti.adapt import Precomputed
from mti.joint import footprints

L = 3
pats = np.array(list(itertools.product([0, 1], repeat=L - 1)))           # pattern order (00,01,10,11)
fps = footprints(L)                                                         # state order: none, flip0, flip1, flip2
def pat_index(f): return int(np.where((pats == f).all(1))[0][0])
S_of_state = [pat_index(f) for f in fps]
H = np.array([[(-1) ** int(np.dot(A, x)) for x in pats] for A in pats])   # Walsh-Hadamard: chi_A(x)
absA = pats.sum(1)

def F_from_w(w):
    F = np.zeros(len(pats))
    for k, i in enumerate(S_of_state): F[i] += w[k]
    return F
def w_from_F(F):
    return np.array([F[i] for i in S_of_state])
def ridge_w(w, eta, eta2):
    t = (1 - 2 * eta) / (1 - 2 * eta2)
    Fh = H @ F_from_w(w)
    return w_from_F((H @ (Fh * t ** absA)) / len(pats))

def eta_hi(w, eta):
    # largest eta2 >= eta with ridge_w >= 0 (bisection on t; F'(x) is a quadratic in t)
    lo, hi = eta, 0.4999
    if ridge_w(w, eta, eta + 1e-9).min() < -1e-12: return eta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if ridge_w(w, eta, mid).min() >= 0 else (lo, mid)
    return lo

out = {}
for pname in ("lead", "sec"):
    phys = PHYS[pname].with_(L=L)
    ts = TestSet(phys, N_EVAL, cell_id=0, master=MASTER_EVAL)
    ex = ExactRecords(ts); tl = TrueLaw(ts, ex); K = tl.K; voi = tl.voi_clean(); n, V = ts.n, ts.V
    for tname, truth in truths(L):
        w0 = truth.view_weights(); eta0 = phys.eta
        m2t = tl.reference(truth)
        V2 = float(((K * (ex.naive - m2t) ** 2) @ w0).mean())
        delta = float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi))
        eh = eta_hi(w0, eta0)
        grid = np.unique(np.r_[np.linspace(0, eta0, 21), np.linspace(eta0, eh, 21)])
        logZ0 = None; rows = []; MU = []
        llref = ex.ll[:, 0] + np.log((np.exp(ex.ll - ex.ll[:, [0]]) * w0).sum(1))   # log p(S) under reference
        for e2 in grid:
            w2 = ridge_w(w0, eta0, e2); w2 = np.maximum(w2, 0); w2 /= w2.sum()
            e = ex if abs(e2 - eta0) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(e2)))
            ll2 = e.ll[:, 0] + np.log((np.exp(e.ll - e.ll[:, [0]]) * w2).sum(1))
            mu = Precomputed.build(e.ll, e.mu).compose(w2).reshape(n, V)
            MU.append(mu)
            reg = float(((K * (mu - m2t) ** 2) @ w0).mean())
            rows.append(dict(eta=float(e2), beta=float(1 - w2[0]), w=w2.round(6).tolist(), regret_over_voi=reg / voi,
                             max_abs_ll_diff=float(np.abs(ll2 - llref).max())))
        MU = np.array(MU)                                                    # (G, n, V)
        # monotonicity of mu in eta' per record view (betweenness): fraction of (record,view) with monotone path
        dmu = np.diff(MU, axis=0)
        mono = float(np.mean((np.all(dmu >= -1e-12, 0)) | (np.all(dmu <= 1e-12, 0))))
        mu_lo, mu_hi = MU.min(0), MU.max(0)
        diam_env = float(((K * (mu_hi - mu_lo) ** 2) @ w0).mean())          # E kappa (envelope width)^2
        # pairwise P-distances between ridge endpoints and least-favourable prior over the grid
        G = len(grid)
        Wk = (K * w0[None, :])                                                # (n,V) weights
        M = np.einsum('anv,bnv,nv->ab', MU, MU, Wk) / n
        d = np.diag(M).copy()
        D2 = d[:, None] + d[None, :] - 2 * M                                  # E kappa (mu_a - mu_b)^2
        # maximise pi.d - pi M pi over simplex (concave) by Frank-Wolfe
        pi = np.full(G, 1.0 / G)
        for it in range(20000):
            g = d - 2 * M @ pi
            s = np.zeros(G); s[np.argmax(g)] = 1
            gap = g @ (s - pi)
            if gap < 1e-14: break
            dirn = s - pi; a = dirn @ M @ dirn
            step = min(1.0, (g @ dirn) / (2 * a)) if a > 0 else 1.0
            pi = pi + step * dirn
        minimax_val = float(pi @ d - pi @ M @ pi)
        b_mm = np.tensordot(pi, MU, 1)                                        # minimax rule: fixed mixture of ridge beliefs
        worst_mm = float(max(((K * (b_mm - MU[g_]) ** 2) @ w0).mean() for g_ in range(G)))
        b_mid = 0.5 * (mu_lo + mu_hi)
        worst_mid = float(max(((K * (b_mid - MU[g_]) ** 2) @ w0).mean() for g_ in range(G)))
        worst_bottom = float(D2[0].max()); worst_truth = float(D2[np.argmin(abs(grid - eta0))].max())
        key = f"{pname}_L3_{tname}"
        out[key] = dict(voi_ref=voi, delta_over_voi=delta / voi, V2_over_voi=V2 / voi, eta_hi=eh, beta_at_eta0=float(1 - ridge_w(w0, eta0, 0)[0]),
                        monotone_frac=mono, diam_env_over_voi=diam_env / voi, diameter_max_pair_over_voi=float(D2.max() / voi),
                        minimax_value_over_voi=minimax_val / voi, lf_prior={f"{grid[i]:.4f}": round(float(pi[i]), 4) for i in range(G) if pi[i] > 1e-4},
                        worst_minimax_rule_over_voi=worst_mm / voi, worst_pointwise_mid_over_voi=worst_mid / voi,
                        worst_bottom_selection_over_voi=worst_bottom / voi, worst_truth_eta_point_over_voi=worst_truth / voi,
                        rows=rows)
        print(key, json.dumps({k: v for k, v in out[key].items() if k != 'rows'}))
        for r in rows:
            if any(abs(r['eta'] - x) < 1e-9 for x in (0.0, 0.005, 0.03, 0.05)) or r is rows[-1]:
                print('   ', r)
json.dump(out, open('ridge_l3.json', 'w'), indent=1)
