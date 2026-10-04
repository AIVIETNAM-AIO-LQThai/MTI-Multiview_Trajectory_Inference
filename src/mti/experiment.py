"""Chunk workers for the oracle information/opportunity experiment (spec sections 7-12).

`rb_chunk`   : Rao-Blackwellised estimator. For each independent clean prefix all L+1 corruption views
               (none, T_0H..T_{L-1}H) are evaluated exactly and averaged with weights [1-beta, q_j].
`check_chunk`: independent single-draw estimator (theta sampled from the channel stream) plus simulated
               decision-branch costs (common random numbers) for the physical-cost check.
`fit_chunk`  : raw single-draw (|s_L|, mu~, mu_2, kappa) records for the recalibration diagnostic.
Chunks return additive sums only; ordering of chunks and number of workers cannot change results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .decision import a_star, base_cost, kappa
from .inference import candidate_route, clean_belief, compatibility_residual, folded_route
from .params import ChannelSpec, Physics
from .simulate import Prefix, apply_channel, decision_state, sample_theta, simulate

METRICS = ["V2", "I", "X", "AW", "D"]
LAG_METRICS = METRICS
REL_EDGES = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-12])
PROF_TYPES = ["record S", "cand repair at true loc", "cand repair elsewhere", "folded at true loc", "folded elsewhere"]
PROF_STATS = ["ll", "d_ll", "d_ll2", "below_p01", "below_p05"]
MASS_STATS = ["cand_contam_count", "fold_contam_count", "cand_mass_true", "fold_mass_true"]


@dataclass(frozen=True)
class Cell:
    cell_id: int
    name: str
    role: str
    phys: Physics
    chan: ChannelSpec


def eval_views(H: Prefix, phys: Physics, chan: ChannelSpec):
    """Records S_v = T_vH for views v=0 (none), j+1 (flip at j), stacked as B*(L+1) records."""
    B, L = H.n, phys.L
    V = L + 1
    s = np.repeat(H.s[:, None, :], V, axis=1)
    a = np.repeat(H.a[:, None, :], V, axis=1)
    j = np.arange(L)
    a[:, j + 1, j] *= -1.0
    rec = Prefix(s.reshape(B * V, L + 1), a.reshape(B * V, L))
    return rec, candidate_route(rec, phys, chan)


def column_names(cell: Cell, recal_names):
    names = []
    for m in METRICS:
        names += [m, f"{m}_none", f"{m}_cor"]
    names += ["cost_zero", "cost_mode", "cost_clean", "voi_clean", "cost_naive", "cost_aware", "kappa",
              "logscore_cat", "brier_cat", "logloss_any", "brier_any", "top1_cor"]
    for r in recal_names:
        names += [f"rc:{r}:A", f"rc:{r}:R", f"rc:{r}:X"]
    return names


def flag_names(L: int):
    return [f"E{k}" for k in range(1, L + 1)] + ["any_prefix", "none_all"]


def _rs(x, B, V):
    return x.reshape((B, V) + x.shape[1:])


def rb_chunk(cell: Cell, master: int, chunk_id: int, n: int, recal: dict, thr: dict, do_profile: bool):
    phys, chan = cell.phys, cell.chan
    L = phys.L
    V = L + 1
    smp = simulate(phys, n, master, cell.cell_id, chunk_id)
    H = smp.prefix
    rec, cr = eval_views(H, phys, chan)
    B = n
    mu_t = _rs(cr.mu, B, V)
    mu2 = _rs(cr.mu2, B, V)
    var = _rs(cr.var_muH, B, V)
    sL = H.s[:, -1]
    kap = kappa(phys, sL)
    mH = mu_t[:, 0]
    mL = smp.modes[:, -1]
    w = chan.view_weights()
    pi = chan.pi_arr
    K = kap[:, None]

    Vm = {
        "V2": K * (mu_t - mu2) ** 2,
        "I": K * var,
        "X": K * (mu_t - mH[:, None]) ** 2,
        "AW": K * (mu2 - mH[:, None]) ** 2,
        "D": K * ((mu_t - mL[:, None]) ** 2 - (mu2 - mL[:, None]) ** 2),
    }
    cols = {}
    for m, arr in Vm.items():
        cols[m] = arr @ w
        cols[f"{m}_none"] = arr[:, 0]
        cols[f"{m}_cor"] = arr[:, 1:] @ pi
    base = base_cost(phys, sL)
    cols["cost_zero"] = base
    cols["cost_mode"] = base - kap
    cols["cost_clean"] = base - kap * mH**2
    cols["voi_clean"] = kap * mH**2
    cols["cost_naive"] = cols["cost_clean"] + cols["X"]
    cols["cost_aware"] = cols["cost_clean"] + cols["AW"]
    cols["kappa"] = kap

    # exact corruption posterior as an oracle diagnostic (class index of view v is v)
    lp = _rs(cr.log_post, B, V)                                  # (B,V,V)
    ar = np.arange(V)
    diag = lp[:, ar, ar]
    P = np.exp(lp)
    pos = (w > 0)[None, :]            # views with zero channel mass never occur; their scores are undefined (-inf)
    cols["logscore_cat"] = np.where(pos, -diag, 0.0) @ w
    cols["brier_cat"] = ((P**2).sum(-1) - 2.0 * np.exp(diag) + 1.0) @ w
    log_any = logsumexp(lp[..., 1:], axis=-1)                    # log(1 - p_none), no cancellation
    q_any = np.exp(log_any)
    y_any = (ar >= 1)[None, :]
    cols["logloss_any"] = np.where(pos, np.where(y_any, -log_any, -lp[..., 0]), 0.0) @ w
    cols["brier_any"] = (q_any - y_any) ** 2 @ w
    top = np.argmax(P[..., 1:], axis=-1)                         # (B,V)
    cols["top1_cor"] = (top[:, 1:] == np.arange(L)[None, :]).astype(float) @ pi
    rel_bin = np.digitize(q_any, REL_EDGES) - 1
    wv = np.broadcast_to(w, q_any.shape)
    nb = len(REL_EDGES) - 1
    rel = np.stack([np.bincount(rel_bin.ravel(), weights=x.ravel(), minlength=nb)
                    for x in (wv, wv * q_any, wv * y_any.astype(float))], axis=1)

    # restricted-summary recalibration variants (fixed before evaluation)
    for name, g in recal.items():
        gv = g(np.abs(sL)[:, None], mu_t)
        A = K * (mu_t - gv) ** 2
        R = K * (gv - mu2) ** 2
        X = 2.0 * K * (mu_t - gv) * (gv - mu2)
        cols[f"rc:{name}:A"], cols[f"rc:{name}:R"], cols[f"rc:{name}:X"] = A @ w, R @ w, X @ w

    names = column_names(cell, list(recal))
    Y = np.stack([cols[k] for k in names], axis=1)
    m_ = smp.modes
    E = m_[:, 1:] != m_[:, :-1]                                  # edge k = m_{k-1} -> m_k, k=1..L
    Fm = np.concatenate([E, E[:, : L - 1].any(axis=1, keepdims=True), ~E.any(axis=1, keepdims=True)], axis=1).astype(float)
    lagm = np.stack([Vm[m] for m in LAG_METRICS], axis=0)        # (nm,B,V)
    out = dict(
        n=n, S1=Y.sum(0), S2=(Y**2).sum(0), SF=Fm.sum(0), SFY=Fm.T @ Y, SFY2=Fm.T @ Y**2,
        lag_S1=lagm.sum(1), lag_S2=(lagm**2).sum(1), rel=rel,
    )
    if do_profile:
        out.update(_profile_and_verify(cell, rec, cr, B, thr))
    return out


def _profile_and_verify(cell: Cell, rec: Prefix, cr, B: int, thr: dict):
    phys, chan = cell.phys, cell.chan
    L = phys.L
    V = L + 1
    fr = folded_route(rec, phys, chan)
    C = compatibility_residual(fr, cr)
    verify = dict(
        max_abs_G_minus_M=float(np.max(np.abs(fr.G - cr.mu2))),
        max_abs_C=float(np.max(np.abs(C))),
        max_abs_logZ_diff=float(np.max(np.abs(fr.log_Z - cr.log_Z))),
        max_abs_logell_diff=float(np.max(np.abs(fr.log_ell - cr.log_ell))),
        max_abs_pnorm=float(np.max(np.abs(cr.p_none + cr.p.sum(1) - 1.0))),
        max_abs_wnorm=float(np.max(np.abs(fr.w0 + fr.w.sum(1) - 1.0))),
        min_log_r=float(fr.log_r.min()), min_log_1mr=float(fr.log_1mr.min()),
    )
    llS, llT, llF = _rs(cr.ll, B, V), _rs(cr.ll_flip, B, V), _rs(fr.ll_fold, B, V)
    llH = llS[:, :1]
    v_idx = np.arange(V)
    theta = v_idx - 1
    attacked = v_idx >= 1
    # true-location masks (V,L): True where k == theta_v
    tm = np.zeros((V, L), bool)
    tm[v_idx[1:], theta[1:]] = True
    verify["max_abs_repair_true_minus_llH"] = float(np.max(np.abs(llT[:, v_idx[1:], theta[1:]] - llH)))
    verify["max_abs_folded_O_theta_invariance"] = float(
        np.max(np.abs(llF[:, v_idx[1:], theta[1:]] - llF[:, 0, :])))
    # Exact (V,L) contamination counts relative to H (action sign positions that differ)
    cnt_S = attacked.astype(float)                                           # (V,)
    cnt_T = np.where(tm, 0.0, cnt_S[:, None] + 1.0)                          # (V,L)
    cnt_F = (attacked[:, None] & ~tm).astype(float)                          # (V,L)

    def type_stats(ll, mask_v_k, thr_pair):
        """Per-record average over the queries selected by mask (V,L) of [ll, d_ll, d_ll^2, <p01, <p05]."""
        cntm = mask_v_k.sum(1)                                               # (V,)
        ok = cntm > 0
        den = np.where(ok, cntm, 1)[None, :]
        m = mask_v_k[None, :, :]
        a = (ll * m).sum(2) / den
        d = ll - llH[:, :, None]
        dd = (d * m).sum(2) / den
        dd2 = ((d**2) * m).sum(2) / den
        b1 = ((ll < thr_pair[0]) * m).sum(2) / den
        b5 = ((ll < thr_pair[1]) * m).sum(2) / den
        return np.stack([a, dd, dd2, b1, b5], axis=-1), ok                   # (B,V,5), (V,)

    tf = (thr["full_p01"], thr["full_p05"])
    tg = (thr["fold_p01"], thr["fold_p05"])
    allk = np.ones((V, L), bool)
    recs = []
    # type 0: record S itself
    a0 = np.stack([llS, llS - llH, (llS - llH) ** 2, (llS < tf[0]).astype(float), (llS < tf[1]).astype(float)], -1)
    recs.append((a0, np.ones(V, bool)))
    recs.append(type_stats(llT, tm, tf))                 # cand repair at true location (attacked only)
    recs.append(type_stats(llT, ~tm, tf))                # cand repair elsewhere (all k for view 0)
    recs.append(type_stats(llF, tm, tg))                 # folded at true location
    recs.append(type_stats(llF, ~tm, tg))
    pi = chan.pi_arr
    prof = np.zeros((2, len(PROF_TYPES), len(PROF_STATS)))
    for t, (arr, ok) in enumerate(recs):
        prof[0, t] = arr[:, 0, :].sum(0) if ok[0] else 0.0
        att_w = pi[None, :, None] * arr[:, 1:, :]
        prof[1, t] = att_w.sum((0, 1)) if ok[1:].all() else 0.0
    # oracle weight masses on contaminated queries and on the true location
    p_none, p = _rs(cr.p_none, B, V), _rs(cr.p, B, V)
    w0, wf = _rs(fr.w0, B, V), _rs(fr.w, B, V)
    cand_cnt = p_none * cnt_S[None, :] + (p * cnt_T[None]).sum(2)
    fold_cnt = w0 * cnt_S[None, :] + (wf * cnt_F[None]).sum(2)
    cand_true = np.where(attacked[None, :], (p * tm[None]).sum(2), p_none)
    fold_true = np.where(attacked[None, :], (wf * tm[None]).sum(2), w0)
    mass = np.zeros((2, len(MASS_STATS)))
    for i, arr in enumerate((cand_cnt, fold_cnt, cand_true, fold_true)):
        mass[0, i] = arr[:, 0].sum()
        mass[1, i] = (arr[:, 1:] * pi[None, :]).sum()
    return dict(prof=prof, mass=mass, verify=verify, prof_n=B)


def clean_thresholds(cell: Cell, master: int, n: int = 20000, chunk_id: int = 99999):
    """1% / 5% quantiles of the clean transition log-likelihood ll(H) and of the folded-query ll(O_k(H))."""
    phys = cell.phys
    smp = simulate(phys, n, master, cell.cell_id, chunk_id)
    _, ll = clean_belief(smp.prefix, phys)
    fr = folded_route(smp.prefix, phys, ChannelSpec.uniform(phys.L, 0.0))
    q = (0.01, 0.05)
    return dict(full_p01=float(np.quantile(ll, q[0])), full_p05=float(np.quantile(ll, q[1])),
                fold_p01=float(np.quantile(fr.ll_fold, q[0])), fold_p05=float(np.quantile(fr.ll_fold, q[1])))


def _single_draw(cell: Cell, master: int, chunk_id: int, n: int):
    phys, chan = cell.phys, cell.chan
    smp = simulate(phys, n, master, cell.cell_id, chunk_id)
    theta = sample_theta(chan, n, master, cell.cell_id, chunk_id)
    S = apply_channel(smp.prefix, theta)
    cr = candidate_route(S, phys, chan)
    mH, _ = clean_belief(smp.prefix, phys)
    return smp, theta, cr, mH


def fit_chunk(cell: Cell, master: int, chunk_id: int, n: int):
    smp, theta, cr, mH = _single_draw(cell, master, chunk_id, n)
    sL = smp.prefix.s[:, -1]
    return dict(abs_s=np.abs(sL), mu=cr.mu, mu2=cr.mu2, kap=kappa(cell.phys, sL))


CHECK_COLS = ["V2", "I", "X", "AW", "D", "cal_mu2", "cal_mH", "any_cor", "any_pred",
              "sim_zero", "sim_clean", "sim_naive", "sim_aware", "sim_mode",
              "ana_zero", "ana_clean", "ana_naive", "ana_aware", "ana_mode"]


def check_chunk(cell: Cell, master: int, chunk_id: int, n: int):
    """Single-draw estimator plus simulated decision-branch costs (common random numbers)."""
    phys = cell.phys
    smp, theta, cr, mH = _single_draw(cell, master, chunk_id, n)
    sL = smp.prefix.s[:, -1]
    kap = kappa(phys, sL)
    mL = smp.modes[:, -1]
    base = base_cost(phys, sL)
    c = {}
    c["V2"] = kap * (cr.mu - cr.mu2) ** 2
    c["I"] = kap * cr.var_muH
    c["X"] = kap * (cr.mu - mH) ** 2
    c["AW"] = kap * (cr.mu2 - mH) ** 2
    c["D"] = kap * ((cr.mu - mL) ** 2 - (cr.mu2 - mL) ** 2)
    c["cal_mu2"] = mL * cr.mu2 - cr.mu2**2
    c["cal_mH"] = mL * mH - mH**2
    c["any_cor"] = (theta >= 0).astype(float)
    c["any_pred"] = 1.0 - cr.p_none
    actions = dict(zero=np.zeros(n), clean=a_star(phys, sL, mH), naive=a_star(phys, sL, cr.mu),
                   aware=a_star(phys, sL, cr.mu2), mode=a_star(phys, sL, mL))
    for k, act in actions.items():
        s1 = decision_state(phys, smp, act)
        c[f"sim_{k}"] = s1**2 + phys.lam * act**2
    # analytic conditional expectations given (H, theta): E[m_L | H, theta] = mu_H (exogenous channel)
    from .decision import cond_cost
    for k, act in actions.items():
        u = mL if k == "mode" else mH
        c[f"ana_{k}"] = cond_cost(phys, sL, u, act)
    Y = np.stack([c[k] for k in CHECK_COLS], axis=1)
    sim = np.stack([c[f"sim_{k}"] for k in ("zero", "clean", "naive", "aware", "mode")], axis=1)
    ana = np.stack([c[f"ana_{k}"] for k in ("zero", "clean", "naive", "aware", "mode")], axis=1)
    D_ = sim - ana
    return dict(n=n, S1=Y.sum(0), S2=(Y**2).sum(0), dS1=D_.sum(0), dS2=(D_**2).sum(0))


def merge_sums(parts):
    """Add the additive arrays across chunks; keep per-chunk lists for batch-means errors."""
    out = {}
    for k in ("n", "S1", "S2", "SF", "SFY", "SFY2", "lag_S1", "lag_S2", "rel", "dS1", "dS2", "prof", "mass", "prof_n"):
        vals = [p[k] for p in parts if k in p]
        if vals:
            out[k] = sum(vals)
    if any("rel" in p for p in parts):
        out["rel_chunks"] = np.stack([p["rel"] for p in parts])
    prof_parts = [p for p in parts if "prof" in p]
    if prof_parts:
        out["prof_chunks"] = np.stack([p["prof"] for p in prof_parts])
        out["mass_chunks"] = np.stack([p["mass"] for p in prof_parts])
        ver = {}
        for p in prof_parts:
            for k, v in p["verify"].items():
                ver[k] = min(ver.get(k, np.inf), v) if k.startswith("min_") else max(ver.get(k, 0.0), v)
        out["verify"] = ver
    return out
