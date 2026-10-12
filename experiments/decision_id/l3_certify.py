"""L = 3 decision identification: exact arc, decision diameter, continuous endpoint-midpoint certificate and Frank-Wolfe bracket (closeout spec II.3).

Exact-law / evaluation only: no fitting, no training. Evaluation measure = the E4 evaluation prefixes (master 8601, cell 0, 20,000 prefixes, all L+1 views
weighted by the reference truth). One prefix with its views is one evaluation unit.

Per (physics, truth) it reports
  * the arc eta' in [0, eta_hi] (exact roots; bisection cross-check; uniform footprint law -> non-compact, skipped),
  * agreement of the four-function belief representation with the HMM at >= 9 arc points, and record-law equality at the endpoints,
  * the endpoint-pair LOWER BOUND d^2/4 (valid for every belief rule on this measure) and its prefix-level SE,
  * the CONTINUOUS worst case of the endpoint midpoint (real roots of a degree <= 8 polynomial) and the endpoint-ball certificate,
  * a Frank-Wolfe (pairwise) bracket for the minimax value: lower bound = Bayes value of the FW prior on a 401-point grid, upper bound = continuous
    worst case of the FW mixture rule; FW gap, iterations, termination reason and support tolerance are saved,
  * the Jung upper bound on V* (an unconditional certificate on this measure),
  * descriptive grid quantities: worst risk of the pointwise-envelope midpoint, of the bottom-endpoint selection and of naive.
Normalisers: raw cost, E_P[kappa] (identified) and the reference truth's VOI_clean (labelled). The margin is the COMMON reference margin of the reference
truth (clip(0.1 V2, 0.002 VOI, 0.02 VOI)); it is not recomputed per compatible world.

    python experiments/decision_id/l3_certify.py            ->  results/decision_id/l3_certify.json
"""
import json
import os
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridgelib as R  # noqa: E402
from did_common import ETA_TRUE, N_EVAL, OUT, PHYS, TestSet, reference_margin, save_json, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.misspec import ExactRecords, TrueLaw  # noqa: E402

TOL = dict(hmm=1e-10, loglik=1e-12, root_residual=1e-8, gap_rel=0.01)
GRID = 401


def lp(e, w):
    with np.errstate(divide="ignore"):
        return logsumexp(e.ll + np.log(w)[None, :], axis=1)


def certify_cell(pname, tname, truth, ts_cache):
    phys = PHYS[pname].with_(L=3)
    ts = ts_cache.setdefault(pname, TestSet(phys, N_EVAL, cell_id=0, master=8601))
    ex = ts_cache.setdefault(pname + "_ex", ExactRecords(ts))
    tl = ts_cache.setdefault(pname + "_tl", TrueLaw(ts, ex))
    n, V = ts.n, ts.V
    w0, eta0 = truth.view_weights(), phys.eta
    delta_ref, V2_ref, voi = reference_margin(phys, truth, ts, ex, tl)
    Ekap = float(ts.kap.mean())
    cell = dict(physics=pname, truth=tname, eta=eta0, w_truth=w0, voi_ref=voi, margin_ref_raw=delta_ref, margin_ref_over_voi=delta_ref / voi,
                V2_ref_over_voi=V2_ref / voi, E_kappa=Ekap, n_prefixes=n, checks={})
    if R.is_uniform_F(w0):
        cell["arc"] = "non-compact (uniform footprint law): every eta' < 1/2 admissible; skipped"
        return cell
    eh, t_hi = R.eta_hi(w0, eta0)
    eh_b = R.eta_hi_bisect(w0, eta0)
    cell["arc"] = dict(eta_lo=0.0, eta_hi=eh, t_hi=t_hi, eta_hi_bisection_crosscheck=eh_b, abs_diff=abs(eh - eh_b))
    r_hi = 1 - 2 * eh
    Phat = R.phat_from(w0, eta0)
    C = R.J_coef(Phat)
    psi, den = R.psi_features(ts.s_rec, ts.a_rec, phys.c, phys.rho, phys.q_w, Phat)
    kw = np.repeat(ts.kap, V) * np.tile(w0, n)                                # (n*V,) weights kappa_i * w_v (prefix-major, V views per prefix)
    M = (psi * kw[:, None]).T @ psi / n

    # --- HMM agreement at >= 9 arc points and record-law equality at the endpoints
    arc_pts = np.linspace(0.0, eh, 9)
    worst_hmm = 0.0
    ll_diffs = {}
    llref = lp(ex, w0)
    for e2 in arc_pts:
        w2 = np.maximum(R.ridge_w(w0, eta0, e2), 0.0)
        w2 /= w2.sum()
        e = ex if abs(e2 - eta0) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(e2)))
        mu_hmm = Precomputed.build(e.ll, e.mu).compose(w2)
        mu_rep = psi @ R.J_at(C, 1 - 2 * e2)
        worst_hmm = max(worst_hmm, float(np.abs(mu_hmm - mu_rep).max()))
        if e2 in (arc_pts[0], arc_pts[-1]):
            ll_diffs[f"{e2:.5f}"] = float(np.abs(lp(e, w2) - llref).max())
    cell["checks"] = dict(max_abs_hmm_vs_four_function=worst_hmm, record_loglik_diff_endpoints=ll_diffs,
                          hmm_ok=worst_hmm <= TOL["hmm"], loglik_ok=max(ll_diffs.values()) <= TOL["loglik"])

    # --- decision diameter on a grid + endpoint pair
    eta_g = np.linspace(0.0, eh, GRID)
    Jg = np.array([R.J_at(C, 1 - 2 * e) for e in eta_g])
    Mg = Jg @ M @ Jg.T
    d = np.diag(Mg)
    D2 = d[:, None] + d[None, :] - 2 * Mg
    J_bot, J_top = Jg[0], Jg[-1]
    delta_v = J_top - J_bot
    d2_end = float(delta_v @ M @ delta_v)
    cell["diameter_sq"] = dict(endpoint_pair_raw=d2_end, grid_max_pair_raw=float(D2.max()), grid_max_attained_by_endpoints=bool(abs(D2.max() - d2_end) <= 1e-12 * d2_end))
    # prefix-level SE of d^2/4 (fits/prior/arc fixed): per-prefix M_i
    Mi = np.einsum("nvk,nvl,nv->nkl", psi.reshape(n, V, 4), psi.reshape(n, V, 4), kw.reshape(n, V))
    d2_i = np.einsum("k,nkl,l->n", delta_v, Mi, delta_v)
    v_i = ts.kap * tl.muH ** 2
    ratio = d2_i.mean() / v_i.mean()
    cell["lower_bound"] = dict(
        d2_over_4_raw=d2_end / 4, d2_over_4_over_Ekappa=d2_end / 4 / Ekap, d2_over_4_over_voi_ref=d2_end / 4 / voi,
        se_prefix_over_voi_fixed_voi=float(d2_i.std(ddof=1) / np.sqrt(n) / 4 / voi),
        se_prefix_over_voi_delta_method=float(np.sqrt(np.var(d2_i - ratio * v_i, ddof=1) / n) / v_i.mean() / 4),
        mean_prefix_d2_matches_M=float(abs(d2_i.mean() - d2_end) / d2_end),
        exceeds_common_reference_margin=bool(d2_end / 4 > delta_ref), ratio_to_margin=float(d2_end / 4 / delta_ref))

    # --- continuous endpoint-midpoint certificate
    interval = [(r_hi, 1.0)]
    c_mid = 0.5 * (J_bot + J_top)
    sm = R.sup_risk(C, M, c_mid, interval)
    excess = sm["sup"] / (d2_end / 4) - 1
    cell["endpoint_midpoint"] = dict(sup_risk_raw=sm["sup"], d2_over_4_raw=d2_end / 4, excess_over_lower_bound_rel=excess, argmax_r=sm["argmax_r"],
                                     max_root_residual_rel=sm["max_root_residual_rel"], dense_grid_max_raw=sm["dense_grid_max"],
                                     dense_minus_sup_rel=sm["dense_minus_sup_rel"], certified_minimax_on_evaluation_measure=bool(excess <= 1e-9))
    cell["checks"]["root_residual_ok"] = sm["max_root_residual_rel"] <= TOL["root_residual"]

    # --- farthest compatible point from the reference truth (continuous worst case of the truth's own belief over the arc)
    J_truth = R.J_at(C, 1 - 2 * eta0)
    sf = R.sup_risk(C, M, J_truth, interval)
    cell["farthest_from_truth"] = dict(sup_risk_raw=sf["sup"], over_voi_ref=sf["sup"] / voi, over_Ekappa=sf["sup"] / Ekap, argmax_r=sf["argmax_r"],
                                       max_root_residual_rel=sf["max_root_residual_rel"],
                                       note="max over compatible parameters of E_P[kappa (mu_truth - mu_xi)^2]: the decision cost of acting on the truth's belief when another compatible world is true")
    cell["checks"]["root_residual_ok"] = bool(cell["checks"]["root_residual_ok"] and sf["max_root_residual_rel"] <= TOL["root_residual"])

    # --- Frank-Wolfe bracket
    fw = R.pairwise_fw(Mg)
    c_fw = fw["pi"] @ Jg
    s_fw = R.sup_risk(C, M, c_fw, interval)
    lb, ub = fw["value"], s_fw["sup"]
    cell["frank_wolfe"] = dict(grid_points=GRID, iterations=fw["iterations"], termination=fw["termination"], fw_gap_raw=fw["fw_gap"], tol_abs=fw["tol_abs"],
                               **R.support_report(fw["pi"], eta_g), lower_bound_raw=lb, upper_bound_raw=ub, bracket_gap_raw=ub - lb,
                               bracket_gap_rel=(ub - lb) / lb, upper_root_residual_rel=s_fw["max_root_residual_rel"],
                               lower_le_upper=bool(lb <= ub + 1e-12 * float(d.max())), noncertifiable_gap_over_1pct=bool((ub - lb) / lb > TOL["gap_rel"]),
                               endpoint_mass=float(fw["pi"][0] + fw["pi"][-1]))
    cell["checks"]["bracket_ok"] = bool(lb <= ub + 1e-12 * float(d.max()))   # absolute tolerance scaled by max diag(M_g): FW value is a difference of O(max d) terms
    cell["checks"]["root_residual_ok"] = bool(cell["checks"]["root_residual_ok"] and s_fw["max_root_residual_rel"] <= TOL["root_residual"])

    # --- Jung bound: V* <= n/(2(n+1)) diam^2, n = affine dimension of the set (<= 4 here)
    lam, U = np.linalg.eigh(M)
    sv = np.linalg.svd((U * np.sqrt(np.clip(lam, 0, None))).T @ (Jg - Jg[0]).T, compute_uv=False)
    n_eff = int((sv > 1e-9 * sv.max()).sum())
    diam2 = float(D2.max())
    cell["jung"] = dict(affine_dim_numerical=n_eff, upper_bound_raw_dim4=0.4 * diam2, upper_bound_raw_dim_eff=n_eff / (2 * (n_eff + 1)) * diam2,
                        lower_bound_raw=diam2 / 4)

    # --- descriptive grid quantities (not certificates)
    MUmin = np.full(psi.shape[0], np.inf)
    MUmax = np.full(psi.shape[0], -np.inf)
    for lo in range(0, GRID, 50):
        blk = psi @ Jg[lo:lo + 50].T
        MUmin = np.minimum(MUmin, blk.min(1))
        MUmax = np.maximum(MUmax, blk.max(1))
    b_pw = 0.5 * (MUmin + MUmax)
    naive = ex.naive.ravel()
    wk = kw / n
    worst_pw = worst_naive = 0.0
    for lo in range(0, GRID, 50):
        blk = psi @ Jg[lo:lo + 50].T
        worst_pw = max(worst_pw, float(((blk - b_pw[:, None]) ** 2 * wk[:, None]).sum(0).max()))
        worst_naive = max(worst_naive, float(((blk - naive[:, None]) ** 2 * wk[:, None]).sum(0).max()))
    sb = R.sup_risk(C, M, J_bot, interval)
    cell["descriptive_grid"] = dict(worst_pointwise_envelope_midpoint_raw=worst_pw, worst_bottom_endpoint_continuous_raw=sb["sup"], worst_naive_raw=worst_naive,
                                    note="grid (401 pts) quantities are descriptive, not certificates; the bottom-endpoint value is a continuous worst case")
    cell["checks"]["all_ok"] = bool(all(v for k, v in cell["checks"].items() if k.endswith("_ok")))
    return cell


def main():
    out = dict(tolerances=TOL, grid_points=GRID, cells={})
    ts_cache = {}
    bad = []
    arch = os.path.join(OUT, "archive_received_2026-10-11", "diam_se.json")
    archive = json.load(open(arch)) if os.path.exists(arch) else {}
    for pname in PHYS:
        for tname, truth in truths(3):
            c = certify_cell(pname, tname, truth, ts_cache)
            key = f"{pname}_L3_{tname}"
            if f"{pname}_{tname}" in archive and isinstance(c.get("lower_bound"), dict):
                a = archive[f"{pname}_{tname}"]
                c["reproduction_vs_received_archive"] = dict(archive_Vstar_raw=a["Vstar_raw"], here_d2_over_4_raw=c["lower_bound"]["d2_over_4_raw"],
                                                            rel_diff=abs(a["Vstar_raw"] - c["lower_bound"]["d2_over_4_raw"]) / a["Vstar_raw"])
            out["cells"][key] = c
            ok = c.get("checks", {}).get("all_ok", True)
            if not ok:
                bad.append(key)
            if "lower_bound" in c:
                lb = c["lower_bound"]
                print(f"{key:14s} arc 0..{c['arc']['eta_hi']:.5f}  d2/4/VOI {lb['d2_over_4_over_voi_ref']:.5f} (SE {lb['se_prefix_over_voi_delta_method']:.5f})  "
                      f"margin/VOI {c['margin_ref_over_voi']:.4f}  mid-sup excess {c['endpoint_midpoint']['excess_over_lower_bound_rel']:.2e}  "
                      f"FW LB..UB gap_rel {c['frank_wolfe']['bracket_gap_rel']:.2e}  hmm {c['checks']['max_abs_hmm_vs_four_function']:.1e}  ok={ok}")
            else:
                print(key, c["arc"])
    out["failed_cells"] = bad
    path = save_json("l3_certify.json", out, __file__)
    print("saved", path, "| FAILED:" if bad else "| all checks passed", bad if bad else "")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
