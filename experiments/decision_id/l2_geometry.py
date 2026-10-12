"""L = 2 decision geometry on the E4 evaluation measure (closeout spec II.2). Exact-law / evaluation only.

For both physics and the E4 truths at L = 2 (identified h = r(1 - 2 beta) > 0): the compatible target triangle, its metric G = E_P[kappa phi phi'],
the minimax value V* (circumradius^2 when the triangle is acute in the G metric, else 1/4 longest side^2), the least-favourable prior, the endpoint-pair
lower bound d^2/4, the Jung bound (1/3) diam^2, the worst risk of the farthest-pair midpoint, and the eta-known segment (V* = d^2/4 exactly).
Checks: belief formula phi.(u,v) vs the HMM and record-law equality for the three vertex worlds and the truth; G cross-check on a fresh Monte Carlo
sample (master 1234, 1e6 records; the evaluation measure itself uses master 8601, cell 0, 20,000 prefixes).

    python experiments/decision_id/l2_geometry.py   ->  results/decision_id/l2_geometry.json
"""
import os
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2lib as T  # noqa: E402
from did_common import N_EVAL, PHYS, TestSet, reference_margin, save_json, truths  # noqa: E402

from mti.decision import kappa  # noqa: E402
from mti.misspec import ExactRecords, TrueLaw  # noqa: E402
from mti.params import ChannelSpec  # noqa: E402
from mti.simulate import apply_channel, sample_theta, simulate  # noqa: E402

TOL = dict(hmm=1e-12, loglik=1e-12)


def chan_of(q0, q1):
    b = q0 + q1
    return ChannelSpec(b, (0.5, 0.5) if b == 0 else (q0 / b, q1 / b))


def phi_of(s, a, phys, h):
    z = np.tanh(phys.c * a * (s[:, 1:] - phys.rho * s[:, :-1]) / phys.q_w)
    return np.stack([z[:, 1], z[:, 0]], axis=1) / (1 + h * z[:, 0] * z[:, 1])[:, None]


def logp(e, w):
    with np.errstate(divide="ignore"):
        return logsumexp(e.ll + np.log(w)[None, :], axis=1)


def cell(pname, tname, truth, cache, mc_records):
    phys = PHYS[pname].with_(L=2)
    ts = cache.setdefault(pname, TestSet(phys, N_EVAL, cell_id=0, master=8601))
    ex = cache.setdefault(pname + "_ex", ExactRecords(ts))
    tl = cache.setdefault(pname + "_tl", TrueLaw(ts, ex))
    n, V = ts.n, ts.V
    w0, eta0 = truth.view_weights(), phys.eta
    q0, q1 = truth.q
    h = T.h_of(eta0, q0, q1)
    assert 0 < h < 1
    delta_ref, V2_ref, voi = reference_margin(phys, truth, ts, ex, tl)
    Ekap = float(ts.kap.mean())
    kw = np.repeat(ts.kap, V) * np.tile(w0, n)
    phi = phi_of(ts.s_rec, ts.a_rec, phys, h)
    G = (phi * kw[:, None]).T @ phi / n
    X = T.vertices(h)
    uv_truth = T.uv_of(eta0, q0, q1)
    # --- checks: formula vs HMM and record-law equality (truth and the three vertex worlds)
    worlds = dict(truth=(eta0, q0, q1), **T.vertex_worlds(h))
    llref = logp(ex, w0)
    chk = {}
    for nm, (e, a0, a1) in worlds.items():
        wl = np.array([1 - a0 - a1, a0, a1])
        er = ex if abs(e - eta0) < 1e-12 and nm == "truth" else ExactRecords(ts, phys.with_(eta=float(e)))
        mu_hmm = er.mu2(chan_of(a0, a1)).ravel()
        mu_f = phi @ T.uv_of(e, a0, a1)
        chk[nm] = dict(eta=e, q0=a0, q1=a1, uv=T.uv_of(e, a0, a1), max_abs_formula_vs_hmm=float(np.abs(mu_hmm - mu_f).max()),
                       max_abs_loglik_diff_vs_truth=float(np.abs(logp(er, wl) - llref).max()))
    # --- fresh Monte Carlo cross-check of G (independent sample, master 1234)
    nmc = mc_records
    smp = simulate(phys, nmc, 1234, 0, 0)
    S = apply_channel(smp.prefix, sample_theta(truth, nmc, 1234, 0, 0))
    kap_mc = kappa(phys, smp.prefix.s[:, -1])
    phi_mc = phi_of(S.s, S.a, phys, h)
    G_mc = (phi_mc * kap_mc[:, None]).T @ phi_mc / nmc
    # --- minimax on the triangle and the eta-known segment
    tm = T.triangle_minimax(X, G)
    names = "ABC"
    diam2 = tm["max_side_sq"]
    cosines = {}
    for o in range(3):
        i, j = [k for k in range(3) if k != o]
        cosines["ABC"[o]] = float(tm["angle_inner"]["ABC"[o]] / np.sqrt(T.metric_d2(X[i], X[o], G) * T.metric_d2(X[j], X[o], G))) if T.metric_d2(X[i], X[o], G) > 0 and T.metric_d2(X[j], X[o], G) > 0 else float("nan")
    tm_mc = T.triangle_minimax(X, G_mc)
    mid_worst = max(T.metric_d2((X[i] + X[j]) / 2, X[k], G) for i, j in [max(tm["d2"], key=tm["d2"].get)] for k in range(3))
    r = 1 - 2 * eta0
    seg = np.array([[h, r * r], [r, r * h]])
    d2_seg = T.metric_d2(seg[0], seg[1], G)
    out = dict(
        physics=pname, truth=tname, eta=eta0, beta=truth.beta, h=h, r=r, voi_ref=voi, margin_ref_raw=delta_ref, margin_ref_over_voi=delta_ref / voi, E_kappa=Ekap,
        n_prefixes=n, truth_uv=uv_truth, truth_in_triangle=T.in_triangle(uv_truth, h), vertices={names[i]: X[i] for i in range(3)},
        vertex_worlds={k: dict(eta=v[0], q0=v[1], q1=v[2]) for k, v in T.vertex_worlds(h).items()}, G=G, G_monte_carlo_1e6=G_mc,
        G_rel_diff_mc=float(np.abs(G - G_mc).max() / np.abs(G).max()), angle_inner_products=tm["angle_inner"], angle_cosines=cosines, checks=chk,
        minimax_on_fresh_mc_G=dict(kind=tm_mc["kind"], V_star_raw=tm_mc["value"], least_favourable_weights={"ABC"[i]: float(tm_mc["weights"][i]) for i in range(3)}),
        minimax=dict(kind=tm["kind"], V_star_raw=tm["value"], center_uv=tm["center"], least_favourable_weights={names[i]: float(tm["weights"][i]) for i in range(3)},
                     n_support=int((tm["weights"] > 1e-9).sum())),
        endpoint_pair_lower_bound=dict(d2_over_4_raw=diam2 / 4, V_over_lower=tm["value"] / (diam2 / 4)),
        jung_upper_bound_raw=diam2 / 3, farthest_pair_midpoint_worst_raw=mid_worst,
        eta_known_segment=dict(endpoints_uv=seg, V_star_raw=d2_seg / 4, note="affine dimension 1: V* = d^2/4 exactly; the endpoint midpoint is minimax"),
        normalised=dict(V_star_over_Ekappa=tm["value"] / Ekap, V_star_over_voi_ref=tm["value"] / voi, lower_over_voi_ref=diam2 / 4 / voi,
                        segment_V_over_voi_ref=d2_seg / 4 / voi, V_star_over_margin=tm["value"] / delta_ref, segment_V_over_margin=d2_seg / 4 / delta_ref))
    out["checks_ok"] = bool(all(c["max_abs_formula_vs_hmm"] <= TOL["hmm"] and c["max_abs_loglik_diff_vs_truth"] <= TOL["loglik"] for c in chk.values()) and out["truth_in_triangle"])
    return out


def main():
    mc = 1_000_000
    res = dict(tolerances=TOL, cells={})
    cache, bad = {}, []
    for pname in PHYS:
        for tname, truth in truths(2):
            c = cell(pname, tname, truth, cache, mc)
            key = f"{pname}_L2_{tname}"
            res["cells"][key] = c
            if not c["checks_ok"]:
                bad.append(key)
            mm = c["minimax"]
            print(f"{key:14s} h={c['h']:.2f} kind={mm['kind']:12s} LFP={[round(v, 3) for v in mm['least_favourable_weights'].values()]} "
                  f"V*={mm['V_star_raw']:.5f} d2/4={c['endpoint_pair_lower_bound']['d2_over_4_raw']:.5f} (x{c['endpoint_pair_lower_bound']['V_over_lower']:.3f}) "
                  f"midpoint-worst={c['farthest_pair_midpoint_worst_raw']:.5f} seg V*={c['eta_known_segment']['V_star_raw']:.5f} "
                  f"G-MC diff {c['G_rel_diff_mc']:.1e} ok={c['checks_ok']}")
    res["failed_cells"] = bad
    print("saved", save_json("l2_geometry.json", res, __file__), "| FAILED:" if bad else "| all checks passed", bad or "")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
