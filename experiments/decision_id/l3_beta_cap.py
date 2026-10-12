"""One explicit family of side information at L = 3: a declared cap beta' <= beta_bar on the corruption rate (closeout spec II.4). Exact-law / evaluation only.

For each reference truth (T3, T4, T6; both physics) and a grid of caps beta_bar from beta_true to the largest beta' on the arc:
  * the restricted compatible set {eta' in [0, eta_hi] : beta'(eta') <= beta_bar} (computed exactly as a union of intervals; monotonicity is not assumed),
  * whether the truth is retained (always, since beta_bar >= beta_true),
  * the restricted decision bracket: lower bounds (largest pairwise d^2/4 on the set; Frank-Wolfe Bayes value) and the certified upper bound (continuous
    worst case of the Frank-Wolfe mixture rule over the restricted set, via the degree <= 8 polynomial), with the gap,
  * the smallest cap in the grid at which the certified upper bound falls below the common reference margin.
A cap BOUNDS the worst-case decision cost; it does not by itself make the decision identified (the residual set is reported).

    python experiments/decision_id/l3_beta_cap.py   ->  results/decision_id/l3_beta_cap.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridgelib as R  # noqa: E402
from did_common import N_EVAL, PHYS, TestSet, reference_margin, save_json, truths  # noqa: E402

from mti.misspec import ExactRecords, TrueLaw  # noqa: E402

N_CAPS = 25
PER_INTERVAL = 201
FINE = 4001


def beta_of(w0, eta0, e2):
    w2 = R.ridge_w(w0, eta0, e2)
    return float(1 - w2[0])


def restricted_intervals(w0, eta0, eh, cap):
    """Exact union of eta'-intervals within [0, eh] where beta'(eta') <= cap (grid detection, endpoints refined by bisection)."""
    g = np.linspace(0.0, eh, FINE)
    ok = np.array([beta_of(w0, eta0, e) <= cap + 1e-13 for e in g])
    out, i = [], 0
    while i < len(g):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(g) and ok[j + 1]:
            j += 1
        lo, hi = g[i], g[j]
        if i > 0:
            a, b = g[i - 1], g[i]
            for _ in range(80):
                m = 0.5 * (a + b)
                (a, b) = (a, m) if beta_of(w0, eta0, m) <= cap + 1e-13 else (m, b)
            lo = b
        if j < len(g) - 1:
            a, b = g[j], g[j + 1]
            for _ in range(80):
                m = 0.5 * (a + b)
                (a, b) = (m, b) if beta_of(w0, eta0, m) <= cap + 1e-13 else (a, m)
            hi = a
        out.append((float(lo), float(hi)))
        i = j + 1
    return out


def cell(pname, tname, truth, cache):
    phys = PHYS[pname].with_(L=3)
    ts = cache.setdefault(pname, TestSet(phys, N_EVAL, cell_id=0, master=8601))
    ex = cache.setdefault(pname + "_ex", ExactRecords(ts))
    tl = cache.setdefault(pname + "_tl", TrueLaw(ts, ex))
    n, V = ts.n, ts.V
    w0, eta0 = truth.view_weights(), phys.eta
    delta_ref, V2_ref, voi = reference_margin(phys, truth, ts, ex, tl)
    Ekap = float(ts.kap.mean())
    eh, _ = R.eta_hi(w0, eta0)
    Phat = R.phat_from(w0, eta0)
    C = R.J_coef(Phat)
    psi, _ = R.psi_features(ts.s_rec, ts.a_rec, phys.c, phys.rho, phys.q_w, Phat)
    kw = np.repeat(ts.kap, V) * np.tile(w0, n)
    M = (psi * kw[:, None]).T @ psi / n
    fine = np.linspace(0.0, eh, FINE)
    bfine = np.array([beta_of(w0, eta0, e) for e in fine])
    beta_true, beta_max = float(1 - w0[0]), float(bfine.max())
    caps = np.linspace(beta_true, beta_max, N_CAPS)
    rows, first_ub = [], None
    for cap in caps:
        ivs = restricted_intervals(w0, eta0, eh, cap)
        etas = np.unique(np.concatenate([np.linspace(lo, hi, PER_INTERVAL) if hi > lo else np.array([lo]) for lo, hi in ivs]))
        Jg = np.array([R.J_at(C, 1 - 2 * e) for e in etas])
        Mg = Jg @ M @ Jg.T
        d = np.diag(Mg)
        D2 = d[:, None] + d[None, :] - 2 * Mg
        fw = R.pairwise_fw(Mg)
        r_ivs = [(1 - 2 * hi, 1 - 2 * lo) for lo, hi in ivs]
        ub = R.sup_risk(C, M, fw["pi"] @ Jg, r_ivs)
        lb_fw, ubv = fw["value"], ub["sup"]
        row = dict(beta_cap=float(cap), intervals_eta=ivs, n_intervals=len(ivs), truth_retained=bool(any(lo - 1e-12 <= eta0 <= hi + 1e-12 for lo, hi in ivs)),
                   residual_set_is_point=bool(len(ivs) == 1 and ivs[0][1] - ivs[0][0] < 1e-12), pair_lower_bound_raw=float(D2.max() / 4),
                   fw_lower_bound_raw=lb_fw, upper_bound_raw=ubv, bracket_gap_raw=ubv - lb_fw, fw_gap_raw=fw["fw_gap"], fw_iterations=fw["iterations"],
                   fw_termination=fw["termination"], upper_root_residual_rel=ub["max_root_residual_rel"],
                   upper_over_margin=ubv / delta_ref, lower_over_margin=lb_fw / delta_ref, upper_below_margin=bool(ubv < delta_ref),
                   lower_below_margin=bool(lb_fw < delta_ref), lower_le_upper=bool(lb_fw <= ubv + 1e-12 * float(d.max())), abs_scale_for_tolerance=float(d.max()))
        rows.append(row)
        if first_ub is None and row["upper_below_margin"]:
            first_ub = float(cap)
    ok_caps = [r["beta_cap"] for r in rows if r["upper_below_margin"]]
    largest = float(max(ok_caps)) if ok_caps else None
    cross = None
    for r0, r1 in zip(rows[:-1], rows[1:]):
        if r0["upper_below_margin"] and not r1["upper_below_margin"]:
            a0, a1 = r0["upper_over_margin"], r1["upper_over_margin"]
            cross = float(r0["beta_cap"] + (1 - a0) / (a1 - a0) * (r1["beta_cap"] - r0["beta_cap"]))
            break
    return dict(physics=pname, truth=tname, eta=eta0, beta_true=beta_true, beta_max_on_arc=beta_max, eta_hi=eh, voi_ref=voi, margin_ref_raw=delta_ref,
                margin_ref_over_voi=delta_ref / voi, E_kappa=Ekap,
                beta_monotone_decreasing_in_eta=bool(np.all(np.diff(bfine) <= 1e-12)), smallest_cap_with_certified_upper_below_margin=first_ub,
                largest_grid_cap_with_certified_upper_below_margin=largest, interpolated_cap_where_upper_bound_crosses_margin=cross,
                note_nested_sets="restricted sets are nested, so the upper bound is nondecreasing in the cap and the smallest such cap is trivially beta_true; the informative statistic is the largest cap",
                full_arc_lower_bound_over_margin=rows[-1]["fw_lower_bound_raw"] / delta_ref, rows=rows)


def main():
    out = dict(n_caps=N_CAPS, per_interval_points=PER_INTERVAL, cells={})
    cache, bad = {}, []
    for pname in PHYS:
        for tname, truth in truths(3):
            if tname == "T0":
                continue
            c = cell(pname, tname, truth, cache)
            key = f"{pname}_L3_{tname}"
            out["cells"][key] = c
            if not all(r["truth_retained"] and r["lower_le_upper"] and r["upper_root_residual_rel"] <= 1e-8 for r in c["rows"]):
                bad.append(key)
            r0, r1 = c["rows"][0], c["rows"][-1]
            print(f"{key:12s} beta {c['beta_true']:.3f}..{c['beta_max_on_arc']:.3f} monotone={c['beta_monotone_decreasing_in_eta']} "
                  f"UB/margin at cap=beta_true {r0['upper_over_margin']:.3f} (point={r0['residual_set_is_point']}), at full arc {r1['upper_over_margin']:.2f}; "
                  f"largest grid cap with UB<margin: {c['largest_grid_cap_with_certified_upper_below_margin']} (crossing ~{c['interpolated_cap_where_upper_bound_crosses_margin']})")
    out["failed_cells"] = bad
    print("saved", save_json("l3_beta_cap.json", out, __file__), "| FAILED:" if bad else "| all checks passed", bad or "")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
