"""Evaluation-uncertainty accounting for the E4 P1 contrasts (L = 8, T0, n = 1e4), from the saved estimates (closeout spec II.5). Evaluation only.

Notation: D[r, i] = per-prefix contrast of replicate r (fit/adaptation set) on evaluation prefix i; R replicates, N prefixes (one prefix with its views is one unit).
Reported (all over VOI_clean of the reference law; VOI-ratio versions use the delta method with VOI estimated on the same prefixes):
  (a) rep interval        : 90% t-interval over replicates, conditional on the fixed evaluation set (fit uncertainty only).
  (b) shared-prefix SE    : sd_i(Dbar_i)/sqrt(N)  (replicate-average contrast, fits and VOI fixed). Identity: SE_shared^2 = SE_indep^2 +
                            (R^2 N)^-1 sum_{r != r'} Cov_i(D_r, D_r'), so the independent-set figure is NOT a bound in general.
  (c) independent-set approx: sqrt(sum_r Var_i(D_r)/N)/R  (treats every replicate as having its own independent prefixes; an approximation, not a bound).
  (d) delta method        : ratio-of-means SE with VOI estimated on the same prefixes.
  (e) crossed ANOVA       : two-way random-effects model D_ri = mu + a_r + b_i + e_ri with independently sampled fits (rows) and prefixes (columns):
                            Var(grand mean) = sigma_a^2/R + sigma_b^2/N + sigma_e^2/(RN), estimated unbiasedly by (MS_R + MS_C - MS_E)/(RN).
                            Assumptions: additive effects, rows and columns exchangeable and independent. Negative estimates are floored at 0 and flagged.
  (f) old "crossed" figure: s^2(replicate means)/R + s^2(prefix means)/N = (MS_R + MS_C)/(RN): it adds sigma_e^2/(RN) in expectation, i.e. double
                            counts the interaction; kept only to show the difference.
Also saved: reproduction of the saved regrets (max relative error), cross-replicate correlation summary.

    python experiments/decision_id/se_e4.py   ->  results/decision_id/se_e4.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from did_common import N_EVAL, PHYS, ROOT, TestSet, save_json, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.misspec import ExactRecords, TrueLaw  # noqa: E402

N_P1 = 10000
ARMS = ("J-MLE", "EX-MLE", "ETA--MLE")


def crossed_anova(D):
    """Unbiased variance estimate of the grand mean of a two-way random-effects layout. Returns dict(var, var_floored, flagged_negative, MS_R, MS_C, MS_E)."""
    R_, N = D.shape
    grand = D.mean()
    rows, cols = D.mean(1), D.mean(0)
    ms_r = N * np.sum((rows - grand) ** 2) / (R_ - 1)
    ms_c = R_ * np.sum((cols - grand) ** 2) / (N - 1)
    res = D - rows[:, None] - cols[None, :] + grand
    ms_e = np.sum(res ** 2) / ((R_ - 1) * (N - 1))
    var = (ms_r + ms_c - ms_e) / (R_ * N)
    return dict(var=float(var), var_floored=float(max(var, 0.0)), flagged_negative=bool(var < 0), MS_R=float(ms_r), MS_C=float(ms_c), MS_E=float(ms_e))


def main():
    out = {}
    for pname in PHYS:
        phys = PHYS[pname]
        L = phys.L
        tname, truth = truths(L)[0]
        wt = truth.view_weights()
        ts = TestSet(phys, N_EVAL, cell_id=0, master=8601)
        n, V = ts.n, ts.V
        ex = ExactRecords(ts)
        tl = TrueLaw(ts, ex)
        K = tl.K
        m2t = tl.reference(truth)
        voi = tl.voi_clean()
        v_i = tl.kap * tl.muH ** 2
        cache = {}

        def pre_at(eta):
            k = round(float(eta), 9)
            if k not in cache:
                e = ex if abs(eta - phys.eta) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(eta)))
                cache[k] = Precomputed.build(e.ll, e.mu)
            return cache[k]

        files = sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage1", "units", f"{pname}_L{L}_T0_r*.json")), key=lambda f: int(f.rsplit("_r", 1)[-1][:-5]))
        Y = {a: [] for a in ARMS}
        relerr = []
        for f in files:
            u = json.load(open(f))
            for a in ARMS:
                arm = u["arms"][a][str(N_P1)]
                y = (K * (pre_at(arm["eta"]).compose(np.asarray(arm["w"])).reshape(n, V) - m2t) ** 2) @ wt
                Y[a].append(y)
                relerr.append(abs(y.mean() - arm["regret"]) / arm["regret"])
            assert abs(u["voi"] - voi) / voi < 1e-12
        Y = {a: np.array(v) for a, v in Y.items()}
        R_ = len(files)
        res = dict(R=R_, N=n, voi=voi, max_rel_err_vs_saved=float(max(relerr)))
        tq = stats.t.ppf(0.95, R_ - 1)
        for cname, D in (("J_minus_EX", Y["J-MLE"] - Y["EX-MLE"]), ("ETAm_minus_J", Y["ETA--MLE"] - Y["J-MLE"])):
            rep_means, Dbar = D.mean(1), D.mean(0)
            est = Dbar.mean()
            ratio = est / voi
            se_rep = rep_means.std(ddof=1) / np.sqrt(R_)
            se_shared = Dbar.std(ddof=1) / np.sqrt(n)
            se_indep = np.sqrt(np.sum(D.var(1, ddof=1) / n)) / R_
            C = np.corrcoef(D)
            offd = C[~np.eye(R_, dtype=bool)]
            se_delta = np.sqrt(np.var(Dbar - ratio * v_i, ddof=1) / n) / voi
            an = crossed_anova(D)
            an_lin = crossed_anova(D - ratio * v_i[None, :])          # linearised numerator/denominator on the same prefixes (shared denominator)
            old = np.sqrt(rep_means.var(ddof=1) / R_ + Dbar.var(ddof=1) / n)
            res[cname] = dict(
                mean_over_voi=ratio, rep_interval90_over_voi=[(est - tq * se_rep) / voi, (est + tq * se_rep) / voi], se_rep_over_voi=se_rep / voi,
                se_shared_prefix_fixed_fits_fixed_voi=se_shared / voi, se_independent_set_approximation_not_a_bound=se_indep / voi, shared_over_indep=se_shared / se_indep,
                mean_cross_replicate_corr=float(offd.mean()), min_cross_replicate_corr=float(offd.min()), max_cross_replicate_corr=float(offd.max()),
                se_delta_method_shared_prefix=se_delta, se_crossed_anova_fixed_voi=float(np.sqrt(an["var_floored"]) / voi), crossed_anova=an,
                se_crossed_anova_delta_method=float(np.sqrt(an_lin["var_floored"]) / voi), crossed_anova_delta=an_lin,
                se_old_crossed_pigeonhole_double_counts_interaction=float(old / voi))
        out[pname] = res
        print(pname, json.dumps({k: (v if not isinstance(v, dict) else {a: (round(b, 7) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, (dict, list))}) for k, v in res.items()}, indent=1))
    print("saved", save_json("se_e4.json", out, __file__))


if __name__ == "__main__":
    main()
