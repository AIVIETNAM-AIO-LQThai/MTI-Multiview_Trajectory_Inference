"""Shared-prefix evaluation SE for the E4 P1 contrasts (L=8, T0, n=1e4), from the saved estimates; evaluation only.
Reports: saved-regret reproduction; (a) replicate interval conditional on the fixed evaluation set; (b) shared-prefix SE of the
replicate-average contrast with fits fixed, sd_i(Dbar_i)/sqrt(N); (c) the script's independent-set approximation; (d) mean pairwise
cross-replicate correlation of prefix-level contrasts; (e) ratio-of-means SE with VOI_clean estimated on the same prefixes (delta method);
(f) crossed (pigeonhole) total SE combining replicate and prefix variation."""
import glob, json, os
import numpy as np
from scipy import stats
from e4common import PHYS, truths, MASTER_EVAL, N_EVAL, TestSet
from mti.misspec import ExactRecords, TrueLaw
from mti.adapt import Precomputed
ROOT = '/home/claude/mti'; N_P1 = 10000; ARMS = ("J-MLE", "EX-MLE", "ETA--MLE")
out = {}
for pname in PHYS:
    phys = PHYS[pname]; L = phys.L
    tname, truth = truths(L)[0]; wt = truth.view_weights()
    ts = TestSet(phys, N_EVAL, cell_id=0, master=MASTER_EVAL); n, V = ts.n, ts.V
    ex = ExactRecords(ts); tl = TrueLaw(ts, ex); K = tl.K; m2t = tl.reference(truth); voi = tl.voi_clean()
    v_i = tl.kap * tl.muH ** 2                                   # per-prefix VOI_clean contribution
    cache = {}
    def pre_at(eta):
        k = round(float(eta), 9)
        if k not in cache:
            e = ex if abs(eta - phys.eta) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(eta)))
            cache[k] = Precomputed.build(e.ll, e.mu)
        return cache[k]
    files = sorted(glob.glob(f"{ROOT}/results/e4/stage1/units/{pname}_L{L}_T0_r*.json"), key=lambda f: int(f.split('_r')[-1][:-5]))
    Y = {a: [] for a in ARMS}; relerr = []
    for f in files:
        u = json.load(open(f))
        for a in ARMS:
            arm = u["arms"][a][str(N_P1)]
            y = (K * (pre_at(arm["eta"]).compose(np.asarray(arm["w"])).reshape(n, V) - m2t) ** 2) @ wt
            Y[a].append(y); relerr.append(abs(y.mean() - arm["regret"]) / arm["regret"])
        assert abs(u["voi"] - voi) / voi < 1e-12
    Y = {a: np.array(v) for a, v in Y.items()}; R = len(files)
    res = dict(R=R, N=n, voi=voi, max_rel_err_vs_saved=float(max(relerr)))
    for cname, D in (("J_minus_EX", Y["J-MLE"] - Y["EX-MLE"]), ("ETAm_minus_J", Y["ETA--MLE"] - Y["J-MLE"])):
        rep_means = D.mean(1)                                     # (R,)
        Dbar = D.mean(0)                                          # (N,)
        se_shared = Dbar.std(ddof=1) / np.sqrt(n)
        se_indep = np.sqrt(np.sum(D.var(1, ddof=1) / n)) / R
        C = np.corrcoef(D); offd = C[~np.eye(R, dtype=bool)]
        tq = stats.t.ppf(0.95, R - 1)
        se_rep = rep_means.std(ddof=1) / np.sqrt(R)
        est = Dbar.mean()
        ratio = est / voi
        se_ratio = np.sqrt(np.var(Dbar - ratio * v_i, ddof=1) / n) / voi   # delta method, numerator and denominator on same prefixes
        se_cross = np.sqrt(rep_means.var(ddof=1) / R + Dbar.var(ddof=1) / n)  # pigeonhole-style (slightly conservative)
        res[cname] = dict(mean_over_voi=ratio, rep_interval90_over_voi=[(est - tq * se_rep) / voi, (est + tq * se_rep) / voi],
                          se_rep_over_voi=se_rep / voi, se_shared_fixedfits_fixedvoi=se_shared / voi, se_indep_approx=se_indep / voi,
                          shared_over_indep=se_shared / se_indep, mean_cross_rep_corr=float(offd.mean()), min_cross_rep_corr=float(offd.min()),
                          se_ratio_delta_method=se_ratio, se_crossed_total_over_voi=se_cross / voi)
    out[pname] = res
    print(pname, json.dumps(res, indent=1))
json.dump(out, open('se_shared.json', 'w'), indent=1)
