import sys, json
import numpy as np
from scipy.special import logsumexp
from ridgelib import *
def logp(e, w):
    with np.errstate(divide='ignore'):
        return logsumexp(e.ll + np.log(w)[None, :], axis=1)
res = {}
for pname in ("lead", "sec"):
    phys = PHYS[pname].with_(L=L)
    ts = TestSet(phys, N_EVAL, cell_id=0, master=MASTER_EVAL)
    ex = ExactRecords(ts); tl = TrueLaw(ts, ex); K = tl.K; voi = tl.voi_clean(); n, V = ts.n, ts.V
    for tname, truth in truths(L):
        w0 = truth.view_weights(); eta0 = phys.eta; m2t = tl.reference(truth)
        eh = eta_hi(w0, eta0)
        ends = {}
        for e2 in (0.0, eh):
            w2 = np.maximum(ridge_w(w0, eta0, e2), 0); w2 /= w2.sum()
            e = ex if abs(e2 - eta0) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(e2)))
            ends[e2] = (Precomputed.build(e.ll, e.mu).compose(w2).reshape(n, V), float(np.abs(logp(e, w2) - logp(ex, w0)).max()))
        lo, hi = ends[0.0][0], ends[eh][0]
        mid = 0.5 * (lo + hi)
        r = lambda b, ref: float(((K * (b - ref) ** 2) @ w0).mean()) / voi
        naive = ex.naive
        res[f"{pname}_{tname}"] = dict(eta_hi=round(eh, 5), llcheck_eta0=ends[0.0][1], llcheck_etahi=ends[eh][1],
            regret_at_truth={"midpoint": r(mid, m2t), "bottom(eta'=0)": r(lo, m2t), "top": r(hi, m2t), "naive": r(naive, m2t)},
            worst_case_over_endpoints={"midpoint": max(r(mid, lo), r(mid, hi)), "bottom": r(lo, hi), "naive": max(r(naive, lo), r(naive, hi))},
            quarter_diam_sq=0.25 * r(lo, hi))
        print(pname, tname, json.dumps(res[f"{pname}_{tname}"]))
json.dump(res, open('ridge_l3_extra.json', 'w'), indent=1)
