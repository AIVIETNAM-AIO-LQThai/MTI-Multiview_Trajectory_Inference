import json, numpy as np
from ridgelib import *
out = {}
for pname in ("lead", "sec"):
    phys = PHYS[pname].with_(L=L)
    ts = TestSet(phys, N_EVAL, cell_id=0, master=MASTER_EVAL)
    ex = ExactRecords(ts); tl = TrueLaw(ts, ex); K = tl.K; voi = tl.voi_clean(); n, V = ts.n, ts.V
    Ekap = float(ts.kap.mean())
    for tname, truth in truths(L):
        w0 = truth.view_weights(); eh = eta_hi(w0, phys.eta)
        mus = []
        for e2 in (0.0, eh):
            w2 = np.maximum(ridge_w(w0, phys.eta, e2), 0); w2 /= w2.sum()
            e = ex if abs(e2 - phys.eta) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(e2)))
            mus.append(Precomputed.build(e.ll, e.mu).compose(w2).reshape(n, V))
        d_i = (K * (mus[0] - mus[1]) ** 2) @ w0                 # per-prefix decision diameter contribution (all views of a prefix = one unit)
        v_i = tl.kap * tl.muH ** 2
        D = d_i.mean(); r = D / voi
        se_fixed = d_i.std(ddof=1) / np.sqrt(n) / voi
        se_ratio = np.sqrt(np.var(d_i - r * v_i, ddof=1) / n) / voi
        out[f"{pname}_{tname}"] = dict(eta_hi=eh, Vstar_raw=D / 4, Vstar_over_voi=r / 4, se_Vstar_over_voi_fixedvoi=se_fixed / 4,
                                      se_Vstar_over_voi_ratio=se_ratio / 4, Vstar_over_Ekappa=D / 4 / Ekap, Ekappa=Ekap, voi=voi)
        print(pname, tname, {k: round(v, 5) for k, v in out[f"{pname}_{tname}"].items()})
json.dump(out, open('diam_se.json', 'w'), indent=1)
