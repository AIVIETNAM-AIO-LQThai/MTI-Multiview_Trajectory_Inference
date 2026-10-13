import itertools, numpy as np
from scipy.optimize import linprog
rng = np.random.default_rng(0)
def check(L, k, eta=0.05, alts=(0.02, 0.035, 0.065, 0.08)):
    subsets = [b for b in itertools.product([0,1], repeat=L) if sum(b) <= k]
    D = lambda b: tuple((b[j] ^ b[j+1]) for j in range(L-1))
    fp = [D(b) for b in subsets]
    inj = len(set(fp)) == len(fp)
    pats = list(itertools.product([0,1], repeat=L-1)) if L > 1 else [()]
    full = set(fp) == set(pats)
    comp_pairs = sum(1 for b in subsets if tuple(1-x for x in b) in subsets and sum(b) < L - sum(b) or (tuple(1-x for x in b) in subsets and sum(b)==L-sum(b) and b < tuple(1-x for x in b)))
    # kernel K[pattern, subset] = P(apparent pattern | subset) under eta
    def K(e):
        M = np.zeros((len(pats), len(subsets)))
        for s, f in enumerate(fp):
            for p, x in enumerate(pats):
                d = [(x[i] + f[i]) % 2 for i in range(L-1)]
                M[p, s] = np.prod([e if di else 1-e for di in d]) if L > 1 else 1.0
        return M
    w = rng.dirichlet(np.ones(len(subsets)) * 2.0)            # interior channel over allowed subsets
    P = K(eta) @ w
    tvs = []
    for e2 in alts:
        K2 = K(e2); G, m = K2.shape
        c = np.r_[np.zeros(m), 0.5*np.ones(G)]
        A = np.block([[K2, -np.eye(G)], [-K2, -np.eye(G)]])
        r = linprog(c, A_ub=A, b_ub=np.r_[P, -P], A_eq=np.r_[np.ones(m), np.zeros(G)][None], b_eq=[1], bounds=[(0,None)]*(m+G), method="highs")
        tvs.append(r.fun)
    # decision relevance of complement pairs: last-position sign differs between b and complement (always, since complement flips every coordinate)
    return dict(L=L, k=k, n_subsets=len(subsets), cube=len(pats), injective=inj, image_full=full, complement_pairs_allowed=comp_pairs,
                pred_known=(L >= 2*k+1), pred_joint=(L >= 2*k+2), min_tv_alt=[round(t, 8) for t in tvs])
bad = 0
for k in (1, 2, 3):
    for L in range(2, 10):
        r = check(L, k)
        ok_known = r['injective'] == r['pred_known']
        ok_joint = ((not r['image_full']) and r['injective']) == r['pred_joint']
        ambiguous = min(r['min_tv_alt']) < 1e-9
        ok_lp = ambiguous == (not r['pred_joint'])
        bad += not (ok_known and ok_joint and ok_lp)
        print(r, 'OK' if ok_known and ok_joint and ok_lp else 'MISMATCH')
print('mismatches', bad)
