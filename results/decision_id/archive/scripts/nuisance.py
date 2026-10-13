import sys, numpy as np
from e4common import PHYS, truths
from mti.simulate import simulate, apply_channel, sample_theta
from mti.inference import flip_batch
from ridgelib import ridge_w, eta_hi, L as L3
for pname in ("lead", "sec"):
    phys = PHYS[pname]
    tname, truth = truths(8)[2]                                   # T4: beta=0.35 concentrated on last action
    n = 200_000
    smp = simulate(phys, n, 1234, 0, 0)
    S = apply_channel(smp.prefix, sample_theta(truth, n, 1234, 0, 0))
    s, a = S.s, S.a
    x, y = s[:, :-1].ravel(), s[:, 1:].ravel()
    rho = (x @ y) / (x @ x)
    d = (s[:, 1:] - rho * s[:, :-1]).ravel(); a2 = (a ** 2).ravel()
    qa = a2.mean(); c2 = np.cov(a2, d ** 2)[0, 1] / a2.var(ddof=1); qw = (d ** 2).mean() - c2 * qa
    print(pname, f"rho {rho:.4f} (0.9)  q_a {qa:.4f} (1)  c^2 {c2:.4f} (1)  q_w {qw:.4f} ({phys.q_w})  [beta=0.35 corrupted records]")
# trusted-outcome identity at L=3 on the ridge: E[u (y - rho s_L) | S]/q_u = c mu2(S) under the TRUE law; ridge alternatives give different mu2
phys = PHYS["lead"].with_(L=3); tname, truth = truths(3)[1]
n = 400_000
smp = simulate(phys, n, 4321, 0, 0)
th = sample_theta(truth, n, 4321, 0, 0); S = apply_channel(smp.prefix, th)
u, y = smp.probe.a, smp.probe.s_next
z = u * (y - phys.rho * S.s[:, -1]) / phys.q_a                     # unbiased for c*mu2(S)
from mti.adapt import Precomputed
def mu2_of(eta2):
    w2 = np.maximum(ridge_w(truth.view_weights(), phys.eta, eta2), 0); w2 /= w2.sum()
    ll, mu = flip_batch(S, phys.with_(eta=float(eta2)))
    return Precomputed.build(ll, mu).compose(w2)
for e2 in (0.0, 0.05, eta_hi(truth.view_weights(), phys.eta)):
    m = mu2_of(e2)
    # projection test: regress z on the candidate's c*mu2; under the truth the coefficient is 1 and residual mean-square is minimal
    print(f"eta'={e2:.4f}: mean (z - c mu2)^2 = {np.mean((z - phys.c * m) ** 2):.5f}   slope of z on mu2 = {np.dot(z, m) / np.dot(m, m):.4f}")
