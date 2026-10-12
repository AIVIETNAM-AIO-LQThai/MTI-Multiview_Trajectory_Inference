"""Channel-invariant identification of the physics and a trusted-outcome check (closeout appendix; discussion sections 5.3-5.4). No fitting of models;
moment estimators only, on freshly simulated records with masters 1234 (nuisance) and 4321 (trusted outcomes), unused by E4.

Part 1. Channel-invariant moments (states and magnitudes are intact under the channel): rho = E[s_j s_{j+1}]/E[s_j^2], q_a = E[a^2],
        c^2 = Cov(a^2, delta^2)/Var(a^2), q_w = E[delta^2] - c^2 q_a. Checked on 2e5 records with beta = 0.35 on the last action (E4 truth T4, L = 8).
Part 2. Trusted outcome at L = 3 (lead physics, truth T3): z = u (y - rho s_L)/q_a has conditional mean c mu_2(S) under the TRUE law (u = clean probe action,
        independent of m_L given S; y = next state). Regressing z on candidate beliefs along the identified arc: slope ~ 1 at the truth, different at the arc ends.

    python experiments/decision_id/nuisance.py   ->  results/decision_id/nuisance.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridgelib as R  # noqa: E402
from did_common import PHYS, save_json, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.inference import flip_batch  # noqa: E402
from mti.simulate import apply_channel, sample_theta, simulate  # noqa: E402


def main():
    out = dict(physics_moments={}, trusted_outcome={})
    for pname in ("lead", "sec"):
        phys = PHYS[pname]
        tname, truth = truths(8)[2]
        n = 200_000
        smp = simulate(phys, n, 1234, 0, 0)
        S = apply_channel(smp.prefix, sample_theta(truth, n, 1234, 0, 0))
        s, a = S.s, S.a
        x, y = s[:, :-1].ravel(), s[:, 1:].ravel()
        rho = (x @ y) / (x @ x)
        d = (s[:, 1:] - rho * s[:, :-1]).ravel()
        a2 = (a ** 2).ravel()
        qa = a2.mean()
        c2 = np.cov(a2, d ** 2)[0, 1] / a2.var(ddof=1)
        qw = (d ** 2).mean() - c2 * qa
        out["physics_moments"][pname] = dict(truth=tname, n=n, rho=float(rho), rho_true=phys.rho, q_a=float(qa), q_a_true=phys.q_a, c2=float(c2), c2_true=phys.c ** 2,
                                              q_w=float(qw), q_w_true=phys.q_w)
        print(pname, f"rho {rho:.4f} (0.9)  q_a {qa:.4f} (1)  c^2 {c2:.4f} (1)  q_w {qw:.4f} ({phys.q_w})  [beta=0.35 corrupted records]")
    phys = PHYS["lead"].with_(L=3)
    tname, truth = truths(3)[1]
    n = 400_000
    smp = simulate(phys, n, 4321, 0, 0)
    th = sample_theta(truth, n, 4321, 0, 0)
    S = apply_channel(smp.prefix, th)
    u, y = smp.probe.a, smp.probe.s_next
    z = u * (y - phys.rho * S.s[:, -1]) / phys.q_a
    w0 = truth.view_weights()
    eh, _ = R.eta_hi(w0, phys.eta)
    rows = []
    for e2 in (0.0, 0.05, eh):
        w2 = np.maximum(R.ridge_w(w0, phys.eta, e2), 0)
        w2 /= w2.sum()
        ll, mu = flip_batch(S, phys.with_(eta=float(e2)))
        m = Precomputed.build(ll, mu).compose(w2)
        rows.append(dict(eta_prime=float(e2), mean_sq_resid=float(np.mean((z - phys.c * m) ** 2)), slope_z_on_mu2=float(np.dot(z, m) / np.dot(m, m))))
        print(f"eta'={e2:.4f}: mean (z - c mu2)^2 = {rows[-1]['mean_sq_resid']:.5f}   slope of z on mu2 = {rows[-1]['slope_z_on_mu2']:.4f}")
    out["trusted_outcome"] = dict(truth=tname, physics="lead", L=3, n=n, arc_points=rows)
    print("saved", save_json("nuisance.json", out, __file__))


if __name__ == "__main__":
    main()
