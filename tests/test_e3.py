"""E3: identification controls and estimators (protocol docs/e3_protocol.md section 6)."""
import inspect

import numpy as np

from _record import rec
from mti import adapt
from mti.adapt import Precomputed, chan_from_w, em_channel, mom_channel, pp_channel
from mti.inference import candidate_from_queries, flip_batch
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate


def phys(L, qw=0.02, eta=0.05):
    return Physics(0.9, 1.0, 1.0, qw, 0.1, eta, L)


def records(p, chan, n, master):
    smp = simulate(p, n, master)
    return apply_channel(smp.prefix, sample_theta(chan, n, master))


def test_TE3_0_fast_composition_equals_log_domain_route():
    p = phys(8)
    S = records(p, ChannelSpec.interp(8, 0.3, 0.5), 3000, 1)
    ll, mu = flip_batch(S, p)
    pre = Precomputed.build(ll, mu)
    worst = 0.0
    for beta, alpha in ((0.0, 0.0), (0.05, 0.0), (0.2, 0.5), (0.5, 1.0)):
        ch = ChannelSpec.interp(8, beta, alpha)
        ref = candidate_from_queries(ll, mu, ch).mu2
        worst = max(worst, np.abs(pre.compose(ch.view_weights()) - ref).max())
    rec("TE3_0_max_abs_err_fast_vs_logdomain", worst)
    assert worst <= 1e-10


def test_TE3_1_identification_negative_controls_exact():
    # L = 1: record law is the same for every beta, yet mu2 = (1-2 beta) mu
    p1 = phys(1, 0.5)
    S = simulate(p1, 2000, 5).prefix
    ll, mu = flip_batch(S, p1)
    assert np.abs(ll[:, 1] - ll[:, 0]).max() == 0.0 and np.abs(mu[:, 1] + mu[:, 0]).max() == 0.0
    for b in (0.1, 0.3):
        cr = candidate_from_queries(ll, mu, ChannelSpec(b, (1.0,)))
        assert np.allclose(np.exp(cr.log_Z), 1.0) and np.abs(cr.mu2 - (1 - 2 * b) * mu[:, 0]).max() <= 1e-12
    # L = 2: the allocation does not change the record law, but changes the belief; beta is identified
    p2 = phys(2, 0.5)
    S = simulate(p2, 2000, 6).prefix
    ll, mu = flip_batch(S, p2)
    assert np.abs(ll[:, 1] - ll[:, 2]).max() == 0.0 and np.abs(mu[:, 1] + mu[:, 2]).max() == 0.0
    a, b, c = (candidate_from_queries(ll, mu, ChannelSpec(0.2, pi)) for pi in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)))
    assert np.allclose(a.log_Z, b.log_Z) and np.allclose(a.log_Z, c.log_Z)
    gap = np.abs(a.mu2 - b.mu2).max()
    rec("TE3_1_L2_belief_gap_between_allocations", gap, kind="set")
    assert gap > 0.5
    assert not np.allclose(a.log_Z, candidate_from_queries(ll, mu, ChannelSpec(0.4, (0.5, 0.5))).log_Z)


def test_TE3_2_cross_moment_formula_and_recovery():
    L, eta = 8, 0.05
    p = phys(L, 0.5, eta)
    ch = ChannelSpec(0.3, tuple(np.random.default_rng(1).dirichlet(np.ones(L))))
    n = 300_000
    S = records(p, ch, n, 7)
    z = S.a * (S.s[:, 1:] - p.rho * S.s[:, :-1])
    worst = 0.0
    for i in range(L):
        for j in range(i + 1, L):
            prod = z[:, i] * z[:, j]
            pred = p.c**2 * p.q_a**2 * (1 - 2 * eta) ** (j - i) * (1 - 2 * (ch.q[i] + ch.q[j]))
            worst = max(worst, abs(prod.mean() - pred) / (prod.std(ddof=1) / np.sqrt(n)))
    rec("TE3_2_max_abs_z_cross_moment_formula", worst, kind="set")
    assert worst <= 4.0
    w, raw = mom_channel(S.s, S.a, p, eta)
    assert np.abs(w[1:] - ch.q).max() < 0.01 and abs(raw - ch.beta) < 0.02


def test_TE3_3_em_monotone_and_recovers_rate_with_exact_law():
    p = phys(8)
    ch = ChannelSpec.interp(8, 0.2, 0.5)
    S = records(p, ch, 20000, 8)
    ll, _ = flip_batch(S, p)
    logell = ll[:, 1:] - ll[:, [0]]
    w, llv, it, tr = em_channel(logell, trace=True)
    assert np.all(np.diff(tr) >= -1e-9)
    assert abs((1 - w[0]) - 0.2) < 0.02 and abs(w.sum() - 1) < 1e-12
    rec("TE3_3_abs_err_beta_hat_L8_n20000", abs((1 - w[0]) - 0.2), kind="set")


def test_TE3_4_pp_concentrates_on_true_grid_point():
    p = phys(8)
    truth = ChannelSpec.interp(8, 0.2, 0.5)
    grid = [ChannelSpec.interp(8, b, a) for b in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    G = np.stack([c.view_weights() for c in grid])
    S = records(p, truth, 3000, 9)
    ll, _ = flip_batch(S, p)
    wpp, post = pp_channel(ll[:, 1:] - ll[:, [0]], G)
    k = [i for i, c in enumerate(grid) if abs(c.beta - 0.2) < 1e-12 and abs(c.pi_arr[-1] - truth.pi_arr[-1]) < 1e-12][0]
    assert post[k] > 0.9 and np.allclose(wpp, post @ G) and abs(post.sum() - 1) < 1e-12


def test_TE3_5_map_limits():
    p = phys(8)
    S = records(p, ChannelSpec.interp(8, 0.2, 0.5), 2000, 10)
    ll, _ = flip_batch(S, p)
    logell = ll[:, 1:] - ll[:, [0]]
    prior = ChannelSpec.interp(8, 0.2033, 0.5).view_weights()
    w_mle, *_ = em_channel(logell)
    w_small, *_ = em_channel(logell, prior=prior, kappa=1e-6)
    w_big, *_ = em_channel(logell, prior=prior, kappa=1e9, iters=50)
    assert np.abs(w_small - w_mle).max() < 1e-3
    assert np.abs(w_big - prior).max() < 1e-3


def test_TE3_6_absorption_sign_of_moment_estimator():
    """Clean model too persistent (eta_hat < eta) -> spurious positive beta at beta = 0; too flexible (eta_hat > eta) -> negative raw estimate."""
    p = phys(8, 0.5, 0.05)
    S = records(p, ChannelSpec.uniform(8, 0.0), 200_000, 11)
    raw_low = mom_channel(S.s, S.a, p, 0.03)[1]
    raw_ok = mom_channel(S.s, S.a, p, 0.05)[1]
    raw_high = mom_channel(S.s, S.a, p, 0.08)[1]
    rec("TE3_6_raw_beta_eta03_eta05_eta08", raw_low, kind="set")
    assert raw_low > 0.02 and abs(raw_ok) < 0.02 and raw_high < -0.02
    # exact form for one pair: s_hat = (1 - (1-2s)((1-2eta)/(1-2eta_hat))^d) / 2 with s = 0
    w, _ = mom_channel(S.s, S.a, p, 0.03)
    assert (1 - w[0]) > 0.01                                                       # survives the nonnegativity projection


def test_TE3_7_access_and_disjoint_streams():
    for fn in (adapt.em_channel, adapt.pp_channel):
        assert not any(k in inspect.signature(fn).parameters for k in ("probe", "modes", "theta"))
    assert list(inspect.signature(adapt.mom_channel).parameters) == ["s", "a", "phys", "eta_hat"]
    masters = dict(adapt=8301, eval=8401, e2_screen=8101, e2_confirm=8201, train=7001, val=7002, pilot_test=8001)
    assert len(set(masters.values())) == len(masters)
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments"))
    import run_e3
    assert run_e3.MASTER_ADAPT == 8301 and run_e3.MASTER_EVAL == 8401
