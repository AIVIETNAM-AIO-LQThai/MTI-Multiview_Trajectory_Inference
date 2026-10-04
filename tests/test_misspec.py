"""E2 (channel misspecification): convex-combination identity, safe-region inequality, controls, RB-vs-MC, priors, decomposition."""
import numpy as np
import pytest
import torch

from _record import rec
from mti.learned.data import sample_family
from mti.learned.evalset import TestSet
from mti.misspec import ExactRecords, TrueLaw, convex_form, no_channel, odds, predicted_safe, record_violations
from mti.params import ChannelSpec, Physics
from mti.simulate import sample_theta

L = 8


def make(qw, n=3000, seed_cell=0):
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=qw, lam=0.1, eta=0.05, L=L)
    ts = TestSet(phys, n, cell_id=seed_cell, master=8101)
    ex = ExactRecords(ts)
    return phys, ts, ex, TrueLaw(ts, ex)


@pytest.mark.parametrize("qw", [0.02, 0.5])
def test_TE2_1_convex_combination_identity(qw):
    phys, ts, ex, truth = make(qw)
    worst = 0.0
    for alpha in (0.0, 0.5, 1.0):
        pi = ChannelSpec.interp(L, 0.2, alpha).pi
        for beta in (0.0, 0.05, 0.2, 0.5):
            mu_S, mbar, A, w = convex_form(ex, pi, beta)
            m2 = ex.mu2(ChannelSpec(beta, pi)).reshape(-1)
            worst = max(worst, np.abs((1 - w) * mu_S + w * mbar - m2).max())
    rec("TE2_1_max_abs_err_convex_identity", worst)
    assert worst <= 1e-10


@pytest.mark.parametrize("qw", [0.02, 0.5])
def test_TE2_2_safe_region_has_no_per_record_violations_and_test_has_power(qw):
    phys, ts, ex, truth = make(qw)
    n_checked, bad = 0, 0
    for alpha in (0.0, 1.0):
        for bt in (0.0, 0.05, 0.2, 0.5):
            ct = ChannelSpec.interp(L, bt, alpha)
            for ba in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 1 / 3, 0.4, 0.5):
                ca = ChannelSpec.interp(L, ba, alpha)
                v = record_violations(truth, ct, ca)
                if predicted_safe(ba, bt, True):
                    n_checked += 1
                    bad += v
    rec("TE2_2_violations_in_safe_region", bad)
    assert bad == 0 and n_checked > 20
    # power: outside the predicted region violations do occur (overstating beta four-fold)
    assert record_violations(truth, ChannelSpec.interp(L, 0.05, 0.0), ChannelSpec.interp(L, 0.5, 0.0)) > 0
    # beta_t = 0 and any beta_a > 0: harmful wherever mbar != mu (prediction iv)
    assert record_violations(truth, ChannelSpec.interp(L, 0.0, 0.0), ChannelSpec.interp(L, 0.1, 0.0)) > 0


def test_TE2_3_controls_are_exact():
    phys, ts, ex, truth = make(0.02)
    for alpha in (0.0, 1.0):
        for beta in (0.05, 0.2, 0.5):
            ch = ChannelSpec.interp(L, beta, alpha)
            r, y = truth.score(ex.mu2(ch), ch, ex.naive)
            assert np.all(y == 0.0) and r["E"] == 0.0                              # q_assumed == q_true: R1 == R0 exactly
    assert np.array_equal(ex.mu2(no_channel(L)), ex.naive)                         # beta_a = 0: composition == naive exactly
    ct = ChannelSpec.interp(L, 0.2, 0.0)
    r, _ = truth.score(ex.naive, ct, ex.naive)
    assert r["E"] > 0 and r["D"] == 0.0 and r["D_none"] == 0.0                     # naive vs itself: no realised-cost difference


def test_TE2_4_rb_estimate_matches_single_draw_monte_carlo():
    phys, ts, ex, truth = make(0.02, n=40000)
    ct, ca = ChannelSpec.interp(L, 0.2, 0.5), ChannelSpec.interp(L, 0.3, 0.0)
    m2a = ex.mu2(ca)
    r, y_rb = truth.score(m2a, ct, ex.naive)
    theta = sample_theta(ct, ts.n, master=77)
    v = theta + 1
    y_mc = truth.kap * (m2a[np.arange(ts.n), v] - truth.reference(ct)[np.arange(ts.n), v]) ** 2
    z = (y_mc.mean() - y_rb.mean()) / np.sqrt(y_mc.var(ddof=1) / ts.n + y_rb.var(ddof=1) / ts.n)
    rec("TE2_4_abs_z_rb_vs_mc", abs(z))
    assert abs(z) <= 4


def test_TE2_5_interpolated_prior_and_coverage_family():
    u, last = ChannelSpec.interp(L, 0.2, 0.0), ChannelSpec.interp(L, 0.2, 1.0)
    assert np.allclose(u.pi_arr, 1 / L) and np.allclose(last.pi_arr, ChannelSpec.point_mass(L, 0.2, L - 1).pi_arr)
    mid = ChannelSpec.interp(L, 0.2, 0.5)
    assert abs(mid.pi_arr.sum() - 1) < 1e-12 and abs(mid.pi_arr[-1] - (0.5 / L + 0.5)) < 1e-12
    assert np.allclose(mid.pi_arr[:-1], 0.5 / L)
    gen = torch.Generator().manual_seed(0)
    probs, q = sample_family("cov", 40000, L, gen)
    assert torch.allclose(probs.sum(1), torch.ones(40000), atol=1e-5) and (1 - probs[:, 0]).max() <= 0.5 + 1e-6
    pi = q / q.sum(1, keepdim=True)
    broad_probs, bq = sample_family("broad", 40000, L, gen)
    bpi = bq / bq.sum(1, keepdim=True)
    p_cov, p_broad = float((pi[:, -1] > 0.9).float().mean()), float((bpi[:, -1] > 0.9).float().mean())
    rec("TE2_5_P_last_gt_0.9_cov", p_cov, kind="set")
    assert p_cov > 0.03 and p_broad < 1e-3                                         # the control actually covers the vertex region


def test_TE2_6_decomposition_is_exact_per_sample():
    phys, ts, ex, truth = make(0.5)
    rng = np.random.default_rng(1)
    ct, ca = ChannelSpec.interp(L, 0.2, 1.0), ChannelSpec.interp(L, 0.1, 0.0)
    m2a = ex.mu2(ca)
    for muhat in (m2a + 0.1 * rng.normal(size=m2a.shape), ex.naive, m2a):
        r, y = truth.score(muhat, ct, ex.naive, mu2a=m2a)
        rec("TE2_6_max_abs_err_decomposition", r["decomp_max_abs_err"])
        assert r["decomp_max_abs_err"] <= 1e-12
    r, _ = truth.score(ex.naive, ct, ex.naive, mu2a=m2a)
    assert r["learn"] > 0 and r["channel"] > 0
    # prediction in section 3: beta_a <= beta_t with the right pi => channel term below the naive regret
    ca2 = ChannelSpec.interp(L, 0.1, 1.0)
    rn, _ = truth.score(ex.naive, ct, ex.naive)
    rr, _ = truth.score(ex.mu2(ca2), ct, ex.naive)
    assert rr["E"] <= rn["E"]
