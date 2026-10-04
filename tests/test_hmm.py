"""T1 enumeration of mode paths, T3 normalisation basics, T5 final prediction, T13 symmetries, T16 extreme regime."""
import numpy as np
import pytest

from _record import rec
from conftest import random_chan, random_phys
from mti import enumerate as en
from mti.inference import candidate_route, clean_belief, folded_route
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate

CONFIGS = [(L, eta, qw) for L in (1, 2, 3, 5, 8) for eta in (0.0, 0.05, 0.5) for qw in (0.02, 0.5)]


@pytest.mark.parametrize("L,eta,qw", CONFIGS)
def test_T1_hmm_vs_path_enumeration(L, eta, qw):
    rng = np.random.default_rng(1000 + L)
    phys = random_phys(rng, L, eta, qw)
    smp = simulate(phys, 3, master=11, cell_id=L)
    chan = random_chan(rng, L)
    theta = sample_theta(chan, 3, master=11, cell_id=L)
    for prefix in (smp.prefix, apply_channel(smp.prefix, theta)):
        mu, ll = clean_belief(prefix, phys)
        for i in range(3):
            s, a = prefix.s[i], prefix.a[i]
            ll_enum = en.enum_transition_loglik(s, a, phys)
            full_enum = en.enum_loglik(s, a, phys)
            lq = (-0.5 * np.log(2 * np.pi * phys.var_s0) - s[0] ** 2 / (2 * phys.var_s0)
                  + np.sum(-0.5 * np.log(2 * np.pi * phys.q_a) - a**2 / (2 * phys.q_a)))
            mu_enum = en.enum_belief(s, a, phys)
            e_ll = abs(ll[i] - ll_enum) / max(1.0, abs(ll_enum))
            e_full = abs(ll[i] + lq - full_enum) / max(1.0, abs(full_enum))
            e_mu = abs(mu[i] - mu_enum)
            rec("T1_max_rel_err_transition_loglik", e_ll)
            rec("T1_max_rel_err_fullprefix_loglik", e_full)
            rec("T1_max_abs_err_belief", e_mu)
            assert e_ll <= 1e-9 and e_full <= 1e-9 and e_mu <= 1e-10


@pytest.mark.parametrize("L,eta", [(3, 0.05), (5, 0.2), (4, 0.0)])
def test_T5_final_prediction_step_exact(L, eta):
    """mu(H) = (1-2 eta) E[m_{L-1} | H]: the filter includes the m_{L-1} -> m_L transition."""
    rng = np.random.default_rng(L)
    phys = random_phys(rng, L, eta, 0.5)
    smp = simulate(phys, 4, master=12, cell_id=L)
    mu, _ = clean_belief(smp.prefix, phys)
    for i in range(4):
        s, a = smp.prefix.s[i], smp.prefix.a[i]
        m_prev = en.enum_marginal_mode(s, a, phys, L - 1)
        rec("T5_max_err_final_step", abs(mu[i] - (1 - 2 * eta) * m_prev))
        assert abs(mu[i] - (1 - 2 * eta) * m_prev) <= 1e-10
        assert abs(mu[i] - en.enum_belief(s, a, phys)) <= 1e-10


def _binned_z(belief, target, nb=10):
    order = np.argsort(belief)
    zs = []
    for idx in np.array_split(order, nb):
        d = target[idx] - belief[idx]
        zs.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    return np.array(zs)


def test_T5_calibration_and_mutation(primary):
    """Simulated calibration of mu_H and mu_2 as predictors of m_L; the same test aimed at m_{L-1} must fail."""
    phys, chan = primary
    n = 60000
    smp = simulate(phys, n, master=13)
    theta = sample_theta(chan, n, master=13)
    S = apply_channel(smp.prefix, theta)
    muH, _ = clean_belief(smp.prefix, phys)
    cr = candidate_route(S, phys, chan)
    mL, mprev = smp.modes[:, -1], smp.modes[:, -2]
    zH = _binned_z(muH, mL)
    z2 = _binned_z(cr.mu2, mL)
    rec("T5_max_abs_z_calibration_muH", np.abs(zH).max())
    rec("T5_max_abs_z_calibration_mu2", np.abs(z2).max())
    assert np.abs(zH).max() <= 4 and np.abs(z2).max() <= 4
    d = muH * (mprev - muH)
    z_mut = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    rec("T5_mutation_abs_z_target_m_Lminus1", abs(z_mut), kind="set")
    assert abs(z_mut) > 8, "the calibration test must be able to detect a wrong target (m_{L-1} instead of m_L)"


@pytest.mark.parametrize("eta,qw,L", [(0.05, 0.5, 6), (0.0, 0.02, 4)])
def test_T3_normalisation_beta0_zero_masses(eta, qw, L):
    rng = np.random.default_rng(5)
    phys = random_phys(rng, L, eta, qw)
    smp = simulate(phys, 200, master=14, cell_id=L)
    # beta = 0: p_none = 1 exactly and mu_2 == mu, variance 0 (no clipping involved)
    c0 = ChannelSpec.uniform(L, 0.0)
    cr = candidate_route(smp.prefix, phys, c0)
    assert np.all(cr.p_none == 1.0) and np.all(cr.p == 0.0) and np.all(cr.mu2 == cr.mu) and np.all(cr.var_muH == 0.0)
    # random priors with true zero masses
    chan = random_chan(rng, L, zeros=True)
    cr = candidate_route(smp.prefix, phys, chan)
    err = np.abs(cr.p_none + cr.p.sum(1) - 1.0).max()
    rec("T3_max_abs_normalisation_error", err)
    assert err <= 1e-12
    zero = chan.q == 0.0
    assert np.all(cr.p[:, zero] == 0.0)
    assert np.all(np.isfinite(cr.mu2)) and np.all(cr.var_muH >= 0.0)


def test_T13_symmetries(primary):
    phys, chan = primary
    smp = simulate(phys, 200, master=15)
    S = apply_channel(smp.prefix, sample_theta(chan, 200, master=15))
    neg_a = Prefix(S.s, -S.a)
    neg_both = Prefix(-S.s, -S.a)
    cr, cr_a, cr_b = (candidate_route(p, phys, chan) for p in (S, neg_a, neg_both))
    errs = [np.abs(cr_a.ll - cr.ll).max() / 100.0,            # ll invariant under a -> -a
            np.abs(cr_a.mu + cr.mu).max(), np.abs(cr_a.mu2 + cr.mu2).max(),
            np.abs(cr_b.mu - cr.mu).max(), np.abs(cr_b.mu2 - cr.mu2).max(),   # (s,a) -> (-s,-a)
            np.abs(cr_b.p - cr.p).max()]
    rec("T13_max_symmetry_error", max(errs))
    assert max(errs) <= 1e-10


def test_T16_extreme_regime_finite():
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.02, lam=0.1, eta=0.05, L=32)
    chan = ChannelSpec.uniform(32, 0.5)
    smp = simulate(phys, 300, master=16)
    S = apply_channel(smp.prefix, sample_theta(chan, 300, master=16))
    cr, fr = candidate_route(S, phys, chan), folded_route(S, phys, chan)
    for name, arr in dict(mu=cr.mu, mu2=cr.mu2, var=cr.var_muH, p=cr.p, log_ell=cr.log_ell, log_r=fr.log_r,
                          log_1mr=fr.log_1mr, psi=fr.psi, G=fr.G, w=fr.w).items():
        assert np.all(np.isfinite(arr)), name
    assert np.abs(cr.p_none + cr.p.sum(1) - 1).max() <= 1e-12
    assert np.abs(fr.w0 + fr.w.sum(1) - 1).max() <= 1e-12
    rec("T16_min_log_r", fr.log_r.min(), kind="set")
    rec("T16_min_log_1mr", fr.log_1mr.min(), kind="set")
