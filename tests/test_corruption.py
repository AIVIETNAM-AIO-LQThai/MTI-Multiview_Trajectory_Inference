"""T2 posterior vs hypothesis enumeration, T4 known-location limits, T6 folded observer vs enumeration,
T7 candidate/folded equality, T8 folded observer invariance, T12 access control."""
import inspect

import numpy as np
import pytest

from _record import rec
from conftest import random_chan, random_phys
from mti import enumerate as en
from mti import inference
from mti.inference import (candidate_route, clean_belief, compatibility_residual, folded_observer, folded_route)
from mti.params import ChannelSpec, Physics
from mti.simulate import (Prefix, Probe, Sample, apply_channel, flip_location, sample_theta, simulate)


@pytest.mark.parametrize("L,eta,qw,zeros", [(2, 0.05, 0.5, False), (3, 0.0, 0.02, False), (5, 0.1, 0.5, True),
                                            (7, 0.05, 0.1, True), (4, 0.5, 0.5, False)])
def test_T2_corruption_posterior_vs_enumeration(L, eta, qw, zeros):
    rng = np.random.default_rng(200 + L)
    phys = random_phys(rng, L, eta, qw)
    chan = random_chan(rng, L, zeros=zeros)
    smp = simulate(phys, 3, master=21, cell_id=L)
    S = apply_channel(smp.prefix, sample_theta(chan, 3, master=21, cell_id=L))
    cr, fr = candidate_route(S, phys, chan), folded_route(S, phys, chan)
    for i in range(3):
        e = en.enum_corruption_posterior(S.s[i], S.a[i], phys, chan)
        post = np.concatenate([[cr.p_none[i]], cr.p[i]])
        errs = [np.abs(post - e["p_theta"]).max(), abs(cr.mu2[i] - e["mu2"]), abs(cr.var_muH[i] - e["var"]),
                abs(fr.G[i] - e["mu2"]), np.abs(np.concatenate([[cr.mu[i]], cr.mu_flip[i]]) - e["mu_theta"]).max()]
        rec("T2_max_abs_err_posterior_mu2_var_G", max(errs))
        assert max(errs) <= 1e-10
        # Var(mu_H|S) in the problem's alternative form
        alt = cr.p_none[i] * cr.mu[i] ** 2 + np.sum(cr.p[i] * cr.mu_flip[i] ** 2) - cr.mu2[i] ** 2
        assert abs(alt - cr.var_muH[i]) <= 1e-10


@pytest.mark.parametrize("L,beta", [(3, 0.3), (5, 0.1), (6, 0.5)])
def test_T4_known_location_limits(L, beta):
    rng = np.random.default_rng(40 + L)
    phys = random_phys(rng, L, 0.05, 0.5)
    smp = simulate(phys, 50, master=22, cell_id=L)
    for k in range(L):
        chan = ChannelSpec.point_mass(L, beta, k)
        S = apply_channel(smp.prefix, np.full(50, k))
        cr, fr = candidate_route(S, phys, chan), folded_route(S, phys, chan)
        ell = np.exp(cr.log_ell[:, k])
        e1 = np.abs(cr.p[:, k] - beta * ell / (1 - beta + beta * ell)).max()
        rec("T4_max_abs_err_known_location_p", e1)
        assert e1 <= 1e-12
        assert np.all(np.delete(cr.p, k, axis=1) == 0.0)
        if beta == 0.5:
            e2 = np.abs(cr.mu2 - fr.psi[:, k]).max()          # aware belief == the folded belief at k (incl. k = L-1)
            rec("T4_max_abs_err_beta_half_equals_folded", e2)
            assert e2 <= 1e-10


@pytest.mark.parametrize("L,eta,qw", [(3, 0.05, 0.5), (4, 0.0, 0.1), (5, 0.1, 0.02)])
def test_T6_folded_vs_enumeration_and_ratio(L, eta, qw):
    rng = np.random.default_rng(60 + L)
    phys = random_phys(rng, L, eta, qw)
    chan = random_chan(rng, L)
    smp = simulate(phys, 3, master=23, cell_id=L)
    S = apply_channel(smp.prefix, sample_theta(chan, 3, master=23, cell_id=L))
    cr, fr = candidate_route(S, phys, chan), folded_route(S, phys, chan)
    for i in range(3):
        for j in range(L):
            psi, pplus, llf = en.enum_folded(S.s[i], S.a[i], phys, j)
            rec_pos = S.a[i, j] >= 0
            r_enum = (1 - pplus) if rec_pos else pplus
            e_psi = abs(fr.psi[i, j] - psi)
            e_r = abs(np.exp(fr.log_r[i, j]) - r_enum)
            e_1mr = abs(np.exp(fr.log_1mr[i, j]) - (1 - r_enum))
            e_ll = abs(fr.ll_fold[i, j] - llf) / max(1.0, abs(llf))
            rec("T6_max_abs_err_psi", e_psi)
            rec("T6_max_abs_err_r", max(e_r, e_1mr))
            rec("T6_max_rel_err_folded_loglik", e_ll)
            assert e_psi <= 1e-10 and e_r <= 1e-10 and e_1mr <= 1e-10 and e_ll <= 1e-9
    # candidate density ratio equals the odds of the opposite sign from the independent folded smoother
    d = np.abs(cr.log_ell - fr.log_ell) / np.maximum(1.0, np.abs(cr.log_ell))
    rec("T6_max_rel_err_log_ell_vs_odds", d.max())
    assert d.max() <= 1e-9


@pytest.mark.parametrize("L,qw", [(8, 0.5), (8, 0.02), (32, 0.5), (32, 0.02)])
def test_T7_candidate_folded_equality(L, qw):
    rng = np.random.default_rng(70 + L)
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=qw, lam=0.1, eta=0.05, L=L)
    for beta in (0.2, 0.5):
        chan = ChannelSpec.uniform(L, beta)
        smp = simulate(phys, 150, master=24, cell_id=L)
        for prefix in (smp.prefix, apply_channel(smp.prefix, sample_theta(chan, 150, master=24, cell_id=L))):
            cr, fr = candidate_route(prefix, phys, chan), folded_route(prefix, phys, chan)
            C = compatibility_residual(fr, cr)
            e_C = np.abs(C).max()
            e_G = np.abs(fr.G - cr.mu2).max()
            e_Z = (np.abs(fr.log_Z - cr.log_Z) / np.maximum(1.0, np.abs(cr.log_Z))).max()
            e_ell = (np.abs(fr.log_ell - cr.log_ell) / np.maximum(1.0, np.abs(cr.log_ell))).max()
            rec("T7_max_abs_C_j", e_C)
            rec("T7_max_abs_G_minus_M", e_G)
            rec("T7_max_rel_logZ_diff", e_Z)
            rec("T7_max_rel_logell_diff", e_ell)
            assert e_C <= 1e-10 and e_G <= 1e-10 and e_Z <= 1e-9 and e_ell <= 1e-9


def test_T8_folded_observer_invariance(primary):
    phys, chan = primary
    smp = simulate(phys, 100, master=25)
    H = smp.prefix
    frH = folded_route(H, phys, chan)
    for j in range(phys.L):
        Hj = flip_location(H, j)
        s0, a0, m0 = folded_observer(H, j)
        s1, a1, m1 = folded_observer(Hj, j)
        assert np.array_equal(s0, s1) and np.array_equal(a0, a1) and np.array_equal(m0, m1)
        assert np.all(a0[:, j] == 0.0) and np.array_equal(m0, np.abs(H.a[:, j]))
        fr1 = folded_route(Hj, phys, chan)
        pplus0 = np.where(H.a[:, j] >= 0, np.exp(frH.log_1mr[:, j]), np.exp(frH.log_r[:, j]))
        pplus1 = np.where(Hj.a[:, j] >= 0, np.exp(fr1.log_1mr[:, j]), np.exp(fr1.log_r[:, j]))
        e = max(np.abs(frH.psi[:, j] - fr1.psi[:, j]).max(), np.abs(pplus0 - pplus1).max(),
                np.abs(frH.ll_fold[:, j] - fr1.ll_fold[:, j]).max())
        rec("T8_max_abs_err_O_j_invariance", e)
        assert e <= 1e-12


def test_T12_inference_sees_only_the_prefix(primary):
    phys, chan = primary
    smp = simulate(phys, 50, master=26)
    S = apply_channel(smp.prefix, sample_theta(chan, 50, master=26))
    other = Sample(smp.prefix, -smp.modes, Probe(smp.probe.a + 5.0, smp.probe.s_next - 3.0), smp.decision_noise * 7.0)
    r1, r2 = candidate_route(smp.prefix, phys, chan), candidate_route(other.prefix, phys, chan)
    f1, f2 = folded_route(S, phys, chan), folded_route(Prefix(S.s.copy(), S.a.copy()), phys, chan)
    assert np.array_equal(r1.mu2, r2.mu2) and np.array_equal(r1.log_post, r2.log_post)
    assert np.array_equal(f1.G, f2.G)
    for fn in (inference.candidate_route, inference.folded_route, inference.clean_belief, inference.flip_batch):
        assert list(inspect.signature(fn).parameters)[:2] == ["prefix", "phys"]
        with pytest.raises(TypeError):
            fn(smp, phys, chan) if fn in (inference.candidate_route, inference.folded_route) else fn(smp, phys)
    assert set(Prefix.__dataclass_fields__) == {"s", "a"}
