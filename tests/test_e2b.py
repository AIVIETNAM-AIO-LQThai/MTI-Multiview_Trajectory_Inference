"""E2b: mean-prior algebra and Bayes optimality of composition with q_bar (protocol e2_protocol.md section 10)."""
import numpy as np

from _record import rec
from mti.inference import candidate_from_queries
from mti.learned.evalset import TestSet
from mti.misspec import ExactRecords, TrueLaw, alpha_of, mean_prior, no_channel
from mti.params import ChannelSpec, Physics

L = 8
BETAS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRUTHS = [ChannelSpec.interp(L, b, a) for b in BETAS for a in ALPHAS]


def make(qw=0.02, n=4000):
    phys = Physics(0.9, 1.0, 1.0, qw, 0.1, 0.05, L)
    ts = TestSet(phys, n, cell_id=0, master=8101)
    ex = ExactRecords(ts)
    return phys, ts, ex, TrueLaw(ts, ex)


def test_TE2b_1_mean_prior_algebra_and_bayes_identity():
    qbar = mean_prior(TRUTHS)
    vw = sum(c.view_weights() for c in TRUTHS) / len(TRUTHS)
    assert np.allclose(vw, qbar.view_weights(), atol=1e-15)                              # mixture of view weights == view weights of q_bar
    assert abs(qbar.beta - np.mean(BETAS)) < 1e-12 and np.allclose(qbar.pi_arr, ChannelSpec.interp(L, 0.1, 0.5).pi_arr)
    # linearity => compound posterior mean: Zbar * mu2(q_bar) = sum_t w_t Z_t * mu2(q_t) on every record
    phys, ts, ex, truth = make()
    Zt = np.stack([np.exp(candidate_from_queries(ex.ll, ex.mu, c).log_Z) for c in TRUTHS])      # (T, R)
    mu_t = np.stack([ex.mu2(c).reshape(-1) for c in TRUTHS])
    Zbar = np.exp(candidate_from_queries(ex.ll, ex.mu, qbar).log_Z)
    lhs = Zbar * ex.mu2(qbar).reshape(-1)
    rhs = (Zt * mu_t).mean(axis=0)
    err = np.abs(lhs - rhs).max() / max(1.0, np.abs(rhs).max())
    rec("TE2b_1_max_rel_err_bayes_identity", err)
    assert err <= 1e-10 and np.abs(Zbar - Zt.mean(axis=0)).max() <= 1e-10 * max(1.0, Zbar.max())


def test_TE2b_2_mean_prior_minimises_uniform_average_regret():
    """K1(candidate) - K1(q_bar) = E_compound[kappa (mu_cand - mu_2(q_bar))^2] >= 0 (exact in expectation): q_bar is not beaten by any candidate."""
    phys, ts, ex, truth = make(0.5, n=8000)
    qbar = mean_prior(TRUTHS)
    m2bar = ex.mu2(qbar)
    wbar = qbar.view_weights()
    cands = [ChannelSpec.interp(L, b, a) for b in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    refs = [truth.reference(c) for c in TRUTHS]

    def k1(m):                                                           # per-prefix uniform-average regret over the 30 truths
        return np.mean([(truth.K * (m - r) ** 2) @ c.view_weights() for r, c in zip(refs, TRUTHS)], axis=0)

    y_bar = k1(m2bar)
    worst_z, gaps = np.inf, []
    for c in cands + [no_channel(L)]:
        m = ex.mu2(c) if c.beta > 0 else ex.naive
        d = k1(m) - y_bar
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        worst_z = min(worst_z, z)
        gaps.append(d.mean())
        # the same difference from the compound-law formula (different estimator of the same expectation)
        d2 = (truth.K * (m - m2bar) ** 2) @ wbar
        zc = (d.mean() - d2.mean()) / np.sqrt((d.var(ddof=1) + d2.var(ddof=1)) / len(d))
        assert abs(zc) <= 4
    rec("TE2b_2_min_z_candidate_minus_qbar", worst_z, kind="set")
    assert worst_z > -4 and min(gaps) > -1e-3


def test_TE2b_3_singleton_uncertainty_set():
    phys, ts, ex, truth = make()
    ct = ChannelSpec.interp(L, 0.2, 0.5)
    qbar = mean_prior([ct])
    assert qbar.beta == ct.beta and np.allclose(qbar.pi_arr, ct.pi_arr)
    r, y = truth.score(ex.mu2(qbar), ct, ex.naive)
    assert r["E"] == 0.0 and np.all(y == 0.0)


def test_alpha_roundtrip_and_qbar_alpha():
    """Regression test for a bug caught in review: the prior parameter alpha must survive spec -> number -> interp round trips."""
    for a in (0.0, 0.125, 0.5, 0.9, 1.0):
        c = ChannelSpec.interp(L, 0.2, a)
        assert abs(alpha_of(c) - a) < 1e-12
        assert np.allclose(ChannelSpec.interp(L, c.beta, alpha_of(c)).pi_arr, c.pi_arr, atol=1e-15)
    qbar = mean_prior(TRUTHS)
    assert abs(alpha_of(qbar) - 0.5) < 1e-12 and abs(qbar.pi_arr[-1] - 0.5625) < 1e-12
    rebuilt = ChannelSpec.interp(L, qbar.beta, alpha_of(qbar))
    assert np.allclose(rebuilt.view_weights(), qbar.view_weights(), atol=1e-15)
