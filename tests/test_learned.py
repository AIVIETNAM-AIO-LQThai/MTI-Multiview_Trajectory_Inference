"""Learned-pilot infrastructure: access control, feature/query construction, composition identity, training sanity."""
import inspect

import numpy as np
import pytest
import torch

from _record import rec
from mti.inference import candidate_route, clean_belief, folded_route
from mti.learned import data as ldata
from mti.learned.compose import compose, signed_sign_probs
from mti.learned.evalset import TestSet, compose_F, query_density, query_F, query_full
from mti.learned.features import make_features, onehot_mask, to_tensor
from mti.learned.hmm_fit import fit_eta
from mti.learned.models import CausalDensity, MaskedBelief
from mti.learned.train import Recalibrator, TrainCfg, mixture_logpdf, train_A1, _tokens
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate

torch.set_num_threads(2)
PHYS = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.05, L=8)


def test_probe_set_has_no_hidden_fields_and_features_ignore_probe():
    ps = ldata.make_probe_set(PHYS, 64, 12, 0, 0)
    assert set(ps.__dataclass_fields__) == {"s", "a", "a_pr", "d_pr"}
    assert list(inspect.signature(make_features).parameters) == ["s", "a", "mask", "phys"]
    nomask = torch.zeros(64, PHYS.L, dtype=torch.bool)
    x1 = make_features(ps.s, ps.a, nomask, PHYS)
    ps2 = ldata.ProbeSet(ps.s, ps.a, ps.a_pr + 3.0, ps.d_pr - 2.0)
    assert torch.equal(x1, make_features(ps2.s, ps2.a, nomask, PHYS))     # inputs cannot depend on the probe


def test_folded_features_do_not_carry_the_sign():
    ps = ldata.make_probe_set(PHYS, 40, 13, 0, 0)
    for j in range(PHYS.L):
        a_flip = ps.a.clone()
        a_flip[:, j] = -a_flip[:, j]
        mask = onehot_mask(torch.full((40,), j), PHYS.L)
        assert torch.equal(make_features(ps.s, ps.a, mask, PHYS), make_features(ps.s, a_flip, mask, PHYS))   # O_j(T_jH) = O_j(H)
        nomask = torch.zeros_like(mask)
        assert not torch.equal(make_features(ps.s, ps.a, nomask, PHYS), make_features(ps.s, a_flip, nomask, PHYS))


def test_probe_loss_is_minimised_by_the_clean_conditional_mean():
    """Population minimiser of (delta_L - c a_L h(I))^2 is E[m_L|I]: perturbing the exact belief increases the loss."""
    n = 60000
    smp = simulate(PHYS, n, 14)
    mu, _ = clean_belief(smp.prefix, PHYS)
    dL = smp.probe.s_next - PHYS.rho * smp.prefix.s[:, -1]
    base = (dL - PHYS.c * smp.probe.a * mu) ** 2
    for eps in (0.3, -0.3):
        diff = (dL - PHYS.c * smp.probe.a * (mu + eps)) ** 2 - base
        z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
        rec("learned_probe_loss_minimiser_min_z", z, kind="set")
        assert z > 4


def test_compose_reproduces_mu2_from_exact_components_and_gap_identity():
    rng = np.random.default_rng(3)
    for beta, pi in ((0.2, None), (0.5, None), (0.4, "zeros")):
        L = PHYS.L
        if pi is None:
            chan = ChannelSpec.uniform(L, beta)
        else:
            w = rng.dirichlet(np.ones(L)); w[[1, 4]] = 0.0
            chan = ChannelSpec(beta=beta, pi=tuple(w / w.sum()))
        smp = simulate(PHYS, 200, 15)
        S = apply_channel(smp.prefix, sample_theta(chan, 200, 15))
        cr, fr = candidate_route(S, PHYS, chan), folded_route(S, PHYS, chan)
        c = compose(cr.mu, fr.psi, fr.log_r, fr.log_1mr, cr.mu_flip, chan)
        e = max(np.abs(c.G - cr.mu2).max(), np.abs(c.M - cr.mu2).max(), np.abs(c.C).max(), np.abs(c.gap).max())
        rec("learned_compose_exact_max_err", e)
        assert e <= 1e-10
        # perturbed (non-exact) components: the signed route gap equals the residual sum algebraically
        mu_p = cr.mu + 0.05 * rng.normal(size=cr.mu.shape)
        psi_p = fr.psi + 0.05 * rng.normal(size=fr.psi.shape)
        mf_p = cr.mu_flip + 0.05 * rng.normal(size=cr.mu_flip.shape)
        c2 = compose(mu_p, psi_p, fr.log_r, fr.log_1mr, mf_p, chan)
        ident = np.abs(c2.gap - c2.gap_identity).max()
        rec("learned_gap_identity_max_err", ident)
        assert ident <= 1e-10 and np.abs(c2.C).max() > 1e-3


def test_query_F_matches_individual_model_calls():
    torch.manual_seed(0)
    model = MaskedBelief(16).eval()
    smp = simulate(PHYS, 5, 16)
    s, a = smp.prefix.s, smp.prefix.a
    mu, psi, sl, muf = query_F(model, s, a, PHYS, chunk=3)
    L = PHYS.L
    with torch.no_grad():
        for i in range(5):
            si, ai = to_tensor(s[i:i + 1]), to_tensor(a[i:i + 1])
            full, _ = model(make_features(si, ai, torch.zeros(1, L, dtype=torch.bool), PHYS))
            assert abs(full.item() - mu[i]) < 1e-5
            for j in range(L):
                b, sg = model(make_features(si, ai, onehot_mask(torch.tensor([j]), L), PHYS))
                af = ai.clone(); af[0, j] = -af[0, j]
                bf, _ = model(make_features(si, af, torch.zeros(1, L, dtype=torch.bool), PHYS))
                assert abs(b.item() - psi[i, j]) < 1e-5 and abs(sg[0, j].item() - sl[i, j]) < 1e-5 and abs(bf.item() - muf[i, j]) < 1e-5
    assert np.allclose(query_full(model, s, a, PHYS), mu, atol=1e-5)


def test_query_density_matches_manual_loglik():
    torch.manual_seed(1)
    model = CausalDensity(16).eval()
    smp = simulate(PHYS, 4, 17)
    ll, mu = query_density(model, smp.prefix.s, smp.prefix.a, PHYS, chunk=3)
    L = PHYS.L
    with torch.no_grad():
        for i in range(4):
            for q in range(L + 1):
                a = to_tensor(smp.prefix.a[i:i + 1]).clone()
                if q > 0:
                    a[0, q - 1] = -a[0, q - 1]
                s = to_tensor(smp.prefix.s[i:i + 1])
                logits = model(_tokens(s, a, PHYS))
                delta = s[:, 1:] - PHYS.rho * s[:, :-1]
                man = mixture_logpdf(delta, a, logits[:, :L], PHYS).sum().item()
                assert abs(man - ll[i, q]) < 1e-3 and abs(torch.tanh(logits[0, L] / 2).item() - mu[i, q]) < 1e-5


@pytest.mark.parametrize("eta", [0.05, 0.2])
def test_hmm_em_recovers_persistence(eta):
    phys = PHYS.with_(eta=eta)
    smp = simulate(phys, 20000, 18)
    est, it, _ = fit_eta(smp.prefix.s, smp.prefix.a, smp.probe.s_next, smp.probe.a, phys)
    rec("learned_hmm_em_abs_err_eta", abs(est - eta))
    assert abs(est - eta) < 0.01 and it < 300


def test_channel_sampler_and_prior_families():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(20000, PHYS.L)
    chan = ChannelSpec.uniform(PHYS.L, 0.2)
    out = ldata.corrupt(a, ldata.channel_probs(chan), gen)
    changed = (out != a).sum(1)
    assert changed.max() == 1 and abs((changed == 1).float().mean().item() - 0.2) < 0.01
    probs, q = ldata.sample_family("narrow", 5000, PHYS.L, gen)
    assert probs[:, 1:].sum(1).max() <= 0.3 + 1e-6 and torch.allclose(q, q[:, :1].expand_as(q))   # uniform pi
    probs, q = ldata.sample_family("broad", 5000, PHYS.L, gen)
    assert abs(probs.sum(1) - 1).max() < 1e-5 and q.sum(1).max() <= 0.5 + 1e-6 and q.std(1).mean() > 0.01


def test_recalibrator_starts_as_identity_and_is_odd():
    g = Recalibrator()
    f = torch.linspace(-0.9, 0.9, 21)
    s = torch.rand(21)
    assert torch.equal(g(s, f), f)
    for p in g.parameters():
        torch.nn.init.normal_(p, 0, 0.5)
    assert torch.allclose(g(s, -f), -g(s, f), atol=1e-6)


def test_A1_training_reduces_the_probe_loss():
    tr = ldata.make_probe_set(PHYS, 4000, ldata.MASTER_TRAIN, 9, 0)
    va = ldata.make_probe_set(PHYS, 1000, ldata.MASTER_VAL, 9, 0)
    torch.manual_seed(0)
    untrained = MaskedBelief(32)
    p0, _ = untrained(make_features(va.s, va.a, torch.zeros(1000, PHYS.L, dtype=torch.bool), PHYS))
    loss0 = ldata.probe_loss(p0, va.d_pr, va.a_pr, PHYS).item()
    model, info = train_A1(tr, va, PHYS, TrainCfg(epochs=4, hidden=32, seed=0))
    assert info.best_val < loss0 and info.steps > 0
