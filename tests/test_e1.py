"""E1 density-direct arms: objective targets mu_2 on corrupted records, read-out, access control, training sanity."""
import numpy as np
import torch

from _record import rec
from mti.inference import candidate_route
from mti.learned import data as ldata
from mti.learned.evalset import query_density_belief
from mti.learned.features import to_tensor
from mti.learned.models import CausalDensity
from mti.learned.train import TrainCfg, _a6d_nll, _tokens, mixture_logpdf, train_A6d, train_A6pd
from mti.params import ChannelSpec, Physics
from mti.simulate import apply_channel, sample_theta, simulate

torch.set_num_threads(2)
PHYS = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.05, L=8)


def test_probe_transition_density_is_minimised_at_the_aware_belief():
    """On channel-simulated records with the ORIGINAL clean probe, the NLL of the probe transition is minimised by w_L = (1 + mu_2)/2."""
    chan = ChannelSpec.uniform(PHYS.L, 0.3)
    n = 60000
    smp = simulate(PHYS, n, 21)
    S = apply_channel(smp.prefix, sample_theta(chan, n, 21))
    mu2 = candidate_route(S, PHYS, chan).mu2
    d = torch.as_tensor(smp.probe.s_next - PHYS.rho * smp.prefix.s[:, -1], dtype=torch.float64)
    a = torch.as_tensor(smp.probe.a, dtype=torch.float64)

    def nll(w):
        logit = torch.log(torch.as_tensor(w) / (1 - torch.as_tensor(w)))
        c = PHYS.c * a
        const = -0.5 * np.log(2 * np.pi * PHYS.q_w)
        lp = torch.log(torch.as_tensor(w)) + const - (d - c) ** 2 / (2 * PHYS.q_w)
        lm = torch.log(1 - torch.as_tensor(w)) + const - (d + c) ** 2 / (2 * PHYS.q_w)
        return -torch.logaddexp(lp, lm)

    w0 = (1 + mu2) / 2
    zs = []
    for eps in (0.08, -0.08):
        diff = (nll(np.clip(w0 + eps, 0.01, 0.99)) - nll(w0)).numpy()
        zs.append(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
    rec("e1_density_minimiser_min_z", min(zs), kind="set")
    assert min(zs) > 4


def test_readout_equals_tanh_of_last_logit_and_ignores_probe():
    torch.manual_seed(0)
    model = CausalDensity(16).eval()
    smp = simulate(PHYS, 6, 22)
    s, a = smp.prefix.s, smp.prefix.a
    mu = query_density_belief(model, s, a, PHYS)
    with torch.no_grad():
        logits = model(_tokens(to_tensor(s), to_tensor(a), PHYS))
    assert np.allclose(mu, torch.tanh(logits[:, -1] / 2).numpy(), atol=1e-6)
    assert np.all(np.abs(mu) < 1)
    # the read-out is a function of the prefix only: the probe is not an argument
    import inspect
    assert list(inspect.signature(query_density_belief).parameters)[:3] == ["model", "s_rec", "a_rec"]


def test_conditioned_model_uses_the_prior_and_default_is_unchanged():
    torch.manual_seed(0)
    plain = CausalDensity(16)
    cond = CausalDensity(16, cond_dim=PHYS.L)
    assert sum(p.numel() for p in plain.parameters()) < sum(p.numel() for p in cond.parameters())
    smp = simulate(PHYS, 4, 23)
    tok = _tokens(to_tensor(smp.prefix.s), to_tensor(smp.prefix.a), PHYS)
    q1, q2 = torch.zeros(4, PHYS.L), torch.full((4, PHYS.L), 0.1)
    with torch.no_grad():
        assert not torch.equal(cond(tok, q1), cond(tok, q2))
        assert plain(tok).shape == (4, PHYS.L + 1)


def test_density_direct_training_reduces_the_corrupted_record_nll():
    chan = ChannelSpec.uniform(PHYS.L, 0.3)
    tr = ldata.make_probe_set(PHYS, 4000, ldata.MASTER_TRAIN, 9, 0)
    va = ldata.make_probe_set(PHYS, 1000, ldata.MASTER_VAL, 9, 0)
    g = torch.Generator().manual_seed(0)
    va_a = ldata.corrupt(va.a, ldata.channel_probs(chan), g)
    torch.manual_seed(0)
    untrained = CausalDensity(32)
    with torch.no_grad():
        l0 = _a6d_nll(untrained, va.s, va_a, va.d_pr, va.a_pr, PHYS).item()
    _, info = train_A6d(tr, va, PHYS, chan, TrainCfg(epochs=4, hidden=32, seed=0))
    assert info.best_val < l0 and info.steps > 0
    _, info2 = train_A6pd(tr, va, PHYS, "broad", TrainCfg(epochs=3, hidden=32, seed=0))
    assert np.isfinite(info2.best_val) and info2.n_params > info.n_params
