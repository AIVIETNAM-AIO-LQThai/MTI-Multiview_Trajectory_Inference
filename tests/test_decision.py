"""T9 decision identities and simulated physical costs, T10 decomposition, T11 exact-zero controls."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from _record import rec
from conftest import random_phys
from mti.decision import a_star, cond_cost, kappa
from mti.experiment import Cell, check_chunk, eval_views, rb_chunk
from mti.inference import candidate_route, clean_belief
from mti.params import ChannelSpec, Physics
from mti.simulate import apply_channel, sample_theta, simulate


def test_T9_cost_identities():
    rng = np.random.default_rng(9)
    for _ in range(25):
        phys = random_phys(rng, 4, 0.05, 0.5)
        s, u, uhat = rng.normal(0, 2), rng.uniform(-1, 1), rng.uniform(-1, 1)
        a = a_star(phys, s, uhat)
        # direct expectation over m_L in {+-1} (Pr(+1) = (1+u)/2) and Gaussian noise: E[(x+w)^2] = x^2 + q_w
        direct = 0.0
        for m, pm in ((1.0, (1 + u) / 2), (-1.0, (1 - u) / 2)):
            direct += pm * ((phys.rho * s + phys.c * m * a) ** 2 + phys.q_w + phys.lam * a**2)
        e1 = abs(direct - cond_cost(phys, s, u, a))
        # a*(s,u) is the argmin of the conditional cost
        res = minimize_scalar(lambda x: cond_cost(phys, s, u, x), bracket=(-5, 0, 5), tol=1e-12)
        e2 = abs(res.x - a_star(phys, s, u))
        # excess cost of acting on uhat instead of u equals kappa (uhat - u)^2
        excess = cond_cost(phys, s, u, a_star(phys, s, uhat)) - cond_cost(phys, s, u, a_star(phys, s, u))
        e3 = abs(excess - kappa(phys, s) * (uhat - u) ** 2)
        rec("T9_max_abs_err_cost_identities", max(e1, e2, e3))
        assert e1 <= 1e-10 and e2 <= 1e-6 and e3 <= 1e-10


def test_T9_simulated_costs_match_analytic(primary):
    phys, chan = primary
    cell = Cell(0, "t", "t", phys, chan)
    parts = [check_chunk(cell, 3000 + 1, k, 20000) for k in range(3)]
    n = sum(p["n"] for p in parts)
    dS1 = sum(p["dS1"] for p in parts)
    dS2 = sum(p["dS2"] for p in parts)
    mean = dS1 / n
    se = np.sqrt((dS2 / n - mean**2) / n)
    z = mean / se
    rec("T9_max_abs_z_simulated_minus_analytic_cost", np.abs(z).max())
    assert np.all(np.abs(z) <= 4), z


def test_T10_decomposition(primary):
    phys, chan = primary
    n = 600
    smp = simulate(phys, n, master=27)
    S = apply_channel(smp.prefix, sample_theta(chan, n, master=27))
    cr = candidate_route(S, phys, chan)
    kap = kappa(phys, S.s[:, -1])
    # exact conditional identity given S: E[k(mu~ - mu_H)^2 | S] = k(mu~-mu_2)^2 + k Var(mu_H|S)
    lhs = kap * (cr.p_none * (cr.mu - cr.mu) ** 2 + np.sum(cr.p * (cr.mu[:, None] - cr.mu_flip) ** 2, axis=1))
    rhs = kap * (cr.mu - cr.mu2) ** 2 + kap * cr.var_muH
    e = np.abs(lhs - rhs).max() / max(1.0, np.abs(rhs).max())
    rec("T10_max_rel_err_conditional_decomposition", e)
    assert e <= 1e-10


def test_T10_expectation_decomposition(primary):
    """X = V_2 + I_loss in expectation, on independent prefixes with the Rao-Blackwellised per-prefix values."""
    phys, chan = primary
    n = 40000
    smp = simulate(phys, n, master=28)
    rec_, cr = eval_views(smp.prefix, phys, chan)
    V = phys.L + 1
    mu_t, mu2, var = (x.reshape(n, V) for x in (cr.mu, cr.mu2, cr.var_muH))
    kap = kappa(phys, smp.prefix.s[:, -1])[:, None]
    w = chan.view_weights()
    mH = mu_t[:, :1]
    diff = (kap * (mu_t - mH) ** 2 - kap * (mu_t - mu2) ** 2 - kap * var) @ w
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    rec("T10_abs_z_X_minus_V2_minus_Iloss", abs(z))
    assert abs(z) <= 4
    # aware excess over clean oracle E[k(mu_2 - mu_H)^2] equals I_loss in expectation as well
    d2 = (kap * (mu2 - mH) ** 2 - kap * var) @ w
    z2 = d2.mean() / (d2.std(ddof=1) / np.sqrt(n))
    rec("T10_abs_z_AW_minus_Iloss", abs(z2))
    assert abs(z2) <= 4


@pytest.mark.parametrize("beta,eta", [(0.0, 0.05), (0.2, 0.5)])
def test_T11_controls_are_exact_zeros(beta, eta):
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=eta, L=8)
    cell = Cell(0, "ctl", "control", phys, ChannelSpec.uniform(8, beta))
    out = rb_chunk(cell, 29, 0, 500, {}, {}, False)
    from mti.experiment import column_names
    names = column_names(cell, [])
    for k in ("V2", "I", "X", "AW"):
        assert out["S1"][names.index(k)] == 0.0 and out["S2"][names.index(k)] == 0.0, k
    if eta == 0.5:
        smp = simulate(phys, 200, master=30)
        mu, _ = clean_belief(smp.prefix, phys)
        assert np.all(mu == 0.0)
