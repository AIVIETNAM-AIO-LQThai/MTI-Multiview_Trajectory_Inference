"""T14 simulator and channel moments (|z| <= 4)."""
import numpy as np

from _record import rec
from mti.params import ChannelSpec, Physics
from mti.simulate import apply_channel, sample_theta, simulate

N = 300000


def _z_var(x, var):
    """z-score of the sample variance of a mean-zero Gaussian variable."""
    return (np.mean(x**2) - var) / (var * np.sqrt(2.0 / len(x)))


def test_T14_simulator_moments():
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.05, L=8)
    smp = simulate(phys, N, master=31)
    s, a, m = smp.prefix.s, smp.prefix.a, smp.modes
    L = phys.L
    zs = {
        "var_s0": _z_var(s[:, 0], phys.var_s0),
        "var_sL": _z_var(s[:, L], phys.var_s0),           # stationary: Var(s_k) is constant in k
        "var_a": _z_var(a.ravel(), phys.q_a),
        "var_w": _z_var((s[:, 1:] - phys.rho * s[:, :-1] - phys.c * m[:, :-1] * a).ravel(), phys.q_w),
    }
    flips = (m[:, 1:] != m[:, :-1])
    zs["flip_rate"] = (flips.mean() - phys.eta) / np.sqrt(phys.eta * (1 - phys.eta) / flips.size)
    zs["m0_balance"] = m[:, 0].mean() / np.sqrt(1.0 / N)
    zs["mean_s_L"] = s[:, L].mean() / np.sqrt(phys.var_s0 / N)
    # probe branch uses the physical mode m_L and independent noise
    pr = smp.probe.s_next - phys.rho * s[:, L] - phys.c * m[:, L] * smp.probe.a
    zs["var_probe_noise"] = _z_var(pr, phys.q_w)
    zs["var_probe_a"] = _z_var(smp.probe.a, phys.q_a)
    # probe/decision streams are independent of the prefix noises (zero correlation)
    w0 = s[:, 1] - phys.rho * s[:, 0] - phys.c * m[:, 0] * a[:, 0]
    zs["corr_probe_prefix_noise"] = np.corrcoef(pr, w0)[0, 1] * np.sqrt(N)
    zs["corr_decision_prefix_noise"] = np.corrcoef(smp.decision_noise, w0)[0, 1] * np.sqrt(N)
    zs["var_decision_noise"] = _z_var(smp.decision_noise, phys.q_w)
    rec("T14_max_abs_z_simulator_moments", max(abs(v) for v in zs.values()))
    assert all(abs(v) <= 4 for v in zs.values()), zs


def test_T14_eta_zero_modes_constant_and_half_independent():
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.0, L=8)
    m = simulate(phys, 1000, master=32).modes
    assert np.all(m == m[:, :1])
    m = simulate(phys.with_(eta=0.5), N, master=32).modes
    z = np.corrcoef(m[:, 3], m[:, 4])[0, 1] * np.sqrt(N)
    assert abs(z) <= 4


def test_T14_channel_frequencies_and_states_untouched():
    L = 8
    for chan in (ChannelSpec.uniform(L, 0.2),
                 ChannelSpec(beta=0.4, pi=(0.0, 0.5, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0))):
        theta = sample_theta(chan, N, master=33)
        zs = [((theta == -1).mean() - (1 - chan.beta)) / np.sqrt(chan.beta * (1 - chan.beta) / N)]
        for j in range(L):
            q = chan.q[j]
            f = (theta == j).mean()
            if q == 0.0:
                assert f == 0.0
            else:
                zs.append((f - q) / np.sqrt(q * (1 - q) / N))
        rec("T14_max_abs_z_channel_frequencies", max(abs(v) for v in zs))
        assert all(abs(v) <= 4 for v in zs), zs
    smp = simulate(Physics(0.9, 1.0, 1.0, 0.5, 0.1, 0.05, L), 1000, master=34)
    theta = sample_theta(ChannelSpec.uniform(L, 0.5), 1000, master=34)
    S = apply_channel(smp.prefix, theta)
    assert np.array_equal(S.s, smp.prefix.s)                       # states are never corrupted
    changed = S.a != smp.prefix.a
    assert np.array_equal(changed.sum(1), (theta >= 0).astype(int))  # exactly one sign flip per corrupted record
    rows = np.flatnonzero(theta >= 0)
    assert np.array_equal(S.a[rows, theta[rows]], -smp.prefix.a[rows, theta[rows]])
