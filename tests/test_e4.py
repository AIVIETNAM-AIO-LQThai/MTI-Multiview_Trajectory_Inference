"""E4: joint identification of clean persistence and channel (protocol docs/e4_protocol.md section 6)."""
import hashlib
import inspect
import json
import os

import numpy as np

from _record import rec
from mti import joint
from mti.inference import candidate_from_queries, flip_batch
from mti.joint import (ETA_GRID, clean_eta_mle, fit_channel, interleave_persistence, jmom, pattern_tv, profile_fisher,
                       profile_grid, refine_eta, _queries)
from mti.params import ChannelSpec, Physics
from mti.simulate import Prefix, apply_channel, sample_theta, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULATE_SHA16 = "9e156769b596bbff"


def phys(L, qw=0.02, eta=0.05):
    return Physics(0.9, 1.0, 1.0, qw, 0.1, eta, L)


def records(p, chan, n, master, cell=0, chunk=0):
    smp = simulate(p, n, master, cell, chunk)
    return apply_channel(smp.prefix, sample_theta(chan, n, master, cell, chunk))


def w_of(chan):
    return chan.view_weights()


def test_TE4_0_L3_ridge_record_level():
    p = phys(3, 0.5)
    ch = ChannelSpec.interp(3, 0.2, 0.5)
    d, w2 = pattern_tv(3, 0.05, w_of(ch), 0.03)
    assert d < 1e-9
    ch2 = ChannelSpec(1 - w2[0], tuple(w2[1:] / w2[1:].sum()))
    S = records(p, ch, 20000, 3)
    ll, mu = flip_batch(S, p)
    ll2, mu2 = flip_batch(S, p.with_(eta=0.03))
    a, b = candidate_from_queries(ll, mu, ch), candidate_from_queries(ll2, mu2, ch2)
    diff = np.abs((ll[:, 0] + a.log_Z) - (ll2[:, 0] + b.log_Z)).max()
    rec("TE4_0_max_record_loglik_diff_L3_ridge", diff)
    assert diff <= 1e-12
    gap = np.abs(a.mu2 - b.mu2).max()
    rec("TE4_0_max_belief_gap_L3_ridge", gap, kind="set")
    assert gap > 0.1


def test_TE4_1_pattern_lp():
    ch3, ch4, ch8 = (ChannelSpec.interp(L, 0.2, 0.5) for L in (3, 4, 8))
    assert pattern_tv(3, 0.05, w_of(ch3), 0.03)[0] < 1e-9
    d4, d8 = pattern_tv(4, 0.05, w_of(ch4), 0.03)[0], pattern_tv(8, 0.05, w_of(ch8), 0.03)[0]
    rec("TE4_1_pattern_tv_L4_eta03", d4, kind="set")
    rec("TE4_1_pattern_tv_L8_eta03", d8, kind="set")
    assert d4 > 1e-3 and d8 > 1e-3
    assert pattern_tv(4, 0.05, w_of(ch4), 0.05)[0] < 1e-9                        # identity at the truth


def test_TE4_2_jmle_recovers_eta_and_channel_L8():
    p = phys(8)
    ch = ChannelSpec.interp(8, 0.2, 0.5)
    S = records(p, ch, 20000, 5)
    pg = profile_grid(S, p, (20000,))[20000]
    g = int(np.argmax(pg["ll"]))
    eta_hat, w, ll = refine_eta(S, p, 20000, float(ETA_GRID[g]), pg["w"][g])
    rec("TE4_2_abs_err_eta_hat_L8_n20000", abs(eta_hat - 0.05), kind="set")
    rec("TE4_2_abs_err_beta_hat_L8_n20000", abs((1 - w[0]) - 0.2), kind="set")
    assert abs(eta_hat - 0.05) < 0.01 and abs((1 - w[0]) - 0.2) < 0.03
    assert ll >= pg["ll"].max() - 1e-9                                           # refinement never lowers the profile maximum
    # reported eta maximises the profile over the (grid + refined) points
    assert ll >= pg["ll"][g] - 1e-9


def test_TE4_3_profile_fisher_ridge_and_identified():
    S3 = records(phys(3, 0.5), ChannelSpec.interp(3, 0.2, 0.5), 200_000, 6)
    f3 = profile_fisher(S3, phys(3, 0.5), ChannelSpec.interp(3, 0.2, 0.5))
    S8 = records(phys(8, 0.5), ChannelSpec.interp(8, 0.2, 0.5), 200_000, 7)
    f8 = profile_fisher(S8, phys(8, 0.5), ChannelSpec.interp(8, 0.2, 0.5))
    rec("TE4_3_Iprof_over_Ieta_L3", f3["I_prof"] / f3["I_eta"], kind="set")
    rec("TE4_3_Iprof_over_Ieta_L8", f8["I_prof"] / f8["I_eta"], kind="set")
    assert f3["I_prof"] <= 1e-3 * f3["I_eta"]
    assert f8["I_prof"] > 0 and f8["I_prof"] > 1e-3 * f8["I_eta"]


def test_TE4_4_f1_generator_and_simulator_unchanged():
    p = phys(8)
    H, cls, modes = interleave_persistence(p, 40000, 8501, 8901, 0, 0)
    assert H.n == 40000 and np.all(cls[0::2] == 0) and np.all(cls[1::2] == 1)
    edges = (modes[:, 1:] != modes[:, :-1]).astype(float)
    rate = edges.mean()
    se = edges.std(ddof=1) / np.sqrt(edges.size)
    rec("TE4_4_f1_marginal_edge_rate_z", abs(rate - 0.05) / se, kind="set")
    assert abs(rate - 0.05) <= 4 * se
    assert abs(edges[cls == 0].mean() - 0.02) < 0.004 and abs(edges[cls == 1].mean() - 0.08) < 0.004
    src = open(os.path.join(ROOT, "src", "mti", "simulate.py"), "rb").read()
    assert hashlib.sha256(src).hexdigest()[:16] == SIMULATE_SHA16


def test_TE4_5_access_and_disjoint_streams():
    for fn in (joint.profile_grid, joint.refine_eta, joint.jmom, joint.clean_eta_mle):
        params = inspect.signature(fn).parameters
        assert not any(k in params for k in ("probe", "modes", "theta", "eta", "eta_hat"))
    # results do not depend on the eta stored in the physics object
    p1, p2 = phys(4, 0.5, 0.05), phys(4, 0.5, 0.2)
    S = records(p1, ChannelSpec.interp(4, 0.2, 0.5), 3000, 9)
    a1 = profile_grid(S, p1, (3000,))[3000]["ll"]
    a2 = profile_grid(S, p2, (3000,))[3000]["ll"]
    assert np.array_equal(a1, a2)
    assert jmom(S.s, S.a, p1)[0] == jmom(S.s, S.a, p2)[0]
    masters = dict(adapt=8501, eval=8601, clean=8701, stage0=8801, f1b=8901)
    earlier = {7001, 7002, 8001, 8101, 8201, 8301, 8401}
    assert len(set(masters.values())) == len(masters) and not (set(masters.values()) & earlier)
    import sys
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import run_e4
    assert (run_e4.MASTER_ADAPT, run_e4.MASTER_EVAL, run_e4.MASTER_CLEAN, run_e4.MASTER_F1B) == (8501, 8601, 8701, 8901)


def test_TE4_6_jmom_recovers_eta():
    p = phys(8, 0.5)
    S = records(p, ChannelSpec.interp(8, 0.2, 0.5), 300_000, 11)
    eta_hat, w = jmom(S.s, S.a, p)
    rec("TE4_6_abs_err_eta_hat_jmom", abs(eta_hat - 0.05), kind="set")
    assert abs(eta_hat - 0.05) < 0.01 and abs((1 - w[0]) - 0.2) < 0.03


def test_TE4_7_ex_mle_reproduces_e3():
    """Same records as E3 (master 8301, truth T3 = cell 3, replicate 0): the known-eta full-simplex MLE returns the saved E3 estimate."""
    f = os.path.join(ROOT, "results", "e3", "units", "EX__T3.json")
    saved = json.load(open(f))
    ni = saved["ns"].index(10_000)
    beta_e3 = saved["A"]["bhat"][0][ni][0]
    p = phys(8)
    S = records(p, ChannelSpec.interp(8, 0.2, 0.5), 10_000, 8301, cell=3, chunk=0)
    llS, logell = _queries(S, p, 0.05)
    from mti.adapt import em_channel
    w, _, _ = em_channel(logell)
    rec("TE4_7_abs_diff_beta_hat_vs_e3", abs((1 - w[0]) - beta_e3), kind="set")
    assert abs((1 - w[0]) - beta_e3) < 1e-6
    wf, _, _ = fit_channel(llS, logell)
    assert abs((1 - wf[0]) - beta_e3) < 1e-4
