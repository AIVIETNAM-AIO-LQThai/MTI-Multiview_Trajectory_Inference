"""Decision-identification closeout checks (docs/decision_identification_closeout.md, spec II.7). Exact-law / evaluation only; small samples."""
import os
import sys

import numpy as np
import pytest

from _record import rec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments", "decision_id"))

import l2lib as T  # noqa: E402
import ridgelib as R  # noqa: E402
from did_common import PHYS, TestSet, truths  # noqa: E402

from mti.adapt import Precomputed  # noqa: E402
from mti.misspec import ExactRecords  # noqa: E402
from mti.params import ChannelSpec  # noqa: E402


def chan(q0, q1):
    b = q0 + q1
    return ChannelSpec(b, (0.5, 0.5) if b == 0 else (q0 / b, q1 / b))


def lp(e, w):
    from scipy.special import logsumexp
    with np.errstate(divide="ignore"):
        return logsumexp(e.ll + np.log(w)[None, :], axis=1)


def test_TD_0_numpy_testset_replica_equals_evalset_testset():
    pytest.importorskip("torch")
    from mti.learned.evalset import TestSet as RealTestSet
    phys = PHYS["lead"].with_(L=3)
    a, b = TestSet(phys, 400, cell_id=0, master=8601), RealTestSet(phys, 400, cell_id=0, master=8601)
    assert np.array_equal(a.s_rec, b.s_rec) and np.array_equal(a.a_rec, b.a_rec) and np.array_equal(a.kap, b.kap)
    assert np.array_equal(a.sw, b.sw) and np.array_equal(a.mL, b.mL)


def test_TD_1_l2_belief_formula_and_vertex_worlds_equal_record_law():
    phys = PHYS["sec"].with_(L=2)
    for tname, truth in truths(2):
        q0, q1 = truth.q
        h = T.h_of(phys.eta, q0, q1)
        ts = TestSet(phys, 1500, cell_id=0, master=8601)
        ex = ExactRecords(ts)
        z = np.tanh(phys.c * ts.a_rec * (ts.s_rec[:, 1:] - phys.rho * ts.s_rec[:, :-1]) / phys.q_w)
        phi = np.stack([z[:, 1], z[:, 0]], 1) / (1 + h * z[:, 0] * z[:, 1])[:, None]
        llref = lp(ex, truth.view_weights())
        worst_mu = worst_ll = 0.0
        for nm, (e, a0, a1) in dict(truth=(phys.eta, q0, q1), **T.vertex_worlds(h)).items():
            er = ExactRecords(ts, phys.with_(eta=float(e)))
            worst_mu = max(worst_mu, float(np.abs(er.mu2(chan(a0, a1)).ravel() - phi @ T.uv_of(e, a0, a1)).max()))
            worst_ll = max(worst_ll, float(np.abs(lp(er, np.array([1 - a0 - a1, a0, a1])) - llref).max()))
            assert abs(T.h_of(e, a0, a1) - h) < 1e-12
        rec("TD_1_l2_formula_vs_hmm", worst_mu)
        rec("TD_1_l2_vertex_worlds_loglik", worst_ll)
        assert worst_mu <= 1e-12 and worst_ll <= 1e-12


def test_TD_2_triangle_feasibility_round_trip():
    rng = np.random.default_rng(5)
    for h in (0.27, 0.54, 0.9):
        for _ in range(300):
            u = rng.uniform(h, 1.0)
            v = rng.uniform(h * u, 1 + h - u)
            w = T.world_from_uv(u, v, h)
            assert w is not None
            eta, q0, q1 = w
            assert 0 <= eta < 0.5 and q0 >= 0 and q1 >= 0 and q0 + q1 <= 1
            assert abs(T.h_of(eta, q0, q1) - h) < 1e-9
            assert np.allclose(T.uv_of(eta, q0, q1), [u, v], atol=1e-9)
        # outside the triangle there is no valid world
        assert T.world_from_uv(h - 0.05, h * h, h) is None
        assert T.world_from_uv(0.5 * (1 + h), 1.5, h) is None


def test_TD_3_l3_four_function_formula_matches_hmm_on_the_arc():
    phys = PHYS["lead"].with_(L=3)
    ts = TestSet(phys, 800, cell_id=0, master=8601)
    ex = ExactRecords(ts)
    worst = 0.0
    for tname, truth in truths(3)[1:4:2]:                                   # T3 (two-sided arc) and T6
        w0 = truth.view_weights()
        eh, _ = R.eta_hi(w0, phys.eta)
        C = R.J_coef(R.phat_from(w0, phys.eta))
        psi, _ = R.psi_features(ts.s_rec, ts.a_rec, phys.c, phys.rho, phys.q_w, R.phat_from(w0, phys.eta))
        for e2 in np.linspace(0.0, eh, 9):
            w2 = np.maximum(R.ridge_w(w0, phys.eta, e2), 0)
            w2 /= w2.sum()
            e = ex if abs(e2 - phys.eta) < 1e-12 else ExactRecords(ts, phys.with_(eta=float(e2)))
            worst = max(worst, float(np.abs(Precomputed.build(e.ll, e.mu).compose(w2) - psi @ R.J_at(C, 1 - 2 * e2)).max()))
            if e2 in (0.0, eh):
                assert float(np.abs(lp(e, w2) - lp(ex, w0)).max()) <= 1e-12
    rec("TD_3_l3_four_function_vs_hmm", worst)
    assert worst <= 1e-10


def test_TD_4_eta_hi_roots_vs_bisection_and_uniform_case():
    for tname, truth in truths(3):
        w = truth.view_weights()
        eh, _ = R.eta_hi(w, 0.05)
        assert abs(eh - R.eta_hi_bisect(w, 0.05)) < 1e-9
    assert R.eta_hi(truths(3)[0][1].view_weights(), 0.05)[0] == 0.05          # T0: zero masses block the upper branch
    uniform = np.array([0.25, 0.25, 0.25, 0.25])                            # uniform footprint law (beta = 3/4): non-compact
    assert R.is_uniform_F(uniform) and R.eta_hi(uniform, 0.05)[0] == np.inf


def test_TD_5_sup_risk_equals_dense_grid_max_and_pairwise_fw_on_a_segment():
    rng = np.random.default_rng(11)
    for _ in range(5):
        C = rng.normal(size=(4, 5)) * 0.1
        B = rng.normal(size=(4, 4))
        M = B @ B.T + 0.1 * np.eye(4)
        c = rng.normal(size=4) * 0.1
        s = R.sup_risk(C, M, c, [(0.85, 1.0)], grid_per_interval=20001)
        assert abs(s["dense_grid_max"] - s["sup"]) <= 1e-9 * max(s["sup"], 1e-12) + 1e-12 and s["dense_grid_max"] <= s["sup"] * (1 + 1e-12)
        assert s["max_root_residual_rel"] <= 1e-8
    # a 1-D set (segment) in a metric: pairwise FW value = d^2/4 with the two endpoints, and the endpoint midpoint satisfies the ball criterion
    x = np.linspace(0.0, 1.0, 41)
    G = 2.5
    Mg = G * np.outer(x, x)
    fw = R.pairwise_fw(Mg)
    assert abs(fw["value"] - G * 1.0 / 4) < 1e-10 and fw["fw_gap"] < 1e-10 and fw["pi"][0] + fw["pi"][-1] > 1 - 1e-9


def test_TD_6_endpoint_ball_criterion_and_triangle_minimax():
    G = np.eye(2)
    # right/obtuse triangle: longest-side prior; midpoint of the longest side contains the third vertex
    X = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.5]])
    tm = T.triangle_minimax(X, G)
    assert tm["kind"] == "longest_side" and abs(tm["value"] - 4.0) < 1e-12 and tm["weights"][2] == 0
    assert (X[2] - X[0]) @ G @ (X[2] - X[1]) <= 0                            # endpoint-ball criterion holds for the longest side (third vertex in the ball)
    # acute triangle: circumcentre, three-point prior, value = circumradius^2 and the endpoint-ball criterion FAILS for every side
    X = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
    tm = T.triangle_minimax(X, G)
    assert tm["kind"] == "circumcentre" and (tm["weights"] > 0).all() and abs(tm["weights"].sum() - 1) < 1e-12
    d_max = tm["max_side_sq"]
    assert tm["value"] > d_max / 4 + 1e-6                                   # strictly above the two-point lower bound
    for i, j in [(0, 1), (0, 2), (1, 2)]:                                   # criterion (4): <third - a, third - c> <= 0 holds for no side of an acute triangle
        k = 3 - i - j
        assert (X[k] - X[i]) @ G @ (X[k] - X[j]) > 0
    # anisotropic metric: acute in the plane but obtuse in the G metric
    Gm = np.array([[1.0, 0.9], [0.9, 1.0]])
    tm2 = T.triangle_minimax(np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]]), Gm)
    assert tm2["value"] >= tm2["max_side_sq"] / 4 - 1e-12


def test_TD_7_crossed_anova_is_unbiased_and_old_formula_double_counts():
    import se_e4
    R_, N, sa, sb, se = 10, 50, 1.0, 1.0, 1.0
    true_var = sa ** 2 / R_ + sb ** 2 / N + se ** 2 / (R_ * N)
    rng = np.random.default_rng(3)
    est, old = [], []
    for _ in range(4000):
        D = rng.normal(0, sa, (R_, 1)) + rng.normal(0, sb, (1, N)) + rng.normal(0, se, (R_, N))
        est.append(se_e4.crossed_anova(D)["var"])
        old.append(D.mean(1).var(ddof=1) / R_ + D.mean(0).var(ddof=1) / N)
    assert abs(np.mean(est) - true_var) < 0.004
    assert abs(np.mean(old) - true_var - se ** 2 / (R_ * N)) < 0.004        # the old figure exceeds the truth by sigma_e^2/(RN)


def test_TD_8_conftest_record_path_does_not_touch_committed_artifact_by_default():
    import conftest
    assert conftest.record_path({}).endswith(os.path.join(".pytest_cache", "test_max_errors.json"))
    assert conftest.record_path({"MTI_WRITE_TEST_RECORD": "0"}).endswith(os.path.join(".pytest_cache", "test_max_errors.json"))
    assert conftest.record_path({"MTI_WRITE_TEST_RECORD": "1"}).endswith(os.path.join("results", "oracle_v1", "test_max_errors.json"))


def test_TD_9_uniform_footprint_branch_belief_is_nonzero_at_L3():
    """Proposition 6(b) of docs/decision_identification_chapter.md: at L = 3 with P uniform, the branch world (eta' < 1/2, q_unif) has belief
    1/2 r z2 - 1/2 r^2 z0 z1 z2 + 1/2 r^2 z1 + 1/2 r^3 z0 (not identically zero), while eta' = 1/2 gives mu = 0 for every channel."""
    assert np.allclose(R.GHAT[R.A_IDX], [0.5, -0.5, 0.5, 0.5])
    phys = PHYS["sec"].with_(L=3)
    ts = TestSet(phys, 1500, cell_id=0, master=8601)
    z = np.tanh(phys.c * ts.a_rec * (ts.s_rec[:, 1:] - phys.rho * ts.s_rec[:, :-1]) / phys.q_w)
    q_unif = np.full(4, 0.25)
    worst = 0.0
    for e2 in (0.0, 0.2, 0.4):
        e = ExactRecords(ts, phys.with_(eta=e2))
        mu = Precomputed.build(e.ll, e.mu).compose(q_unif)
        r = 1 - 2 * e2
        form = 0.5 * r * z[:, 2] - 0.5 * r ** 2 * z[:, 0] * z[:, 1] * z[:, 2] + 0.5 * r ** 2 * z[:, 1] + 0.5 * r ** 3 * z[:, 0]
        worst = max(worst, float(np.abs(mu - form).max()))
        assert np.abs(mu).mean() > 0.01
    rec("TD_9_uniform_branch_formula_vs_hmm", worst)
    assert worst <= 1e-12
    e = ExactRecords(ts, phys.with_(eta=0.5))
    assert np.abs(Precomputed.build(e.ll, e.mu).compose(np.array([0.6, 0.1, 0.2, 0.1]))).max() <= 1e-15
