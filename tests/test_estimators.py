"""T15 main (Rao-Blackwellised) vs independent single-draw estimator; recalibration algebra; chunk invariance."""
import numpy as np

from _record import rec
from mti.experiment import Cell, check_chunk, column_names, merge_sums, rb_chunk
from mti.metrics import mean_ci
from mti.recal import BinnedRecal


def test_T15_rb_vs_single_draw_agree(primary):
    phys, chan = primary
    cell = Cell(0, "t", "t", phys, chan)
    names = column_names(cell, [])
    main = merge_sums([rb_chunk(cell, 2001, k, 2000, {}, {}, False) for k in range(20)])
    chk = merge_sums([check_chunk(cell, 3001, k, 2000) for k in range(20)])
    m, _, se, _ = mean_ci(main["n"], main["S1"], main["S2"])
    from mti.experiment import CHECK_COLS
    cm, _, cse, _ = mean_ci(chk["n"], chk["S1"], chk["S2"])
    zs = {}
    for k in ("V2", "I", "X", "AW", "D"):
        i, j = names.index(k), CHECK_COLS.index(k)
        zs[k] = (m[i] - cm[j]) / np.hypot(se[i], cse[j])
    rec("T15_max_abs_z_main_vs_check", max(abs(v) for v in zs.values()))
    assert all(abs(v) <= 4 for v in zs.values()), zs


def test_chunks_are_independent_of_ordering_and_grouping(primary):
    phys, chan = primary
    cell = Cell(0, "t", "t", phys, chan)
    a = merge_sums([rb_chunk(cell, 2001, k, 500, {}, {}, False) for k in (0, 1, 2, 3)])
    b = merge_sums([rb_chunk(cell, 2001, k, 500, {}, {}, False) for k in (3, 2, 1, 0)])
    assert np.allclose(a["S1"], b["S1"], rtol=1e-12, atol=0) and np.allclose(a["S2"], b["S2"], rtol=1e-12, atol=0)


def test_recalibration_three_term_identity_and_symmetry(primary):
    phys, chan = primary
    cell = Cell(0, "t", "t", phys, chan)
    rng = np.random.default_rng(3)
    abs_s, mu = np.abs(rng.normal(0, 3, 4000)), rng.uniform(-0.9, 0.9, 4000)
    mu2 = np.clip(mu + rng.normal(0, 0.2, 4000), -1, 1)
    kap = rng.uniform(0, 3, 4000)
    g = BinnedRecal(6, 6).fit(abs_s, mu, mu2, kap)
    gv = g(abs_s, mu)
    assert np.allclose(g(abs_s, -mu), -gv)                          # odd in mu~
    lhs = kap * (mu - mu2) ** 2
    rhs = kap * (mu - gv) ** 2 + kap * (gv - mu2) ** 2 + 2 * kap * (mu - gv) * (gv - mu2)
    assert np.allclose(lhs, rhs, atol=1e-12)
    out = rb_chunk(cell, 2001, 0, 400, {"g": g}, {}, False)
    names = column_names(cell, ["g"])
    i = names.index("V2")
    tot = sum(out["S1"][names.index(f"rc:g:{t}")] for t in "ARX")
    assert abs(tot - out["S1"][i]) <= 1e-9 * max(1.0, abs(out["S1"][i]))
