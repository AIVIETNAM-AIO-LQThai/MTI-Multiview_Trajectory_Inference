"""Shared helpers for the decision-identification closeout scripts (docs/decision_identification_closeout.md, Part I.6).

Evaluation / exact-law only: no fitting, no training. numpy + scipy; no torch (the TestSet replica below makes identical `simulate` calls, so it does
not import mti.learned.evalset, which imports torch). Paths are relative to the repository root; outputs go to results/decision_id/ and never overwrite
historical results (the received archive lives in results/decision_id/archive_received_2026-10-11/).
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
OUT = os.path.join(ROOT, "results", "decision_id")

from mti.decision import kappa  # noqa: E402
from mti.params import ChannelSpec, Physics  # noqa: E402
from mti.simulate import simulate  # noqa: E402

MASTER_EVAL, N_EVAL, ETA_TRUE = 8601, 20_000, 0.05
PHYS = {"lead": Physics(0.9, 1.0, 1.0, 0.02, 0.1, ETA_TRUE, 8), "sec": Physics(0.9, 1.0, 1.0, 0.5, 0.1, ETA_TRUE, 8)}


def truths(L):
    """Replica of experiments/run_e4.truths."""
    w = [2.0 ** -(L - 1 - j) for j in range(L)]
    return [("T0", ChannelSpec.uniform(L, 0.0)), ("T3", ChannelSpec.interp(L, 0.2, 0.5)), ("T4", ChannelSpec.interp(L, 0.35, 1.0)),
            ("T6", ChannelSpec(0.2, tuple(x / sum(w) for x in w)))]


class TestSet:
    """Numpy-only replica of mti.learned.evalset.TestSet.__init__ (identical `simulate` call). Equality is asserted in tests/test_decision_id.py."""
    __test__ = False

    def __init__(self, phys, n, cell_id, chunk_id=0, master=MASTER_EVAL):
        self.phys, self.n, self.L, self.V = phys, n, phys.L, phys.L + 1
        smp = simulate(phys, n, master, cell_id, chunk_id)
        self.H = smp.prefix
        self.mL = smp.modes[:, -1]
        self.sw = smp.modes[:, -2] != smp.modes[:, -3] if phys.L >= 3 else np.zeros(n, bool)
        self.kap = kappa(phys, self.H.s[:, -1])
        s = np.repeat(self.H.s[:, None, :], self.V, axis=1)
        a = np.repeat(self.H.a[:, None, :], self.V, axis=1)
        jj = np.arange(self.L)
        a[:, jj + 1, jj] *= -1.0
        self.s_rec, self.a_rec = s.reshape(n * self.V, -1), a.reshape(n * self.V, -1)


def reference_margin(phys, truth, ts, ex, tl):
    """Common reference margin of the E4 comparisons for the *reference truth*: clip(0.1 V2, 0.002 VOI, 0.02 VOI). Returns (delta, V2, voi)."""
    w0 = truth.view_weights()
    m2t = tl.reference(truth)
    V2 = float(((tl.K * (ex.naive - m2t) ** 2) @ w0).mean())
    voi = tl.voi_clean()
    return float(min(max(0.10 * V2, 0.002 * voi), 0.02 * voi)), V2, voi


def provenance(script_path):
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    dirty = git("status", "--porcelain", "--", "src", "experiments", "tests").strip() != ""
    return dict(git_commit=git("rev-parse", "HEAD").strip(), working_tree_dirty_code=dirty,
                script=os.path.relpath(script_path, ROOT).replace("\\", "/"), script_sha256=hashlib.sha256(open(script_path, "rb").read()).hexdigest()[:16],
                numpy=np.__version__, python=sys.version.split()[0])


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(name, obj, script_path, out_dir=None):
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    obj = dict(obj)
    obj["_provenance"] = provenance(script_path)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1)
    return path
