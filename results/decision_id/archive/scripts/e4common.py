"""Numpy-only replicas of run_e4.PHYS/truths and learned.evalset.TestSet.__init__ (identical simulate calls), avoiding the torch import."""
import sys
import numpy as np
sys.path.insert(0, '/home/claude/mti/src')
from mti.decision import kappa
from mti.params import ChannelSpec, Physics
from mti.simulate import simulate

MASTER_EVAL, N_EVAL, ETA_TRUE = 8601, 20_000, 0.05
PHYS = {"lead": Physics(0.9, 1.0, 1.0, 0.02, 0.1, ETA_TRUE, 8), "sec": Physics(0.9, 1.0, 1.0, 0.5, 0.1, ETA_TRUE, 8)}

def truths(L):
    w = [2.0 ** -(L - 1 - j) for j in range(L)]
    return [("T0", ChannelSpec.uniform(L, 0.0)), ("T3", ChannelSpec.interp(L, 0.2, 0.5)), ("T4", ChannelSpec.interp(L, 0.35, 1.0)),
            ("T6", ChannelSpec(0.2, tuple(x / sum(w) for x in w)))]

class TestSet:
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
