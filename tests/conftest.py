import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _record import RECORD  # noqa: E402

from mti.params import ChannelSpec, Physics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("MTI_MUTATION") or session.testscollected < 60:
        return   # only a full-suite run may overwrite the recorded maxima (subset runs would drop entries)
    out = os.path.join(ROOT, "results", "oracle_v1")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "test_max_errors.json"), "w") as f:
        json.dump(dict(exit_status=int(exitstatus), max_errors=RECORD), f, indent=1, sort_keys=True)


def random_phys(rng, L, eta, q_w):
    """Random physics inside the core domain (|rho|<1, c != 0, q_a > 0, lambda >= 0)."""
    return Physics(rho=float(rng.uniform(-0.95, 0.95)), c=float(rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)),
                   q_a=float(rng.uniform(0.5, 2.0)), q_w=q_w, lam=float(rng.uniform(0.0, 1.0)), eta=eta, L=L)


def random_chan(rng, L, beta=None, zeros=False):
    pi = rng.dirichlet(np.ones(L))
    if zeros and L >= 3:
        pi[rng.choice(L, size=max(1, L // 3), replace=False)] = 0.0
        pi = pi / pi.sum()
    return ChannelSpec(beta=float(rng.uniform(0.05, 0.5)) if beta is None else beta, pi=tuple(pi))


@pytest.fixture
def primary():
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.5, lam=0.1, eta=0.05, L=8)
    return phys, ChannelSpec.uniform(8, 0.2)
