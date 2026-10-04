"""One-step decision quantities (spec section 7). Cost = s_{L+1}^2 + lambda a^2."""
from __future__ import annotations

import numpy as np

from .params import Physics


def kappa(phys: Physics, sL):
    return phys.rho**2 * phys.c**2 * np.asarray(sL) ** 2 / phys.D


def a_star(phys: Physics, sL, u):
    return -phys.rho * phys.c * np.asarray(sL) * np.asarray(u) / phys.D


def cond_cost(phys: Physics, s, u, a):
    """E[cost | info] when E[m_L | info] = u and action a is chosen: rho^2 s^2 + 2 rho c s u a + D a^2 + q_w."""
    return phys.rho**2 * s**2 + 2.0 * phys.rho * phys.c * s * u * a + phys.D * a**2 + phys.q_w


def base_cost(phys: Physics, sL):
    """Expected cost of the zero action: rho^2 s^2 + q_w."""
    return phys.rho**2 * np.asarray(sL) ** 2 + phys.q_w
