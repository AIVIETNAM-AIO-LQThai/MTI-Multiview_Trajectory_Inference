"""Monte Carlo summaries from additive sufficient statistics (spec section 11).

Chunks return sums; the driver adds them. The unit of replication is the independent clean prefix: every
per-prefix value Y(H) already contains the Rao-Blackwellised average over the L+1 corruption views, so the
usual normal interval over prefixes is valid.
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def mean_ci(n, s1, s2):
    """Mean, sd, standard error and 95% half-width from n, sum y, sum y^2 (arrays broadcast)."""
    n = float(n)
    mean = s1 / n
    var = np.maximum((s2 - n * mean**2) / (n - 1.0), 0.0)
    sd = np.sqrt(var)
    se = sd / np.sqrt(n)
    return mean, sd, se, Z95 * se


def ratio_ci(sf, sfy, sfy2):
    """Stratum mean R = sum(f y)/sum(f) for a 0/1 flag f, with delta-method standard error (prefix unit)."""
    sf = np.asarray(sf, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sfy / sf
        resid2 = sfy2 - 2.0 * r * sfy + r**2 * sf   # sum f (y - R)^2
        se = np.sqrt(np.maximum(resid2, 0.0)) / sf
    return r, se


def batch_means_se(values: np.ndarray):
    """Standard error of the mean of per-chunk statistics (rows)."""
    k = values.shape[0]
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(k)
