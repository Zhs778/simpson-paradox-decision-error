"""Performance measures for the simulation study (manuscript Sec. 3.4).

Primary outcome: three-band decision error at threshold delta.
Comparator: sign error. Also bias, RMSE, bootstrap percentile coverage,
failure rate and Monte Carlo standard errors.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def recommend(value: float | np.ndarray, delta: float) -> int | np.ndarray:
    """Three-band rule with a CLOSED band: |RD| <= delta is never a recommendation.

    Returns -1 for 'treat' (value < -delta), +1 for 'avoid' (value > +delta),
    0 for 'uncertain' otherwise. Values exactly on +/-delta are uncertain.
    """
    v = np.asarray(value)
    out = np.zeros(v.shape, dtype=int)
    out[v < -delta] = -1
    out[v > +delta] = +1
    return int(out) if out.ndim == 0 else out


def decision_error(estimates: Iterable[float], truth: float, delta: float) -> float:
    """P(rec(theta_hat) != rec(theta)) using the three-band rule."""
    est = np.asarray(estimates, dtype=float)
    return float(np.mean(recommend(est, delta) != recommend(truth, delta)))


def sign_error(estimates: Iterable[float], truth: float) -> float:
    """P(sign(theta_hat) != sign(theta)); zero estimate has sign 0."""
    est = np.asarray(estimates, dtype=float)
    return float(np.mean(np.sign(est) != np.sign(truth)))


def bootstrap_percentile_coverage(
    point_estimates: np.ndarray,
    boot_estimates: np.ndarray,
    truth: float,
    alpha: float = 0.05,
) -> float:
    """Coverage of percentile bootstrap intervals.

    point_estimates: shape (R,)
    boot_estimates : shape (R, B), bootstrap replicates per Monte Carlo repetition
    """
    lo = np.quantile(boot_estimates, alpha / 2.0, axis=1)
    hi = np.quantile(boot_estimates, 1.0 - alpha / 2.0, axis=1)
    return float(np.mean((lo <= truth) & (truth <= hi)))


def rate_mcse(rate: float, n: int) -> float:
    """Binomial Monte Carlo standard error sqrt(p(1-p)/n)."""
    if n <= 0:
        return float("nan")
    return float(np.sqrt(rate * (1.0 - rate) / n))


def paired_difference(est_adjusted: np.ndarray, est_crude: np.ndarray, truth: float, delta: float) -> float:
    """Within-replicate mean of [I(error adjusted) - I(error crude)].

    Positive => adjustment increases decision error.
    """
    a = np.asarray(est_adjusted, dtype=float)
    b = np.asarray(est_crude, dtype=float)
    ra = recommend(a, delta) != recommend(truth, delta)
    rb = recommend(b, delta) != recommend(truth, delta)
    return float(np.mean(ra.astype(float) - rb.astype(float)))


def cell_metrics(
    estimates: Sequence[float],
    truth: float,
    delta: float,
    *,
    boot_estimates: np.ndarray | None = None,
    failure_rate: float = 0.0,
) -> Dict[str, float]:
    """Full per-cell metric set for one estimator in one scenario/cell."""
    est = np.asarray([x for x in estimates if np.isfinite(x)], dtype=float)
    n = int(est.size)
    if n == 0:
        raise ValueError("No finite estimates supplied")
    bias = float(np.mean(est) - truth)
    rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
    de = decision_error(est, truth, delta)
    se = sign_error(est, truth)
    out = {
        "n_used": n,
        "truth": float(truth),
        "bias": bias,
        "rmse": rmse,
        "sd": float(np.std(est, ddof=1)) if n > 1 else 0.0,
        "decision_error": de,
        "decision_error_mcse": rate_mcse(de, n),
        "sign_error": se,
        "sign_error_mcse": rate_mcse(se, n),
        "failure_rate": float(failure_rate),
    }
    if boot_estimates is not None:
        out["coverage"] = bootstrap_percentile_coverage(est, boot_estimates, truth)
        out["coverage_mcse"] = rate_mcse(out["coverage"], n)
    return out


def family_summary(
    scenario_deltas: Dict[str, float],
    n_reps: int = 2000,
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Scenario-equal-weight family mean with a scenario-resampling interval.

    scenario_deltas maps scenario id -> paired difference d_s. The interval is a
    design-sensitivity range (B resamples of scenarios), not a superpopulation CI.
    """
    ids = list(scenario_deltas)
    vals = np.array([scenario_deltas[k] for k in ids], dtype=float)
    K = vals.size
    mean = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, K), replace=True).mean(axis=1)
    return {
        "K": K,
        "mean": mean,
        "median": float(np.median(vals)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "ri_lo": float(np.quantile(draws, 0.025)),
        "ri_hi": float(np.quantile(draws, 0.975)),
    }
