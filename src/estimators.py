"""Estimators compared in the study (manuscript Sec. 3.3).

B0 crude RD; A1 standardisation to the sample C distribution; A2
standardisation to the treated C distribution (estimand theta_t); A3
g-computation from a deliberately misspecified main-effects logistic model;
A1Z1 (Experiment 3) standardisation over the Z1 marginal only.

For a single binary covariate, saturated g-computation and Hajek IPW equal A1
algebraically (manuscript: agreement to 2.3e-10 / 1.1e-14); they are therefore
NOT separate strategies, only implementation checks.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _stratum_risks(a: np.ndarray, c: np.ndarray, y: np.ndarray, empty_cell_rule: str = "empty_arm_only") -> np.ndarray:
    """p_hat[a, c] with the pre-specified empty-cell rule.

    empty_arm_only (primary): if an arm is empty in a stratum, drop that stratum
    from the standardisation weights; every replicate still yields an estimate.
    """
    p = np.full((2, 2), np.nan)
    for av in (0, 1):
        for cv in (0, 1):
            mask = (a == av) & (c == cv)
            n = int(mask.sum())
            p[av, cv] = float(y[mask].mean()) if n > 0 else np.nan
    return p


def crude_rd(a: np.ndarray, y: np.ndarray) -> float:
    """B0: marginal P(Y=1|A=1) - P(Y=1|A=0)."""
    y1 = y[a == 1]
    y0 = y[a == 0]
    if y1.size == 0 or y0.size == 0:
        return float("nan")
    return float(y1.mean() - y0.mean())


def standardise(
    a: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    empty_cell_rule: str = "empty_arm_only",
) -> float:
    """Direct standardisation of stratum-specific RDs to `weights` over c."""
    p = _stratum_risks(a, c, y, empty_cell_rule)
    rd_c = p[1, :] - p[0, :]
    valid = np.isfinite(rd_c)
    if not np.any(valid):
        return float("nan")
    w = np.asarray(weights, dtype=float) * valid
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * np.where(valid, rd_c, 0.0)) / w.sum())


def standardize_sample(a: np.ndarray, c: np.ndarray, y: np.ndarray, empty_cell_rule: str = "empty_arm_only") -> float:
    """A1: weights = empirical distribution of C in the sample."""
    w = np.array([(c == 0).mean(), (c == 1).mean()])
    return standardise(a, c, y, w, empty_cell_rule)


def standardize_treated(a: np.ndarray, c: np.ndarray, y: np.ndarray, empty_cell_rule: str = "empty_arm_only") -> float:
    """A2: weights = distribution of C among the treated (estimand theta_t)."""
    treated = a == 1
    if treated.sum() == 0:
        return float("nan")
    w = np.array([(c[treated] == 0).mean(), (c[treated] == 1).mean()])
    return standardise(a, c, y, w, empty_cell_rule)


def _logit_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Small Newton-Raphson logistic fit (intercept included by caller)."""
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(iters):
        eta = x @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = x.T @ (y - p)
        hess = x.T @ (x * w[:, None]) + np.eye(x.shape[1]) * 1e-8
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def gcomp_main_effects(a: np.ndarray, c: np.ndarray, y: np.ndarray, delta: float = 0.0) -> float:
    """A3: g-computation from a main-effects logistic model Y ~ A + C.

    Deliberately omits the A*C interaction. Standardises to the sample C
    distribution. `delta` is accepted for API symmetry but unused.
    """
    x = np.column_stack([np.ones(a.size), a, c])
    beta = _logit_fit(x, y)
    x1 = x.copy(); x1[:, 1] = 1.0
    x0 = x.copy(); x0[:, 1] = 0.0
    p1 = 1.0 / (1.0 + np.exp(-np.clip(x1 @ beta, -30, 30)))
    p0 = 1.0 / (1.0 + np.exp(-np.clip(x0 @ beta, -30, 30)))
    return float(np.mean(p1 - p0))


def standardize_z1_only(z1: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    """A1Z1 (Exp3): standardise over the Z1 marginal, omitting Z2."""
    rd = np.full(2, np.nan)
    for z in (0, 1):
        m = z1 == z
        y1 = y[m & (a == 1)]
        y0 = y[m & (a == 0)]
        if y1.size and y0.size:
            rd[z] = y1.mean() - y0.mean()
    valid = np.isfinite(rd)
    if not np.any(valid):
        return float("nan")
    w = np.array([(z1 == 0).mean(), (z1 == 1).mean()])
    w = w * valid
    return float(np.sum(w * np.where(valid, rd, 0.0)) / w.sum())


def saturated_gcomp_check(a: np.ndarray, c: np.ndarray, y: np.ndarray) -> float:
    """Implementation check only: saturated g-computation should equal A1."""
    p = _stratum_risks(a, c, y)
    w = np.array([(c == 0).mean(), (c == 1).mean()])
    rd = p[1, :] - p[0, :]
    valid = np.isfinite(rd)
    w = w * valid
    return float(np.sum(w * np.where(valid, rd, 0.0)) / w.sum())


def run_all_strategies(a: np.ndarray, c: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Exp1/Exp3 core strategies on one replicate (single-covariate case)."""
    return {
        "B0": crude_rd(a, y),
        "A1": standardize_sample(a, c, y),
        "A2": standardize_treated(a, c, y),
        "A3": gcomp_main_effects(a, c, y),
    }
