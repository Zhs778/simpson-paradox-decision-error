"""Experiment 1 data-generating mechanism (manuscript Sec. 2.1 / 3.2).

Order: C -> A | C -> Y | A, C, with Bernoulli sampling throughout.
C = 1 with probability c_rate; e0/e1 are P(A=1|C=0/1);
p00,p10,p01,p11 are P(Y=1|A=0,C=0), P(Y=1|A=1,C=0),
P(Y=1|A=0,C=1), P(Y=1|A=1,C=1).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_scenario(scenario: Dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate one replicate from a single-covariate scenario."""
    n = int(scenario["n"])
    c_rate = float(scenario["c_rate"])
    e0, e1 = float(scenario["e0"]), float(scenario["e1"])
    p00, p10 = float(scenario["p00"]), float(scenario["p10"])
    p01, p11 = float(scenario["p01"]), float(scenario["p11"])

    c = (rng.random(n) < c_rate).astype(int)
    e = np.where(c == 1, e1, e0)
    a = (rng.random(n) < e).astype(int)
    p = np.where(
        c == 1,
        np.where(a == 1, p11, p01),
        np.where(a == 1, p10, p00),
    )
    y = (rng.random(n) < p).astype(int)
    return {"C": c, "A": a, "Y": y}


def analytic_truth(scenario: Dict) -> float:
    """theta = P(C=0)(p10-p00) + P(C=1)(p11-p01)."""
    pi = float(scenario["c_rate"])
    rd0 = float(scenario["p10"]) - float(scenario["p00"])
    rd1 = float(scenario["p11"]) - float(scenario["p01"])
    return float((1 - pi) * rd0 + pi * rd1)


def analytic_truth_treated(scenario: Dict) -> float:
    """theta_t = P(C=c|A=1) weighted stratum-specific RD."""
    pi = float(scenario["c_rate"])
    e0, e1 = float(scenario["e0"]), float(scenario["e1"])
    rd0 = float(scenario["p10"]) - float(scenario["p00"])
    rd1 = float(scenario["p11"]) - float(scenario["p01"])
    w0 = e0 * (1 - pi)
    w1 = e1 * pi
    denom = w0 + w1
    if denom <= 0:
        return float("nan")
    return float((w0 * rd0 + w1 * rd1) / denom)
