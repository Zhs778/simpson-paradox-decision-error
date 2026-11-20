"""Experiment 3 data-generating mechanism: two binary covariates (Sec. 3.7).

Balanced (Z1, Z2) cells of size n_cell. Assignment e(z1,z2) = 0.20+0.45*z1+d2*z2;
baseline risk r(z1,z2) = 0.10+0.15*z1+b2*z2; conditional effect
eff(z1,z2) = base_eff + z1_eff*z1 + dY2*z2; P(Y=1|A=1) = r + eff.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def level_params(level: Dict) -> Dict[str, float]:
    """Fill defaults for levels L0-L2 (z1_eff = -0.25) and L3 (explicit)."""
    return {
        "d2": float(level["d2"]),
        "b2": float(level["b2"]),
        "base_eff": float(level["base_eff"]),
        "dY2": float(level["dY2"]),
        "z1_eff": float(level.get("z1_eff", -0.25)),
    }


def sim_exp3_replicate(cfg: Dict, level: Dict, n_cell: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate one replicate with two covariates in balanced cells."""
    lp = level_params(level)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    z1 = np.concatenate([np.full(n_cell, z1v) for z1v, _ in cells]).astype(int)
    z2 = np.concatenate([np.full(n_cell, z2v) for _, z2v in cells]).astype(int)

    e = 0.20 + 0.45 * z1 + lp["d2"] * z2
    a = (rng.random(z1.size) < e).astype(int)

    r = 0.10 + 0.15 * z1 + lp["b2"] * z2
    eff = lp["base_eff"] + lp["z1_eff"] * z1 + lp["dY2"] * z2
    p = r + a * eff
    p = np.clip(p, 0.0, 1.0)
    y = (rng.random(z1.size) < p).astype(int)
    return {"Z1": z1, "Z2": z2, "A": a, "Y": y}


def analytic_truth_exp3(level: Dict) -> float:
    """theta = E[eff] under the balanced Z1,Z2 distribution."""
    lp = level_params(level)
    return float(lp["base_eff"] + 0.5 * lp["z1_eff"] + 0.5 * lp["dY2"])


def analytic_truth_treated_exp3(level: Dict) -> float:
    """theta_t = E[eff | A=1], weighted by assignment propensity."""
    lp = level_params(level)
    ws, es = [], []
    for z1 in (0, 1):
        for z2 in (0, 1):
            e = 0.20 + 0.45 * z1 + lp["d2"] * z2
            eff = lp["base_eff"] + lp["z1_eff"] * z1 + lp["dY2"] * z2
            ws.append(e)
            es.append(eff)
    ws = np.asarray(ws)
    return float(np.sum(ws * np.asarray(es)) / ws.sum())
