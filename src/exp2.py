"""Experiment 2: three boundary probes on a fixed-truth skeleton (Sec. 3.6).

Skeleton: Z -> A, Z -> Y, A -> Y. True population RD = 0.05 in every cell.
  P(A=1|Z=0)=0.90, P(A=1|Z=1)=0.10
  q(0,0)=0.10, q(0,1)=0.15, q(1,0)=0.10+Delta, q(1,1)=0.15+Delta

E1 (grid):  Delta in {0,0.05,0.10,0.20} x n_z in {50,200,1000} x
            w in {0.5,0.8,0.95} -> 36 cells, 2000 reps (72,000).
            seed = 20261117 + 100000*cell + replicate.
E2 (structural): four arms x 36 cells (216 cells, 432,000 reps).
            seed = 20261118 + 2000000*arm + 2000*cell + replicate.
            arms: collider, mediator, unmeasured, wrong_link.
E3 (criteria): no sampling; replays E1 estimates, three definitions of a
            wrong decision at delta in {0.01,0.05,0.09}.

All seeds come from run_config.yaml; no absolute paths, no hard-coded local dirs.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable

import numpy as np


DELTA_VALUES = [0.0, 0.05, 0.10, 0.20]
NZ_VALUES = [50, 200, 1000]
W_VALUES = [0.5, 0.8, 0.95]
TRUE_RD = 0.05


def grid_cells() -> list[dict]:
    cells = []
    for i, (delta, nz, w) in enumerate(product(DELTA_VALUES, NZ_VALUES, W_VALUES)):
        cells.append({"cell": i, "Delta": delta, "n_z": nz, "w": w, "truth": TRUE_RD})
    return cells


def _empty_cell_risk(y: np.ndarray, a: np.ndarray, rule: str) -> float:
    if y.size == 0:
        if rule in ("all_haldane", "degenerate_only"):
            return 0.5  # Haldane-style 0.5 correction for empty arms
        return float("nan")
    return float(y.mean())


def sim_skeleton(delta_conf: float, nz: int, w: int | float, arm: str, rng: np.random.Generator,
                 empty_cell_rule: str = "empty_arm_only") -> Dict[str, np.ndarray | float]:
    """One replicate of the fixed-truth skeleton, optionally with an E2 arm."""
    N = 2 * int(nz)
    n0 = int((1 - w) * N)
    n1 = N - n0
    z = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    e = np.where(z == 1, 0.10, 0.90)
    a = (rng.random(N) < e).astype(int)

    q00, q01 = 0.10, 0.15
    q10, q11 = 0.10 + delta_conf, 0.15 + delta_conf
    base = np.where(z == 1, np.where(a == 1, q11, q01), np.where(a == 1, q10, q00))

    extra: Dict[str, np.ndarray] = {}
    if arm == "collider":
        # common effect C of A and Y: P(C=1)=0.3+0.3A+0.3Y
        # generate Y first from base, then C
        y = (rng.random(N) < base).astype(int)
        pc = 0.3 + 0.3 * a + 0.3 * y
        extra["C"] = (rng.random(N) < pc).astype(int)
        return {"Z": z, "A": a, "Y": y, "base_risk": base, **extra}
    if arm == "mediator":
        pm = 0.20 + 0.50 * a
        m = (rng.random(N) < pm).astype(int)
        # total effect = 0.02 + 0.06*0.5 = 0.05
        risk = base + 0.02 * a + 0.06 * m
        y = (rng.random(N) < np.clip(risk, 0, 1)).astype(int)
        extra["M"] = m
        return {"Z": z, "A": a, "Y": y, "base_risk": base, **extra}
    if arm == "unmeasured":
        u = (rng.random(N) < 0.5).astype(int)
        # proxy Z carried from U; u_share is fraction not carried
        return {"Z": z, "A": a, "U": u, "base_risk": base,
                "Y": (rng.random(N) < base).astype(int), **extra}
    # wrong_link / no arm
    y = (rng.random(N) < base).astype(int)
    return {"Z": z, "A": a, "Y": y, "base_risk": base, **extra}


def seed_grid(cell: int, rep: int, seed_base: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed_base, cell, rep]))


def seed_struct(arm: int, cell: int, rep: int, seed_base: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed_base, arm, cell, rep]))
