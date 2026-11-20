"""Reproducible random-number handling.

The manuscript specifies SeedSequence([seed_s, r]) for repetition r and
SeedSequence([seed_s, r, 1]) for the bootstrap draws of that repetition.
We follow that scheme exactly, and additionally fix `random` and `torch`
(optional import) so that all three common RNG sources are deterministic.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fix random, numpy and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(int(seed) % (2**32))
    try:
        import torch  # optional; the study itself is pure NumPy
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def make_rng(seed_parts: Iterable[int]) -> np.random.Generator:
    """Build a Generator from an explicit SeedSequence (non-overlapping streams)."""
    return np.random.default_rng(np.random.SeedSequence(list(int(x) for x in seed_parts)))


def repetition_rng(seed_s: int, repetition: int, bootstrap: bool = False) -> np.random.Generator:
    """Reproduction of the manuscript's seed convention."""
    parts = [int(seed_s), int(repetition)]
    if bootstrap:
        parts.append(1)
    return make_rng(parts)


def scenario_seed(seed_base: int, scenario_index: int) -> int:
    """Per-scenario seed_s = base + scenario index (indices are 0-based)."""
    return int(seed_base) + int(scenario_index)


def describe(seed_s: Optional[int] = None, repetition: Optional[int] = None) -> str:
    """Human-readable seed descriptor for logs (no environment specifics)."""
    if seed_s is None or repetition is None:
        return "SeedSequence([seed_s, r])"
    return f"SeedSequence([{seed_s}, {repetition}])"
