"""Analytic decision-error approximation and design nomogram (Sec. 3.4, 4.5, 5.3).

The approximation treats an estimator as N(theta + b, sigma^2):
  outside the band (|theta| > delta):
      error ~= Phi((delta - theta - b)/sigma)   for theta > delta
             = Phi((theta + b + delta)/sigma)   for theta < -delta
  inside the band (|theta| <= delta):
      error ~= sum of the two crossing probabilities
Unified form for |theta| > delta:  Phi(-(u + s*b/sigma)),
  u = (|theta| - delta)/sigma, s = +1 if theta > delta else -1.
"""
from __future__ import annotations

from math import erf, sqrt
from typing import Dict


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def analytic_decision_error(theta: float, bias: float, sigma: float, delta: float) -> float:
    """Predicted three-band decision error under the normal approximation."""
    if sigma <= 0:
        return 0.0 if abs(theta + bias) > delta else 0.5
    m = theta + bias
    if theta > delta:
        return phi((delta - m) / sigma)
    if theta < -delta:
        return phi((m + delta) / sigma)
    # truth inside the band: sum of crossing probabilities
    return phi((-delta - m) / sigma) + phi((m - delta) / sigma)


def decision_noise_floor(theta: float, delta: float, sigma: float) -> float:
    """u = (|theta| - delta)/sigma; the design diagnostic of the manuscript."""
    if sigma <= 0:
        return float("inf")
    return (abs(theta) - delta) / sigma


def nomogram(theta: float, delta: float, sigma: float, bias: float = 0.0) -> Dict[str, float]:
    """Convenience wrapper returning u, bias/sigma, sign and predicted error."""
    s = 1.0 if theta > delta else -1.0
    u = decision_noise_floor(theta, delta, sigma)
    return {
        "theta": theta,
        "delta": delta,
        "sigma": sigma,
        "bias": bias,
        "u": u,
        "bias_over_sigma": bias / sigma if sigma > 0 else float("inf"),
        "sign_s": s,
        "predicted_decision_error": analytic_decision_error(theta, bias, sigma, delta),
    }
