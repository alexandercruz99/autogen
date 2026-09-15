"""Predictive distributions over official daily max (°F) and coherent contract probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

import numpy as np
from scipy import stats

from kalshi_bot.models.weather.settlement_rules import TempInterval, yes_from_observation
from kalshi_bot.money import D, ONE, ZERO, clamp01


@dataclass
class PredictiveDistribution:
    """Discrete support over integer °F with probabilities summing to 1."""

    temps_f: list[int]
    probs: list[float]
    method: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.temps_f) != len(self.probs):
            raise ValueError("temps_f and probs length mismatch")
        s = float(sum(self.probs))
        if s <= 0:
            raise ValueError("empty distribution")
        self.probs = [p / s for p in self.probs]

    def mean(self) -> float:
        return float(sum(t * p for t, p in zip(self.temps_f, self.probs)))

    def quantile(self, q: float) -> int:
        c = 0.0
        for t, p in zip(self.temps_f, self.probs):
            c += p
            if c >= q:
                return t
        return self.temps_f[-1]

    def p_interval(self, interval: TempInterval) -> Decimal:
        total = 0.0
        for t, p in zip(self.temps_f, self.probs):
            if yes_from_observation(float(t), interval):
                total += p
        return clamp01(D(f"{total:.8f}"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mean_f": self.mean(),
            "q10": self.quantile(0.1),
            "q50": self.quantile(0.5),
            "q90": self.quantile(0.9),
            "support": list(zip(self.temps_f, [round(p, 6) for p in self.probs])),
            "details": self.details,
        }


def from_empirical_residuals(
    point_forecast_f: float,
    residuals_f: Sequence[float],
    *,
    support_low: int | None = None,
    support_high: int | None = None,
    method: str = "empirical_residual",
) -> PredictiveDistribution:
    """Shift historical (outcome − forecast) residuals onto today's point forecast.

    Residuals must be from settlement-aligned outcomes vs same-vintage forecasts.
    """
    if not residuals_f:
        raise ValueError("need residuals")
    res = np.asarray(list(residuals_f), dtype=float)
    # Map residual samples → integer temps via rounding (CLI is whole °F)
    samples = np.rint(point_forecast_f + res).astype(int)
    low = int(support_low if support_low is not None else samples.min() - 2)
    high = int(support_high if support_high is not None else samples.max() + 2)
    temps = list(range(low, high + 1))
    counts = {t: 0 for t in temps}
    for s in samples:
        t = int(np.clip(s, low, high))
        counts[t] += 1
    # Laplace smooth so empty bins near support get tiny mass
    probs = [(counts[t] + 1e-3) for t in temps]
    return PredictiveDistribution(
        temps_f=temps,
        probs=probs,
        method=method,
        details={
            "n_residuals": len(res),
            "point_forecast_f": point_forecast_f,
            "resid_mean": float(res.mean()),
            "resid_std": float(res.std(ddof=1)) if len(res) > 1 else 0.0,
        },
    )


def from_normal(
    mu: float,
    sigma: float,
    *,
    support_low: int,
    support_high: int,
    method: str = "normal_discretized",
) -> PredictiveDistribution:
    """Discretize N(mu, sigma^2) onto integer °F bins with continuity correction."""
    if sigma <= 0:
        sigma = 0.5
    temps = list(range(support_low, support_high + 1))
    probs = []
    for t in temps:
        # P(T = t) ≈ Φ(t+0.5) − Φ(t−0.5)
        p = float(stats.norm.cdf(t + 0.5, loc=mu, scale=sigma) - stats.norm.cdf(t - 0.5, loc=mu, scale=sigma))
        probs.append(max(p, 0.0))
    return PredictiveDistribution(temps_f=temps, probs=probs, method=method, details={"mu": mu, "sigma": sigma})


def truncate_below(dist: PredictiveDistribution, floor_f: float, *, reason: str) -> PredictiveDistribution:
    """Same-day: observed max so far implies official max ≥ floor (preliminary evidence)."""
    floor_i = int(np.floor(floor_f))
    temps = []
    probs = []
    for t, p in zip(dist.temps_f, dist.probs):
        if t >= floor_i:
            temps.append(t)
            probs.append(p)
    if not temps:
        # Degenerate: put mass on floor
        temps = [floor_i]
        probs = [1.0]
    return PredictiveDistribution(
        temps_f=temps,
        probs=probs,
        method=f"{dist.method}+same_day_floor",
        details={**dist.details, "same_day_floor_f": floor_f, "same_day_reason": reason},
    )


def coherent_bracket_probabilities(
    dist: PredictiveDistribution,
    intervals: list[TempInterval],
) -> list[Decimal]:
    """P(YES) for each contract from one shared distribution (sums ≈ 1 if exhaustive ME)."""
    return [dist.p_interval(iv) for iv in intervals]


def assert_brackets_sum_near_one(probs: list[Decimal], *, tol: Decimal = D("0.02")) -> None:
    s = sum(probs, ZERO)
    if abs(s - ONE) > tol:
        raise AssertionError(f"bracket probabilities sum to {s}, not ~1")
