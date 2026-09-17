"""Probabilistic forecast metrics for discrete °F predictive distributions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kalshi_bot.models.weather.distribution import PredictiveDistribution
from kalshi_bot.models.weather.settlement_rules import TempInterval, yes_from_observation


def crps_discrete(dist: PredictiveDistribution, y: float) -> float:
    """CRPS for a discrete distribution on integer support (Gneiting & Raftery form).

    For integer-valued Y with PMF p: CRPS = sum_k (F(k) - 1{y <= k})^2
    where F is the CDF on the support grid (extended if needed).
    """
    y_i = int(round(y))
    temps = list(dist.temps_f)
    probs = list(dist.probs)
    if not temps:
        return float("nan")
    lo = min(temps[0], y_i - 2)
    hi = max(temps[-1], y_i + 2)
    pmap = {t: 0.0 for t in range(lo, hi + 1)}
    for t, p in zip(temps, probs):
        if t in pmap:
            pmap[t] += p
        elif t < lo:
            pmap[lo] += p
        else:
            pmap[hi] += p
    cdf = 0.0
    crps = 0.0
    for k in range(lo, hi + 1):
        cdf += pmap[k]
        ind = 1.0 if y_i <= k else 0.0
        crps += (cdf - ind) ** 2
    return float(crps)


def quantile_loss(y: float, q_hat: float, tau: float) -> float:
    e = y - q_hat
    return float((tau - (1.0 if e < 0 else 0.0)) * e)


def interval_coverage(y: float, q_lo: float, q_hi: float) -> bool:
    return q_lo <= y <= q_hi


def contract_brier(dist: PredictiveDistribution, y: float, interval: TempInterval) -> float:
    p = float(dist.p_interval(interval))
    o = 1.0 if yes_from_observation(float(y), interval) else 0.0
    return (p - o) ** 2


def contract_log_loss(dist: PredictiveDistribution, y: float, interval: TempInterval, eps: float = 1e-6) -> float:
    p = min(max(float(dist.p_interval(interval)), eps), 1.0 - eps)
    o = 1.0 if yes_from_observation(float(y), interval) else 0.0
    return float(-(o * np.log(p) + (1.0 - o) * np.log(1.0 - p)))


def p_inclusive_band(dist: PredictiveDistribution, a: int, b: int) -> float:
    """P(a ≤ Y ≤ b) = F(b) − F(a−1) on integer support."""
    cdf = 0.0
    f_b = 0.0
    f_am1 = 0.0
    seen = set()
    for t, p in zip(dist.temps_f, dist.probs):
        cdf += p
        seen.add(t)
        if t == b:
            f_b = cdf
        if t == a - 1:
            f_am1 = cdf
    # If b or a-1 missing from support, accumulate appropriately
    if b not in seen:
        f_b = sum(p for t, p in zip(dist.temps_f, dist.probs) if t <= b)
    if (a - 1) not in seen:
        f_am1 = sum(p for t, p in zip(dist.temps_f, dist.probs) if t <= (a - 1))
    return float(max(0.0, min(1.0, f_b - f_am1)))


def summarize_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str | None = None,
) -> dict[str, Any]:
    """Aggregate MAE/CRPS/coverage from scored row dicts with error_f, crps, etc."""
    if not rows:
        return {"n": 0}
    days = {r.get("climate_day") for r in rows}
    mae = float(np.mean([abs(r["error_f"]) for r in rows]))
    bias = float(np.mean([r["error_f"] for r in rows]))
    crps = float(np.mean([r["crps"] for r in rows if r.get("crps") is not None]))
    cov80 = [r["cov80"] for r in rows if r.get("cov80") is not None]
    width80 = [r["width80"] for r in rows if r.get("width80") is not None]
    brier = [r["brier_between_1f"] for r in rows if r.get("brier_between_1f") is not None]
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "n_unique_days": len(days),
        "mae_f": mae,
        "bias_f": bias,
        "crps": crps,
        "interval_80_coverage": float(np.mean(cov80)) if cov80 else None,
        "interval_80_width": float(np.mean(width80)) if width80 else None,
        "brier_between_1f_around_median": float(np.mean(brier)) if brier else None,
        "within_1f": float(np.mean([1.0 if abs(r["error_f"]) <= 1 else 0.0 for r in rows])),
        "within_2f": float(np.mean([1.0 if abs(r["error_f"]) <= 2 else 0.0 for r in rows])),
    }
    if group_key:
        out["group"] = group_key
    return out
