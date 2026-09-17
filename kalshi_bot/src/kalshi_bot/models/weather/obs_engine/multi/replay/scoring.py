"""Exact discrete CRPS and forecast scoring helpers for historical replay.

CRPS for PMF {t: p_t} and outcome y (user-specified energy form):

  CRPS = Σ_t p_t |t − y| − ½ Σ_t Σ_u p_t p_u |t − u|

Do not use Monte Carlo when exact computation is available.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kalshi_bot.models.weather.distribution import PredictiveDistribution
from kalshi_bot.models.weather.settlement_rules import TempInterval, yes_from_observation


def crps_pmf(temps_f: Sequence[int], probs: Sequence[float], y: float) -> float:
    """Exact energy-score CRPS for a discrete temperature PMF."""
    t = np.asarray(list(temps_f), dtype=float)
    p = np.asarray(list(probs), dtype=float)
    if t.size == 0 or p.size != t.size:
        return float("nan")
    s = float(p.sum())
    if s <= 0:
        return float("nan")
    p = p / s
    yf = float(y)
    term1 = float(np.sum(p * np.abs(t - yf)))
    # pairwise |t-u| weighted
    diff = np.abs(t[:, None] - t[None, :])
    term2 = float(0.5 * np.sum(p[:, None] * p[None, :] * diff))
    return term1 - term2


def crps_distribution(dist: PredictiveDistribution, y: float) -> float:
    return crps_pmf(dist.temps_f, dist.probs, y)


def modal_degree(dist: PredictiveDistribution) -> int:
    """Most likely integer °F; ties → lower temperature (stable)."""
    best_p = -1.0
    best_t = dist.temps_f[0]
    for t, p in zip(dist.temps_f, dist.probs):
        if p > best_p or (p == best_p and t < best_t):
            best_p = p
            best_t = t
    return int(best_t)


def yes_probability(dist: PredictiveDistribution, interval: TempInterval) -> float:
    return float(dist.p_interval(interval))


def contract_brier(dist: PredictiveDistribution, y: float, interval: TempInterval, *, side: str = "yes") -> float:
    p_yes = yes_probability(dist, interval)
    p = p_yes if side.lower() == "yes" else (1.0 - p_yes)
    o = 1.0 if yes_from_observation(float(y), interval) else 0.0
    if side.lower() == "no":
        o = 1.0 - o
    return (p - o) ** 2


def contract_log_loss(
    dist: PredictiveDistribution,
    y: float,
    interval: TempInterval,
    *,
    side: str = "yes",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Log loss with explicit zero-probability handling.

    If the purchased side has exact probability 0 and the outcome wins that side,
    flag impossible_event_error=True rather than silently clipping.
    """
    p_yes = yes_probability(dist, interval)
    p = p_yes if side.lower() == "yes" else (1.0 - p_yes)
    yes_wins = yes_from_observation(float(y), interval)
    o = 1.0 if yes_wins else 0.0
    if side.lower() == "no":
        o = 1.0 - o
    impossible = bool(o == 1.0 and p <= 0.0)
    if impossible:
        return {
            "log_loss": float("inf"),
            "impossible_event_error": True,
            "p": p,
            "outcome": o,
            "treatment": "exact_zero_probability_on_winning_side",
        }
    pp = min(max(p, eps), 1.0 - eps)
    ll = float(-(o * np.log(pp) + (1.0 - o) * np.log(1.0 - pp)))
    return {
        "log_loss": ll,
        "impossible_event_error": False,
        "p": p,
        "outcome": o,
        "treatment": f"clip_to_[{eps},1-{eps}]_only_when_p_in_(0,1)",
    }


def summarize_forecast_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0, "n_unique_days": 0}
    abs_err = np.asarray([abs(r["error_f"]) for r in rows], dtype=float)
    err = np.asarray([r["error_f"] for r in rows], dtype=float)
    crps = np.asarray([r["crps"] for r in rows], dtype=float)
    days = {r["climate_day"] for r in rows}
    modal_hit = [1.0 if r.get("modal_hit") else 0.0 for r in rows]
    cov80 = [r["cov80"] for r in rows if r.get("cov80") is not None]
    width80 = [r["width80"] for r in rows if r.get("width80") is not None]
    return {
        "n_rows": len(rows),
        "n_unique_days": len(days),
        "mae_f": float(np.mean(abs_err)),
        "bias_f": float(np.mean(err)),
        "mean_crps": float(np.mean(crps)),
        "exact_degree_hit_rate_modal": float(np.mean(modal_hit)),
        "pct_median_within_0p1f": float(np.mean(abs_err <= 0.1)),
        "pct_median_within_0p5f": float(np.mean(abs_err <= 0.5)),
        "pct_median_within_1f": float(np.mean(abs_err <= 1.0)),
        "pct_median_within_2f": float(np.mean(abs_err <= 2.0)),
        "abs_error_p80": float(np.percentile(abs_err, 80)),
        "abs_error_p95": float(np.percentile(abs_err, 95)),
        "interval_80_coverage": float(np.mean(cov80)) if cov80 else None,
        "interval_80_mean_width": float(np.mean(width80)) if width80 else None,
        "note_0p1f": (
            "±0.1°F rate vs whole-degree settlement labels does not establish "
            "tenth-degree measurement accuracy."
        ),
    }
