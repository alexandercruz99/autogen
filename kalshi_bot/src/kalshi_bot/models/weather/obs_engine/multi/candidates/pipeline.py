"""Shared prediction primitives for all candidates (train/eval/replay parity)."""

from __future__ import annotations

from typing import Any

import numpy as np

from kalshi_bot.models.weather.distribution import (
    PredictiveDistribution,
    from_empirical_residuals,
    truncate_below,
)
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import STATION_V2_FEATURES
from kalshi_bot.models.weather.settlement_rules import TempInterval


def fit_quantile_gbms(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor

    models: dict[str, Any] = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            max_depth=2,
            n_estimators=120,
            learning_rate=0.05,
            random_state=0,
        )
        m.fit(X, y)
        models[name] = m
    return models


def matrix_from_rows(rows: list[dict[str, Any]], *, feature_key: str = "features") -> np.ndarray:
    return np.nan_to_num(np.asarray([r[feature_key] for r in rows], dtype=float), nan=-999.0)


def correct_quantile_crossing(q10: float, q50: float, q90: float) -> tuple[float, float, float, bool]:
    raw = (q10, q50, q90)
    ordered = tuple(sorted(raw))
    return ordered[0], ordered[1], ordered[2], raw != ordered


def remain_point_from_quantiles(
    max_so_far: float,
    q50_remain: float,
    *,
    bias: float = 0.0,
    clamp_to_max_so_far: bool = True,
) -> float:
    """Point forecast with bias applied BEFORE the physical floor (shared ordering).

    Ordering (corrected):
      1. raw = max_so_far + q50_remain + bias
      2. if clamp: point = max(raw, max_so_far)
    Applying bias after a floor can push the point below the intended bound when bias < 0;
    this helper applies bias first so the floor is the final operation on the point.
    """
    raw = float(max_so_far) + float(q50_remain) + float(bias)
    if clamp_to_max_so_far:
        return max(raw, float(max_so_far))
    return raw


def predict_remain_quantiles(models: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q10 = models["q10"].predict(X)
    q50 = models["q50"].predict(X)
    q90 = models["q90"].predict(X)
    crossed = np.zeros(len(X), dtype=bool)
    out10 = np.empty_like(q10)
    out50 = np.empty_like(q50)
    out90 = np.empty_like(q90)
    for i in range(len(X)):
        a, b, c, cr = correct_quantile_crossing(float(q10[i]), float(q50[i]), float(q90[i]))
        out10[i], out50[i], out90[i] = a, b, c
        crossed[i] = cr
    return out10, out50, out90, crossed


def hour_bias_map(residuals_by_hour: dict[str, list[float]], *, min_n: int = 8) -> dict[str, float]:
    out: dict[str, float] = {}
    for hour in ("8", "11", "14"):
        vals = residuals_by_hour.get(hour) or []
        if len(vals) >= min_n:
            out[hour] = float(np.median(vals))
        elif residuals_by_hour.get("all"):
            out[hour] = float(np.median(residuals_by_hour["all"]))
    return out


def build_residual_dist(
    point: float,
    residuals: list[float],
    *,
    method: str,
    floor_f: float | None = None,
    floor_reason: str | None = None,
    sync_point_to_floor: bool = True,
) -> tuple[PredictiveDistribution, float]:
    """Build empirical residual PMF and optionally condition on a justified integer floor.

    Constrains the *distribution* via truncate_below (conditioning by removing mass and
    renormalizing). Point is synced to max(point, floor) when a floor is applied so the
    reported point cannot disagree with the constrained distribution median's lower bound.
    Clipping samples vs conditioning are not equivalent; we condition the PMF.
    """
    dist = from_empirical_residuals(point, residuals, method=method)
    point_out = float(point)
    if floor_f is not None:
        floor_i = float(int(floor_f))
        dist = truncate_below(dist, floor_i, reason=floor_reason or "justified_settlement_floor")
        if sync_point_to_floor:
            point_out = max(point_out, floor_i)
        # If essentially degenerate certainty forced by floor with no mass above: conflict flag
        if len(dist.temps_f) == 1 and dist.details.get("same_day_floor_int") == dist.temps_f[0]:
            dist.details["model_data_conflict"] = (
                "floor left single-bin support; do not interpret as genuine certainty"
            )
    return dist, point_out


def from_standardized_residuals(
    mu: float,
    s: float,
    z_residuals: list[float],
    *,
    method: str = "adaptive_location_scale_residual",
    min_support_width: int = 8,
) -> PredictiveDistribution:
    """Y* = mu + s * z_i — location-scale residual transfer (candidate B)."""
    if s <= 0:
        raise ValueError("spread s must be positive")
    if not z_residuals:
        raise ValueError("need standardized residuals")
    samples = np.rint(mu + s * np.asarray(z_residuals, dtype=float)).astype(int)
    low = int(samples.min() - 2)
    high = int(samples.max() + 2)
    if high - low < min_support_width:
        mid = int(round(mu))
        low = mid - min_support_width // 2
        high = mid + min_support_width // 2
    temps = list(range(low, high + 1))
    counts = {t: 0.0 for t in temps}
    for v in samples:
        t = int(np.clip(int(v), low, high))
        counts[t] += 1.0
    alpha = 0.5 if len(z_residuals) < 30 else 1e-3
    probs = [counts[t] + alpha for t in temps]
    return PredictiveDistribution(
        temps_f=temps,
        probs=probs,
        method=method,
        details={
            "mu": mu,
            "s": s,
            "n_z": len(z_residuals),
            "z_mean": float(np.mean(z_residuals)),
            "z_std": float(np.std(z_residuals, ddof=1)) if len(z_residuals) > 1 else None,
            "note": "location-scale approximation; not guaranteed calibrated",
        },
    )


def from_quantile_grid(
    quantile_values: np.ndarray,
    *,
    method: str,
    min_support_width: int = 8,
) -> PredictiveDistribution:
    """Convert a dense conditional quantile grid into an integer °F PMF."""
    qs = np.asarray(quantile_values, dtype=float).ravel()
    samples = np.rint(qs).astype(int)
    low = int(samples.min() - 2)
    high = int(samples.max() + 2)
    if high - low < min_support_width:
        mid = int(round(float(np.median(qs))))
        low = mid - min_support_width // 2
        high = mid + min_support_width // 2
    temps = list(range(low, high + 1))
    counts = {t: 0.0 for t in temps}
    for v in samples:
        counts[int(np.clip(v, low, high))] += 1.0
    probs = [counts[t] + 1e-3 for t in temps]
    return PredictiveDistribution(
        temps_f=temps,
        probs=probs,
        method=method,
        details={"n_quantile_grid": int(len(qs)), "q50": float(np.median(qs))},
    )


def synthetic_between_interval(center: int, width: int = 0) -> TempInterval:
    """Inclusive [center-width, center+width] band for contract-score probes."""
    return TempInterval(
        "range_inclusive",
        float(center - width),
        float(center + width),
        "eval_probe",
        "synthetic inclusive band for Brier/log-loss probe",
    )


FEATURE_NAMES = list(STATION_V2_FEATURES)
