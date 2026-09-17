"""Train station-specific predictive models from archived forecasts + CLI outcomes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.distribution import PredictiveDistribution, from_empirical_residuals, from_normal

logger = logging.getLogger(__name__)


@dataclass
class EmpiricalArtifact:
    residuals: list[float]
    n: int
    train_end: str
    source: str
    mae: float
    bias: float

    def to_json(self) -> dict[str, Any]:
        return {
            "residuals": self.residuals,
            "n": self.n,
            "train_end": self.train_end,
            "source": self.source,
            "mae": self.mae,
            "bias": self.bias,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "EmpiricalArtifact":
        return cls(
            residuals=list(d["residuals"]),
            n=int(d["n"]),
            train_end=str(d["train_end"]),
            source=str(d.get("source") or "nws_grid"),
            mae=float(d.get("mae") or 0),
            bias=float(d.get("bias") or 0),
        )


def build_pairs(
    archive: WeatherArchive,
    station_key: str,
    *,
    source: str = "nws_grid",
    require_non_preliminary: bool = True,
    allow_retrospective: bool = False,
) -> list[dict[str, Any]]:
    """Join forecasts to CLI outcomes.

    When allow_retrospective=False (default), drop pairs whose forecast available_at
    is after CLI issuance (leakage). Open-Meteo pulls often only know retrieval time,
    so they require allow_retrospective=True and must be labeled as non-decision-time.
    """
    rows = archive.paired_forecast_outcomes(station_key, source=source)
    by_day: dict[str, dict[str, Any]] = {}
    for r in rows:
        if require_non_preliminary and int(r.get("is_preliminary") or 0) == 1:
            continue
        day = r["valid_date"]
        avail = r.get("available_at") or ""
        issued = r.get("issuance_time") or ""
        if not allow_retrospective and issued and avail and avail > issued:
            continue
        prev = by_day.get(day)
        if prev is None or (avail >= (prev.get("available_at") or "")):
            row = dict(r)
            row["retrospective"] = bool(allow_retrospective and issued and avail and avail > issued)
            by_day[day] = row
    return [by_day[k] for k in sorted(by_day)]


def train_empirical(
    archive: WeatherArchive,
    station_key: str,
    *,
    source: str = "nws_grid",
    holdout_days: int = 7,
    allow_retrospective: bool = False,
) -> dict[str, Any]:
    pairs = build_pairs(
        archive, station_key, source=source, allow_retrospective=allow_retrospective
    )
    if len(pairs) < 5:
        return {
            "ok": False,
            "reason": f"insufficient paired forecast/CLI rows n={len(pairs)} (need ≥5). Collect more.",
            "n": len(pairs),
            "source": source,
            "retrospective": allow_retrospective,
        }
    cut = max(1, len(pairs) - holdout_days) if len(pairs) > holdout_days + 2 else len(pairs)
    train, hold = pairs[:cut], pairs[cut:]
    residuals = [float(p["outcome_f"]) - float(p["forecast_f"]) for p in train]
    art = EmpiricalArtifact(
        residuals=residuals,
        n=len(residuals),
        train_end=str(train[-1]["valid_date"]),
        source=source,
        mae=float(np.mean(np.abs(residuals))),
        bias=float(np.mean(residuals)),
    )
    metrics: dict[str, Any] = {
        "train_n": len(train),
        "holdout_n": len(hold),
        "mae_train": art.mae,
        "bias_train": art.bias,
        "retrospective": allow_retrospective,
        "source": source,
        "warning": (
            "Residuals fit on retrospective forecast joins (available_at after outcome). "
            "Useful for error shape only — NOT decision-time validated."
            if allow_retrospective
            else ""
        ),
    }
    if hold:
        errs = []
        for p in hold:
            dist = from_empirical_residuals(float(p["forecast_f"]), residuals)
            pred = dist.mean()
            errs.append(float(p["outcome_f"]) - pred)
        metrics["mae_holdout"] = float(np.mean(np.abs(errs))) if errs else None
        metrics["holdout_days"] = [p["valid_date"] for p in hold]
    archive.save_artifact(
        station_key, "empirical_residual", art.to_json(), metrics=metrics, train_end_day=art.train_end
    )
    return {"ok": True, "metrics": metrics, "artifact": art.to_json()}


def train_quantile_gbm(
    archive: WeatherArchive,
    station_key: str,
    *,
    source: str = "nws_grid",
    allow_retrospective: bool = False,
) -> dict[str, Any]:
    """Gradient-boosted quantile regression when sample size allows; else skip honestly."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return {"ok": False, "reason": "scikit-learn not installed"}

    pairs = build_pairs(
        archive, station_key, source=source, allow_retrospective=allow_retrospective
    )
    if len(pairs) < 40:
        return {
            "ok": False,
            "reason": f"n={len(pairs)} < 40 — quantile GBM needs more paired CLI days; empirical baseline only",
            "n": len(pairs),
        }
    # Features: forecast, day-of-year, lead proxy (1.0)
    X, y = [], []
    for p in pairs:
        d = date.fromisoformat(p["valid_date"])
        X.append([float(p["forecast_f"]), float(d.timetuple().tm_yday)])
        y.append(float(p["outcome_f"]))
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    cut = int(len(X_arr) * 0.8)
    models = {}
    metrics = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, max_depth=2, n_estimators=80, random_state=0)
        m.fit(X_arr[:cut], y_arr[:cut])
        pred = m.predict(X_arr[cut:])
        metrics[f"mae_{name}"] = float(np.mean(np.abs(pred - y_arr[cut:]))) if cut < len(y_arr) else None
        # Store tree count only — full sklearn pickle avoided; store training summary + fallback to residual
        models[name] = {"alpha": q, "n_train": cut}
    # Still persist residuals for distribution construction (GBM point used as mu)
    residuals = list(y_arr[:cut] - X_arr[:cut, 0])
    artifact = {
        "type": "quantile_gbm_meta",
        "models": models,
        "residuals": residuals,
        "note": "Full GBM weights not pickled in v1; distribution uses empirical residuals around forecast. Meta trained for metrics.",
    }
    archive.save_artifact(station_key, "quantile_gbm", artifact, metrics=metrics, train_end_day=pairs[cut - 1]["valid_date"])
    return {"ok": True, "metrics": metrics, "n": len(pairs)}


def predict_distribution(
    archive: WeatherArchive,
    station_key: str,
    point_forecast_f: float,
    *,
    model_name: str = "empirical_residual",
) -> PredictiveDistribution:
    row = archive.load_artifact(station_key, model_name)
    if row:
        art = EmpiricalArtifact.from_json(__import__("json").loads(row["artifact_json"]))
        if art.residuals:
            return from_empirical_residuals(point_forecast_f, art.residuals)
    # Fallback: wide normal prior — flagged in method name
    return from_normal(point_forecast_f, 4.5, support_low=int(point_forecast_f) - 25, support_high=int(point_forecast_f) + 25, method="fallback_normal_unfitted")
