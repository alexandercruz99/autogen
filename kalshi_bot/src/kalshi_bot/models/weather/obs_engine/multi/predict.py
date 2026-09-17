"""Unified prediction path shared by calibration, evaluation, replay, and operating inference.

All callers must supply an explicit decision clock and ForecastContext. Feature processing,
quantile → residual distribution construction, observation constraints, and CLI floors use
the same code path — no separate evaluator approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.distribution import PredictiveDistribution, from_empirical_residuals, truncate_below
from kalshi_bot.models.weather.obs_engine.feeds.calibration import load_calibration, pick_residuals
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    STATION_V2_FEATURES,
    vector_for_model,
)
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext


DEFAULT_MODEL_PATH = Path("data/obs_engine/feeds/models/station_corrected_v2.joblib")
DEFAULT_CALIB_PATH = Path("data/obs_engine/feeds/models/station_corrected_v2_calibration.json")


@dataclass
class UnifiedPrediction:
    ok: bool
    status: str
    context: ForecastContext
    point_median_f: float | None = None
    remain_q10_q50_q90: list[float] | None = None
    quantiles_crossed: bool = False
    distribution: PredictiveDistribution | None = None
    probabilities_available: bool = False
    calibration_status: str | None = None
    observation_constraint: dict[str, Any] = field(default_factory=dict)
    model_artifact: str | None = None
    model_version: str | None = None
    reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "context": self.context.as_dict(),
            "point_median_f": self.point_median_f,
            "remain_q10_q50_q90": self.remain_q10_q50_q90,
            "quantiles_crossed": self.quantiles_crossed,
            "distribution": self.distribution.as_dict() if self.distribution else None,
            "probabilities_available": self.probabilities_available,
            "calibration_status": self.calibration_status,
            "observation_constraint": self.observation_constraint,
            "model_artifact": self.model_artifact,
            "model_version": self.model_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "reason": self.reason,
            "extras": self.extras,
        }


def load_operating_model(path: Path | None = None) -> tuple[dict[str, Any] | None, str, str]:
    import joblib

    model_path = path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None, str(model_path), "model_missing"
    blob = joblib.load(model_path)
    names = list(blob.get("feature_names") or [])
    if names != STATION_V2_FEATURES:
        return None, str(model_path), "schema_mismatch"
    return blob, str(model_path), "ok"


def predict_station_v2(
    *,
    context: ForecastContext,
    features: dict[str, float | None],
    max_so_far: float,
    coverage_adequate: bool,
    decision_hour_local: int | None,
    cli_applied: dict[str, Any] | None = None,
    model_blob: dict[str, Any] | None = None,
    model_path: Path | None = None,
    calib_path: Path | None = None,
    calibration: dict[str, Any] | None = None,
    require_supported_hour: bool = True,
    require_location_id: bool = True,
    clamp_point_to_max_so_far: bool = True,
) -> UnifiedPrediction:
    """Single prediction function for train/eval/replay/operating.

    ``clamp_point_to_max_so_far`` encodes the physical constraint that the day's
    maximum cannot be below the observed max-so-far (continuous). Integer CLI
    floors are applied separately via ``cli_applied``.
    """
    ctx = context
    if decision_hour_local is not None:
        ctx.decision_hour_local = decision_hour_local

    if not coverage_adequate:
        return UnifiedPrediction(
            ok=False,
            status="insufficient_data",
            context=ctx,
            reason="Essential station coverage inadequate",
        )

    if require_supported_hour and decision_hour_local not in (8, 11, 14):
        return UnifiedPrediction(
            ok=False,
            status="unsupported_decision_time",
            context=ctx,
            reason=f"decision_hour_local={decision_hour_local} not in (8,11,14)",
        )

    if model_blob is None:
        model_blob, artifact, load_status = load_operating_model(model_path)
        if model_blob is None:
            return UnifiedPrediction(
                ok=False,
                status="model_unavailable",
                context=ctx,
                model_artifact=artifact,
                reason=load_status,
            )
    else:
        artifact = str(model_path or "in_memory")

    feature_names = list(model_blob.get("feature_names") or STATION_V2_FEATURES)
    if feature_names != STATION_V2_FEATURES:
        return UnifiedPrediction(
            ok=False,
            status="schema_mismatch",
            context=ctx,
            model_artifact=artifact,
            reason="Artifact feature_names do not match station_v2.1",
        )

    vec = vector_for_model(features, feature_names)
    X = np.nan_to_num(np.asarray([vec], dtype=float), nan=-999.0)
    models = model_blob["models"]
    by_hour = model_blob.get("models_by_hour") or {}
    if decision_hour_local is not None and str(decision_hour_local) in by_hour:
        models = by_hour[str(decision_hour_local)]
    q10 = float(models["q10"].predict(X)[0])
    q50 = float(models["q50"].predict(X)[0])
    q90 = float(models["q90"].predict(X)[0])
    q_sorted = sorted([q10, q50, q90])
    crossed = (q10, q50, q90) != tuple(q_sorted)
    q10, q50, q90 = q_sorted

    # Point = max_so_far + remaining rise (q50). Physical floor applied after bias below.
    raw_point = float(max_so_far) + q50
    point = max(raw_point, float(max_so_far)) if clamp_point_to_max_so_far else raw_point

    obs_constraint: dict[str, Any] = {
        "max_so_far_f": float(max_so_far),
        "clamp_point_to_max_so_far": clamp_point_to_max_so_far,
        "integer_floor_applied": False,
        "note": "Fractional METAR max_so_far is not an official whole-°F CLI minimum",
    }

    if calibration is not None:
        calib = calibration
    elif calib_path is None:
        return UnifiedPrediction(
            ok=True,
            status="probabilities_unavailable",
            context=ctx,
            point_median_f=point,
            remain_q10_q50_q90=[q10, q50, q90],
            quantiles_crossed=crossed,
            probabilities_available=False,
            calibration_status="missing_location_calibration",
            observation_constraint=obs_constraint,
            model_artifact=artifact,
            model_version=str(model_blob.get("model_version") or ""),
            reason="missing_location_calibration",
            extras={"research_point_forecast_only": True},
        )
    else:
        calib = load_calibration(calib_path)
        if calib is None:
            return UnifiedPrediction(
                ok=True,
                status="probabilities_unavailable",
                context=ctx,
                point_median_f=point,
                remain_q10_q50_q90=[q10, q50, q90],
                quantiles_crossed=crossed,
                probabilities_available=False,
                calibration_status="missing_location_calibration",
                observation_constraint=obs_constraint,
                model_artifact=artifact,
                model_version=str(model_blob.get("model_version") or ""),
                reason="missing_location_calibration",
                extras={"research_point_forecast_only": True, "calib_path": str(calib_path)},
            )

    if require_location_id:
        calib_loc = (calib.get("meta") or {}).get("location_id")
        if calib_loc and calib_loc != ctx.location_id:
            reason = f"calibration_location_mismatch:{calib_loc}!={ctx.location_id}"
            return UnifiedPrediction(
                ok=True,
                status="probabilities_unavailable",
                context=ctx,
                point_median_f=point,
                remain_q10_q50_q90=[q10, q50, q90],
                quantiles_crossed=crossed,
                probabilities_available=False,
                calibration_status=reason,
                observation_constraint=obs_constraint,
                model_artifact=artifact,
                model_version=str(model_blob.get("model_version") or ""),
                reason=reason,
                extras={"research_point_forecast_only": True},
            )

    residuals, calib_status = pick_residuals(calib, decision_hour_local)

    if residuals is None:
        return UnifiedPrediction(
            ok=True,
            status="probabilities_unavailable",
            context=ctx,
            point_median_f=point,
            remain_q10_q50_q90=[q10, q50, q90],
            quantiles_crossed=crossed,
            probabilities_available=False,
            calibration_status=calib_status,
            observation_constraint=obs_constraint,
            model_artifact=artifact,
            model_version=str(model_blob.get("model_version") or ""),
            reason=calib_status,
            extras={"research_point_forecast_only": True},
        )

    # Settlement-source hour bias: apply BEFORE the physical floor so a negative
    # bias cannot leave the point below max_so_far after an earlier clamp.
    bias_map = (calib.get("meta") or {}).get("point_bias_by_hour") or {}
    hour_key = str(decision_hour_local) if decision_hour_local is not None else None
    bias = 0.0
    if hour_key and hour_key in bias_map:
        try:
            bias = float(bias_map[hour_key])
        except (TypeError, ValueError):
            bias = 0.0
    raw_point = float(max_so_far) + q50 + bias
    point = max(raw_point, float(max_so_far)) if clamp_point_to_max_so_far else raw_point
    if bias:
        obs_constraint["point_bias_f"] = bias
        obs_constraint["point_bias_hour"] = hour_key
        obs_constraint["bias_ordering"] = "bias_then_floor"

    dist = from_empirical_residuals(
        point,
        residuals,
        method=f"calibrated_residual_hour_{decision_hour_local}",
    )
    if cli_applied and cli_applied.get("max_temp_f") is not None:
        # Justified whole-°F settlement floor: condition the full distribution
        # (truncate+renormalize), then sync point to the constrained support.
        floor = float(int(cli_applied["max_temp_f"]))
        dist = truncate_below(dist, floor, reason="CLI same-day whole °F floor")
        point = max(point, floor)
        if len(dist.temps_f) == 1:
            dist.details["model_data_conflict"] = (
                "floor left single-bin support; not genuine forecast certainty"
            )
        obs_constraint["integer_floor_applied"] = True
        obs_constraint["cli_floor_f"] = int(cli_applied["max_temp_f"])
        obs_constraint["cli_is_preliminary"] = cli_applied.get("is_preliminary")
        obs_constraint["constraint_mode"] = "condition_distribution_then_sync_point"

    return UnifiedPrediction(
        ok=True,
        status="ok",
        context=ctx,
        point_median_f=point,
        remain_q10_q50_q90=[q10, q50, q90],
        quantiles_crossed=crossed,
        distribution=dist,
        probabilities_available=True,
        calibration_status=calib_status,
        observation_constraint=obs_constraint,
        model_artifact=artifact,
        model_version=str(model_blob.get("model_version") or ""),
    )


def predict_from_rows(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    calibration: dict[str, Any] | None,
    feature_key: str = "features",
    clamp_point_to_max_so_far: bool = True,
    require_location_id: bool = False,
) -> list[dict[str, Any]]:
    """Batch helper for train/eval using the same point + residual rules as operating."""
    out: list[dict[str, Any]] = []
    for r in rows:
        hour = int(r["decision_hour"])
        feats = r[feature_key]
        if isinstance(feats, list):
            fmap = {n: feats[i] if i < len(feats) else None for i, n in enumerate(STATION_V2_FEATURES)}
        else:
            fmap = dict(feats)
        ctx = ForecastContext(
            location_id=str(r.get("location_id") or "unknown"),
            series_ticker=str(r.get("series_ticker") or ""),
            measurement=str(r.get("measurement") or "daily_max_temp_f"),
            settlement_source_family=str(r.get("settlement_source_family") or "nws_cli"),
            climate_day=r["climate_day"] if hasattr(r["climate_day"], "isoformat") else r["climate_day"],
            decision_time_utc=r.get("decision_time_utc") or datetime.now(timezone.utc),
            horizon="same_day",
            decision_hour_local=hour,
            mode="HISTORICAL_REPLAY",
        )
        if isinstance(ctx.climate_day, str):
            from datetime import date as _date

            ctx.climate_day = _date.fromisoformat(ctx.climate_day)
        if isinstance(ctx.decision_time_utc, str):
            raw_ts = ctx.decision_time_utc.replace("Z", "+00:00")
            ctx.decision_time_utc = datetime.fromisoformat(raw_ts)
            if ctx.decision_time_utc.tzinfo is None:
                ctx.decision_time_utc = ctx.decision_time_utc.replace(tzinfo=timezone.utc)
        pred = predict_station_v2(
            context=ctx,
            features=fmap,
            max_so_far=float(r["max_so_far"]),
            coverage_adequate=bool(r.get("coverage_adequate", True)),
            decision_hour_local=hour,
            cli_applied=r.get("cli_applied"),
            model_blob=models if "models" in models else {"models": models, "feature_names": STATION_V2_FEATURES, "model_version": "batch"},
            calibration=calibration,
            require_supported_hour=True,
            require_location_id=require_location_id,
            clamp_point_to_max_so_far=clamp_point_to_max_so_far,
        )
        out.append(pred.as_dict())
    return out
