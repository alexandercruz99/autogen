"""Train station_v2 operating model with chronological calib residuals (baseline frozen)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.feeds.calibration import (
    chronological_day_splits,
    residuals_by_hour,
    save_calibration_artifact,
)
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import SUPPORTED_DECISION_HOURS_LOCAL, lst_climate_day
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    STATION_V2_FEATURES,
    build_station_v2_features,
)

logger = logging.getLogger(__name__)


def _enumerate_station_v2(obs, labels: dict[date, float]) -> tuple[list[dict[str, Any]], int]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(NYC_TARGET.timezone)
    # Candidate days from LST climate day of observations
    days = sorted({lst_climate_day(o.valid_utc) for o in obs})
    out: list[dict[str, Any]] = []
    neg_remain = 0
    for day in days:
        if day not in labels:
            continue
        label = float(labels[day])
        for hour in SUPPORTED_DECISION_HOURS_LOCAL:
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            # Note: hour is civil-local on the calendar date equal to LST day — disclosed approximation
            bundle = build_station_v2_features(
                obs,
                decision_utc,
                climate_day=day,
                availability_assumption="archive_valid_utc_equals_availability_DISCLOSED",
            )
            if bundle is None or not bundle.coverage.adequate:
                continue
            remain = label - bundle.max_so_far
            if remain < 0:
                neg_remain += 1  # keep — do not clip
            out.append(
                {
                    "climate_day": day.isoformat(),
                    "decision_hour": hour,
                    "decision_time_utc": decision_utc.isoformat(),
                    "features": [float("nan") if v is None else float(v) for v in bundle.values],
                    "feature_map": bundle.feature_map,
                    "max_so_far": bundle.max_so_far,
                    "label_tmax_f": label,
                    "remain_f": remain,
                    "coverage": bundle.coverage.as_dict(),
                    "negative_remain": remain < 0,
                }
            )
    return out, neg_remain


def _fit_quantile_models(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingRegressor

    models = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
        )
        m.fit(X, y)
        models[name] = m
    return models


def _eval_production_distribution(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    residuals_by_h: dict[str, list[float]],
) -> dict[str, Any]:
    """Evaluate the same residual distribution path used in production (not raw q50 MAE alone)."""
    from kalshi_bot.models.weather.distribution import from_empirical_residuals

    by_hour: dict[str, Any] = {}
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        hrs = [r for r in rows if int(r["decision_hour"]) == hour]
        if not hrs:
            continue
        res = residuals_by_h.get(str(hour)) or residuals_by_h.get("all") or []
        if len(res) < 10:
            by_hour[str(hour)] = {"n": len(hrs), "status": "calib_thin"}
            continue
        abs_err = []
        hit80 = []
        for r in hrs:
            X = np.nan_to_num(np.asarray([r["features"]], dtype=float), nan=-999.0)
            rem = float(models["q50"].predict(X)[0])
            point = max(float(r["max_so_far"]) + rem, float(r["max_so_far"]))
            dist = from_empirical_residuals(point, res, method="eval")
            med = dist.quantile(0.5)
            y = float(r["label_tmax_f"])
            abs_err.append(abs(y - med))
            q10, q90 = dist.quantile(0.1), dist.quantile(0.9)
            hit80.append(1.0 if q10 <= y <= q90 else 0.0)
        by_hour[str(hour)] = {
            "n_rows": len(hrs),
            "n_independent_days": len({r["climate_day"] for r in hrs}),
            "production_median_mae_f": float(np.mean(abs_err)),
            "interval_80_coverage": float(np.mean(hit80)),
            "note": "MAE of production residual distribution median, not raw GBM q50 alone",
        }
    return by_hour


def train_station_corrected(*, data_dir: Path | None = None, satrad_max_days: int = 0) -> dict[str, Any]:
    """Train station_v2.1 GBM + chronological calibration; freeze baseline untouched."""
    import joblib

    data_dir = data_dir or default_data_dir()
    out_dir = Path("data/obs_engine/feeds/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    rows, n_neg = _enumerate_station_v2(obs, labels)
    days = sorted({r["climate_day"] for r in rows})
    splits = chronological_day_splits(days)
    train = [r for r in rows if r["climate_day"] in splits["train"]]
    calib = [r for r in rows if r["climate_day"] in splits["calib"]]
    test = [r for r in rows if r["climate_day"] in splits["test"]]

    Xtr = np.nan_to_num(np.asarray([r["features"] for r in train], dtype=float), nan=-999.0)
    ytr = np.asarray([r["remain_f"] for r in train], dtype=float)
    models = _fit_quantile_models(Xtr, ytr)

    resid_map = residuals_by_hour(calib, models)
    calib_path = out_dir / "station_corrected_v2_calibration.json"
    save_calibration_artifact(
        calib_path,
        residuals_by_hour=resid_map,
        meta={
            "n_calib_days": len(splits["calib"]),
            "n_calib_rows": len(calib),
            "train_end_day": max(splits["train"]) if splits["train"] else None,
            "calib_end_day": max(splits["calib"]) if splits["calib"] else None,
            "availability_assumption": "archive_valid_utc_equals_availability_DISCLOSED",
        },
    )

    prod_eval = _eval_production_distribution(test, models, resid_map)

    # Raw q50 MAE for reference only (not claimed as production distribution validation)
    def raw_mae(rows_):
        by = {}
        for hour in SUPPORTED_DECISION_HOURS_LOCAL:
            hrs = [r for r in rows_ if int(r["decision_hour"]) == hour]
            if not hrs:
                continue
            X = np.nan_to_num(np.asarray([r["features"] for r in hrs], dtype=float), nan=-999.0)
            rem = models["q50"].predict(X)
            pred = np.maximum(np.asarray([r["max_so_far"] for r in hrs]) + rem, np.asarray([r["max_so_far"] for r in hrs]))
            y = np.asarray([r["label_tmax_f"] for r in hrs], dtype=float)
            by[str(hour)] = float(np.mean(np.abs(y - pred)))
        return by

    report: dict[str, Any] = {
        "ok": True,
        "model_version": "weather.obs_nyc.station_corrected.v2",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": STATION_V2_FEATURES,
        "n_train_days": len(splits["train"]),
        "n_calib_days": len(splits["calib"]),
        "n_test_days": len(splits["test"]),
        "n_train_rows": len(train),
        "n_calib_rows": len(calib),
        "n_test_rows": len(test),
        "negative_remain_labels_kept": n_neg,
        "raw_q50_mae_test_reference_only": raw_mae(test),
        "production_distribution_eval_test": prod_eval,
        "frozen_baseline_path": "data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib",
        "calibration_path": str(calib_path),
        "calib_residual_counts": {k: len(v) for k, v in resid_map.items()},
        "live_eligible": False,
        "supported_decision_hours_local": list(SUPPORTED_DECISION_HOURS_LOCAL),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sat_radar_training": {
            "status": "deferred",
            "note": (
                "Sat/radar candidate remains unpromoted. Future comparisons must use matched dates "
                "and a station-only control trained on the same days; n=9 cannot establish improvement."
            ),
        },
    }

    path = out_dir / "station_corrected_v2.joblib"
    joblib.dump(
        {
            "models": models,
            "feature_names": STATION_V2_FEATURES,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_set": FEATURE_SCHEMA_VERSION,
            "model_version": report["model_version"],
            "calibration_path": str(calib_path),
            # intentionally NO remain_residuals from train — use calibration file only
            "train_end_day": max(splits["train"]) if splits["train"] else None,
            "promoted_for_inference": True,
            "live_eligible": False,
        },
        path,
    )
    report["artifact"] = str(path)
    (out_dir / "station_corrected_v2_report.json").write_text(json.dumps(report, indent=2, default=str))
    # Keep a pointer report name used by ops
    (out_dir / "station_corrected_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
