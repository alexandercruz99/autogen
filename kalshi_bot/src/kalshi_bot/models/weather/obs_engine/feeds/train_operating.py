"""Train station_v2 operating model with chronological calib residuals (baseline frozen).

Supports NYC (default) and location-specific training (e.g. Chicago Midway) writing
into partitioned multi/artifacts directories.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import (
    default_data_dir,
    load_ghcnd_tmax,
    load_hourly_asos_bundle,
    load_nyc_hourly_bundle,
)
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

# Location training profiles — settlement-matched ASOS + GHCND labels only.
LOCATION_TRAIN_PROFILES: dict[str, dict[str, Any]] = {
    "nyc_central_park": {
        "location_id": "nyc_central_park",
        "measurement": "daily_max_temp_f",
        "series_ticker": "HIGHNY",
        "display_name": "New York City Central Park",
        "timezone": "America/New_York",
        "lat": NYC_TARGET.lat,
        "lon": NYC_TARGET.lon,
        "metar_id": "KNYC",
        "iem_station": "NYC",
        "asos_glob": "asos_NYC_*.csv",
        "asos_subdir": None,  # files live in data/obs_engine/
        "label_csv": "nyc_central_park_ghcnd_tmax_f.csv",
        "ghcnd_id": "USW00094728",
        "model_version": "weather.obs_nyc.station_corrected.v2",
        "write_legacy_feeds_models": True,
    },
    "chi_midway": {
        "location_id": "chi_midway",
        "measurement": "daily_max_temp_f",
        "series_ticker": "HIGHCHI",
        "display_name": "Chicago Midway",
        "timezone": "America/Chicago",
        "lat": 41.7868,
        "lon": -87.7522,
        "elev_m": 189.0,
        "metar_id": "KMDW",
        "iem_station": "MDW",
        "asos_glob": "asos_MDW_*.csv",
        "asos_subdir": "chicago",
        "label_csv": "chicago/midway_ghcnd_tmax_f.csv",
        "ghcnd_id": "USW00014819",
        "cli_location_id": "MDW",
        "model_version": "weather.obs_chi_midway.station_corrected.v1",
        "write_legacy_feeds_models": False,
        "label_note": (
            "GHCND USW00014819 Midway daily TMAX (°F rounded) is a documented proxy for "
            "NWS CLI MAXIMUM MDW when historical CLI text is incomplete; settlement remains CLI."
        ),
    },
    "lax_airport": {
        "location_id": "lax_airport",
        "measurement": "daily_max_temp_f",
        "series_ticker": "KXHIGHLAX",
        "display_name": "Los Angeles International (LAX)",
        "timezone": "America/Los_Angeles",
        "lat": 33.9425,
        "lon": -118.4081,
        "elev_m": 38.0,
        "metar_id": "KLAX",
        "iem_station": "LAX",
        "asos_glob": "asos_LAX_*.csv",
        "asos_subdir": "la",
        "label_csv": "la/lax_ghcnd_tmax_f.csv",
        "ghcnd_id": "USW00023174",
        "cli_location_id": "LAX",
        "model_version": "weather.obs_lax_airport.station_corrected.v1",
        "write_legacy_feeds_models": False,
        "label_note": (
            "GHCND USW00023174 LAX daily TMAX (°F) trains the station model for TWC CLILAX / "
            "KXHIGHLAX settlement (same ICAO). Progressive evidence uses KLAX + TWC portal."
        ),
    },
}


def _enumerate_station_v2(
    obs,
    labels: dict[date, float],
    *,
    tz_name: str,
    lat: float,
    lon: float,
    station_id: str,
) -> tuple[list[dict[str, Any]], int]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    days = sorted({lst_climate_day(o.valid_utc, tz_name) for o in obs})
    out: list[dict[str, Any]] = []
    neg_remain = 0
    for day in days:
        if day not in labels:
            continue
        label = float(labels[day])
        for hour in SUPPORTED_DECISION_HOURS_LOCAL:
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            bundle = build_station_v2_features(
                obs,
                decision_utc,
                climate_day=day,
                tz_name=tz_name,
                lat=lat,
                lon=lon,
                station_id=station_id,
                availability_assumption="archive_valid_utc_equals_availability_DISCLOSED",
            )
            if bundle is None or not bundle.coverage.adequate:
                continue
            remain = label - bundle.max_so_far
            if remain < 0:
                neg_remain += 1
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
                    "location_id": station_id,
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
    from kalshi_bot.models.weather.obs_engine.multi.predict import predict_from_rows

    calib = {"residuals_by_hour": residuals_by_h, "meta": {}}
    preds = predict_from_rows(rows, models, calibration=calib, feature_key="features")
    by_hour: dict[str, Any] = {}
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        hour_pairs = [
            (r, p) for r, p in zip(rows, preds) if int(r["decision_hour"]) == hour
        ]
        if not hour_pairs:
            continue
        res = residuals_by_h.get(str(hour)) or residuals_by_h.get("all") or []
        if len(res) < 10:
            by_hour[str(hour)] = {"n": len(hour_pairs), "status": "calib_thin"}
            continue
        abs_err = []
        hit80 = []
        bias = []
        for r, pred in hour_pairs:
            dist = pred.get("distribution")
            if not pred.get("probabilities_available") or not dist:
                continue
            med = float(dist["q50"])
            y = float(r["label_tmax_f"])
            abs_err.append(abs(y - med))
            bias.append(med - y)
            q10, q90 = float(dist["q10"]), float(dist["q90"])
            hit80.append(1.0 if q10 <= y <= q90 else 0.0)
        if not abs_err:
            by_hour[str(hour)] = {"n": len(hour_pairs), "status": "calib_thin"}
            continue
        by_hour[str(hour)] = {
            "n_rows": len(hour_pairs),
            "n_independent_days": len({r["climate_day"] for r, _ in hour_pairs}),
            "production_median_mae_f": float(np.mean(abs_err)),
            "production_median_bias_f": float(np.mean(bias)),
            "interval_80_coverage": float(np.mean(hit80)),
            "note": "MAE of production residual distribution median, not raw GBM q50 alone",
        }
    return by_hour


def train_location_station_corrected(
    location_id: str = "nyc_central_park",
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Train + calibrate a location-specific station_v2 model."""
    import joblib

    if location_id not in LOCATION_TRAIN_PROFILES:
        return {
            "ok": False,
            "error": f"unknown location_id={location_id}; known={list(LOCATION_TRAIN_PROFILES)}",
        }
    profile = LOCATION_TRAIN_PROFILES[location_id]
    data_dir = data_dir or default_data_dir()

    asos_dir = data_dir / profile["asos_subdir"] if profile.get("asos_subdir") else data_dir
    if location_id == "nyc_central_park" and not profile.get("asos_subdir"):
        obs = load_nyc_hourly_bundle(data_dir)
    else:
        obs = load_hourly_asos_bundle(
            asos_dir,
            glob_pattern=profile["asos_glob"],
            station=profile["iem_station"],
        )
    label_path = data_dir / profile["label_csv"]
    labels = load_ghcnd_tmax(label_path)
    if not obs:
        return {"ok": False, "error": f"no ASOS obs for {location_id} under {asos_dir}/{profile['asos_glob']}"}
    if not labels:
        return {"ok": False, "error": f"no labels at {label_path}"}

    rows, n_neg = _enumerate_station_v2(
        obs,
        labels,
        tz_name=profile["timezone"],
        lat=float(profile["lat"]),
        lon=float(profile["lon"]),
        station_id=profile["metar_id"],
    )
    if len(rows) < 100:
        return {
            "ok": False,
            "error": f"too few training rows ({len(rows)}); need more history/coverage",
            "n_obs": len(obs),
            "n_labels": len(labels),
        }

    days = sorted({r["climate_day"] for r in rows})
    splits = chronological_day_splits(days)
    train = [r for r in rows if r["climate_day"] in splits["train"]]
    calib = [r for r in rows if r["climate_day"] in splits["calib"]]
    test = [r for r in rows if r["climate_day"] in splits["test"]]

    Xtr = np.nan_to_num(np.asarray([r["features"] for r in train], dtype=float), nan=-999.0)
    ytr = np.asarray([r["remain_f"] for r in train], dtype=float)
    models = _fit_quantile_models(Xtr, ytr)

    resid_map = residuals_by_hour(calib, models)
    artifact_dir = Path("data/obs_engine/multi/artifacts") / f"{profile['location_id']}__{profile['measurement']}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    calib_path = artifact_dir / "station_corrected_v2_calibration.json"
    save_calibration_artifact(
        calib_path,
        residuals_by_hour=resid_map,
        meta={
            "location_id": profile["location_id"],
            "measurement": profile["measurement"],
            "series_ticker": profile["series_ticker"],
            "timezone": profile["timezone"],
            "metar_id": profile["metar_id"],
            "ghcnd_id": profile.get("ghcnd_id"),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "decision_hours_local": list(SUPPORTED_DECISION_HOURS_LOCAL),
            "n_calib_days": len(splits["calib"]),
            "n_calib_rows": len(calib),
            "train_end_day": max(splits["train"]) if splits["train"] else None,
            "calib_end_day": max(splits["calib"]) if splits["calib"] else None,
            "availability_assumption": "archive_valid_utc_equals_availability_DISCLOSED",
            "label_note": profile.get("label_note"),
            "pooling": "none_location_specific",
        },
    )

    prod_eval = _eval_production_distribution(test, models, resid_map)

    def raw_mae(rows_):
        by = {}
        for hour in SUPPORTED_DECISION_HOURS_LOCAL:
            hrs = [r for r in rows_ if int(r["decision_hour"]) == hour]
            if not hrs:
                continue
            X = np.nan_to_num(np.asarray([r["features"] for r in hrs], dtype=float), nan=-999.0)
            rem = models["q50"].predict(X)
            pred = np.maximum(
                np.asarray([r["max_so_far"] for r in hrs]) + rem,
                np.asarray([r["max_so_far"] for r in hrs]),
            )
            y = np.asarray([r["label_tmax_f"] for r in hrs], dtype=float)
            by[str(hour)] = float(np.mean(np.abs(y - pred)))
        return by

    date_range = {
        "obs_first_utc": obs[0].valid_utc.isoformat() if obs else None,
        "obs_last_utc": obs[-1].valid_utc.isoformat() if obs else None,
        "label_first": min(labels) if labels else None,
        "label_last": max(labels) if labels else None,
        "train_days": sorted(splits["train"])[:1] + sorted(splits["train"])[-1:],
        "test_days": sorted(splits["test"])[:1] + sorted(splits["test"])[-1:],
    }

    report: dict[str, Any] = {
        "ok": True,
        "location_id": profile["location_id"],
        "display_name": profile["display_name"],
        "series_ticker": profile["series_ticker"],
        "measurement": profile["measurement"],
        "model_version": profile["model_version"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": STATION_V2_FEATURES,
        "timezone": profile["timezone"],
        "metar_id": profile["metar_id"],
        "ghcnd_id": profile.get("ghcnd_id"),
        "n_obs": len(obs),
        "n_labels": len(labels),
        "n_train_days": len(splits["train"]),
        "n_calib_days": len(splits["calib"]),
        "n_test_days": len(splits["test"]),
        "n_train_rows": len(train),
        "n_calib_rows": len(calib),
        "n_test_rows": len(test),
        "negative_remain_labels_kept": n_neg,
        "raw_q50_mae_test_reference_only": raw_mae(test),
        "production_distribution_eval_test": prod_eval,
        "calibration_path": str(calib_path),
        "calib_residual_counts": {k: len(v) for k, v in resid_map.items()},
        "date_range": date_range,
        "live_eligible": False,
        "supported_decision_hours_local": list(SUPPORTED_DECISION_HOURS_LOCAL),
        "noon_plan": {
            "target_local_hour": 11,
            "note": (
                "Actionable decision hours are 08/11/14 local. For trading around noon, "
                "use the 11:00 local forecast window (closest validated hour)."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_note": profile.get("label_note"),
        "nyc_model_not_transferred": location_id != "nyc_central_park",
    }

    model_path = artifact_dir / "station_corrected_v2.joblib"
    blob = {
        "models": models,
        "feature_names": STATION_V2_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_set": FEATURE_SCHEMA_VERSION,
        "model_version": report["model_version"],
        "calibration_path": str(calib_path),
        "location_id": profile["location_id"],
        "measurement": profile["measurement"],
        "timezone": profile["timezone"],
        "metar_id": profile["metar_id"],
        "train_end_day": max(splits["train"]) if splits["train"] else None,
        "promoted_for_inference": True,
        "live_eligible": False,
    }
    joblib.dump(blob, model_path)
    report["artifact"] = str(model_path)
    (artifact_dir / "station_corrected_v2_report.json").write_text(json.dumps(report, indent=2, default=str))

    # NYC also mirrors into legacy feeds/models path for existing worker
    if profile.get("write_legacy_feeds_models"):
        legacy = Path("data/obs_engine/feeds/models")
        legacy.mkdir(parents=True, exist_ok=True)
        joblib.dump(blob, legacy / "station_corrected_v2.joblib")
        save_calibration_artifact(
            legacy / "station_corrected_v2_calibration.json",
            residuals_by_hour=resid_map,
            meta={"location_id": "nyc_central_park", "mirrored_from": str(calib_path)},
        )
        (legacy / "station_corrected_v2_report.json").write_text(json.dumps(report, indent=2, default=str))
        (legacy / "station_corrected_report.json").write_text(json.dumps(report, indent=2, default=str))
        report["legacy_feeds_models_mirrored"] = True

    return report


def train_station_corrected(*, data_dir: Path | None = None, satrad_max_days: int = 0) -> dict[str, Any]:
    """Backward-compatible NYC training entrypoint."""
    return train_location_station_corrected("nyc_central_park", data_dir=data_dir)


def train_chicago_midway(*, data_dir: Path | None = None) -> dict[str, Any]:
    return train_location_station_corrected("chi_midway", data_dir=data_dir)
