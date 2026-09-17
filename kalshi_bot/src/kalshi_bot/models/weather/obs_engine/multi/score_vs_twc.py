"""Score production station_v2 predictions against official TWC settlement highs.

Goal: measure whether the model we run matches actual climate-report maxTemp
(the Kalshi TWC settlement source), not GHCND proxies alone.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_hourly_asos_bundle, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import SUPPORTED_DECISION_HOURS_LOCAL
from kalshi_bot.models.weather.obs_engine.feeds.train_operating import (
    LOCATION_TRAIN_PROFILES,
    _enumerate_station_v2,
)
from kalshi_bot.models.weather.obs_engine.feeds.twc_kalshi import TwcKalshiClient
from kalshi_bot.models.weather.obs_engine.multi.predict import load_operating_model, predict_from_rows

logger = logging.getLogger(__name__)

# TWC cliId → train profile / artifact location
TWC_SCORE_TARGETS: dict[str, dict[str, Any]] = {
    "NYC": {
        "location_id": "nyc_central_park",
        "twc_cli_id": "NYC",
        "icao": "KNYC",
        "artifact_subdir": "nyc_central_park__daily_max_temp_f",
        "fallback_model": "feeds/models/station_corrected_v2.joblib",
        "fallback_calib": "feeds/models/station_corrected_v2_calibration.json",
    },
    "MDW": {
        "location_id": "chi_midway",
        "twc_cli_id": "MDW",
        "icao": "KMDW",
        "artifact_subdir": "chi_midway__daily_max_temp_f",
    },
    "LAX": {
        "location_id": "lax_airport",
        "twc_cli_id": "LAX",
        "icao": "KLAX",
        "artifact_subdir": "lax_airport__daily_max_temp_f",
    },
}


def fetch_twc_actuals(days: list[date]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {YYYY-MM-DD: {cliId: {max_temp_f, status, ...}}}."""
    client = TwcKalshiClient()
    out: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for day in days:
            payload = client.climate_primary(day)
            by_cli: dict[str, dict[str, Any]] = {}
            for row in payload.get("results") or []:
                st = row.get("station") or {}
                cli = (st.get("cliId") or "").upper()
                if not cli:
                    continue
                data = row.get("data") or {}
                by_cli[cli] = {
                    "status": row.get("status"),
                    "max_temp_f": data.get("maxTemp"),
                    "min_temp_f": data.get("minTemp"),
                    "is_official": bool(data.get("isOfficial")) if data else False,
                    "icao": (st.get("icao") or "").upper(),
                }
            out[day.isoformat()] = by_cli
    finally:
        client.close()
    return out


def ensure_nyc_multi_artifacts(data_dir: Path) -> dict[str, str]:
    """Copy NYC feeds model+calib into multi artifacts so TWC sibling resolve works."""
    import shutil

    dest = data_dir / "multi" / "artifacts" / "nyc_central_park__daily_max_temp_f"
    dest.mkdir(parents=True, exist_ok=True)
    src_model = data_dir / "feeds" / "models" / "station_corrected_v2.joblib"
    src_calib = data_dir / "feeds" / "models" / "station_corrected_v2_calibration.json"
    src_report = data_dir / "feeds" / "models" / "station_corrected_v2_report.json"
    copied = {}
    for src, name in (
        (src_model, "station_corrected_v2.joblib"),
        (src_calib, "station_corrected_v2_calibration.json"),
        (src_report, "station_corrected_v2_report.json"),
    ):
        if src.exists():
            target = dest / name
            if not target.exists() or src.stat().st_mtime > target.stat().st_mtime:
                shutil.copy2(src, target)
                copied[name] = "copied"
            else:
                copied[name] = "present"
        else:
            copied[name] = "missing_source"
    return copied


def _load_obs_for_profile(profile: dict[str, Any], data_dir: Path):
    if profile["location_id"] == "nyc_central_park":
        return load_nyc_hourly_bundle(data_dir)
    sub = profile.get("asos_subdir")
    root = data_dir / sub if sub else data_dir
    station = str(profile.get("iem_station") or profile["metar_id"]).replace("K", "")
    return load_hourly_asos_bundle(
        root,
        glob_pattern=profile["asos_glob"],
        station=station,
    )


def score_location_vs_twc(
    *,
    cli_id: str,
    days: list[date],
    twc_actuals: dict[str, dict[str, dict[str, Any]]],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    meta = TWC_SCORE_TARGETS[cli_id]
    loc_id = meta["location_id"]
    profile = LOCATION_TRAIN_PROFILES[loc_id]

    art = data_dir / "multi" / "artifacts" / meta["artifact_subdir"]
    model_path = art / "station_corrected_v2.joblib"
    calib_path = art / "station_corrected_v2_calibration.json"
    if not model_path.exists() and meta.get("fallback_model"):
        model_path = data_dir / meta["fallback_model"]
    if not calib_path.exists() and meta.get("fallback_calib"):
        calib_path = data_dir / meta["fallback_calib"]

    blob, model_art, model_status = load_operating_model(model_path)
    if blob is None:
        return {
            "ok": False,
            "cli_id": cli_id,
            "location_id": loc_id,
            "error": f"model_unavailable:{model_status}",
            "model_path": str(model_path),
        }

    calib = None
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        return {
            "ok": False,
            "cli_id": cli_id,
            "location_id": loc_id,
            "error": "calibration_missing",
            "calib_path": str(calib_path),
        }

    # Labels = TWC official max only (skip days without official report)
    labels: dict[date, float] = {}
    label_meta: dict[str, Any] = {}
    for day in days:
        row = (twc_actuals.get(day.isoformat()) or {}).get(cli_id) or {}
        if row.get("status") != "official" or row.get("max_temp_f") is None:
            label_meta[day.isoformat()] = {"skipped": True, "reason": row.get("status") or "missing"}
            continue
        labels[day] = float(row["max_temp_f"])
        label_meta[day.isoformat()] = {"actual_twc_max_f": labels[day], "status": "official"}

    obs = _load_obs_for_profile(profile, data_dir)
    rows, neg = _enumerate_station_v2(
        obs,
        labels,
        tz_name=profile["timezone"],
        lat=float(profile["lat"]),
        lon=float(profile["lon"]),
        station_id=str(profile["location_id"]),
    )
    if not rows:
        return {
            "ok": False,
            "cli_id": cli_id,
            "location_id": loc_id,
            "error": "no_feature_rows",
            "label_days": list(labels),
            "label_meta": label_meta,
        }

    models = blob.get("models") or {k: blob[k] for k in ("q10", "q50", "q90") if k in blob}
    if not all(k in models for k in ("q10", "q50", "q90")):
        return {
            "ok": False,
            "cli_id": cli_id,
            "location_id": loc_id,
            "error": "model_missing_quantiles",
            "model_keys": list(models),
        }
    preds = predict_from_rows(rows, models, calibration=calib, feature_key="features")

    table: list[dict[str, Any]] = []
    abs_err: list[float] = []
    bias: list[float] = []
    for r, p in zip(rows, preds):
        dist = p.get("distribution") or {}
        med = dist.get("q50")
        if med is None:
            med = p.get("point_median_f")
        if med is None:
            continue
        actual = float(r["label_tmax_f"])
        err = float(med) - actual
        abs_err.append(abs(err))
        bias.append(err)
        table.append(
            {
                "climate_day": r["climate_day"],
                "decision_hour_local": r["decision_hour"],
                "max_so_far_f": r["max_so_far"],
                "predicted_median_f": round(float(med), 2),
                "actual_twc_max_f": actual,
                "error_f": round(err, 2),
                "abs_error_f": round(abs(err), 2),
                "probabilities_available": bool(p.get("probabilities_available")),
                "q10": dist.get("q10"),
                "q90": dist.get("q90"),
            }
        )

    by_hour: dict[str, Any] = {}
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        subset = [t for t in table if t["decision_hour_local"] == hour]
        if not subset:
            continue
        by_hour[str(hour)] = {
            "n": len(subset),
            "mae_f": float(np.mean([t["abs_error_f"] for t in subset])),
            "bias_f": float(np.mean([t["error_f"] for t in subset])),
            "within_1f": float(np.mean([1.0 if t["abs_error_f"] <= 1 else 0.0 for t in subset])),
            "within_2f": float(np.mean([1.0 if t["abs_error_f"] <= 2 else 0.0 for t in subset])),
        }

    return {
        "ok": True,
        "cli_id": cli_id,
        "location_id": loc_id,
        "label_source": "twc_climate_official_maxTemp",
        "settlement_note": "Kalshi KXHIGH* settles on TWC climate report for these stations",
        "model_artifact": model_art,
        "calib_path": str(calib_path),
        "n_rows": len(table),
        "n_days": len({t["climate_day"] for t in table}),
        "negative_remain_rows": neg,
        "mae_f": float(np.mean(abs_err)) if abs_err else None,
        "bias_f": float(np.mean(bias)) if bias else None,
        "within_1f": float(np.mean([1.0 if e <= 1 else 0.0 for e in abs_err])) if abs_err else None,
        "within_2f": float(np.mean([1.0 if e <= 2 else 0.0 for e in abs_err])) if abs_err else None,
        "by_hour": by_hour,
        "label_meta": label_meta,
        "rows": table,
        "match_quality": (
            "strong"
            if abs_err and float(np.mean(abs_err)) <= 1.0
            else "moderate"
            if abs_err and float(np.mean(abs_err)) <= 2.0
            else "weak"
            if abs_err
            else "no_data"
        ),
    }


def score_recent_twc(
    *,
    end_day: date | None = None,
    n_days: int = 5,
    data_dir: Path | None = None,
    ensure_nyc: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    end = end_day or (date.today() - timedelta(days=1))  # prefer completed climate days
    days = [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    nyc_copy = ensure_nyc_multi_artifacts(data_dir) if ensure_nyc else {}
    twc_actuals = fetch_twc_actuals(days)
    locations = []
    for cli in ("NYC", "MDW", "LAX"):
        locations.append(score_location_vs_twc(cli_id=cli, days=days, twc_actuals=twc_actuals, data_dir=data_dir))

    out_dir = data_dir / "multi" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"predict_vs_twc_{stamp}.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal": "Match production predictions to official TWC settlement highs",
        "days": [d.isoformat() for d in days],
        "nyc_artifact_sync": nyc_copy,
        "locations": locations,
        "summary": {
            loc["cli_id"]: {
                "ok": loc.get("ok"),
                "mae_f": loc.get("mae_f"),
                "bias_f": loc.get("bias_f"),
                "within_1f": loc.get("within_1f"),
                "within_2f": loc.get("within_2f"),
                "match_quality": loc.get("match_quality"),
                "n_rows": loc.get("n_rows"),
                "error": loc.get("error"),
            }
            for loc in locations
        },
        "limitations": [
            "ASOS archive must cover decision hours; missing hours drop rows",
            "Model still trained on GHCND labels; scored here against TWC settlement",
            "Transfer residual calib may be biased vs TWC until recalibrated on TWC labels",
            "Sep climate day without official TWC report is excluded",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(out_path)
    latest = out_dir / "predict_vs_twc_latest.json"
    latest.write_text(json.dumps(report, indent=2, default=str))
    return report
