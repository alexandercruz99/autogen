"""Train station-corrected operating model + sat/radar candidate (baseline frozen)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.research.experiments import _enumerate
from kalshi_bot.models.weather.obs_engine.research.features_v2 import LOCAL_V2_FEATURES, build_local_v2, build_baseline
from kalshi_bot.models.weather.obs_engine.feeds.features_live import SATRAD_FEATURES

logger = logging.getLogger(__name__)


def _fit_quantile_models(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingRegressor

    models = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
        )
        m.fit(X, y)
        models[name] = m
    resid = list((y - models["q50"].predict(X)).astype(float))[:500]
    return models, resid


def _mae_by_hour(models, rows: list[dict], feature_key: str = "features") -> dict[str, float]:
    by: dict[str, float] = {}
    for hour in (8, 11, 14):
        hrs = [r for r in rows if int(r["decision_hour"]) == hour]
        if not hrs:
            continue
        X = np.nan_to_num(np.asarray([r[feature_key] for r in hrs], dtype=float), nan=-999.0)
        rem = models["q50"].predict(X)
        max_so = np.asarray([r["max_so_far"] for r in hrs], dtype=float)
        y = np.asarray([r["label_tmax_f"] for r in hrs], dtype=float)
        pred = np.maximum(max_so + rem, max_so)
        by[str(hour)] = float(np.mean(np.abs(y - pred)))
    return by


def _bootstrap_mae_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 400) -> dict[str, float]:
    """Uncertainty on MAE(A)-MAE(B); negative means A better."""
    rng = np.random.default_rng(0)
    n = len(y)
    if n < 5:
        return {"n": float(n), "delta_mae": float("nan"), "ci80_low": float("nan"), "ci80_high": float("nan")}
    err_a = np.abs(y - pred_a)
    err_b = np.abs(y - pred_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(err_a[idx].mean() - err_b[idx].mean()))
    arr = np.asarray(deltas)
    return {
        "n": float(n),
        "delta_mae": float(err_a.mean() - err_b.mean()),
        "ci80_low": float(np.percentile(arr, 10)),
        "ci80_high": float(np.percentile(arr, 90)),
    }


def _backfill_satrad_rows(
    *,
    obs,
    labels: dict[date, float],
    sample_days: list[date],
    decision_hours_local: tuple[int, ...] = (8, 11, 14),
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Geographically scoped historical GOES ACM + NEXRAD N0B features for candidate training."""
    from zoneinfo import ZoneInfo

    from kalshi_bot.models.weather.obs_engine.feeds.goes import _unsigned_s3, acmc_key_near, extract_nyc_cloud_features
    from kalshi_bot.models.weather.obs_engine.feeds import NEXRAD_L3_BUCKET
    from kalshi_bot.models.weather.obs_engine.feeds.nexrad import n0b_key_near, extract_precip_features
    from kalshi_bot.models.weather.obs_engine import NYC_TARGET

    cache_dir.mkdir(parents=True, exist_ok=True)
    goes_dir = cache_dir / "goes"
    rad_dir = cache_dir / "nexrad"
    goes_dir.mkdir(exist_ok=True)
    rad_dir.mkdir(exist_ok=True)
    s3 = _unsigned_s3()
    tz = ZoneInfo(NYC_TARGET.timezone)
    rows: list[dict[str, Any]] = []
    cov = {"goes_ok": 0, "radar_ok": 0, "both_ok": 0, "attempted": 0, "station_only_skipped_satrad": 0}

    for day in sample_days:
        if day not in labels:
            continue
        for hour in decision_hours_local:
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            station = build_local_v2(obs, decision_utc, climate_day=day)
            if station is None:
                continue
            cov["attempted"] += 1
            goes_feat = None
            radar_feat = None
            try:
                hit = acmc_key_near(s3, decision_utc)
                if hit:
                    bucket, key = hit
                    local = goes_dir / f"{bucket}__{Path(key).name}"
                    if not local.exists():
                        s3.download_file(bucket, key, str(local))
                    goes_feat = extract_nyc_cloud_features(local)
                    cov["goes_ok"] += 1
            except Exception as exc:
                logger.debug("goes backfill %s: %s", day, exc)
            try:
                key = n0b_key_near(s3, decision_utc)
                if key:
                    local = rad_dir / key
                    if not local.exists():
                        s3.download_file(NEXRAD_L3_BUCKET, key, str(local))
                    radar_feat = extract_precip_features(local)
                    if radar_feat.get("precip_gate_frac") is not None:
                        cov["radar_ok"] += 1
            except Exception as exc:
                logger.debug("radar backfill %s: %s", day, exc)

            if goes_feat is None and radar_feat is None:
                cov["station_only_skipped_satrad"] += 1
                continue
            if goes_feat is not None and radar_feat is not None and radar_feat.get("precip_gate_frac") is not None:
                cov["both_ok"] += 1

            feat_map = {name: float(v) for name, v in zip(LOCAL_V2_FEATURES, station.values)}
            if goes_feat is not None:
                feat_map["goes_cloud_frac_bcm"] = goes_feat.get("cloud_frac_bcm")
                feat_map["goes_cloudyish_acm"] = goes_feat.get("cloudy_or_probably_frac_acm")
                feat_map["goes_available"] = 1.0
            else:
                feat_map["goes_cloud_frac_bcm"] = None
                feat_map["goes_cloudyish_acm"] = None
                feat_map["goes_available"] = 0.0
            if radar_feat is not None and radar_feat.get("precip_gate_frac") is not None:
                feat_map["radar_precip_frac"] = radar_feat.get("precip_gate_frac")
                feat_map["radar_mean_level"] = radar_feat.get("mean_level_if_any")
                feat_map["radar_available"] = 1.0
            else:
                feat_map["radar_precip_frac"] = None
                feat_map["radar_mean_level"] = None
                feat_map["radar_available"] = 0.0

            vec = [feat_map.get(k) for k in SATRAD_FEATURES]
            # Require sat/radar availability flags present; leave missing as NaN (sentinel at fit)
            if feat_map["goes_available"] < 0.5 and feat_map["radar_available"] < 0.5:
                continue
            rows.append(
                {
                    "climate_day": day.isoformat(),
                    "decision_hour": hour,
                    "features": vec,
                    "station_features": list(station.values),
                    "max_so_far": station.max_so_far,
                    "label_tmax_f": float(labels[day]),
                    "remain_f": float(labels[day]) - float(station.max_so_far),
                    "goes_available": feat_map["goes_available"],
                    "radar_available": feat_map["radar_available"],
                }
            )
    return rows, cov


def train_station_corrected(*, data_dir: Path | None = None, satrad_max_days: int = 60) -> dict[str, Any]:
    """Retrain quantile GBM on local_v2; optionally train satrad candidate on scoped backfill."""
    import joblib

    data_dir = data_dir or default_data_dir()
    out_dir = Path("data/obs_engine/feeds/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    rows = _enumerate(build_local_v2, obs, [], labels)
    base_rows = _enumerate(build_baseline, obs, [], labels)
    days = sorted({r["climate_day"] for r in rows})
    i_tr = int(len(days) * 0.8)
    train_days = set(days[:i_tr])
    test_days = set(days[i_tr:])
    train = [r for r in rows if r["climate_day"] in train_days]
    test = [r for r in rows if r["climate_day"] in test_days]
    base_train = [r for r in base_rows if r["climate_day"] in train_days]
    base_test = [r for r in base_rows if r["climate_day"] in test_days]

    def fit_models(tr, feat_key="features"):
        X = np.nan_to_num(np.asarray([r[feat_key] for r in tr], dtype=float), nan=-999.0)
        y = np.asarray([r["remain_f"] for r in tr], dtype=float)
        return _fit_quantile_models(X, y)

    models, resid = fit_models(train)
    base_models, _ = fit_models(base_train)
    report: dict[str, Any] = {
        "ok": True,
        "model_version": "weather.obs_nyc.station_corrected.v1",
        "feature_set": "local_v2",
        "feature_names": LOCAL_V2_FEATURES,
        "train_end_day": max(train_days),
        "n_train_days": len(train_days),
        "n_test_days": len(test_days),
        "test_mae_station_corrected": _mae_by_hour(models, test),
        "test_mae_baseline_schema": _mae_by_hour(base_models, base_test),
        "frozen_baseline_path": "data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib",
        "live_eligible": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "station_corrected_v1.joblib"
    joblib.dump(
        {
            "models": models,
            "feature_names": LOCAL_V2_FEATURES,
            "feature_set": "local_v2",
            "model_version": report["model_version"],
            "remain_residuals": resid,
            "train_end_day": report["train_end_day"],
            "promoted_for_inference": True,
            "live_eligible": False,
        },
        path,
    )
    report["artifact"] = str(path)

    # --- Sat/radar candidate: scoped backfill on recent labeled days (GOES-19 era preferred) ---
    satrad_report: dict[str, Any] = {
        "status": "not_run",
        "note": "Candidate trained only on days with retrieved sat/radar features; not attached to station-only artifact.",
    }
    try:
        # Prefer recent days where GOES-19/16 + NEXRAD exist; cap downloads
        day_objs = sorted({date.fromisoformat(d) if isinstance(d, str) else d for d in days})
        # Focus on last satrad_max_days within labeled history that overlap GOES ACM availability
        recent = [d for d in day_objs if d >= date(2024, 6, 1)][-satrad_max_days:]
        cache = Path("data/obs_engine/feeds/satrad_backfill")
        # Hour 14 local first (skill peak); expand prospectively via collector
        sat_rows, cov = _backfill_satrad_rows(
            obs=obs,
            labels=labels,
            sample_days=recent,
            decision_hours_local=(14,),
            cache_dir=cache,
        )
        satrad_report["coverage"] = cov
        satrad_report["n_feature_rows"] = len(sat_rows)
        if len(sat_rows) >= 20:
            sat_days = sorted({r["climate_day"] for r in sat_rows})
            cut = int(len(sat_days) * 0.7) or 1
            tr_d, te_d = set(sat_days[:cut]), set(sat_days[cut:])
            sat_train = [r for r in sat_rows if r["climate_day"] in tr_d]
            sat_test = [r for r in sat_rows if r["climate_day"] in te_d]
            # Matching-date station-corrected comparison on same decision hours
            station_by_key = {
                (r["climate_day"], int(r["decision_hour"])): r for r in rows if isinstance(r["climate_day"], str)
            }
            # normalize climate_day keys in rows from _enumerate
            for r in rows:
                station_by_key[(str(r["climate_day"]), int(r["decision_hour"]))] = r

            Xtr = np.nan_to_num(np.asarray([r["features"] for r in sat_train], dtype=float), nan=-999.0)
            ytr = np.asarray([r["remain_f"] for r in sat_train], dtype=float)
            sat_models, sat_resid = _fit_quantile_models(Xtr, ytr)
            # Compare on overlapping sat_test keys
            matched = []
            for r in sat_test:
                st = station_by_key.get((str(r["climate_day"]), int(r["decision_hour"])))
                if st is None:
                    continue
                matched.append((r, st))
            if matched:
                Xs = np.nan_to_num(np.asarray([a["features"] for a, _ in matched], dtype=float), nan=-999.0)
                Xst = np.nan_to_num(np.asarray([b["features"] for _, b in matched], dtype=float), nan=-999.0)
                y = np.asarray([a["label_tmax_f"] for a, _ in matched], dtype=float)
                max_so = np.asarray([a["max_so_far"] for a, _ in matched], dtype=float)
                pred_sat = np.maximum(max_so + sat_models["q50"].predict(Xs), max_so)
                pred_st = np.maximum(max_so + models["q50"].predict(Xst), max_so)
                mae_sat = float(np.mean(np.abs(y - pred_sat)))
                mae_st = float(np.mean(np.abs(y - pred_st)))
                # crude probability calibration proxy: residual sign balance
                resid_sat = y - pred_sat
                resid_st = y - pred_st
                cal = {
                    "satrad_mean_signed_error": float(np.mean(resid_sat)),
                    "station_mean_signed_error": float(np.mean(resid_st)),
                    "satrad_frac_over": float(np.mean(resid_sat > 0)),
                    "station_frac_over": float(np.mean(resid_st > 0)),
                }
                delta = _bootstrap_mae_delta(y, pred_sat, pred_st)
                satrad_report.update(
                    {
                        "status": "evaluated",
                        "n_train_rows": len(sat_train),
                        "n_test_rows": len(sat_test),
                        "n_matched_compare": len(matched),
                        "test_mae_satrad": mae_sat,
                        "test_mae_station_corrected_same_dates": mae_st,
                        "mae_delta_satrad_minus_station": delta,
                        "calibration_proxy": cal,
                        "promoted": False,
                        "promotion_note": (
                            "Not promoted: require CI excluding zero improvement and larger multi-year coverage. "
                            "Operating inference uses station_corrected.v1; sat/radar features stored for prospective use."
                        ),
                    }
                )
                # Save candidate artifact (not for operating inference until promoted)
                cand_path = out_dir / "satrad_candidate_v1.joblib"
                joblib.dump(
                    {
                        "models": sat_models,
                        "feature_names": SATRAD_FEATURES,
                        "feature_set": "satrad_v1",
                        "model_version": "weather.obs_nyc.satrad.candidate.v1",
                        "remain_residuals": sat_resid,
                        "promoted_for_inference": False,
                        "live_eligible": False,
                        "includes_rtm_model_assist_goes_acm": True,
                    },
                    cand_path,
                )
                satrad_report["artifact"] = str(cand_path)
            else:
                satrad_report["status"] = "insufficient_overlap"
        else:
            satrad_report["status"] = "insufficient_rows"
            satrad_report["note"] = f"Only {len(sat_rows)} satrad rows (need >=20); keep collecting prospectively."
    except Exception as exc:
        logger.exception("satrad backfill/train failed")
        satrad_report = {"status": "error", "error": str(exc)}

    report["sat_radar_training"] = satrad_report
    (out_dir / "station_corrected_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
