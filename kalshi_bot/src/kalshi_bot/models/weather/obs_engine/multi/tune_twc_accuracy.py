"""Tune station_v2 accuracy against official TWC settlement highs.

Keeps the existing GBM remain model (GHCND-trained) and learns a chronological
hour-specific point bias + TWC residual calibration so predictions track the
actual Kalshi settlement source more closely.

Holdout is always the most recent TWC days (never used to fit bias/residuals).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir
from kalshi_bot.models.weather.obs_engine.feeds.calibration import (
    MIN_CALIB_RESIDUALS,
    residuals_by_hour,
    save_calibration_artifact,
)
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import SUPPORTED_DECISION_HOURS_LOCAL
from kalshi_bot.models.weather.obs_engine.feeds.train_operating import (
    LOCATION_TRAIN_PROFILES,
    _enumerate_station_v2,
)
from kalshi_bot.models.weather.obs_engine.multi.predict import load_operating_model, predict_from_rows
from kalshi_bot.models.weather.obs_engine.multi.score_vs_twc import (
    TWC_SCORE_TARGETS,
    _load_obs_for_profile,
    ensure_nyc_multi_artifacts,
    fetch_twc_actuals,
)

logger = logging.getLogger(__name__)


def _mae(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([abs(r["error_f"]) for r in rows]))


def _score_rows(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    preds = predict_from_rows(rows, models, calibration=calibration, feature_key="features")
    out: list[dict[str, Any]] = []
    for r, p in zip(rows, preds):
        dist = p.get("distribution") or {}
        med = dist.get("q50")
        if med is None:
            med = p.get("point_median_f")
        if med is None:
            continue
        actual = float(r["label_tmax_f"])
        err = float(med) - actual
        out.append(
            {
                "climate_day": r["climate_day"],
                "decision_hour_local": int(r["decision_hour"]),
                "predicted_median_f": float(med),
                "actual_twc_max_f": actual,
                "error_f": err,
                "abs_error_f": abs(err),
                "max_so_far_f": float(r["max_so_far"]),
            }
        )
    return out


def _hour_bias_from_residuals(resid_map: dict[str, list[float]]) -> dict[str, float]:
    """Median residual (actual − pred) by hour → additive bias to apply to point."""
    out: dict[str, float] = {}
    for hour in ("8", "11", "14"):
        vals = resid_map.get(hour) or []
        if len(vals) >= 8:
            # pred_adj = pred + bias, bias ≈ median(actual - pred) = median(residual)
            out[hour] = float(np.median(vals))
        elif resid_map.get("all"):
            out[hour] = float(np.median(resid_map["all"]))
    return out


def _twc_day_splits(days: list[str], *, holdout_frac: float = 0.25, calib_frac: float = 0.35) -> dict[str, set[str]]:
    """Chronological: early = fit, middle = calib, last = holdout."""
    norm = sorted(days)
    n = len(norm)
    if n < 12:
        # tiny history: last 3 days holdout, rest calib
        return {
            "fit": set(norm[:-max(2, n // 4)]),
            "calib": set(norm[:-max(2, n // 4)]),
            "holdout": set(norm[-max(2, n // 4) :]),
        }
    n_hold = max(5, int(round(n * holdout_frac)))
    n_calib = max(8, int(round(n * calib_frac)))
    holdout = set(norm[-n_hold:])
    calib = set(norm[-(n_hold + n_calib) : -n_hold])
    fit = set(norm[: -(n_hold + n_calib)]) or set(norm[: max(1, n - n_hold)])
    return {"fit": fit, "calib": calib, "holdout": holdout}


def tune_location_twc(
    cli_id: str,
    *,
    data_dir: Path | None = None,
    lookback_days: int = 100,
    promote: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    meta = TWC_SCORE_TARGETS[cli_id]
    loc_id = meta["location_id"]
    profile = LOCATION_TRAIN_PROFILES[loc_id]

    art = data_dir / "multi" / "artifacts" / meta["artifact_subdir"]
    art.mkdir(parents=True, exist_ok=True)
    model_path = art / "station_corrected_v2.joblib"
    if not model_path.exists() and meta.get("fallback_model"):
        model_path = data_dir / meta["fallback_model"]
    baseline_calib_path = art / "station_corrected_v2_calibration.json"
    if not baseline_calib_path.exists() and meta.get("fallback_calib"):
        baseline_calib_path = data_dir / meta["fallback_calib"]

    blob, model_art, status = load_operating_model(model_path)
    if blob is None:
        return {"ok": False, "cli_id": cli_id, "error": f"model_unavailable:{status}"}
    models = blob  # full artifact so models_by_hour is honored in score/calib
    baseline_calib = json.loads(baseline_calib_path.read_text()) if baseline_calib_path.exists() else None
    if baseline_calib is None:
        return {"ok": False, "cli_id": cli_id, "error": "baseline_calibration_missing"}

    end = date.today() - timedelta(days=1)
    days = [end - timedelta(days=i) for i in range(lookback_days - 1, -1, -1)]
    twc = fetch_twc_actuals(days)
    labels: dict[date, float] = {}
    for day in days:
        row = (twc.get(day.isoformat()) or {}).get(cli_id) or {}
        if row.get("status") == "official" and row.get("max_temp_f") is not None:
            labels[day] = float(row["max_temp_f"])
    if len(labels) < 15:
        return {
            "ok": False,
            "cli_id": cli_id,
            "error": f"too_few_twc_official_days:{len(labels)}",
            "hint": "Need more official TWC climate history for this station",
        }

    obs = _load_obs_for_profile(profile, data_dir)
    rows, n_neg = _enumerate_station_v2(
        obs,
        labels,
        tz_name=profile["timezone"],
        lat=float(profile["lat"]),
        lon=float(profile["lon"]),
        station_id=str(profile["location_id"]),
    )
    if len(rows) < 30:
        return {"ok": False, "cli_id": cli_id, "error": f"too_few_feature_rows:{len(rows)}"}

    day_ids = sorted({r["climate_day"] for r in rows})
    splits = _twc_day_splits(day_ids)
    fit_rows = [r for r in rows if r["climate_day"] in splits["fit"]]
    calib_rows = [r for r in rows if r["climate_day"] in splits["calib"]]
    hold_rows = [r for r in rows if r["climate_day"] in splits["holdout"]]
    if len(fit_rows) < 30:
        # Short TWC history: use all non-holdout for fit+calib
        fit_rows = [r for r in rows if r["climate_day"] not in splits["holdout"]]
        calib_rows = fit_rows
    if len(calib_rows) < max(24, MIN_CALIB_RESIDUALS // 2):
        calib_rows = [r for r in rows if r["climate_day"] not in splits["holdout"]]

    # --- Candidate A: baseline GBM + hour bias / TWC residuals ---
    base_scored = _score_rows(hold_rows, models, baseline_calib)

    def make_twc_calib(model_dict: dict[str, Any], tag: str) -> dict[str, Any]:
        resid_map = residuals_by_hour(calib_rows, model_dict)
        bias = _hour_bias_from_residuals(resid_map)
        resid_adj: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
        for hour, vals in resid_map.items():
            if hour == "all":
                continue
            b = bias.get(hour, 0.0)
            adj = [float(v) - b for v in vals]
            resid_adj[hour] = adj
            resid_adj["all"].extend(adj)
        return {
            "method": "twc_hour_bias_plus_empirical_residual",
            "min_residuals_required": MIN_CALIB_RESIDUALS,
            "residuals_by_hour": resid_adj,
            "meta": {
                **(baseline_calib.get("meta") or {}),
                "location_id": loc_id,
                "label_source": "twc_climate_official_maxTemp",
                "tuning": tag,
                "point_bias_by_hour": bias,
                "tuned_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_twc_label_days": len(labels),
                "n_fit_rows": len(fit_rows),
                "n_calib_rows": len(calib_rows),
                "n_holdout_rows": len(hold_rows),
                "holdout_days": sorted(splits["holdout"]),
                "calib_days": sorted(splits["calib"]),
                "fit_days": sorted(splits["fit"]),
                "baseline_model": model_art,
                "settlement_note": "Tuned to match TWC climate maxTemp (Kalshi KXHIGH* source)",
                "live_eligible": False,
            },
        }

    from kalshi_bot.models.weather.obs_engine.feeds.train_operating import _fit_quantile_models

    bias_calib = make_twc_calib(models, "twc_point_bias_keep_ghcnd_gbm")
    bias_scored = _score_rows(hold_rows, models, bias_calib)

    # --- Candidate B: retrain GBM remain on TWC fit days + TWC calib ---
    retrain_models = models
    retrain_calib = bias_calib
    retrain_scored = bias_scored
    retrain_ok = False
    if len(fit_rows) >= 36:
        Xtr = np.nan_to_num(np.asarray([r["features"] for r in fit_rows], dtype=float), nan=-999.0)
        ytr = np.asarray([r["remain_f"] for r in fit_rows], dtype=float)
        retrain_models = _fit_quantile_models(Xtr, ytr)
        retrain_calib = make_twc_calib(retrain_models, "twc_retrain_gbm_plus_bias")
        retrain_scored = _score_rows(hold_rows, retrain_models, retrain_calib)
        retrain_ok = True

    def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
        by_h: dict[str, Any] = {}
        for hour in SUPPORTED_DECISION_HOURS_LOCAL:
            sub = [r for r in scored if r["decision_hour_local"] == hour]
            if not sub:
                continue
            by_h[str(hour)] = {
                "n": len(sub),
                "mae_f": _mae(sub),
                "bias_f": float(np.mean([r["error_f"] for r in sub])),
                "within_1f": float(np.mean([1.0 if r["abs_error_f"] <= 1 else 0.0 for r in sub])),
            }
        return {
            "n": len(scored),
            "mae_f": _mae(scored),
            "bias_f": float(np.mean([r["error_f"] for r in scored])) if scored else None,
            "within_1f": float(np.mean([1.0 if r["abs_error_f"] <= 1 else 0.0 for r in scored]))
            if scored
            else None,
            "mae_14h_f": (by_h.get("14") or {}).get("mae_f"),
            "by_hour": by_h,
        }

    base_sum = summarize(base_scored)
    bias_sum = summarize(bias_scored)
    retrain_sum = summarize(retrain_scored)

    candidates = [
        ("baseline", models, baseline_calib, base_sum, base_scored),
        ("twc_bias", models, bias_calib, bias_sum, bias_scored),
    ]
    if retrain_ok:
        candidates.append(("twc_retrain", retrain_models, retrain_calib, retrain_sum, retrain_scored))

    def rank_key(item: tuple[str, Any, Any, dict[str, Any], Any]) -> tuple[float, float]:
        s = item[3]
        mae = s.get("mae_f")
        mae14 = s.get("mae_14h_f")
        # Primary: overall MAE; secondary: 14h MAE
        return (
            float(mae) if mae is not None else 9e9,
            float(mae14) if mae14 is not None else 9e9,
        )

    best_name, best_models, best_calib, best_sum, _ = min(candidates, key=rank_key)
    improved = best_name != "baseline" and rank_key(("x", None, None, best_sum, None)) < rank_key(
        ("b", None, None, base_sum, None)
    )
    # Require at least 0.05°F overall MAE gain to promote
    if (
        improved
        and best_sum.get("mae_f") is not None
        and base_sum.get("mae_f") is not None
        and (base_sum["mae_f"] - best_sum["mae_f"]) < 0.05
    ):
        improved = False

    out: dict[str, Any] = {
        "ok": True,
        "cli_id": cli_id,
        "location_id": loc_id,
        "n_twc_days": len(labels),
        "n_rows": len(rows),
        "negative_remain_rows": n_neg,
        "splits": {k: sorted(v) for k, v in splits.items()},
        "point_bias_by_hour": (best_calib.get("meta") or {}).get("point_bias_by_hour"),
        "baseline_holdout": base_sum,
        "twc_bias_holdout": bias_sum,
        "twc_retrain_holdout": retrain_sum if retrain_ok else None,
        "best_candidate": best_name,
        "tuned_holdout": best_sum,
        "improved": improved,
        "promoted": False,
        "retrain_attempted": retrain_ok,
    }

    candidate_path = art / "station_corrected_v2_calibration_twc_tuned.json"
    save_calibration_artifact(
        candidate_path,
        residuals_by_hour=best_calib["residuals_by_hour"],
        meta=best_calib["meta"],
    )
    saved = json.loads(candidate_path.read_text())
    saved["method"] = best_calib["method"]
    saved["meta"] = best_calib["meta"]
    candidate_path.write_text(json.dumps(saved, indent=2))
    out["candidate_calib_path"] = str(candidate_path)

    if promote and improved:
        import joblib

        backup = art / "station_corrected_v2_calibration_pre_twc_tune.json"
        if baseline_calib_path.exists() and not backup.exists():
            backup.write_text(baseline_calib_path.read_text())
        baseline_calib_path.write_text(candidate_path.read_text())

        if best_name == "twc_retrain":
            model_backup = art / "station_corrected_v2_pre_twc_tune.joblib"
            active_model = art / "station_corrected_v2.joblib"
            if active_model.exists() and not model_backup.exists():
                import shutil

                shutil.copy2(active_model, model_backup)
            new_blob = {
                **blob,
                "models": best_models,
                "model_version": f"{blob.get('model_version', 'station_v2')}+twc_retrain",
                "label_source": "twc_climate_official_maxTemp",
                "promoted_for_inference": True,
                "live_eligible": False,
                "train_end_day": max(splits["fit"]) if splits["fit"] else None,
            }
            joblib.dump(new_blob, active_model)
            out["promoted_model_path"] = str(active_model)

        if loc_id == "nyc_central_park":
            feeds_calib = data_dir / "feeds" / "models" / "station_corrected_v2_calibration.json"
            if feeds_calib.exists():
                feeds_backup = (
                    data_dir / "feeds" / "models" / "station_corrected_v2_calibration_pre_twc_tune.json"
                )
                if not feeds_backup.exists():
                    feeds_backup.write_text(feeds_calib.read_text())
                feeds_calib.write_text(candidate_path.read_text())
            if best_name == "twc_retrain":
                feeds_model = data_dir / "feeds" / "models" / "station_corrected_v2.joblib"
                if feeds_model.exists():
                    fb = data_dir / "feeds" / "models" / "station_corrected_v2_pre_twc_tune.joblib"
                    if not fb.exists():
                        import shutil

                        shutil.copy2(feeds_model, fb)
                    joblib.dump(new_blob, feeds_model)

        out["promoted"] = True
        out["promoted_calib_path"] = str(baseline_calib_path)
    elif promote and not improved:
        out["promote_skipped"] = "holdout_did_not_improve_enough"

    return out


def tune_all_twc(
    *,
    data_dir: Path | None = None,
    lookback_days: int = 100,
    promote: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    ensure_nyc_multi_artifacts(data_dir)
    results = []
    for cli in ("NYC", "MDW", "LAX"):
        results.append(
            tune_location_twc(cli, data_dir=data_dir, lookback_days=lookback_days, promote=promote)
        )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal": "Improve median accuracy vs official TWC settlement highs",
        "method": "hour-specific point bias + TWC residual recalibration (GBM unchanged)",
        "locations": results,
        "summary": {
            r["cli_id"]: {
                "ok": r.get("ok"),
                "improved": r.get("improved"),
                "promoted": r.get("promoted"),
                "best_candidate": r.get("best_candidate"),
                "baseline_mae": (r.get("baseline_holdout") or {}).get("mae_f"),
                "tuned_mae": (r.get("tuned_holdout") or {}).get("mae_f"),
                "baseline_mae_14h": (r.get("baseline_holdout") or {}).get("mae_14h_f"),
                "tuned_mae_14h": (r.get("tuned_holdout") or {}).get("mae_14h_f"),
                "bias_by_hour": r.get("point_bias_by_hour"),
                "error": r.get("error"),
            }
            for r in results
        },
        "live_eligible": False,
        "note": "Holdout is chronological last TWC days. Promotion only if holdout MAE improves.",
    }
    out_dir = data_dir / "multi" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"twc_tune_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "twc_tune_latest.json").write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(path)
    return report
