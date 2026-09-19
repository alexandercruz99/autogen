"""Chronological historical weather replay + trading replay gate.

Weather replay uses injected decision clocks and the shared predict_station_v2 path.
Trading replay requires historical order books; without them it records NO_TRADE and
does not invent P&L.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import SUPPORTED_DECISION_HOURS_LOCAL
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import FEATURE_SCHEMA_VERSION, STATION_V2_FEATURES
from kalshi_bot.models.weather.obs_engine.feeds.train_operating import LOCATION_TRAIN_PROFILES
from kalshi_bot.models.weather.obs_engine.multi.candidates.data import load_location_rows, write_json
from kalshi_bot.models.weather.obs_engine.multi.candidates.pipeline import (
    fit_quantile_gbms,
    hour_bias_map,
    matrix_from_rows,
    predict_remain_quantiles,
    remain_point_from_quantiles,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.splits import (
    build_split_manifest,
    filter_rows_by_days,
    save_split_manifest,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.targets import (
    SETTLEMENT_TARGETS,
    UNSUPPORTED_OR_INSUFFICIENT,
    targets_manifest,
)
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.predict import predict_station_v2
from kalshi_bot.models.weather.obs_engine.multi.replay.picker import pick_contract
from kalshi_bot.models.weather.obs_engine.multi.replay.policy import DEFAULT_PICKER_POLICY
from kalshi_bot.models.weather.obs_engine.multi.replay.scoring import (
    crps_distribution,
    modal_degree,
    summarize_forecast_rows,
)
from kalshi_bot.models.weather.settlement_rules import TempInterval


REPLAY_VERSION = "historical_replay.v1"
AVAILABILITY_ASSUMPTION = "archive_valid_utc_equals_availability_DISCLOSED"


def _replay_root(data_dir: Path) -> Path:
    return data_dir / "multi" / "replay"


def _fit_replay_model(
    train_rows: list[dict[str, Any]],
    sel_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    Xtr = matrix_from_rows(train_rows)
    ytr = np.asarray([r["remain_f"] for r in train_rows], dtype=float)
    models = fit_quantile_gbms(Xtr, ytr)

    # Bias from selection only
    by: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
    if sel_rows:
        Xs = matrix_from_rows(sel_rows)
        _, q50, _, _ = predict_remain_quantiles(models, Xs)
        for r, rem in zip(sel_rows, q50):
            point = remain_point_from_quantiles(float(r["max_so_far"]), float(rem), bias=0.0)
            resid = float(r["label_tmax_f"]) - point
            h = str(int(r["decision_hour"]))
            by[h].append(resid)
            by["all"].append(resid)
    bias = hour_bias_map(by)

    # Residuals on calib after applying selection bias (no test leakage)
    resid_map: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
    pool = calib_rows if len(calib_rows) >= 40 else (sel_rows + calib_rows)
    if pool:
        Xp = matrix_from_rows(pool)
        _, q50, _, _ = predict_remain_quantiles(models, Xp)
        for r, rem in zip(pool, q50):
            h = str(int(r["decision_hour"]))
            b = float(bias.get(h, 0.0))
            point = remain_point_from_quantiles(float(r["max_so_far"]), float(rem), bias=b)
            resid = float(r["label_tmax_f"]) - point
            # store residual relative to biased point for empirical shift
            # from_empirical_residuals uses (outcome - point); point already includes bias,
            # so residual should be outcome - biased_point. Bias is also in calib meta for predict path.
            resid_map[h].append(resid)
            resid_map["all"].append(resid)

    return {
        "models": models,
        "feature_names": list(STATION_V2_FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": f"{REPLAY_VERSION}.remain_gbm",
        "bias_by_hour": bias,
        "residuals_by_hour": resid_map,
        "live_eligible": False,
    }


def _predict_blind(
    row: dict[str, Any],
    blob: dict[str, Any],
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Run shared predict path; outcome must not be passed into the model."""
    hour = int(row["decision_hour"])
    feats = row["feature_map"] if isinstance(row.get("feature_map"), dict) else {
        n: (row["features"][i] if i < len(row["features"]) else None)
        for i, n in enumerate(STATION_V2_FEATURES)
    }
    # decision_time from row (injected clock)
    dts = row["decision_time_utc"]
    if isinstance(dts, str):
        decision_time = datetime.fromisoformat(dts.replace("Z", "+00:00"))
    else:
        decision_time = dts
    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)

    climate_day = row["climate_day"]
    if isinstance(climate_day, str):
        climate_day_d = date.fromisoformat(climate_day)
    else:
        climate_day_d = climate_day

    ctx = ForecastContext(
        location_id=str(profile["location_id"]),
        series_ticker=str(profile["series_ticker"]),
        measurement=str(profile["measurement"]),
        settlement_source_family="ghcnd_proxy_for_replay",
        climate_day=climate_day_d,
        decision_time_utc=decision_time,
        horizon="same_day",
        model_version=str(blob.get("model_version")),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        decision_hour_local=hour,
        timezone=str(profile["timezone"]),
        metar_id=str(profile.get("metar_id")),
        mode="HISTORICAL_REPLAY",
        extras={"availability_assumption": AVAILABILITY_ASSUMPTION},
    )
    calib = {
        "residuals_by_hour": blob["residuals_by_hour"],
        "meta": {
            "location_id": profile["location_id"],
            "point_bias_by_hour": blob.get("bias_by_hour") or {},
            "live_eligible": False,
        },
        "min_residuals_required": 40,
    }
    model_blob = {
        "models": blob["models"],
        "feature_names": blob["feature_names"],
        "model_version": blob["model_version"],
    }
    pred = predict_station_v2(
        context=ctx,
        features=feats,
        max_so_far=float(row["max_so_far"]),
        coverage_adequate=True,
        decision_hour_local=hour,
        model_blob=model_blob,
        calibration=calib,
        require_location_id=True,
        require_supported_hour=True,
    )
    out = {
        "ok": pred.ok,
        "status": pred.status,
        "decision_time_utc": decision_time.isoformat(),
        "decision_hour_local": hour,
        "climate_day": climate_day_d.isoformat(),
        "location_id": profile["location_id"],
        "timezone": profile["timezone"],
        "max_so_far_f": float(row["max_so_far"]),
        "point_median_f": pred.point_median_f,
        "remain_q10_q50_q90": pred.remain_q10_q50_q90,
        "quantiles_crossed": pred.quantiles_crossed,
        "probabilities_available": pred.probabilities_available,
        "calibration_status": pred.calibration_status,
        "model_version": pred.model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "availability_assumption": AVAILABILITY_ASSUMPTION,
        "observation_constraint": pred.observation_constraint,
        "distribution": pred.distribution.as_dict() if pred.distribution else None,
        "prediction_saved_before_outcome": True,
    }
    if pred.distribution is not None:
        out["modal_degree_f"] = modal_degree(pred.distribution)
        out["q10"] = pred.distribution.quantile(0.1)
        out["q50"] = pred.distribution.quantile(0.5)
        out["q90"] = pred.distribution.quantile(0.9)
    return out, pred


def _score_after_outcome(pred_row: dict[str, Any], pred_obj, actual: float) -> dict[str, Any]:
    dist = pred_obj.distribution
    if dist is None or pred_row.get("point_median_f") is None:
        return {
            **pred_row,
            "actual_settlement_f": actual,
            "label_source": "ghcnd_tmax",
            "label_is_proxy_not_exchange_settlement": True,
            "scored": False,
            "score_reason": "no_distribution",
        }
    median = float(pred_row["point_median_f"])
    err = median - float(actual)
    modal = int(pred_row.get("modal_degree_f") or modal_degree(dist))
    # most-likely single-degree bracket hit
    band = TempInterval("range_inclusive", float(modal), float(modal), "replay")
    from kalshi_bot.models.weather.settlement_rules import yes_from_observation

    return {
        **pred_row,
        "actual_settlement_f": float(actual),
        "label_source": "ghcnd_tmax",
        "label_is_proxy_not_exchange_settlement": True,
        "scored": True,
        "error_f": err,
        "abs_error_f": abs(err),
        "crps": crps_distribution(dist, actual),
        "modal_degree_f": modal,
        "modal_hit": int(round(actual)) == modal,
        "most_likely_bracket_hit": yes_from_observation(float(actual), band),
        "cov80": dist.quantile(0.1) <= actual <= dist.quantile(0.9),
        "width80": dist.quantile(0.9) - dist.quantile(0.1),
        "within_0p1f": abs(err) <= 0.1,
        "within_0p5f": abs(err) <= 0.5,
        "within_1f": abs(err) <= 1.0,
        "within_2f": abs(err) <= 2.0,
    }


def _trading_decision_for_row(
    pred_obj,
    pred_row: dict[str, Any],
    *,
    books: dict[str, dict[str, Any]] | None,
    contracts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    decision_id = (
        f"replay-{pred_row['location_id']}-{pred_row['climate_day']}-h{pred_row['decision_hour_local']}"
    )
    d = pick_contract(
        pred_obj.distribution,
        contracts or [],
        books,
        {
            "cash": "100.00",
            "open_notional": "0",
            "reserved": "0",
            "max_total_exposure": "80.00",
            "max_per_market_exposure": "25.00",
            "open_position_tickers": [],
        },
        pred_row["decision_time_utc"],
        DEFAULT_PICKER_POLICY,
        decision_id=decision_id,
        location_id=pred_row["location_id"],
        climate_day=pred_row["climate_day"],
        point_median_f=pred_row.get("point_median_f"),
    )
    out = d.as_dict()
    out["fill_model"] = "none_no_historical_executable_book"
    out["historical_pnl_claimed"] = False
    return out


def replay_location(location_id: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    if location_id not in LOCATION_TRAIN_PROFILES:
        return {
            "ok": False,
            "location_id": location_id,
            "status": "insufficient_data",
            "error": "no_train_profile",
        }
    loaded = load_location_rows(location_id, data_dir=data_dir)
    if not loaded.get("ok"):
        return {
            "ok": False,
            "location_id": location_id,
            "status": loaded.get("status", "insufficient_data"),
            "error": loaded.get("error"),
        }

    profile = LOCATION_TRAIN_PROFILES[location_id]
    rows = loaded["rows"]
    days = sorted({r["climate_day"] for r in rows})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _replay_root(data_dir) / location_id / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_split_manifest(
        location_id=location_id,
        metric="daily_max_temp_f",
        label_source="ghcnd_tmax",
        climate_days=days,
        decision_hours_local=list(SUPPORTED_DECISION_HOURS_LOCAL),
        notes=[
            AVAILABILITY_ASSUMPTION,
            "Final test untouched for bias/residual fitting.",
            "Labels are GHCND proxy — not verified Kalshi TWC/CLI settlement performance.",
            "Picker policy frozen before final test (picker.policy.v1).",
        ],
    )
    save_split_manifest(out_dir / "split_manifest.json", manifest)
    write_json(out_dir / "picker_policy.json", DEFAULT_PICKER_POLICY.as_dict())
    write_json(out_dir / "target.json", SETTLEMENT_TARGETS[location_id].as_dict())
    write_json(out_dir / "negative_remain_audit.json", loaded["negative_remain"])

    train_rows = filter_rows_by_days(rows, manifest.train_days)
    sel_rows = filter_rows_by_days(rows, manifest.selection_days)
    calib_rows = filter_rows_by_days(rows, manifest.calib_days)
    test_rows = filter_rows_by_days(rows, manifest.test_days)

    blob = _fit_replay_model(train_rows, sel_rows, calib_rows)
    joblib.dump(blob, out_dir / "replay_model.joblib")
    write_json(
        out_dir / "replay_calibration.json",
        {
            "residuals_by_hour": blob["residuals_by_hour"],
            "meta": {
                "location_id": location_id,
                "point_bias_by_hour": blob["bias_by_hour"],
                "model_version": blob["model_version"],
                "live_eligible": False,
            },
        },
    )

    forecast_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    abstentions = 0

    for r in test_rows:
        # 1-5 weather: inject clock via row decision_time, predict blind, then score
        pred_row, pred_obj = _predict_blind(r, blob, profile=profile)
        # Save prediction before attaching outcome fields
        write_json(
            out_dir
            / "predictions_blind"
            / f"{pred_row['climate_day']}_h{pred_row['decision_hour_local']}.json",
            pred_row,
        )
        scored = _score_after_outcome(pred_row, pred_obj, float(r["label_tmax_f"]))
        forecast_rows.append(scored)

        # Trading replay: no historical books → NO_TRADE (honest)
        decision = _trading_decision_for_row(pred_obj, pred_row, books=None, contracts=None)
        if decision["kind"] == "NO_TRADE":
            abstentions += 1
        decision_rows.append(
            {
                **decision,
                "climate_day": pred_row["climate_day"],
                "decision_hour_local": pred_row["decision_hour_local"],
                "location_id": location_id,
            }
        )

    summary = summarize_forecast_rows([r for r in forecast_rows if r.get("scored")])
    by_hour: dict[str, Any] = {}
    for h in SUPPORTED_DECISION_HOURS_LOCAL:
        sub = [r for r in forecast_rows if r.get("scored") and int(r["decision_hour_local"]) == h]
        by_hour[str(h)] = summarize_forecast_rows(sub)

    report = {
        "ok": True,
        "replay_version": REPLAY_VERSION,
        "location_id": location_id,
        "stamp": stamp,
        "label_source": "ghcnd_tmax",
        "evaluation_type": "proxy_weather_evaluation",
        "evaluation_type_note": (
            "Outcomes are GHCND TMAX, not exchange settlement (TWC/CLI). "
            "Do not label as verified contract performance."
        ),
        "availability_assumption": AVAILABILITY_ASSUMPTION,
        "split_manifest_hash": manifest.day_hash(),
        "n_train_days": len(manifest.train_days),
        "n_selection_days": len(manifest.selection_days),
        "n_calib_days": len(manifest.calib_days),
        "n_test_days": len(manifest.test_days),
        "weather_test_summary": summary,
        "weather_by_hour": by_hour,
        "trading_replay": {
            "status": "blocked_no_historical_executable_books",
            "n_decisions": len(decision_rows),
            "n_no_trade": abstentions,
            "n_trades_simulated": 0,
            "historical_pnl_claimed": False,
            "note": (
                "Prospective market_snapshots exist in research/prospective.db (n≈12) but are "
                "insufficient for chronological trading replay. Weather metrics completed; "
                "picker exercised with missing books → NO_TRADE."
            ),
        },
        "picker_policy_version": DEFAULT_PICKER_POLICY.version,
        "live_eligible": False,
        "no_orders_submitted": True,
        "artifact_dir": str(out_dir),
        "negative_remain_count": loaded["negative_remain"]["count"],
    }
    write_json(out_dir / "forecast_rows.json", {"rows": forecast_rows})
    write_json(out_dir / "decision_rows.json", {"rows": decision_rows})
    write_json(out_dir / "summary.json", report)

    latest = _replay_root(data_dir) / location_id / "latest_summary.json"
    write_json(latest, report)
    return report


def run_historical_replay(*, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = _replay_root(data_dir) / "_reports" / stamp
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "targets_manifest.json", targets_manifest())
    write_json(root / "picker_policy.json", DEFAULT_PICKER_POLICY.as_dict())
    write_json(
        root / "system_inventory.json",
        {
            "exists": [
                "predict_station_v2 shared inference",
                "select_forecast_consistent (extended by pick_contract)",
                "chronological candidate splits",
                "paper ledger (live books only)",
                "prospective market_snapshots collector (sparse)",
            ],
            "incorrect_or_limited": [
                "Prior CRPS used CDF form; replay uses exact energy PMF form per spec",
                "select_forecast_consistent ranked by strike distance then p then EV; "
                "pick_contract ranks by expected net profit after fees",
                "Archive availability approximated as valid_utc",
            ],
            "missing": [
                "Dense historical Kalshi order books for trading P&L",
                "Verified TWC/CLI settlement labels for full history",
            ],
        },
    )

    locations = []
    for loc_id in SETTLEMENT_TARGETS:
        locations.append(replay_location(loc_id, data_dir=data_dir))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "replay_version": REPLAY_VERSION,
        "locations": locations,
        "unsupported_or_insufficient": UNSUPPORTED_OR_INSUFFICIENT,
        "trading_replay_global": "blocked_no_historical_executable_books",
        "live_eligible": False,
        "no_orders_submitted": True,
        "profit_evidence": None,
        "report_dir": str(root),
    }
    write_json(root / "executive_summary.json", report)
    write_json(_replay_root(data_dir) / "_reports" / "latest.json", report)
    return report
