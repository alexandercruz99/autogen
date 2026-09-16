"""Multi-location collect → feature → forecast → paper cycle.

One location's failure does not stop others. Live order submission is never called.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.models.weather.obs_engine.feeds.climate_day import (
    SUPPORTED_DECISION_HOURS_LOCAL,
    civil_local,
    is_supported_decision_time,
    lst_climate_day,
    next_supported_decision_utc,
)
from kalshi_bot.models.weather.obs_engine.feeds.cli_feed import collect_cli
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import FEATURE_SCHEMA_VERSION
from kalshi_bot.models.weather.obs_engine.feeds.features_live import build_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.features_twc import build_twc_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.metar import collect_metar
from kalshi_bot.models.weather.obs_engine.feeds.paper_sim import PaperLedger
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.feeds.twc_kalshi import collect_twc_climate, collect_twc_metar
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.markets import fetch_open_series_markets, select_event_markets
from kalshi_bot.models.weather.obs_engine.multi.predict import DEFAULT_MODEL_PATH, predict_station_v2
from kalshi_bot.models.weather.obs_engine.multi.registry import LocationRegistry
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path("data/obs_engine/multi")


def _artifact_dir(location_id: str, measurement: str) -> Path:
    p = ARTIFACT_ROOT / "artifacts" / f"{location_id}__{measurement}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _shared_paper_ledger() -> PaperLedger:
    path = ARTIFACT_ROOT / "paper_ledger.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return PaperLedger(path=path)


def process_location(
    target: dict[str, Any],
    store: FeedStore,
    *,
    now: datetime | None = None,
    do_paper: bool = True,
    collect: bool = True,
    force_decision: bool = False,
) -> dict[str, Any]:
    """Run one location end-to-end. Never raises for expected blocked states.

    ``force_decision``: if off supported hours, snap to the latest trained hour
    already passed today and continue (used for explicit user-requested bets).
    """
    now = now or datetime.now(timezone.utc)
    location_id = target["location_id"]
    measurement = target["measurement"]
    series = target["series_ticker"]
    tz = target.get("timezone") or "America/New_York"
    metars = target.get("metar_ids") or []
    neighbors = target.get("neighbor_metar_ids") or []
    cli_id = target.get("cli_location_id")
    family = target.get("settlement_source_family")
    mapping = target.get("mapping_status")
    validation = target.get("validation_status")

    out: dict[str, Any] = {
        "location_id": location_id,
        "series_ticker": series,
        "measurement": measurement,
        "display_name": target.get("display_name"),
        "mapping_status": mapping,
        "validation_status": validation,
        "settlement_source_family": family,
        "timezone": tz,
        "generated_at_utc": now.isoformat(),
        "live_order_submitted": False,
        "ok": False,
        "force_decision": force_decision,
    }

    if measurement != "daily_max_temp_f":
        out.update(
            {
                "status": "blocked_wrong_measurement",
                "reason": (
                    f"Measurement {measurement} requires a metric-specific adapter; "
                    "daily-max station_v2 model must not be applied"
                ),
            }
        )
        return out
    if family == "weather_company":
        return _process_twc_location(
            target,
            store,
            out=out,
            now=now,
            do_paper=do_paper,
            collect=collect,
            force_decision=force_decision,
        )
    if family != "nws_cli":
        out.update(
            {
                "status": "blocked_unsupported_settlement_source",
                "reason": target.get("notes") or f"settlement_source_family={family}",
            }
        )
        return out
    if mapping not in ("verified", "verified_cli_url"):
        out.update(
            {
                "status": "blocked_incomplete_mapping",
                "reason": target.get("notes") or f"mapping_status={mapping}",
            }
        )
        return out
    if not metars:
        out.update(
            {
                "status": "blocked_no_metar",
                "reason": "No verified METAR/ICAO settlement station in registry",
            }
        )
        return out

    primary = metars[0]
    stations = tuple(dict.fromkeys([*metars, *neighbors]))  # preserve order, unique

    collect_summary: dict[str, Any] = {}
    if collect:
        try:
            collect_summary["metar"] = collect_metar(store, stations=stations, hours=30)
        except Exception as exc:
            collect_summary["metar"] = {"ok": False, "error": str(exc), "usable": False}
        if cli_id:
            try:
                collect_summary["cli"] = collect_cli(store, cli_location_id=cli_id)
            except Exception as exc:
                collect_summary["cli"] = {"ok": False, "error": str(exc), "usable": False}
    out["collection"] = collect_summary

    metar_ok = bool((collect_summary.get("metar") or {}).get("usable")) or bool(
        store.latest_sample(f"metar_{primary}")
    )
    out["data_availability"] = {
        "metar_primary": primary,
        "metar_usable": metar_ok,
        "cli_location_id": cli_id,
        "notes": target.get("notes"),
    }
    if not metar_ok and collect:
        out.update({"status": "blocked_no_usable_metar", "reason": f"No usable METAR for {primary}"})
        return out

    features = build_operating_features(
        store,
        now=now,
        metar_id=primary,
        tz_name=tz,
        lat=target.get("lat"),
        lon=target.get("lon"),
        cli_location_id=cli_id,
        location_id=location_id,
        include_satrad=(location_id == "nyc_central_park"),
    )
    out["features_summary"] = {
        "max_so_far": features.get("max_so_far"),
        "coverage": features.get("coverage"),
        "cli_applied": features.get("cli_applied"),
        "attribution": features.get("attribution"),
        "missing": features.get("missing"),
    }

    local = civil_local(now, tz)
    supported, decision_hour = is_supported_decision_time(now, tz_name=tz)
    climate_day = date.fromisoformat(features["climate_day"])
    ctx = ForecastContext(
        location_id=location_id,
        series_ticker=series,
        measurement=measurement,
        settlement_source_family=family,
        climate_day=climate_day,
        decision_time_utc=now,
        horizon="same_day",
        model_version=None,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        decision_hour_local=decision_hour,
        timezone=tz,
        uses_lst_climate_day=bool(target.get("uses_lst_climate_day", True)),
        metar_id=primary,
        cli_location_id=cli_id,
        lat=target.get("lat"),
        lon=target.get("lon"),
        elev_m=target.get("elev_m"),
        mode="RESEARCH",
    )
    out["forecast_context"] = ctx.as_dict()
    out["decision_hour_local"] = decision_hour
    out["supported_decision_hours_local"] = list(SUPPORTED_DECISION_HOURS_LOCAL)
    out["generated_at_local"] = local.isoformat()

    if not supported:
        nxt = next_supported_decision_utc(now, tz_name=tz)
        out.update(
            {
                "status": "unsupported_decision_time",
                "reason": (
                    f"Actionable forecasts restricted to local hours {SUPPORTED_DECISION_HOURS_LOCAL}; "
                    f"now={local.strftime('%H:%M %Z')} tz={tz}"
                ),
                "next_supported_run_utc": nxt.isoformat(),
                "next_supported_run_local": civil_local(nxt, tz).isoformat(),
                "ok": True,  # legitimate blocked outcome, not a crash
                "paper_decision": {
                    "decision": "blocked_unsupported_decision_time",
                    "reason": "off-hours — not an end-to-end forecast success",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    cov = features.get("coverage") or {}
    if features.get("max_so_far") is None or not cov.get("adequate"):
        out.update(
            {
                "status": "insufficient_data",
                "reason": "Essential station coverage inadequate",
                "coverage_notes": cov.get("notes") if isinstance(cov, dict) else None,
                "ok": True,
                "paper_decision": {
                    "decision": "blocked_insufficient_data",
                    "reason": "coverage inadequate",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    # Model: NYC has trained artifact; others need location-specific training
    model_path = DEFAULT_MODEL_PATH
    loc_model = _artifact_dir(location_id, measurement) / "station_corrected_v2.joblib"
    if loc_model.exists():
        model_path = loc_model
    elif location_id != "nyc_central_park":
        out.update(
            {
                "status": "blocked_no_location_model",
                "reason": (
                    f"No trained model for {location_id}; NYC station_v2 must not be silently transferred. "
                    "Collect/backfill history then train location-specific (or validated pooled) model."
                ),
                "ok": True,
                "model_artifact": str(loc_model),
                "paper_decision": {
                    "decision": "blocked_unsupported_model",
                    "reason": "location model missing",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    pred = predict_station_v2(
        context=ctx,
        features=features.get("features") or {},
        max_so_far=float(features["max_so_far"]),
        coverage_adequate=True,
        decision_hour_local=decision_hour,
        cli_applied=features.get("cli_applied"),
        model_path=model_path,
        calib_path=(
            _artifact_dir(location_id, measurement) / "station_corrected_v2_calibration.json"
            if (_artifact_dir(location_id, measurement) / "station_corrected_v2_calibration.json").exists()
            else None
        ),
    )
    out["prediction"] = pred.as_dict()
    out["status"] = pred.status
    out["probabilities_available"] = pred.probabilities_available
    out["calibration_status"] = pred.calibration_status
    out["point_median_f"] = pred.point_median_f
    out["observation_constraint"] = pred.observation_constraint
    out["attribution"] = features.get("attribution")
    if features.get("attribution") is not None:
        features["attribution"]["features_consumed_by_model"] = list(
            __import__(
                "kalshi_bot.models.weather.obs_engine.feeds.feature_schema",
                fromlist=["STATION_V2_FEATURES"],
            ).STATION_V2_FEATURES
        )
        features["attribution"]["feeds_contributing_to_prediction"] = [f"metar_{primary}"]
        out["attribution"] = features["attribution"]

    if not pred.probabilities_available:
        out.update(
            {
                "ok": True,
                "reason": pred.reason,
                "paper_decision": {
                    "decision": "blocked_calibration_unavailable"
                    if pred.status == "probabilities_unavailable"
                    else f"blocked_{pred.status}",
                    "reason": pred.reason,
                    "live_blocked": True,
                    "research_point_forecast": pred.point_median_f,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    if not do_paper:
        out["ok"] = True
        _write_location_report(location_id, measurement, out)
        return out

    # Paper evaluation against this series' open markets
    try:
        paper = _paper_evaluate(
            pred=pred,
            series_ticker=series,
            climate_day=climate_day,
            decision_hour=decision_hour,
            location_id=location_id,
            max_so_far=features.get("max_so_far"),
        )
        out["paper_decision"] = paper.get("paper_decision")
        out["brackets"] = paper.get("brackets")
        out["ev_evaluations"] = paper.get("ev_evaluations")
        out["paper_ledger"] = paper.get("paper_ledger")
        out["target_date"] = paper.get("target_date")
        out["horizon"] = paper.get("horizon")
        if paper.get("status"):
            out["status"] = paper["status"]
            out["reason"] = paper.get("reason")
        out["ok"] = True
    except Exception as exc:
        logger.exception("paper eval %s", location_id)
        out["paper_decision"] = {
            "decision": "blocked_paper_error",
            "reason": str(exc),
            "live_blocked": True,
        }
        out["ok"] = True
        out["status"] = "paper_error"

    _write_location_report(location_id, measurement, out)
    return out


def _resolve_twc_model_path(target: dict[str, Any], location_id: str, measurement: str) -> tuple[Path | None, str]:
    """Prefer location-specific TWC artifact; else same-ICAO NWS station_v2 transfer."""
    loc_model = _artifact_dir(location_id, measurement) / "station_corrected_v2.joblib"
    if loc_model.exists():
        return loc_model, "location_twc_artifact"
    sibling = (target.get("details") or {}).get("same_station_model_location_id") or target.get(
        "same_station_model_location_id"
    )
    if sibling:
        sib_path = _artifact_dir(sibling, measurement) / "station_corrected_v2.joblib"
        if sib_path.exists():
            return sib_path, f"same_icao_transfer:{sibling}"
        if sibling == "nyc_central_park" and DEFAULT_MODEL_PATH.exists():
            return DEFAULT_MODEL_PATH, "same_icao_transfer:nyc_feeds_default"
    if location_id == "twc_nyc_central_park" and DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH, "same_icao_transfer:nyc_feeds_default"
    return None, "missing"


def _process_twc_location(
    target: dict[str, Any],
    store: FeedStore,
    *,
    out: dict[str, Any],
    now: datetime,
    do_paper: bool,
    collect: bool,
    force_decision: bool = False,
) -> dict[str, Any]:
    """TWC settlement path — never applies NWS CLI floors."""
    location_id = target["location_id"]
    measurement = target["measurement"]
    series = target["series_ticker"]
    tz = target.get("timezone") or "America/New_York"
    metars = target.get("metar_ids") or []
    neighbors = target.get("neighbor_metar_ids") or []
    cli_id = target.get("cli_location_id")
    mapping = target.get("mapping_status")

    if mapping not in ("verified", "verified_cli_url"):
        out.update(
            {
                "status": "blocked_incomplete_mapping",
                "reason": target.get("notes") or f"mapping_status={mapping}",
            }
        )
        return out
    if not metars or not cli_id:
        out.update(
            {
                "status": "blocked_incomplete_mapping",
                "reason": "TWC target requires verified metar_ids and cli_location_id (CLIxxx)",
            }
        )
        return out

    primary = metars[0]
    stations = tuple(dict.fromkeys([*metars, *neighbors]))
    collect_summary: dict[str, Any] = {}
    if collect:
        try:
            collect_summary["aviationweather_metar"] = collect_metar(store, stations=stations, hours=30)
        except Exception as exc:
            collect_summary["aviationweather_metar"] = {"ok": False, "error": str(exc), "usable": False}
        try:
            collect_summary["twc_climate"] = collect_twc_climate(store, cli_ids=[cli_id])
        except Exception as exc:
            collect_summary["twc_climate"] = {"ok": False, "error": str(exc), "usable": False}
        try:
            collect_summary["twc_metar"] = collect_twc_metar(store, icao_ids=[primary])
        except Exception as exc:
            collect_summary["twc_metar"] = {"ok": False, "error": str(exc), "usable": False}
    out["collection"] = collect_summary

    av_ok = bool((collect_summary.get("aviationweather_metar") or {}).get("usable")) or bool(
        store.latest_sample(f"metar_{primary}")
    )
    twc_ok = bool((collect_summary.get("twc_metar") or {}).get("usable")) or bool(
        store.latest_sample(f"twc_metar_{primary}")
    )
    out["data_availability"] = {
        "metar_primary": primary,
        "aviationweather_usable": av_ok,
        "twc_metar_usable": twc_ok,
        "twc_cli_id": cli_id,
        "notes": target.get("notes"),
    }
    if not av_ok and not twc_ok and collect:
        out.update(
            {
                "status": "blocked_no_usable_metar",
                "reason": f"No usable AviationWeather or TWC METAR for {primary}",
            }
        )
        return out

    features = build_twc_operating_features(
        store,
        now=now,
        metar_id=primary,
        twc_cli_id=cli_id,
        tz_name=tz,
        lat=target.get("lat"),
        lon=target.get("lon"),
        location_id=location_id,
        include_satrad=(location_id == "twc_nyc_central_park"),
    )
    out["features_summary"] = {
        "max_so_far": features.get("max_so_far"),
        "coverage": features.get("coverage"),
        "cli_applied": features.get("cli_applied"),
        "twc_metar": features.get("twc_metar"),
        "attribution": features.get("attribution"),
        "missing": features.get("missing"),
    }

    local = civil_local(now, tz)
    supported, decision_hour = is_supported_decision_time(now, tz_name=tz)
    snapped = False
    if not supported and force_decision:
        # Use latest trained hour already reached today (8/11/14); after 14 use 14.
        mins = local.hour * 60 + local.minute
        past = [h for h in SUPPORTED_DECISION_HOURS_LOCAL if h * 60 <= mins]
        if past:
            decision_hour = past[-1]
            supported = True
            snapped = True
            out["decision_hour_snapped"] = True
    climate_day = date.fromisoformat(features["climate_day"])
    ctx = ForecastContext(
        location_id=location_id,
        series_ticker=series,
        measurement=measurement,
        settlement_source_family="weather_company",
        climate_day=climate_day,
        decision_time_utc=now,
        horizon="same_day",
        model_version=None,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        decision_hour_local=decision_hour,
        timezone=tz,
        uses_lst_climate_day=True,
        metar_id=primary,
        cli_location_id=cli_id,
        lat=target.get("lat"),
        lon=target.get("lon"),
        elev_m=target.get("elev_m"),
        mode="RESEARCH",
        extras={"model_family": "twc_daily_max_v1", "decision_hour_snapped": snapped},
    )
    out["forecast_context"] = ctx.as_dict()
    out["decision_hour_local"] = decision_hour
    out["supported_decision_hours_local"] = list(SUPPORTED_DECISION_HOURS_LOCAL)
    out["generated_at_local"] = local.isoformat()

    if not supported:
        nxt = next_supported_decision_utc(now, tz_name=tz)
        out.update(
            {
                "status": "unsupported_decision_time",
                "reason": (
                    f"Actionable forecasts restricted to local hours {SUPPORTED_DECISION_HOURS_LOCAL}; "
                    f"now={local.strftime('%H:%M %Z')} tz={tz}"
                ),
                "next_supported_run_utc": nxt.isoformat(),
                "next_supported_run_local": civil_local(nxt, tz).isoformat(),
                "ok": True,
                "paper_decision": {
                    "decision": "blocked_unsupported_decision_time",
                    "reason": "off-hours — not an end-to-end forecast success",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    cov = features.get("coverage") or {}
    if features.get("max_so_far") is None or not cov.get("adequate"):
        out.update(
            {
                "status": "insufficient_data",
                "reason": "Essential station coverage inadequate for TWC path",
                "coverage_notes": cov.get("notes") if isinstance(cov, dict) else None,
                "ok": True,
                "paper_decision": {
                    "decision": "blocked_insufficient_data",
                    "reason": "coverage inadequate",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    model_path, model_origin = _resolve_twc_model_path(target, location_id, measurement)
    if model_path is None:
        out.update(
            {
                "status": "blocked_no_location_model",
                "reason": (
                    f"No TWC/same-ICAO model for {location_id}; collect history then train, "
                    "or add same_station_model_location_id transfer after validation."
                ),
                "ok": True,
                "model_origin": model_origin,
                "paper_decision": {
                    "decision": "blocked_unsupported_model",
                    "reason": "location model missing",
                    "live_blocked": True,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    calib_candidates = [
        _artifact_dir(location_id, measurement) / "station_corrected_v2_calibration.json",
    ]
    sibling = (target.get("details") or {}).get("same_station_model_location_id") or target.get(
        "same_station_model_location_id"
    )
    if sibling:
        calib_candidates.append(
            _artifact_dir(sibling, measurement) / "station_corrected_v2_calibration.json"
        )
    if sibling == "nyc_central_park" or location_id == "twc_nyc_central_park":
        calib_candidates.append(Path("data/obs_engine/feeds/models/station_corrected_v2_calibration.json"))
    calib_path = next((p for p in calib_candidates if p.exists()), None)

    pred = predict_station_v2(
        context=ctx,
        features=features.get("features") or {},
        max_so_far=float(features["max_so_far"]),
        coverage_adequate=True,
        decision_hour_local=decision_hour,
        cli_applied=features.get("cli_applied"),
        model_path=model_path,
        calib_path=calib_path,
    )
    out["prediction"] = pred.as_dict()
    out["status"] = pred.status
    out["probabilities_available"] = pred.probabilities_available
    out["calibration_status"] = pred.calibration_status
    out["point_median_f"] = pred.point_median_f
    out["observation_constraint"] = pred.observation_constraint
    out["model_origin"] = model_origin
    out["model_artifact"] = str(model_path)
    out["attribution"] = features.get("attribution")
    out["sim_label"] = "twc_adapter_exploratory_paper"
    if "same_icao_transfer" in model_origin:
        out["model_transfer_note"] = (
            "Residual model transferred from same-ICAO ASOS/NWS-label training; "
            "settlement floor and progressive max use TWC portal — labeled exploratory."
        )

    if not pred.probabilities_available:
        out.update(
            {
                "ok": True,
                "reason": pred.reason,
                "paper_decision": {
                    "decision": "blocked_calibration_unavailable"
                    if pred.status == "probabilities_unavailable"
                    else f"blocked_{pred.status}",
                    "reason": pred.reason,
                    "live_blocked": True,
                    "research_point_forecast": pred.point_median_f,
                },
            }
        )
        _write_location_report(location_id, measurement, out)
        return out

    if not do_paper:
        out["ok"] = True
        _write_location_report(location_id, measurement, out)
        return out

    try:
        paper = _paper_evaluate(
            pred=pred,
            series_ticker=series,
            climate_day=climate_day,
            decision_hour=decision_hour,
            location_id=location_id,
            max_so_far=features.get("max_so_far"),
        )
        out["paper_decision"] = paper.get("paper_decision")
        out["brackets"] = paper.get("brackets")
        out["ev_evaluations"] = paper.get("ev_evaluations")
        out["paper_ledger"] = paper.get("paper_ledger")
        out["target_date"] = paper.get("target_date")
        out["horizon"] = paper.get("horizon")
        if paper.get("status"):
            out["status"] = paper["status"]
            out["reason"] = paper.get("reason")
        out["ok"] = True
    except Exception as exc:
        logger.exception("paper eval %s", location_id)
        out["paper_decision"] = {
            "decision": "blocked_paper_error",
            "reason": str(exc),
            "live_blocked": True,
        }
        out["ok"] = True
        out["status"] = "paper_error"

    _write_location_report(location_id, measurement, out)
    return out


def _paper_evaluate(
    *,
    pred,
    series_ticker: str,
    climate_day: date,
    decision_hour: int | None,
    location_id: str,
    max_so_far: float | None = None,
) -> dict[str, Any]:
    from kalshi_bot.api.client import KalshiClient
    from kalshi_bot.api.fees import estimate_net_fee
    from kalshi_bot.api.orderbook import parse_orderbook
    from kalshi_bot.config import load_config
    from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
        MAX_STRIKE_DISTANCE_F,
        MIN_MODEL_P,
        format_bet_rationale,
        select_forecast_consistent,
    )

    cfg = load_config("config.yaml" if Path("config.yaml").exists() else "config.example.yaml")
    client = KalshiClient(cfg.api)
    ledger = _shared_paper_ledger()
    dist = pred.distribution
    out: dict[str, Any] = {"live_order_submitted": False}
    try:
        markets = fetch_open_series_markets(client, [series_ticker])
        target_day, event_markets = select_event_markets(markets, prefer_day=climate_day)
        out["target_date"] = target_day.isoformat() if target_day else None
        if target_day is None:
            out["status"] = "no_open_markets"
            out["paper_decision"] = {
                "decision": "blocked_no_markets",
                "reason": f"no open markets for {series_ticker}",
                "live_blocked": True,
            }
            return out
        if target_day != climate_day:
            out["status"] = "unsupported_horizon"
            out["horizon"] = "other_day"
            out["reason"] = (
                f"Open markets are for {target_day.isoformat()} but forecast climate day is "
                f"{climate_day.isoformat()}"
            )
            out["paper_decision"] = {
                "decision": "blocked_unsupported_horizon",
                "reason": out["reason"],
                "live_blocked": True,
            }
            return out

        out["horizon"] = "same_day"
        brackets = []
        evaluations: list[dict[str, Any]] = []
        quote_ts = datetime.now(timezone.utc).isoformat()
        for m in sorted(event_markets, key=lambda x: (x.get("strike_type") or "", x.get("floor_strike") or 0)):
            iv = interval_from_market(m)
            if iv is None:
                continue
            interval = {"op": iv.op, "low": iv.low, "high": iv.high}
            p_yes = dist.p_interval(iv)
            p_no = ONE - p_yes
            ticker = m.get("ticker")
            yes_ask = no_ask = None
            yes_depth = no_depth = None
            try:
                raw = client.get_orderbook(ticker, depth=5)
                book = parse_orderbook(raw)
                yes_ask = book.best_yes_ask
                no_ask = book.best_no_ask
                yd = book.yes_ask_depth()
                nd = book.no_ask_depth()
                yes_depth = str(yd[0].size) if yd else None
                no_depth = str(nd[0].size) if nd else None
            except Exception as exc:
                logger.info("orderbook %s: %s", ticker, exc)

            brackets.append(
                {
                    "ticker": ticker,
                    "p_yes": str(p_yes),
                    "p_no": str(p_no),
                    "yes_ask": str(yes_ask) if yes_ask is not None else None,
                    "no_ask": str(no_ask) if no_ask is not None else None,
                    "yes_ask_size": yes_depth,
                    "no_ask_size": no_depth,
                    "quote_ts_utc": quote_ts,
                    "interval": interval,
                }
            )
            for side, p, ask, depth_s in (
                ("yes", p_yes, yes_ask, yes_depth),
                ("no", p_no, no_ask, no_depth),
            ):
                if ask is None or ask <= ZERO or ask >= ONE:
                    evaluations.append(
                        {
                            "ticker": ticker,
                            "side": side,
                            "decision": "blocked_unavailable_prices",
                            "reason": f"no executable {side} ask",
                            "p": str(p),
                            "interval": interval,
                        }
                    )
                    continue
                if depth_s is None or D(depth_s) < D("1"):
                    evaluations.append(
                        {
                            "ticker": ticker,
                            "side": side,
                            "decision": "blocked_insufficient_depth",
                            "reason": f"{side} ask depth < 1",
                            "p": str(p),
                            "ask": str(ask),
                            "interval": interval,
                        }
                    )
                    continue
                qty = D("1")
                fees = estimate_net_fee(
                    qty,
                    ask,
                    multiplier=cfg.trading.fee_multiplier,
                    assume_taker=True,
                    balance_precision=cfg.trading.balance_precision,
                )
                buffer = D(cfg.trading.uncertainty_buffer)
                ev = p - ask - (fees / qty) - buffer
                evaluations.append(
                    {
                        "ticker": ticker,
                        "side": side,
                        "decision": "ev_evaluated",
                        "reason": f"EV={ev}",
                        "p": str(p),
                        "ask": str(ask),
                        "fees": str(fees),
                        "uncertainty_buffer": str(buffer),
                        "ev": str(ev),
                        "qty": str(qty),
                        "quote_ts_utc": quote_ts,
                        "interval": interval,
                    }
                )

        for e in evaluations:
            if e.get("ev") is not None and D(e["ev"]) <= ZERO:
                e["decision"] = "paper_skip_negative_ev"
                e["reason"] = f"EV={e['ev']} ≤ 0 after fees/buffer"

        median_f = float(pred.point_median_f) if getattr(pred, "point_median_f", None) is not None else None
        if median_f is None and hasattr(pred, "distribution"):
            # fallback: some predictors expose median elsewhere
            median_f = getattr(pred, "median_f", None)
            median_f = float(median_f) if median_f is not None else None

        paper_decision: dict[str, Any]
        best = None
        rejected: list[dict[str, Any]] = []
        if median_f is not None:
            best, rejected = select_forecast_consistent(
                evaluations, median_f=median_f, brackets=brackets
            )
        out["selection_rejected"] = rejected

        if best is not None and median_f is not None:
            oid = f"paper-{location_id}-{best['ticker']}-{best['side']}-{climate_day.isoformat()}-{decision_hour}"
            why = format_bet_rationale(
                point_median_f=median_f,
                max_so_far=max_so_far,
                ticker=str(best["ticker"]),
                side=str(best["side"]),
                p=best["p"],
                ask=best["ask"],
                interval=best.get("interval"),
            )
            sim = ledger.try_simulate_fill(
                client_order_id=oid,
                ticker=best["ticker"],
                side=best["side"],
                qty=D(best["qty"]),
                price=D(best["ask"]),
                fees=D(best["fees"]),
                decision_reason="paper_sim_fill_unvalidated",
                details={**best, "why_buy": why},
                location_id=location_id,
                series_ticker=series_ticker,
                quote_ts_utc=quote_ts,
                market_ts_utc=quote_ts,
            )
            if sim.get("ok") and sim.get("filled"):
                paper_decision = {
                    **best,
                    "decision": "paper_sim_fill_unvalidated",
                    "reason": "Simulated taker fill; live order NOT submitted",
                    "why_buy": why,
                    "selection_rule": (
                        f"forecast-consistent (≤{MAX_STRIKE_DISTANCE_F}°F from median), "
                        f"model_p≥{MIN_MODEL_P}, then max EV — not cheapest ask"
                    ),
                    "sim": sim,
                    "live_blocked": True,
                    "live_order_submitted": False,
                }
            else:
                paper_decision = {
                    **best,
                    "decision": f"paper_skip_{sim.get('reason') or 'rejected'}",
                    "reason": sim.get("reason"),
                    "why_buy": why,
                    "sim": sim,
                    "live_blocked": True,
                    "live_order_submitted": False,
                }
        elif evaluations:
            # No forecast-consistent +EV row — do not fall back to raw max-EV cheap tickets.
            paper_decision = {
                "decision": "paper_skip_no_forecast_consistent_ev",
                "reason": (
                    "No +EV contract within the forecast band after min model_p filter; "
                    "refusing cheapest-ask fallback"
                ),
                "selection_rejected": rejected,
                "live_blocked": True,
                "live_order_submitted": False,
            }
        else:
            paper_decision = {
                "decision": "paper_skip_no_brackets",
                "reason": "no evaluable brackets",
                "live_blocked": True,
                "live_order_submitted": False,
            }

        out["brackets"] = brackets
        out["ev_evaluations"] = evaluations
        out["paper_decision"] = paper_decision
        out["paper_ledger"] = ledger.snapshot()
        return out
    finally:
        client.close()
        ledger.close()


def _write_location_report(location_id: str, measurement: str, out: dict[str, Any]) -> None:
    d = _artifact_dir(location_id, measurement)
    (d / "latest_report.json").write_text(json.dumps(out, indent=2, default=str))


def run_multi_cycle(
    *,
    registry: LocationRegistry | None = None,
    store: FeedStore | None = None,
    now: datetime | None = None,
    do_paper: bool = True,
    collect: bool = True,
    only_location_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Process every operating NWS CLI daily-max location independently."""
    registry = registry or LocationRegistry()
    store = store or FeedStore()
    now = now or datetime.now(timezone.utc)
    targets = registry.operating_daily_max()
    if only_location_ids:
        targets = [t for t in targets if t["location_id"] in only_location_ids]

    results: list[dict[str, Any]] = []
    for t in targets:
        try:
            results.append(
                process_location(t, store, now=now, do_paper=do_paper, collect=collect)
            )
        except Exception as exc:
            logger.exception("location %s crashed", t.get("location_id"))
            results.append(
                {
                    "location_id": t.get("location_id"),
                    "series_ticker": t.get("series_ticker"),
                    "ok": False,
                    "status": "error",
                    "reason": str(exc),
                    "live_order_submitted": False,
                }
            )

    summary = {
        "generated_at_utc": now.isoformat(),
        "n_targets": len(targets),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_probabilities": sum(1 for r in results if r.get("probabilities_available")),
        "by_status": {},
        "live_order_submitted_any": any(r.get("live_order_submitted") for r in results),
        "locations": [
            {
                "location_id": r.get("location_id"),
                "series_ticker": r.get("series_ticker"),
                "status": r.get("status"),
                "mapping_status": r.get("mapping_status"),
                "probabilities_available": r.get("probabilities_available"),
                "paper_decision": (r.get("paper_decision") or {}).get("decision"),
                "reason": r.get("reason"),
            }
            for r in results
        ],
    }
    for r in results:
        st = r.get("status") or "unknown"
        summary["by_status"][st] = summary["by_status"].get(st, 0) + 1

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_ROOT / "last_multi_cycle.json"
    payload = {"summary": summary, "results": results}
    report_path.write_text(json.dumps(payload, indent=2, default=str))
    return {"ok": True, "summary": summary, "report": str(report_path), "n_results": len(results)}
