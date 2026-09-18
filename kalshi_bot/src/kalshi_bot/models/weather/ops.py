"""CLI operations for weather archive collect / train / validate."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from kalshi_bot.config import AppConfig
from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.cli_reports import CliReportClient
from kalshi_bot.models.weather.forecast_collect import ForecastCollector
from kalshi_bot.models.weather.stations import STATIONS
from kalshi_bot.models.weather.train import build_pairs, train_empirical, train_quantile_gbm
from kalshi_bot.validation.metrics import brier_score

logger = logging.getLogger(__name__)


def archive_from_config(config: AppConfig) -> WeatherArchive:
    path = getattr(config.models.weather, "archive_path", None) or "data/weather_archive.db"
    return WeatherArchive(path)


def weather_collect(config: AppConfig, *, days_ahead: int = 2, open_meteo: bool = True) -> dict[str, Any]:
    archive = archive_from_config(config)
    cli = CliReportClient()
    collector = ForecastCollector(archive)
    summary: dict[str, Any] = {"stations": {}}
    today = date.today()
    try:
        for key, station in STATIONS.items():
            st_sum: dict[str, Any] = {"cli": 0, "forecasts": 0}
            try:
                reports = cli.collect_recent_with_max(station.cli_location_id, limit=35)
                for rep in reports:
                    archive.save_cli(station_key=key, report=rep)
                    st_sum["cli"] += 1
            except Exception as exc:
                st_sum["cli_error"] = str(exc)
            for d in range(0, days_ahead + 1):
                day = today + timedelta(days=d)
                fc = collector.collect_nws_daytime_high(station, day)
                if fc:
                    st_sum["forecasts"] += 1
            if open_meteo:
                try:
                    collector.collect_open_meteo_daily(
                        station, today - timedelta(days=14), today, model="gfs_seamless"
                    )
                    st_sum["open_meteo"] = True
                except Exception as exc:
                    st_sum["open_meteo_error"] = str(exc)
            summary["stations"][key] = st_sum
    finally:
        cli.close()
        collector.close()
    summary["archive_path"] = str(archive.path)
    return summary


def weather_train(config: AppConfig) -> dict[str, Any]:
    archive = archive_from_config(config)
    out: dict[str, Any] = {}
    for key in STATIONS:
        safe = train_empirical(archive, key, source="nws_grid", allow_retrospective=False)
        retro = train_empirical(archive, key, source="open_meteo", allow_retrospective=True)
        if safe.get("ok"):
            best = {"source": "nws_grid", "retrospective": False, "n": safe["metrics"]["train_n"]}
            train_empirical(archive, key, source="nws_grid", allow_retrospective=False)
        elif retro.get("ok"):
            best = {"source": "open_meteo", "retrospective": True, "n": retro["metrics"]["train_n"]}
            train_empirical(archive, key, source="open_meteo", allow_retrospective=True)
        else:
            best = None
        gbm = train_quantile_gbm(
            archive,
            key,
            source=(best or {}).get("source", "open_meteo"),
            allow_retrospective=True,
        )
        out[key] = {"decision_time_safe": safe, "retrospective_open_meteo": retro, "best": best, "quantile_gbm": gbm}
    return out


def weather_validate(config: AppConfig) -> dict[str, Any]:
    """Walk-forward style check on paired rows; does not invent order-book PnL."""
    archive = archive_from_config(config)
    report: dict[str, Any] = {
        "promotion": "docs/PROMOTION_CRITERIA.md",
        "trading_profitability": "UNVALIDATED — no historical executable book replay in this report",
        "stations": {},
    }
    for key in STATIONS:
        pairs = build_pairs(archive, key, source="open_meteo", allow_retrospective=True)
        art = archive.load_artifact(key, "empirical_residual")
        st: dict[str, Any] = {"paired_n": len(pairs), "trained": art is not None}
        if len(pairs) >= 8 and art:
            import json as _json

            residuals = _json.loads(art["artifact_json"]).get("residuals") or []
            # Hold out last 20%
            cut = int(len(pairs) * 0.8)
            hold = pairs[cut:]
            # Binary probe: P(max > forecast) using empirical — weak but leakage-safe
            from kalshi_bot.models.weather.distribution import from_empirical_residuals
            from kalshi_bot.models.weather.settlement_rules import TempInterval

            probs, outcomes = [], []
            for p in hold:
                dist = from_empirical_residuals(float(p["forecast_f"]), residuals)
                # Example contract: greater than floor=forecast (not a Kalshi contract — calibration probe)
                iv = TempInterval("gt", float(p["forecast_f"]), None, "validation_probe")
                probs.append(dist.p_interval(iv))
                outcomes.append(1 if float(p["outcome_f"]) > float(p["forecast_f"]) else 0)
            if probs:
                st["holdout_n"] = len(probs)
                st["brier_probe"] = str(brier_score(probs, outcomes))
                st["note"] = (
                    "Probe is forecast-error calibration, not Kalshi contract Brier. "
                    "Collect forward paper decisions for contract-level validation."
                )
        else:
            st["note"] = "Insufficient paired CLI/forecast history for holdout metrics"
        report["stations"][key] = st
    archive.save_validation_report(report)
    report["meets_live_promotion"] = False
    report["reason"] = (
        "Need ≥100 settled Kalshi decision records with settlement-aligned model + "
        "executable-price paper PnL before live_eligible"
    )
    return report


def weather_obs_train(config: AppConfig) -> dict[str, Any]:
    """Train observation-driven NYC remaining-rise model (no forecast features)."""
    from kalshi_bot.models.weather.obs_engine.train_obs import train_obs_nyc

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    archive_path = getattr(config.models.weather, "archive_path", None) or "data/weather_archive.db"
    return train_obs_nyc(data_dir=data_dir, archive_path=archive_path)


def weather_obs_validate(config: AppConfig) -> dict[str, Any]:
    """Summarize last obs train/backtest reports; does not claim NWS superiority without metrics."""
    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    report_path = data_dir / "last_train_report.json"
    backtest_path = data_dir / "backtest_report.json"
    archive = archive_from_config(config)
    art = archive.load_artifact("NYC", "obs_nyc_remaining_rise_v1")
    out: dict[str, Any] = {
        "engine": "obs_driven_nyc",
        "live_eligible": False,
        "config_obs_engine_live_eligible": bool(
            getattr(config.models.weather, "obs_engine_live_eligible", False)
        ),
        "artifact_present": art is not None,
        "trading_profitability": "UNVALIDATED — no executable-price paper PnL yet",
    }
    if backtest_path.exists():
        bt = json.loads(backtest_path.read_text())
        out["backtest"] = {
            "n_independent_climate_days": bt.get("n_independent_climate_days"),
            "splits": bt.get("splits"),
            "overall_test": bt.get("overall_test"),
            "external_benchmark": bt.get("external_benchmark"),
            "outperforms_established_nws_decision_time": (bt.get("overall_test") or {}).get(
                "outperforms_established_nws_decision_time"
            ),
        }
        out["nws_benchmark_comparison"] = (bt.get("overall_test") or {}).get(
            "outperforms_established_nws_reason"
        )
    else:
        out["nws_benchmark_comparison"] = "Run weather-obs-backtest for decision-time metrics"
    if report_path.exists():
        out["last_train_report"] = json.loads(report_path.read_text())
        metrics = (out["last_train_report"] or {}).get("metrics") or {}
        out["beats_climatology"] = metrics.get("beats_climatology")
        out["beats_continuation"] = metrics.get("beats_continuation")
        out["hold_final_mae_q50"] = metrics.get("hold_final_mae_q50")
        out["baselines_holdout"] = metrics.get("baselines_holdout")
    else:
        out["note"] = "Run weather-obs-train first"
    out["meets_live_promotion"] = False
    out["reason"] = (
        "Need multi-season holdout MAE/Brier vs same-time NWS benchmark + forward paper PnL "
        "before setting models.weather.obs_engine_live_eligible=true"
    )
    return out


def weather_obs_backtest(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.backtest import run_obs_backtest

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    archive_path = getattr(config.models.weather, "archive_path", None) or "data/weather_archive.db"
    return run_obs_backtest(data_dir=data_dir, archive_path=archive_path, fetch_external_benchmark=True)


def weather_obs_reconcile_labels(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.backtest import reconcile_cli_vs_ghcnd

    archive_path = getattr(config.models.weather, "archive_path", None) or "data/weather_archive.db"
    return reconcile_cli_vs_ghcnd(archive_path=archive_path)


def weather_obs_predict_now(config: AppConfig) -> dict[str, Any]:
    """Fresh research prediction — never places live orders."""
    from kalshi_bot.models.weather.obs_engine.predict_now import research_predict_now

    return research_predict_now(config)


def weather_obs_freeze(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.research.freeze import freeze_baseline

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return freeze_baseline(data_dir=data_dir, reproduce_backtest=True)


def weather_obs_collect_once(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.research.collector import run_collect_once

    return run_collect_once(config)


def weather_obs_audit(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.research.audit import audit_measurement_settlement

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return audit_measurement_settlement(data_dir=data_dir)


def weather_obs_diagnose(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.research.experiments import export_diagnostics

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return export_diagnostics(data_dir=data_dir, split="test")


def weather_obs_experiment(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.research.experiments import run_feature_experiments

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return run_feature_experiments(data_dir=data_dir)


def weather_feeds_once(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.feeds.worker import run_collect_cycle

    return run_collect_cycle(do_infer=True)


def weather_feeds_status(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    store = FeedStore()
    pidfile = Path("data/obs_engine/feeds/collector.pid")
    running = False
    pid = None
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)
            running = True
        except Exception:
            running = False
    now = datetime.now(timezone.utc)
    feed_ages = {}
    for feed in ("metar_KNYC", "metar_KLGA", "metar_KJFK", "cli_nyc", "goes19_acmc", "nexrad_okx_n0b"):
        row = store.latest_sample(feed)
        if not row:
            feed_ages[feed] = {"connected": False, "age_hours": None, "valid_utc": None}
            continue
        age_h = None
        if row.get("valid_utc"):
            try:
                age_h = (now - datetime.fromisoformat(row["valid_utc"])).total_seconds() / 3600.0
            except Exception:
                age_h = None
        feed_ages[feed] = {
            "connected": True,
            "age_hours": age_h,
            "valid_utc": row.get("valid_utc"),
            "retrieved_at_utc": row.get("retrieved_at_utc"),
            "source_key": row.get("source_key"),
        }
    out = {
        "running": running,
        "pid": pid,
        "heartbeat": store.get_heartbeat(),
        "checkpoints": store.list_checkpoints(),
        "feed_ages": feed_ages,
        "latest_prediction": store.latest_prediction(),
        "nys_mesonet": {"status": "optional_blocked", "reason": "No permitted access configured"},
    }
    store.close()
    return out


def weather_feeds_train(config: AppConfig) -> dict[str, Any]:
    from kalshi_bot.models.weather.obs_engine.feeds.train_operating import train_station_corrected

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return train_station_corrected(data_dir=data_dir)


def weather_train_location(config: AppConfig, *, location_id: str = "chi_midway") -> dict[str, Any]:
    """Train a location-specific station_v2 model (e.g. chi_midway, nyc_central_park, lax_airport)."""
    from kalshi_bot.models.weather.obs_engine.feeds.train_operating import train_location_station_corrected
    from kalshi_bot.models.weather.obs_engine.multi.registry import LocationRegistry

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    report = train_location_station_corrected(location_id, data_dir=data_dir)
    if report.get("ok") and location_id in ("chi_midway", "lax_airport"):
        reg = LocationRegistry()
        try:
            for t in reg.list_targets(measurement="daily_max_temp_f"):
                if t.get("location_id") == location_id:
                    row = dict(t)
                    row["validation_status"] = "operating_candidate"
                    row["data_availability"] = "public_metar_cli_ghcnd_backfilled"
                    if location_id == "chi_midway":
                        row["model_family"] = "station_v2_nws_cli"
                        row["notes"] = (
                            (row.get("notes") or "")
                            + " | Midway station_v2 trained; GHCND USW00014819 labels; live still blocked"
                        )
                    else:
                        row["model_family"] = "station_v2_twc_proxy"
                        row["notes"] = (
                            (row.get("notes") or "")
                            + " | LAX station_v2 trained; GHCND USW00023174 labels; TWC KXHIGHLAX transfer"
                        )
                    # unwrap json fields expected by upsert
                    row["metar_ids"] = row.get("metar_ids") or []
                    row["neighbor_metar_ids"] = row.get("neighbor_metar_ids") or []
                    reg.upsert_target(row)
        finally:
            reg.close()
        report["registry_validation_status"] = "operating_candidate"
    return report


def weather_discover(config: AppConfig) -> dict[str, Any]:
    """Refresh Kalshi weather market discovery into the location registry."""
    from kalshi_bot.api.client import KalshiClient
    from kalshi_bot.models.weather.obs_engine.multi.discovery import discover_weather_markets
    from kalshi_bot.models.weather.obs_engine.multi.registry import LocationRegistry

    client = KalshiClient(config.api)
    registry = LocationRegistry()
    try:
        return discover_weather_markets(client, registry=registry)
    finally:
        client.close()
        registry.close()


def weather_multi_once(config: AppConfig) -> dict[str, Any]:
    """One multi-location collect/infer/paper cycle (live orders blocked)."""
    from kalshi_bot.models.weather.obs_engine.multi.pipeline import run_multi_cycle
    from kalshi_bot.models.weather.obs_engine.multi.registry import (
        VERIFIED_NWS_CLI_DAILY_MAX,
        VERIFIED_TWC_DAILY_MAX,
        LocationRegistry,
    )

    registry = LocationRegistry()
    # Ensure verified mappings exist even if discovery has not run yet
    if not registry.operating_daily_max():
        for tick, base in VERIFIED_NWS_CLI_DAILY_MAX.items():
            registry.upsert_target(
                {
                    **base,
                    "series_ticker": tick,
                    "measurement": "daily_max_temp_f",
                    "settlement_source_family": "nws_cli",
                    "unit": "F",
                    "uses_lst_climate_day": True,
                    "data_availability": "public_metar_cli",
                }
            )
    # Always refresh verified TWC rows so adapter mappings stay current
    for tick, base in VERIFIED_TWC_DAILY_MAX.items():
        details = {}
        if base.get("same_station_model_location_id"):
            details["same_station_model_location_id"] = base["same_station_model_location_id"]
        registry.upsert_target(
            {
                **{k: v for k, v in base.items() if k != "same_station_model_location_id"},
                "series_ticker": tick,
                "measurement": "daily_max_temp_f",
                "settlement_source_family": "weather_company",
                "settlement_name": "The Weather Company",
                "settlement_url": "https://weather.com/kalshi",
                "unit": "F",
                "uses_lst_climate_day": True,
                "rounding_note": "Whole °F as printed on weather.com/kalshi climate report",
                "data_availability": "twc_kalshi_portal_public",
                "details": details,
            }
        )
    try:
        return run_multi_cycle(registry=registry, do_paper=True, collect=True)
    finally:
        registry.close()


def weather_multi_status(config: AppConfig) -> dict[str, Any]:
    """Per-location mapping / latest report / aggregate summary + capability matrix."""
    import json
    from pathlib import Path

    from kalshi_bot.models.weather.obs_engine.feeds.train_operating import LOCATION_TRAIN_PROFILES
    from kalshi_bot.models.weather.obs_engine.multi.registry import (
        VERIFIED_NWS_CLI_DAILY_MAX,
        VERIFIED_TWC_DAILY_MAX,
        LocationRegistry,
    )

    registry = LocationRegistry()
    try:
        targets = registry.list_targets()
        operating = registry.operating_daily_max()
        root = Path("data/obs_engine/multi")
        last = None
        if (root / "last_multi_cycle.json").exists():
            last = json.loads((root / "last_multi_cycle.json").read_text())
        discovery = None
        if (root / "last_discovery.json").exists():
            discovery = json.loads((root / "last_discovery.json").read_text()).get("summary")
        per_loc = []
        for t in operating:
            report_path = (
                root
                / "artifacts"
                / f"{t['location_id']}__{t['measurement']}"
                / "latest_report.json"
            )
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            per_loc.append(
                {
                    "location_id": t["location_id"],
                    "series_ticker": t["series_ticker"],
                    "mapping_status": t.get("mapping_status"),
                    "validation_status": t.get("validation_status"),
                    "timezone": t.get("timezone"),
                    "metar_ids": t.get("metar_ids"),
                    "cli_location_id": t.get("cli_location_id"),
                    "latest_status": (report or {}).get("status"),
                    "latest_paper": ((report or {}).get("paper_decision") or {}).get("decision"),
                    "probabilities_available": (report or {}).get("probabilities_available"),
                }
            )

        capability_rows: list[dict[str, Any]] = []
        seen_locations: set[str] = set()
        for source, mapping in (
            ("nws_cli", VERIFIED_NWS_CLI_DAILY_MAX),
            ("weather_company", VERIFIED_TWC_DAILY_MAX),
        ):
            for series, base in mapping.items():
                loc = str(base.get("location_id"))
                if loc in seen_locations:
                    continue
                seen_locations.add(loc)
                sibling = base.get("same_station_model_location_id")
                train_id = sibling or loc
                has_train_profile = train_id in LOCATION_TRAIN_PROFILES
                art = root / "artifacts" / f"{train_id}__daily_max_temp_f"
                has_model = (art / "model.joblib").exists() or any(art.glob("*.joblib"))
                has_calib = (art / "calibration.json").exists() or any(
                    art.glob("*calib*.json")
                )
                if has_train_profile and has_model and has_calib:
                    stage = "trained_calibrated_research"
                    prep = (
                        f"PYTHONPATH=src python3 -m kalshi_bot.cli weather-train-location "
                        f"--location {train_id}  # retrain/eval; live still blocked"
                    )
                elif has_train_profile and not has_model:
                    stage = "profile_ready_missing_model_artifact"
                    prep = (
                        f"Place ASOS/GHCND under profile paths then: "
                        f"weather-train-location --location {train_id}"
                    )
                elif not has_train_profile:
                    stage = "needs_historical_backfill_and_train_profile"
                    prep = (
                        "Backfill IEM ASOS + GHCND TMAX for this station, add "
                        f"LOCATION_TRAIN_PROFILES['{train_id}'], then weather-train-location. "
                        "Do not mark operating or live_eligible until evaluated."
                    )
                else:
                    stage = "trained_missing_location_calibration"
                    prep = (
                        f"Calibrate residuals for {train_id} only — no NYC fallback. "
                        "Missing calib → probabilities_unavailable."
                    )
                capability_rows.append(
                    {
                        "location_id": loc,
                        "series_ticker": series,
                        "settlement_source_family": source,
                        "train_profile_id": train_id if has_train_profile else None,
                        "same_icao_transfer": bool(sibling),
                        "has_train_profile": has_train_profile,
                        "has_model_artifact": bool(has_model),
                        "has_location_calibration": bool(has_calib),
                        "stage": stage,
                        "live_eligible": False,
                        "prep_workflow": prep,
                    }
                )

        counts = {
            "verified_nws_series": len(VERIFIED_NWS_CLI_DAILY_MAX),
            "verified_twc_series": len(VERIFIED_TWC_DAILY_MAX),
            "unique_locations": len(seen_locations),
            "train_profiles": len(LOCATION_TRAIN_PROFILES),
            "trained_calibrated_research": sum(
                1 for r in capability_rows if r["stage"] == "trained_calibrated_research"
            ),
            "needs_backfill": sum(
                1 for r in capability_rows if r["stage"] == "needs_historical_backfill_and_train_profile"
            ),
            "live_eligible": 0,
        }
        return {
            "n_registry_targets": len(targets),
            "n_operating_nws_cli_daily_max": len(operating),
            "discovery_summary": discovery,
            "last_cycle_summary": (last or {}).get("summary"),
            "locations": per_loc,
            "capability_matrix": capability_rows,
            "counts": counts,
            "live_orders": False,
            "note": (
                "Registry presence ≠ operating ≠ live_eligible. "
                "Aliases are not additional locations. Daily-low/precip use other families."
            ),
        }
    finally:
        registry.close()


def weather_tune_twc(config: AppConfig, *, lookback_days: int = 100, promote: bool = True) -> dict[str, Any]:
    """Tune hour bias + TWC residual calib; promote only if holdout MAE improves."""
    from kalshi_bot.models.weather.obs_engine.multi.tune_twc_accuracy import tune_all_twc

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return tune_all_twc(data_dir=data_dir, lookback_days=lookback_days, promote=promote)


def weather_eval_candidates(config: AppConfig, *, location_id: str | None = None) -> dict[str, Any]:
    """Chronological A–D candidate evaluation (research only; never live-eligible)."""
    from kalshi_bot.models.weather.obs_engine.multi.candidates.evaluate import (
        evaluate_location,
        run_candidate_evaluation,
    )

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    if location_id:
        return evaluate_location(location_id, data_dir=data_dir)
    return run_candidate_evaluation(data_dir=data_dir)


def weather_historical_replay(config: AppConfig, *, location_id: str | None = None) -> dict[str, Any]:
    """Chronological weather replay + deterministic picker (research; no live orders)."""
    from kalshi_bot.models.weather.obs_engine.multi.replay.runner import (
        replay_location,
        run_historical_replay,
    )

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    if location_id:
        return replay_location(location_id, data_dir=data_dir)
    return run_historical_replay(data_dir=data_dir)


def weather_score_twc(config: AppConfig, *, n_days: int = 5) -> dict[str, Any]:
    """Score production predictions vs official TWC settlement highs (research)."""
    from kalshi_bot.models.weather.obs_engine.multi.score_vs_twc import score_recent_twc

    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    return score_recent_twc(n_days=n_days, data_dir=data_dir, ensure_nyc=True)


def weather_twc_bet(
    config: AppConfig,
    *,
    series_ticker: str = "KXHIGHNY",
    dollars: float = 5.0,
    live: bool = False,
    dry_run: bool = False,
    force_decision: bool = False,
) -> dict[str, Any]:
    """TWC daily-max callout + optional user-requested capped live bet."""
    from kalshi_bot.models.weather.obs_engine.multi.live_bet import weather_twc_bet as _run

    return _run(
        config,
        series_ticker=series_ticker,
        dollars=dollars,
        live=live,
        dry_run=dry_run,
        force_decision=force_decision,
    )
