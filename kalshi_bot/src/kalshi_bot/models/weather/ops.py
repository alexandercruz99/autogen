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
    """Summarize last obs train report + artifact; does not claim NWS superiority without metrics."""
    data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
    report_path = data_dir / "last_train_report.json"
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
        "nws_benchmark_comparison": (
            "Not run in this command — NWS grid forecasts are benchmarks only; "
            "pair same decision-time NWS snapshots in a future validation pass."
        ),
    }
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
