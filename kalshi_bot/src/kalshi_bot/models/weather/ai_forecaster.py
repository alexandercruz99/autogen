"""AI / statistical daily-max weather model aligned to NWS CLI settlement."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_bot.config import WeatherModelConfig
from kalshi_bot.data.store import Store
from kalshi_bot.models.base import Prediction, ProbabilityModel
from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.distribution import PredictiveDistribution
from kalshi_bot.models.weather.forecast_collect import ForecastCollector
from kalshi_bot.models.weather.nws_client import parse_market_date
from kalshi_bot.models.weather.same_day import ObservationClient, apply_same_day
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.models.weather.stations import station_for_ticker
from kalshi_bot.models.weather.train import predict_distribution
from kalshi_bot.money import D, clamp01

logger = logging.getLogger(__name__)

MODEL_VERSION = "weather.ai_cli.v1.0-collecting"


class AIWeatherForecaster(ProbabilityModel):
    """Station-specific predictive distribution over official CLI daily max.

    Probabilities come from archived forecast→CLI residuals (empirical) when trained;
    otherwise a wide fallback prior with UNVALIDATED evidence. Never invents CLI outcomes.
    """

    name = "weather_ai_cli"
    version = MODEL_VERSION
    categories = ["Climate and Weather"]

    def __init__(
        self,
        config: WeatherModelConfig,
        store: Store | None = None,
        archive_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.store = store
        path = archive_path or getattr(config, "archive_path", None) or "data/weather_archive.db"
        self.archive = WeatherArchive(path)
        self.collector = ForecastCollector(self.archive)
        self.obs = ObservationClient()

    def close(self) -> None:
        self.collector.close()
        self.obs.close()

    def supports(self, market: dict[str, Any], category: str) -> bool:
        if not self.config.enabled:
            return False
        # Daily series only — skip hourly (Weather Company settlement)
        ticker = (market.get("ticker") or "").upper()
        if "HOUR" in ticker or "TEMPAT" in ticker:
            return False
        return station_for_ticker(ticker) is not None

    def predict(self, market: dict[str, Any], category: str) -> Prediction:
        now = datetime.now(timezone.utc)
        ticker = market.get("ticker") or ""
        station = station_for_ticker(ticker)
        if station is None:
            return self._skip(ticker, now, "no station registry entry")

        target_day = parse_market_date(ticker)
        if target_day is None:
            return self._skip(ticker, now, "could not parse market date")

        interval = interval_from_market(market)
        if interval is None:
            return self._skip(ticker, now, "could not map strike_type to settlement interval")

        # Refresh NWS forecast snapshot (records init/available/retrieved metadata)
        fc = self.collector.collect_nws_daytime_high(station, target_day)
        if not fc or fc.get("temp_f") is None:
            return self._skip(ticker, now, "NWS daytime high unavailable")

        # Leakage: period must have started
        start = fc.get("start_time")
        if start:
            start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            if start_dt > now:
                return self._skip(ticker, now, "forecast period starts after decision time")

        mu = float(fc["temp_f"])
        art = self.archive.load_artifact(station.city_key, "empirical_residual")
        trained = art is not None
        dist = predict_distribution(self.archive, station.city_key, mu)

        factors = [
            f"Settlement target: {station.settlement_source} @ {station.climate_station_name} ({station.cli_location_id})",
            f"Unit={station.temperature_unit}; {station.rounding_note}",
            f"LST climate day note: {station.notes}",
            f"NWS grid daytime high proxy μ={mu:.1f}°F for {target_day} (not CLI itself)",
            f"Distribution method={dist.method}; mean≈{dist.mean():.1f}°F; q10/50/90={dist.quantile(0.1)}/{dist.quantile(0.5)}/{dist.quantile(0.9)}",
        ]

        # Same-day progressive update
        if target_day == now.astimezone().date() or True:
            # Always attempt METAR evidence for target day == local today; else skip
            local_today = now.astimezone().date()  # server local — also compare via station tz ideally
            try:
                from zoneinfo import ZoneInfo

                local_today = now.astimezone(ZoneInfo(station.timezone)).date()
            except Exception:
                pass
            if target_day == local_today:
                evidence = self.obs.metar_recent_max_f(station)
                dist, same_factors = apply_same_day(dist, evidence)
                factors.extend(same_factors)

        p_yes = dist.p_interval(interval)
        # Conservative: use lower of (p, mass under tighter support via q10 stress)
        # Stress: shift distribution mean down/up by residual MAE if trained
        p_cons = p_yes
        uncertainty = D("0.08")
        if trained:
            import json

            payload = json.loads(art["artifact_json"])
            mae = float(payload.get("mae") or 3.0)
            # Stress by evaluating interval on left-shifted normal-ish: reduce edge
            from kalshi_bot.models.weather.distribution import from_empirical_residuals

            dist_lo = from_empirical_residuals(mu - mae, payload.get("residuals") or [0.0])
            dist_hi = from_empirical_residuals(mu + mae, payload.get("residuals") or [0.0])
            p_lo = dist_lo.p_interval(interval)
            p_hi = dist_hi.p_interval(interval)
            p_cons = min(p_yes, p_lo, p_hi)
            uncertainty = max(abs(p_hi - p_lo), D("0.05"))
            factors.append(f"Trained empirical residual n={payload.get('n')} mae={mae:.2f}°F train_end={payload.get('train_end')}")
            validation = (
                f"PARTIALLY_FITTED: empirical residuals n={payload.get('n')} "
                f"source={payload.get('source')} (may be retrospective Open-Meteo↔CLI). "
                "Decision-time NWS archive still collecting. Trading profitability unvalidated."
            )
        else:
            factors.append("No trained empirical artifact — using wide fallback prior")
            # Extra haircut when unfitted
            p_cons = clamp01(p_yes * D("0.5") + D("0.25"))  # pull toward 0.5
            uncertainty = D("0.15")
            validation = (
                "UNVALIDATED: collecting CLI+forecast archive; no station residual model yet. "
                "Run `kalshi-bot weather-collect` then `weather-train`."
            )

        feature_times = [
            fc.get("fetched_at") or fc.get("start_time"),
            now.isoformat(),
        ]

        return Prediction(
            market_ticker=ticker,
            p_yes=clamp01(p_yes),
            p_yes_conservative=clamp01(min(p_cons, p_yes)),
            uncertainty=clamp01(uncertainty),
            model_version=self.version,
            data_sources=[
                {
                    "name": "nws_grid_forecast",
                    "fetched_at": fc.get("fetched_at"),
                    "available_at": fc.get("start_time"),
                    "url": fc.get("forecast_url"),
                },
                {
                    "name": "settlement_spec",
                    "station": station.climate_station_name,
                    "source": station.settlement_source,
                    "cli_location_id": station.cli_location_id,
                },
            ],
            factors=factors,
            validation_evidence=validation,
            as_of=now,
            supported=True,
            details={
                "city": station.city_key,
                "target_day": target_day.isoformat(),
                "settlement_aligned": True,  # target is CLI; forecast proxy still NWS grid
                "forecast_proxy": "nws_grid_daytime",
                "distribution": dist.as_dict(),
                "contract": {
                    "op": interval.op,
                    "low": interval.low,
                    "high": interval.high,
                    "source": interval.source,
                },
                "trained": trained,
                "feature_times": feature_times,
                "p_yes_high": str(p_yes),
                "model_live_eligible": False,
                "engine": "ai_cli",
                "validation_status": "PARTIALLY_FITTED" if trained else "UNVALIDATED",
            },
        )

    def _skip(self, ticker: str, now: datetime, reason: str) -> Prediction:
        return Prediction(
            market_ticker=ticker,
            p_yes=D("0.5"),
            p_yes_conservative=D("0.5"),
            uncertainty=D("0.5"),
            model_version=self.version,
            data_sources=[],
            factors=[],
            validation_evidence="skipped",
            as_of=now,
            supported=False,
            skip_reason=reason,
            details={"model_live_eligible": False, "engine": "ai_cli"},
        )
