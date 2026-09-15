from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from scipy.stats import norm

from kalshi_bot.config import WeatherModelConfig
from kalshi_bot.data.store import Store
from kalshi_bot.models.base import Prediction, ProbabilityModel
from kalshi_bot.models.weather.nws_client import NWSClient, parse_market_date, parse_temp_contract
from kalshi_bot.money import D, clamp01

logger = logging.getLogger(__name__)

MODEL_VERSION = "weather.high_temp.v0.1-unvalidated"


class WeatherHighTempModel(ProbabilityModel):
    """Daily high-temperature binary markets using NWS forecast + Gaussian error model.

    VALIDATION STATUS: Unvalidated for live edge. The forecast-error sigma is a
    configurable prior, not a walk-forward calibrated estimate. Paper trading and
    chronological validation must accumulate before treating EV as actionable for live.
    """

    name = "weather_high_temp"
    version = MODEL_VERSION
    categories = ["Climate and Weather"]

    def __init__(self, config: WeatherModelConfig, store: Store | None = None) -> None:
        self.config = config
        self.store = store
        self.nws = NWSClient()

    def close(self) -> None:
        self.nws.close()

    def supports(self, market: dict[str, Any], category: str) -> bool:
        if not self.config.enabled:
            return False
        ticker = (market.get("ticker") or "").upper()
        if category and category != "Climate and Weather":
            # Still allow if series prefix matches known cities.
            pass
        for city_cfg in self.config.cities.values():
            for prefix in city_cfg.series_prefixes:
                if ticker.startswith(prefix.upper()):
                    return True
        return False

    def _city_for(self, ticker: str) -> tuple[str, Any] | None:
        t = ticker.upper()
        for name, cfg in self.config.cities.items():
            for prefix in cfg.series_prefixes:
                if t.startswith(prefix.upper()):
                    return name, cfg
        return None

    def predict(self, market: dict[str, Any], category: str) -> Prediction:
        ticker = market.get("ticker") or ""
        title = market.get("title") or market.get("subtitle") or ""
        now = datetime.now(timezone.utc)

        city = self._city_for(ticker)
        if not city:
            return self._skip(ticker, now, "no configured city for series")

        city_name, city_cfg = city
        target_day = parse_market_date(ticker)
        if target_day is None:
            return self._skip(ticker, now, "could not parse market date from ticker")

        contract = parse_temp_contract(ticker, title)
        if contract is None:
            return self._skip(
                ticker,
                now,
                "could not parse temperature threshold/bucket; refusing to invent definition",
            )

        cache_key = f"nws:{city_name}:{target_day.isoformat()}"
        forecast: dict[str, Any] | None = None
        if self.store:
            cached = self.store.get_forecast_cache(cache_key)
            if cached:
                from datetime import datetime as dt

                fetched = dt.fromisoformat(cached["fetched_at"])
                age_h = (now - fetched).total_seconds() / 3600.0
                if age_h <= self.config.max_forecast_age_hours:
                    import json

                    forecast = json.loads(cached["payload_json"])

        if forecast is None:
            try:
                forecast = self.nws.daily_high_forecast(city_cfg.lat, city_cfg.lon, target_day)
            except Exception as exc:
                logger.warning("NWS fetch failed for %s: %s", city_name, exc)
                return self._skip(ticker, now, f"NWS forecast unavailable: {exc}")
            if forecast and self.store:
                self.store.cache_forecast(cache_key, "api.weather.gov", forecast)

        if not forecast or forecast.get("temp_f") is None:
            return self._skip(
                ticker,
                now,
                f"no NWS daytime high for {city_name} on {target_day} (refusing synthetic data)",
            )

        mu = float(forecast["temp_f"])
        sigma = max(float(self.config.forecast_error_sigma_f), float(self.config.min_sigma_f))
        p = self._prob_yes(mu, sigma, contract)
        # Wider-error stress test: recompute with inflated sigma (unvalidated model doubt).
        p_wide = self._prob_yes(mu, sigma * 1.5, contract)
        p_yes_low = min(p, p_wide)   # conservative when buying YES
        p_yes_high = max(p, p_wide)  # implies conservative NO = 1 - high
        # Uncertainty floor acknowledges unvalidated σ and settlement-station mismatch risk.
        uncertainty = max(abs(p - p_wide), 0.08)

        factors = [
            f"NWS forecast high {mu:.1f}°F for {city_name} on {target_day}",
            f"Assumed forecast-error σ={sigma:.1f}°F (configurable prior; not walk-forward calibrated)",
            f"Stress σ={sigma * 1.5:.1f}°F → p_yes {p_wide:.4f} (low={p_yes_low:.4f}, high={p_yes_high:.4f})",
            f"Contract definition: {contract}",
            city_cfg.station_note or "Verify settlement station matches NWS grid before live trading",
        ]

        return Prediction(
            market_ticker=ticker,
            p_yes=clamp01(D(f"{p:.6f}")),
            p_yes_conservative=clamp01(D(f"{p_yes_low:.6f}")),
            uncertainty=clamp01(D(f"{uncertainty:.6f}")),
            model_version=self.version,
            data_sources=[
                {
                    "name": forecast.get("source"),
                    "fetched_at": forecast.get("fetched_at"),
                    "available_at": forecast.get("start_time"),
                    "url": forecast.get("forecast_url"),
                    "station": forecast.get("station_note"),
                }
            ],
            factors=factors,
            validation_evidence=(
                "UNVALIDATED: no held-out chronological score vs Kalshi settlements yet. "
                "Compare to market-implied prices; disagreement ≠ edge. Use paper mode."
            ),
            as_of=now,
            supported=True,
            details={
                "mu_f": mu,
                "sigma_f": sigma,
                "contract": contract,
                "city": city_name,
                "target_day": target_day.isoformat(),
                "p_wide": p_wide,
                "p_yes_high": p_yes_high,
            },
        )

    def _prob_yes(self, mu: float, sigma: float, contract: dict[str, Any]) -> float:
        op = contract["op"]
        if op == "range":
            low, high = float(contract["low"]), float(contract["high"])
            return float(norm.cdf(high, loc=mu, scale=sigma) - norm.cdf(low, loc=mu, scale=sigma))
        if op == "gt":
            thr = float(contract["low"])
            # P(temp > thr) — for integer reported highs, approx P(T >= thr+epsilon)
            return float(1.0 - norm.cdf(thr, loc=mu, scale=sigma))
        if op == "lt":
            thr = float(contract["high"])
            return float(norm.cdf(thr, loc=mu, scale=sigma))
        raise ValueError(f"unknown op {op}")

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
        )
