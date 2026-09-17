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
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.money import D, clamp01

logger = logging.getLogger(__name__)

MODEL_VERSION = "weather.high_temp.v0.3-overconfidence-guards"


class WeatherHighTempModel(ProbabilityModel):
    """Daily high-temperature binaries.

    Settlement (Kalshi rules): max temp at city CLINYC/... per **The Weather Company**,
    not NWS. This model uses NWS daytime forecast highs as a **proxy** forecast with a
    Gaussian error prior. LIVE TRADING BLOCKED until settlement-aligned source + walk-forward.

    Decision-time discipline: only NWS periods whose start_time <= decision time are used;
    forecast payload timestamps are stored.
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

        contract_interval = interval_from_market(market)
        contract: dict[str, Any] | None
        if contract_interval is not None:
            contract = {
                "op": contract_interval.op,
                "low": contract_interval.low,
                "high": contract_interval.high,
                "raw": contract_interval.source,
                "rules_primary": contract_interval.rules_primary,
            }
        else:
            parsed = parse_temp_contract(ticker, title)
            if parsed is None:
                return self._skip(
                    ticker,
                    now,
                    "could not parse temperature threshold from market strikes/title; refusing to invent",
                )
            contract = parsed

        # Settlement source audit from rules text when present.
        rules = (market.get("rules_primary") or "") + " " + (market.get("rules_secondary") or "")
        if "Weather Company" in rules or "CLINYC" in rules or "CLILA" in rules:
            settlement_note = (
                "Kalshi settles to The Weather Company station (e.g. CLINYC). "
                "NWS grid forecast is a proxy only — not live_eligible."
            )
        else:
            settlement_note = (
                city_cfg.station_note
                or "Verify settlement station matches forecast source before live trading"
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
                    # Leakage guard: period start must not be after decision time
                    start = forecast.get("start_time")
                    if start:
                        start_dt = dt.fromisoformat(start.replace("Z", "+00:00"))
                        if start_dt > now:
                            forecast = None

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

        # Leakage: do not use a forecast period that starts after decision time
        start = forecast.get("start_time")
        if start:
            from datetime import datetime as dt

            start_dt = dt.fromisoformat(start.replace("Z", "+00:00"))
            if start_dt > now:
                return self._skip(ticker, now, "forecast period starts after decision time (leakage guard)")

        mu = float(forecast["temp_f"])
        # Proxy vs Weather Company settlement: inflate σ so we do not treat NWS as truth.
        sigma = max(float(self.config.forecast_error_sigma_f), float(self.config.min_sigma_f))
        sigma = sigma + float(self.config.proxy_sigma_extra_f)
        p = self._prob_yes(mu, sigma, contract)
        p_wide = self._prob_yes(mu, sigma * 1.5, contract)
        p_yes_low = min(p, p_wide)
        p_yes_high = max(p, p_wide)
        # Floor uncertainty higher while settlement-misaligned (was 0.08).
        uncertainty = max(abs(p - p_wide), 0.12)

        factors = [
            f"NWS forecast high {mu:.1f}°F for {city_name} on {target_day}",
            f"Assumed forecast-error σ={sigma:.1f}°F (includes +proxy mismatch; not walk-forward calibrated)",
            f"Stress σ={sigma * 1.5:.1f}°F → p_yes {p_wide:.4f} (low={p_yes_low:.4f}, high={p_yes_high:.4f})",
            f"Contract definition: {contract} (integer °F continuity correction on gt/lt)",
            settlement_note,
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
                    "settlement_source_kalshi": "The Weather Company (see rules_primary)",
                }
            ],
            factors=factors,
            validation_evidence=(
                "UNVALIDATED: NWS proxy vs Weather Company CLINYC settlement; "
                "no held-out chronological score vs Kalshi settlements yet. "
                "Overconfidence guards: continuity correction, inflated σ, market shrink in EV."
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
                "settlement_aligned": False,
                "decision_time": now.isoformat(),
                "model_live_eligible": False,
                "engine": "legacy_high_temp",
                "validation_status": "UNVALIDATED",
            },
        )

    def _prob_yes(self, mu: float, sigma: float, contract: dict[str, Any]) -> float:
        """Gaussian forecast-error model with integer-degree continuity correction.

        Kalshi weather settles on whole °F. Strict 'greater than 82' means ≥83 observed,
        so P(X>82) ≈ 1−Φ(82.5), not 1−Φ(82) (which wrongly gives 50% when μ=82).
        """
        op = contract["op"]
        if op == "range":
            low, high = float(contract["low"]), float(contract["high"])
            return float(norm.cdf(high, loc=mu, scale=sigma) - norm.cdf(low, loc=mu, scale=sigma))
        if op == "range_inclusive":
            low, high = float(contract["low"]), float(contract["high"])
            # Inclusive integer degrees: approximate as [low-0.5, high+0.5]
            return float(
                norm.cdf(high + 0.5, loc=mu, scale=sigma) - norm.cdf(low - 0.5, loc=mu, scale=sigma)
            )
        if op == "gt":
            thr = float(contract["low"])
            # Strict greater-than on integer °F → mass above thr+0.5
            return float(1.0 - norm.cdf(thr + 0.5, loc=mu, scale=sigma))
        if op == "lt":
            thr = float(contract["high"])
            # Strict less-than on integer °F → mass below thr-0.5
            return float(norm.cdf(thr - 0.5, loc=mu, scale=sigma))
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
            details={"model_live_eligible": False, "engine": "legacy_high_temp"},
        )
