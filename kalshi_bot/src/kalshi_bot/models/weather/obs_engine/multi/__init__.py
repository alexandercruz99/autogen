"""Multi-location package: discovery, registry, unified predict, paper ledger."""

from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.discovery import discover_weather_markets
from kalshi_bot.models.weather.obs_engine.multi.pipeline import process_location, run_multi_cycle
from kalshi_bot.models.weather.obs_engine.multi.predict import predict_station_v2
from kalshi_bot.models.weather.obs_engine.multi.registry import LocationRegistry, VERIFIED_NWS_CLI_DAILY_MAX

__all__ = [
    "ForecastContext",
    "LocationRegistry",
    "VERIFIED_NWS_CLI_DAILY_MAX",
    "discover_weather_markets",
    "predict_station_v2",
    "process_location",
    "run_multi_cycle",
]
