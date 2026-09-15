"""Observation-driven NYC Central Park daily-max engine (research / paper only).

Target station: NY CITY CENTRAL PARK (GHCND USW00094728 / CLI location NYC / METAR KNYC).
Settlement source for Kalshi daily highs: NWS Daily Climate Report MAXIMUM (°F, LST climate day).

Training labels: GHCND daily TMAX for USW00094728 (same station; documented proxy for CLI
when historical CLI text is unavailable). Live inference uses CLI prelim when present.

This engine does NOT ingest third-party forecasts as features. NWS forecasts are benchmarks only.
model_live_eligible is always False until promotion criteria are met and config flag is set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NycCentralParkTarget:
    city_key: str = "NYC"
    display_name: str = "New York City Central Park"
    ghcnd_id: str = "USW00094728"
    cli_location_id: str = "NYC"
    metar_id: str = "KNYC"
    iem_asos_id: str = "NYC"
    lat: float = 40.77898
    lon: float = -73.96925
    elev_m: float = 42.7
    timezone: str = "America/New_York"
    uses_lst_climate_day: bool = True
    unit: str = "F"
    rounding: str = "whole °F as printed in CLI MAXIMUM / GHCND TMAX rounded to °F"
    settlement_source: str = "NWS Daily Climate Report (CLI) for Central Park"
    series_prefixes: tuple[str, ...] = ("KXHIGHNY", "HIGHNY")
    # Fixed research decision hours (local civil time); LST climate-day caveat documented.
    decision_hours_local: tuple[int, ...] = (10, 13, 16)
    neighbor_asos: tuple[str, ...] = ("LGA",)  # optional spatial features when present


NYC_TARGET = NycCentralParkTarget()

MODEL_VERSION = "weather.obs_nyc.v1.0-research"
ARTIFACT_NAME = "obs_nyc_remaining_rise_v1"
