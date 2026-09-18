"""Settlement targets and label-source compatibility (research registry).

GHCND, NWS CLI, ASOS max-so-far, and TWC climate maxTemp are not interchangeable.
This module records what each supported location settles on and what we train on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SettlementTarget:
    location_id: str
    display_name: str
    series_tickers: tuple[str, ...]
    metric: str  # daily_max_temp_f
    units: str
    precision: str  # integer_fahrenheit
    timezone: str
    climate_day_rule: str
    observation_window: str
    settlement_source_family: str
    settlement_station: str
    settlement_rules_summary: str
    train_label_source: str
    train_label_id: str
    train_label_matches_settlement: bool
    label_transfer_note: str
    metar_ids: tuple[str, ...]
    lat: float
    lon: float
    ghcnd_id: str | None = None
    twc_cli_id: str | None = None
    nws_cli_id: str | None = None
    hours_to_20_is_feature_only: bool = True
    hard_floor_policy: str = (
        "METAR/ASOS max_so_far is evidence, not a verified settlement lower bound. "
        "CLI/TWC whole-°F preliminary floors may truncate the predictive distribution "
        "only when source+station+window match settlement; otherwise report as evidence."
    )
    live_eligible: bool = False
    data_status: str = "supported"  # supported | insufficient_data | unsupported
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Explicit registry — only locations with enough ASOS+label history for A–D.
SETTLEMENT_TARGETS: dict[str, SettlementTarget] = {
    "nyc_central_park": SettlementTarget(
        location_id="nyc_central_park",
        display_name="NYC Central Park",
        series_tickers=("HIGHNY", "KXHIGHNY"),
        metric="daily_max_temp_f",
        units="fahrenheit",
        precision="integer_fahrenheit",
        timezone="America/New_York",
        climate_day_rule="LST civil calendar day in America/New_York (DST-aware ZoneInfo)",
        observation_window=(
            "Settlement uses the official daily maximum for the climate day as published by "
            "the market's settlement source (NWS CLI Central Park for HIGHNY; Weather Company "
            "climate maxTemp for KXHIGHNY). The calendar day does NOT end at 20:00 local; "
            "hours_to_20_local is a named feature only."
        ),
        settlement_source_family="nws_cli_or_twc_by_series",
        settlement_station="KNYC / Central Park",
        settlement_rules_summary=(
            "Inclusive integer °F bands for between contracts; gt/lt use strike floors/caps "
            "per market.strike_type. P(a≤Y≤b)=F(b)-F(a-1) on integer support."
        ),
        train_label_source="ghcnd_tmax",
        train_label_id="USW00094728",
        train_label_matches_settlement=False,
        label_transfer_note=(
            "GHCND TMAX at USW00094728 is a same-station proxy for NWS CLI MAXIMUM; "
            "TWC KXHIGHNY uses Weather Company climate maxTemp — transfer must be evaluated "
            "on paired TWC days. Median bias alone does not make transfer live-eligible."
        ),
        metar_ids=("KNYC",),
        lat=40.7789,
        lon=-73.9692,
        ghcnd_id="USW00094728",
        twc_cli_id="NYC",
        nws_cli_id="NYC",
        data_status="supported",
    ),
    "chi_midway": SettlementTarget(
        location_id="chi_midway",
        display_name="Chicago Midway",
        series_tickers=("HIGHCHI", "KXHIGHCHI"),
        metric="daily_max_temp_f",
        units="fahrenheit",
        precision="integer_fahrenheit",
        timezone="America/Chicago",
        climate_day_rule="LST civil calendar day in America/Chicago (DST-aware ZoneInfo)",
        observation_window=(
            "Official daily maximum for the climate day from NWS CLI MDW or TWC climate "
            "maxTemp (series-dependent). Not truncated at 20:00 local."
        ),
        settlement_source_family="nws_cli_or_twc_by_series",
        settlement_station="KMDW",
        settlement_rules_summary="Integer °F inclusive between-bands; gt/lt per strike_type.",
        train_label_source="ghcnd_tmax",
        train_label_id="USW00014819",
        train_label_matches_settlement=False,
        label_transfer_note=(
            "GHCND USW00014819 is a documented proxy when historical CLI text is incomplete; "
            "settlement remains CLI/TWC. Paired TWC evaluation required for KXHIGHCHI."
        ),
        metar_ids=("KMDW",),
        lat=41.7868,
        lon=-87.7522,
        ghcnd_id="USW00014819",
        twc_cli_id="MDW",
        nws_cli_id="MDW",
        data_status="supported",
    ),
    "lax_airport": SettlementTarget(
        location_id="lax_airport",
        display_name="Los Angeles International",
        series_tickers=("HIGHLAX", "KXHIGHLAX"),
        metric="daily_max_temp_f",
        units="fahrenheit",
        precision="integer_fahrenheit",
        timezone="America/Los_Angeles",
        climate_day_rule="LST civil calendar day in America/Los_Angeles (DST-aware ZoneInfo)",
        observation_window=(
            "Official daily maximum for the climate day (NWS CLI LAX / TWC CLILAX). "
            "Not truncated at 20:00 local."
        ),
        settlement_source_family="nws_cli_or_twc_by_series",
        settlement_station="KLAX",
        settlement_rules_summary="Integer °F inclusive between-bands; gt/lt per strike_type.",
        train_label_source="ghcnd_tmax",
        train_label_id="USW00023174",
        train_label_matches_settlement=False,
        label_transfer_note=(
            "GHCND USW00023174 same ICAO as TWC CLILAX / KXHIGHLAX; still a source transfer "
            "until paired TWC residuals validate calibration."
        ),
        metar_ids=("KLAX",),
        lat=33.9425,
        lon=-118.4081,
        ghcnd_id="USW00023174",
        twc_cli_id="LAX",
        nws_cli_id="LAX",
        data_status="supported",
    ),
}

# Known registry locations without enough local ASOS+label train data here.
UNSUPPORTED_OR_INSUFFICIENT: dict[str, dict[str, str]] = {
    "mia_cli": {"status": "insufficient_data", "reason": "No LOCATION_TRAIN_PROFILES ASOS+GHCND bundle in this checkout"},
    "aus_cli": {"status": "insufficient_data", "reason": "No LOCATION_TRAIN_PROFILES ASOS+GHCND bundle in this checkout"},
    "den_cli": {"status": "insufficient_data", "reason": "No LOCATION_TRAIN_PROFILES ASOS+GHCND bundle in this checkout"},
    "hou_cli": {"status": "insufficient_data", "reason": "No LOCATION_TRAIN_PROFILES ASOS+GHCND bundle in this checkout"},
}


def targets_manifest() -> dict[str, Any]:
    return {
        "metric": "daily_max_temp_f",
        "supported": {k: v.as_dict() for k, v in SETTLEMENT_TARGETS.items()},
        "unsupported_or_insufficient": UNSUPPORTED_OR_INSUFFICIENT,
        "notes": [
            "NYC calibration must never be silently reused for other locations.",
            "hours_to_20_local is a feature name only; settlement window is the full climate day.",
            "live_eligible is false for all candidate artifacts in this evaluation.",
        ],
    }
