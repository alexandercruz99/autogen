"""Station registry for Kalshi daily-max temperature markets.

Settlement source (Kalshi Help Center, Jul 2026 + NHIGH/CHIHIGH contract terms):
  Daily high/low markets → final NWS Daily Climate Report (CLI product), °F.
  Hourly markets → The Weather Company (NOT handled by this module).

NWS CLI reports use local *standard* time for the climate day (DST note in Kalshi docs).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StationSpec:
    """One settlement station for daily-max markets."""

    city_key: str
    display_name: str
    series_prefixes: tuple[str, ...]
    # NWS CLI product location id used in /products/types/CLI/locations/{id}
    cli_location_id: str
    # Issuing WFO office id (e.g. KOKX)
    wfo_office: str
    # Human station name as it appears in CLI text / contract PDFs
    climate_station_name: str
    # IANA timezone for local civil time (CLI climate day is LST — see notes)
    timezone: str
    uses_local_standard_time: bool = True
    temperature_unit: str = "F"
    rounding_note: str = "Whole degrees Fahrenheit as printed in CLI MAXIMUM row"
    settlement_source: str = "NWS Daily Climate Report (CLI)"
    contract_pdf_note: str = ""
    # Approximate lat/lon for NWS grid / Open-Meteo (forecast proxy; settlement is CLI)
    lat: float = 0.0
    lon: float = 0.0
    # Optional METAR for progressive same-day evidence (not settlement)
    metar_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


# Verified against Kalshi weather help + NHIGH / CHIHIGH terms where available.
STATIONS: dict[str, StationSpec] = {
    "NYC": StationSpec(
        city_key="NYC",
        display_name="New York City (Central Park)",
        series_prefixes=("KXHIGHNY", "HIGHNY", "KXLOWNY", "LOWNY"),
        cli_location_id="NYC",
        wfo_office="KOKX",
        climate_station_name="CENTRAL PARK NY",
        timezone="America/New_York",
        lat=40.7789,
        lon=-73.9692,
        metar_ids=("KNYC",),
        contract_pdf_note="NHIGH: max temp in NWS Daily Climate Report for Central Park",
        notes="Climate day reported in LST; during DST the climate day spans 1AM–12:59AM next day local.",
    ),
    "LAX": StationSpec(
        city_key="LAX",
        display_name="Los Angeles International (LAX)",
        series_prefixes=("KXHIGHLAX", "HIGHLAX", "KXLOWLAX", "LOWLAX"),
        cli_location_id="LAX",
        wfo_office="KLOX",
        climate_station_name="LOS ANGELES INTL",
        timezone="America/Los_Angeles",
        lat=33.9425,
        lon=-118.4081,
        metar_ids=("KLAX",),
        notes="KXHIGHLAX settles TWC CLILAX at KLAX airport; NWS grid coords match settlement ASOS.",
    ),
    "CHI": StationSpec(
        city_key="CHI",
        display_name="Chicago Midway",
        series_prefixes=("KXHIGHTCHI", "HIGHCHI", "KXHIGHCHI", "KXLOWCHI", "LOWCHI"),
        cli_location_id="MDW",
        wfo_office="KLOT",
        climate_station_name="CHICAGO MIDWAY",
        timezone="America/Chicago",
        lat=41.7868,
        lon=-87.7522,
        metar_ids=("KMDW",),
        contract_pdf_note="CHIHIGH: max temp in NWS Daily Climate Report for Chicago Midway",
        notes="LOT WFO CLI location MDW for Midway (not O'Hare).",
    ),
}


def station_for_ticker(ticker: str) -> StationSpec | None:
    t = (ticker or "").upper()
    for spec in STATIONS.values():
        for prefix in spec.series_prefixes:
            if t.startswith(prefix.upper()):
                return spec
    return None


def all_stations() -> list[StationSpec]:
    return list(STATIONS.values())
