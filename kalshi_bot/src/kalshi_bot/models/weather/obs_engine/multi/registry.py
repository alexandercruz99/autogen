"""Persistent registry of Kalshi weather markets → settlement targets."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS location_targets (
  location_id TEXT NOT NULL,
  series_ticker TEXT NOT NULL,
  measurement TEXT NOT NULL,
  display_name TEXT,
  settlement_source_family TEXT NOT NULL,
  settlement_name TEXT,
  settlement_url TEXT,
  contract_url TEXT,
  cli_location_id TEXT,
  wfo_office TEXT,
  climate_station_name TEXT,
  metar_ids_json TEXT,
  neighbor_metar_ids_json TEXT,
  lat REAL,
  lon REAL,
  elev_m REAL,
  timezone TEXT,
  uses_lst_climate_day INTEGER DEFAULT 1,
  unit TEXT DEFAULT 'F',
  rounding_note TEXT,
  mapping_status TEXT NOT NULL,
  data_availability TEXT,
  validation_status TEXT,
  model_family TEXT,
  notes TEXT,
  rule_version TEXT,
  updated_at_utc TEXT NOT NULL,
  details_json TEXT,
  PRIMARY KEY (series_ticker, measurement)
);
CREATE TABLE IF NOT EXISTS discovery_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ran_at_utc TEXT NOT NULL,
  n_series INTEGER,
  n_mapped INTEGER,
  n_ambiguous INTEGER,
  n_unsupported INTEGER,
  summary_json TEXT
);
"""


# Evidence-backed NWS CLI daily-max mappings (contract terms / settlement_sources URL).
# Nearby airports are features only — never substitute for settlement without evidence.
VERIFIED_NWS_CLI_DAILY_MAX: dict[str, dict[str, Any]] = {
    "HIGHNY": {
        "location_id": "nyc_central_park",
        "display_name": "New York City Central Park",
        "cli_location_id": "NYC",
        "wfo_office": "OKX",
        "climate_station_name": "CENTRAL PARK NY",
        "metar_ids": ["KNYC"],
        "neighbor_metar_ids": ["KLGA", "KJFK"],
        "lat": 40.77898,
        "lon": -73.96925,
        "elev_m": 42.7,
        "timezone": "America/New_York",
        "mapping_status": "verified",
        "validation_status": "operating_candidate",
        "model_family": "station_v2_nws_cli",
        "notes": "Settlement: NWS CLI MAXIMUM Central Park; METAR KNYC same station progressive evidence",
    },
    "HIGHCHI": {
        "location_id": "chi_midway",
        "display_name": "Chicago Midway",
        "cli_location_id": "MDW",
        "wfo_office": "LOT",
        "climate_station_name": "CHICAGO MIDWAY",
        "metar_ids": ["KMDW"],
        "neighbor_metar_ids": ["KORD"],
        "lat": 41.7868,
        "lon": -87.7522,
        "elev_m": 189.0,
        "timezone": "America/Chicago",
        "mapping_status": "verified",
        "validation_status": "operating_candidate",
        "model_family": "station_v2_nws_cli",
        "notes": "CHIHIGH terms: Midway CLI — not O'Hare; GHCND USW00014819 labels for training",
    },
    "HIGHMIA": {
        "location_id": "mia_cli",
        "display_name": "Miami (CLI MIA)",
        "cli_location_id": "MIA",
        "wfo_office": "MFL",
        "climate_station_name": "MIAMI",
        "metar_ids": ["KMIA"],
        "neighbor_metar_ids": [],
        "lat": 25.7959,
        "lon": -80.2870,
        "elev_m": 3.0,
        "timezone": "America/New_York",
        "mapping_status": "verified_cli_url",
        "validation_status": "needs_historical_backfill",
        "model_family": "station_v2_nws_cli",
        "notes": "CLI issuedby=MIA from series settlement_sources; confirm station name vs airport",
    },
    "HIGHAUS": {
        "location_id": "aus_cli",
        "display_name": "Austin (CLI AUS)",
        "cli_location_id": "AUS",
        "wfo_office": "EWX",
        "climate_station_name": "AUSTIN",
        "metar_ids": ["KAUS"],
        "neighbor_metar_ids": [],
        "lat": 30.1945,
        "lon": -97.6699,
        "elev_m": 165.0,
        "timezone": "America/Chicago",
        "mapping_status": "verified_cli_url",
        "validation_status": "needs_historical_backfill",
        "model_family": "station_v2_nws_cli",
        "notes": "AUSHIGH NWS CLI issuedby=AUS",
    },
    "KXDENHIGH": {
        "location_id": "den_cli",
        "display_name": "Denver (CLI DEN)",
        "cli_location_id": "DEN",
        "wfo_office": "BOU",
        "climate_station_name": "DENVER",
        "metar_ids": ["KDEN"],
        "neighbor_metar_ids": [],
        "lat": 39.8561,
        "lon": -104.6737,
        "elev_m": 1655.0,
        "timezone": "America/Denver",
        "mapping_status": "verified_cli_url",
        "validation_status": "needs_historical_backfill",
        "model_family": "station_v2_nws_cli",
        "notes": "CLI issuedby=DEN; confirm exact climate station vs airport",
    },
    "KXHIGHOU": {
        "location_id": "hou_cli",
        "display_name": "Houston (CLI HOU)",
        "cli_location_id": "HOU",
        "wfo_office": "HGX",
        "climate_station_name": "HOUSTON",
        "metar_ids": ["KHOU"],
        "neighbor_metar_ids": ["KIAH"],
        "lat": 29.6454,
        "lon": -95.2789,
        "elev_m": 13.0,
        "timezone": "America/Chicago",
        "mapping_status": "verified_cli_url",
        "validation_status": "needs_historical_backfill",
        "model_family": "station_v2_nws_cli",
        "notes": "HOUHIGH NWS CLI; aliases KXHIGHHOU/KXHOUHIGH may duplicate",
    },
}


# Evidence-backed Weather Company daily-max mappings.
# Station IDs come from Kalshi rules_primary (CLINYC/…) + weather.com/kalshi portal
# ``cliId`` / ``icao`` fields. Do NOT mix these series with the NWS CLI pipeline.
VERIFIED_TWC_DAILY_MAX: dict[str, dict[str, Any]] = {
    "KXHIGHNY": {
        "location_id": "twc_nyc_central_park",
        "display_name": "NYC Central Park (TWC CLINYC)",
        "cli_location_id": "NYC",
        "climate_station_name": "CLINYC",
        "metar_ids": ["KNYC"],
        "neighbor_metar_ids": ["KLGA", "KJFK"],
        "lat": 40.77898,
        "lon": -73.96925,
        "elev_m": 42.7,
        "timezone": "America/New_York",
        "mapping_status": "verified",
        "validation_status": "operating_candidate",
        "model_family": "twc_daily_max_v1",
        "same_station_model_location_id": "nyc_central_park",
        "notes": (
            "Settles on The Weather Company CLINYC via weather.com/kalshi; "
            "progressive evidence from TWC portal METAR + AviationWeather KNYC features; "
            "residual model may reuse same-ICAO station_v2 artifact (exploratory transfer)"
        ),
    },
    "KXHIGHCHI": {
        "location_id": "twc_chi_midway",
        "display_name": "Chicago Midway (TWC CLIMDW)",
        "cli_location_id": "MDW",
        "climate_station_name": "CLIMDW",
        "metar_ids": ["KMDW"],
        "neighbor_metar_ids": ["KORD"],
        "lat": 41.7868,
        "lon": -87.7522,
        "elev_m": 189.0,
        "timezone": "America/Chicago",
        "mapping_status": "verified",
        "validation_status": "operating_candidate",
        "model_family": "twc_daily_max_v1",
        "same_station_model_location_id": "chi_midway",
        "notes": (
            "Settles on The Weather Company CLIMDW (Midway, not O'Hare); "
            "same-ICAO residual transfer from chi_midway when trained"
        ),
    },
    "KXHIGHLAX": {
        "location_id": "twc_lax",
        "display_name": "Los Angeles LAX (TWC CLILAX)",
        "cli_location_id": "LAX",
        "climate_station_name": "CLILAX",
        "metar_ids": ["KLAX"],
        "neighbor_metar_ids": [],
        "lat": 33.9425,
        "lon": -118.4081,
        "elev_m": 38.0,
        "timezone": "America/Los_Angeles",
        "mapping_status": "verified",
        "validation_status": "needs_historical_backfill",
        "model_family": "twc_daily_max_v1",
        "notes": "TWC portal cliId=LAX / ICAO KLAX; no location model yet",
    },
    "KXHIGHAUS": {
        "location_id": "twc_aus",
        "display_name": "Austin Bergstrom (TWC CLIAUS)",
        "cli_location_id": "AUS",
        "climate_station_name": "CLIAUS",
        "metar_ids": ["KAUS"],
        "neighbor_metar_ids": [],
        "lat": 30.1945,
        "lon": -97.6699,
        "elev_m": 165.0,
        "timezone": "America/Chicago",
        "mapping_status": "verified",
        "validation_status": "needs_historical_backfill",
        "model_family": "twc_daily_max_v1",
        "notes": "TWC CLIAUS = Bergstrom (not Camp Mabry)",
    },
    "KXHIGHDEN": {
        "location_id": "twc_den",
        "display_name": "Denver Intl (TWC CLIDEN)",
        "cli_location_id": "DEN",
        "climate_station_name": "CLIDEN",
        "metar_ids": ["KDEN"],
        "neighbor_metar_ids": [],
        "lat": 39.8561,
        "lon": -104.6737,
        "elev_m": 1655.0,
        "timezone": "America/Denver",
        "mapping_status": "verified",
        "validation_status": "needs_historical_backfill",
        "model_family": "twc_daily_max_v1",
        "notes": "TWC portal cliId=DEN / ICAO KDEN",
    },
    "KXHIGHPHIL": {
        "location_id": "twc_phl",
        "display_name": "Philadelphia Intl (TWC CLIPHL)",
        "cli_location_id": "PHL",
        "climate_station_name": "CLIPHL",
        "metar_ids": ["KPHL"],
        "neighbor_metar_ids": [],
        "lat": 39.8744,
        "lon": -75.2424,
        "elev_m": 11.0,
        "timezone": "America/New_York",
        "mapping_status": "verified",
        "validation_status": "needs_historical_backfill",
        "model_family": "twc_daily_max_v1",
        "notes": "TWC portal cliId=PHL / ICAO KPHL (CLIPHL)",
    },
    "KXHIGHMIA": {
        "location_id": "twc_mia",
        "display_name": "Miami (TWC CLIMIA)",
        "cli_location_id": "MIA",
        "climate_station_name": "CLIMIA",
        "metar_ids": ["KMIA"],
        "neighbor_metar_ids": [],
        "lat": 25.7959,
        "lon": -80.2870,
        "elev_m": 3.0,
        "timezone": "America/New_York",
        "mapping_status": "verified",
        "validation_status": "needs_historical_backfill",
        "model_family": "twc_daily_max_v1",
        "notes": "TWC portal cliId=MIA / ICAO KMIA",
    },
}


class LocationRegistry:
    def __init__(self, path: Path | None = None) -> None:
        root = Path("data/obs_engine/multi")
        root.mkdir(parents=True, exist_ok=True)
        self.path = path or (root / "registry.db")
        self._conn = sqlite3.connect(str(self.path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert_target(self, row: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cols = [
            "location_id",
            "series_ticker",
            "measurement",
            "display_name",
            "settlement_source_family",
            "settlement_name",
            "settlement_url",
            "contract_url",
            "cli_location_id",
            "wfo_office",
            "climate_station_name",
            "metar_ids_json",
            "neighbor_metar_ids_json",
            "lat",
            "lon",
            "elev_m",
            "timezone",
            "uses_lst_climate_day",
            "unit",
            "rounding_note",
            "mapping_status",
            "data_availability",
            "validation_status",
            "model_family",
            "notes",
            "rule_version",
            "updated_at_utc",
            "details_json",
        ]
        vals = []
        for c in cols:
            if c == "updated_at_utc":
                vals.append(now)
            elif c.endswith("_json") and c.replace("_json", "") in row and not isinstance(row.get(c), str):
                key = c.replace("_json", "")
                vals.append(json.dumps(row.get(key) or row.get(c) or []))
            elif c == "metar_ids_json":
                vals.append(json.dumps(row.get("metar_ids") or []))
            elif c == "neighbor_metar_ids_json":
                vals.append(json.dumps(row.get("neighbor_metar_ids") or []))
            elif c == "details_json":
                vals.append(json.dumps(row.get("details") or {}))
            elif c == "uses_lst_climate_day":
                vals.append(1 if row.get("uses_lst_climate_day", True) else 0)
            else:
                vals.append(row.get(c))
        placeholders = ",".join("?" * len(cols))
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("series_ticker", "measurement"))
        self._conn.execute(
            f"""INSERT INTO location_targets({",".join(cols)}) VALUES ({placeholders})
                ON CONFLICT(series_ticker, measurement) DO UPDATE SET {updates}""",
            vals,
        )
        self._conn.commit()

    def list_targets(self, *, measurement: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM location_targets"
        args: list[Any] = []
        if measurement:
            q += " WHERE measurement=?"
            args.append(measurement)
        q += " ORDER BY series_ticker"
        out = []
        for r in self._conn.execute(q, args):
            d = dict(r)
            d["metar_ids"] = json.loads(d.pop("metar_ids_json") or "[]")
            d["neighbor_metar_ids"] = json.loads(d.pop("neighbor_metar_ids_json") or "[]")
            d["details"] = json.loads(d.pop("details_json") or "{}")
            out.append(d)
        return out

    def record_discovery_run(self, summary: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO discovery_runs(ran_at_utc, n_series, n_mapped, n_ambiguous, n_unsupported, summary_json)
               VALUES (?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                summary.get("n_series"),
                summary.get("n_mapped"),
                summary.get("n_ambiguous"),
                summary.get("n_unsupported"),
                json.dumps(summary),
            ),
        )
        self._conn.commit()

    def operating_daily_max(self) -> list[dict[str, Any]]:
        """NWS CLI + verified TWC daily-max targets ready for the multi pipeline."""
        out = []
        for t in self.list_targets(measurement="daily_max_temp_f"):
            family = t.get("settlement_source_family")
            mapping = t.get("mapping_status")
            if mapping not in ("verified", "verified_cli_url"):
                continue
            if family == "nws_cli":
                out.append(t)
            elif family == "weather_company" and t.get("model_family") == "twc_daily_max_v1":
                out.append(t)
        return out
