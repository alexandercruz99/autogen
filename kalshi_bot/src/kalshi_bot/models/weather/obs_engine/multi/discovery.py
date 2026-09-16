"""Discover Kalshi weather markets and refresh the location registry."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from kalshi_bot.models.weather.obs_engine.multi.registry import VERIFIED_NWS_CLI_DAILY_MAX, LocationRegistry

logger = logging.getLogger(__name__)


def _classify_measurement(ticker: str, title: str, frequency: str) -> str | None:
    t = (ticker or "").upper()
    title_u = (title or "").upper()
    freq = (frequency or "").lower()
    if "HOUR" in t or "DIRECTION" in title_u or "TEMPAT" in t:
        return "hourly_temp" if freq == "hourly" else "other"
    if freq != "daily":
        if "SNOW" in t or "SNOW" in title_u:
            return "snowfall"
        if "RAIN" in t or "PRECIP" in t:
            return "rainfall"
        return "other_weather"
    # daily
    if ("HIGH" in t or "HIGHEST" in title_u or "MAXIMUM" in title_u or "MAX TEMP" in title_u) and not (
        t.startswith("KXLOW") or t.startswith("LOW")
    ):
        if "LOW" in t and "HIGH" not in t:
            return "daily_min_temp_f"
        return "daily_max_temp_f"
    if "LOW" in t or "LOWEST" in title_u or "MINIMUM" in title_u:
        return "daily_min_temp_f"
    if "SNOW" in t:
        return "snowfall"
    if "RAIN" in t or "PRECIP" in t:
        return "rainfall"
    return "other_daily"


def _source_family(name: str, url: str) -> str:
    u = url or ""
    n = name or ""
    if "weather.gov" in u and "CLI" in u.upper():
        return "nws_cli"
    if "Weather Company" in n or "weather.com" in u:
        return "weather_company"
    if "National Weather Service" in n or "weather.gov" in u:
        return "nws_other"
    return "unknown"


def discover_weather_markets(client, *, registry: LocationRegistry | None = None) -> dict[str, Any]:
    """Refresh discovery from GET /series?category=Climate and Weather (+ per-series detail)."""
    registry = registry or LocationRegistry()
    payload = client.get_series_list(category="Climate and Weather", limit=200)
    series_list = payload.get("series") or []
    n_mapped = n_ambiguous = n_unsupported = 0
    rows_out: list[dict[str, Any]] = []

    for s in series_list:
        tick = s.get("ticker") or ""
        title = s.get("title") or ""
        freq = s.get("frequency") or ""
        measurement = _classify_measurement(tick, title, freq)
        if measurement is None:
            continue
        try:
            detail = client.get(f"/series/{tick}")
            ser = detail.get("series") or detail
        except Exception as exc:
            logger.warning("series detail %s: %s", tick, exc)
            continue
        sources = ser.get("settlement_sources") or []
        src0 = sources[0] if sources else {}
        name = src0.get("name") or ""
        url = src0.get("url") or ""
        family = _source_family(name, url)
        issuedby = None
        wfo = None
        m = re.search(r"issuedby=([A-Z0-9]+)", url, re.I)
        if m:
            issuedby = m.group(1).upper()
        m2 = re.search(r"site=([A-Z0-9]+)", url, re.I)
        if m2:
            wfo = m2.group(1).upper()

        verified = VERIFIED_NWS_CLI_DAILY_MAX.get(tick.upper()) if measurement == "daily_max_temp_f" else None
        if verified and family == "nws_cli":
            row = {
                **verified,
                "series_ticker": tick,
                "measurement": measurement,
                "settlement_source_family": "nws_cli",
                "settlement_name": name,
                "settlement_url": url,
                "contract_url": ser.get("contract_url"),
                "uses_lst_climate_day": True,
                "unit": "F",
                "rounding_note": "Whole °F as printed in CLI MAXIMUM/MINIMUM",
                "data_availability": "public_metar_cli",
                "rule_version": ser.get("contract_url") or "api_settlement_sources",
                "details": {"api_title": title, "frequency": freq},
            }
            # Prefer API issuedby when present
            if issuedby:
                row["cli_location_id"] = issuedby
            n_mapped += 1
        elif family == "nws_cli" and measurement == "daily_max_temp_f" and issuedby:
            row = {
                "location_id": f"cli_{issuedby.lower()}",
                "series_ticker": tick,
                "measurement": measurement,
                "display_name": title,
                "settlement_source_family": "nws_cli",
                "settlement_name": name,
                "settlement_url": url,
                "contract_url": ser.get("contract_url"),
                "cli_location_id": issuedby,
                "wfo_office": wfo,
                "metar_ids": [],
                "neighbor_metar_ids": [],
                "timezone": "America/New_York",  # unknown until verified
                "uses_lst_climate_day": True,
                "unit": "F",
                "rounding_note": "Whole °F CLI",
                "mapping_status": "ambiguous_needs_metar_station",
                "data_availability": "cli_url_only",
                "validation_status": "blocked_incomplete_mapping",
                "model_family": "station_v2_nws_cli",
                "notes": "CLI issuedby known from API; METAR/ICAO settlement station not verified — do not invent",
                "rule_version": ser.get("contract_url"),
                "details": {"api_title": title},
            }
            n_ambiguous += 1
        elif family == "weather_company":
            row = {
                "location_id": f"twc_{tick.lower()}",
                "series_ticker": tick,
                "measurement": measurement,
                "display_name": title,
                "settlement_source_family": "weather_company",
                "settlement_name": name,
                "settlement_url": url,
                "contract_url": ser.get("contract_url"),
                "metar_ids": [],
                "mapping_status": "discovered",
                "data_availability": "twc_settlement_adapter_missing",
                "validation_status": "blocked_unsupported_settlement_source",
                "model_family": None,
                "notes": (
                    "Settles on The Weather Company per series API — NWS CLI / station_v2 pipeline "
                    "must NOT be applied. Metric-specific TWC adapter required."
                ),
                "timezone": "UTC",
                "uses_lst_climate_day": False,
                "unit": "F",
                "details": {"api_title": title, "frequency": freq},
            }
            n_unsupported += 1
        else:
            row = {
                "location_id": f"unk_{tick.lower()}",
                "series_ticker": tick,
                "measurement": measurement or "other",
                "display_name": title,
                "settlement_source_family": family,
                "settlement_name": name,
                "settlement_url": url,
                "contract_url": ser.get("contract_url"),
                "mapping_status": "discovered_unmapped",
                "data_availability": "unknown",
                "validation_status": "blocked_unmapped",
                "model_family": None,
                "notes": "Discovered; settlement station not resolved from evidence",
                "timezone": "UTC",
                "metar_ids": [],
                "details": {"api_title": title, "frequency": freq},
            }
            n_unsupported += 1

        # Alias duplicates for Houston
        if tick.upper() in ("KXHIGHHOU", "KXHOUHIGH") and "KXHIGHOU" in VERIFIED_NWS_CLI_DAILY_MAX:
            base = VERIFIED_NWS_CLI_DAILY_MAX["KXHIGHOU"]
            row = {
                **base,
                "series_ticker": tick,
                "measurement": "daily_max_temp_f",
                "settlement_source_family": "nws_cli",
                "settlement_name": name,
                "settlement_url": url,
                "contract_url": ser.get("contract_url"),
                "uses_lst_climate_day": True,
                "unit": "F",
                "rounding_note": "Whole °F CLI",
                "data_availability": "public_metar_cli",
                "details": {"alias_of": "KXHIGHOU"},
            }
            n_mapped += 1

        registry.upsert_target(row)
        rows_out.append(row)

    summary = {
        "n_series": len(series_list),
        "n_weather_rows": len(rows_out),
        "n_mapped": n_mapped,
        "n_ambiguous": n_ambiguous,
        "n_unsupported": n_unsupported,
        "n_daily_max": sum(1 for r in rows_out if r.get("measurement") == "daily_max_temp_f"),
        "n_daily_min": sum(1 for r in rows_out if r.get("measurement") == "daily_min_temp_f"),
        "operating_nws_cli_daily_max": [
            r["series_ticker"]
            for r in rows_out
            if r.get("measurement") == "daily_max_temp_f"
            and r.get("settlement_source_family") == "nws_cli"
            and r.get("mapping_status") in ("verified", "verified_cli_url")
        ],
    }
    registry.record_discovery_run(summary)
    out_path = Path("data/obs_engine/multi/last_discovery.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "targets": rows_out}, indent=2, default=str))
    return {"ok": True, "summary": summary, "registry_path": str(registry.path), "report": str(out_path)}
