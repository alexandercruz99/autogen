"""METAR adapters for KNYC + neighbors via aviationweather.gov (public, no key)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import hpa_to_inhg
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)

STATIONS = ("KNYC", "KLGA", "KJFK")
# Pull enough history to cover an LST climate day after restart (~26–30h).
DEFAULT_HOURS = 30


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.upper() in ("VRB", "M", "NULL"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_metar_station(station: str, hours: int = DEFAULT_HOURS) -> list[dict[str, Any]]:
    url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json&hours={hours}"
    r = httpx.get(url, timeout=30.0, headers={"User-Agent": "KalshiBotFeeds/0.3"}, follow_redirects=True)
    r.raise_for_status()
    return list(r.json() or [])


def _payload_from_row(station: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    ot = row.get("obsTime")
    if ot is None:
        return None
    valid = datetime.fromtimestamp(int(ot), tz=timezone.utc)
    temp_raw = row.get("tempFloat", row.get("temp"))
    dewp_raw = row.get("dewpFloat", row.get("dewp"))
    precision = "tempFloat" if row.get("tempFloat") is not None else "temp_whole_C"
    tmpf = None if temp_raw is None else float(temp_raw) * 9 / 5 + 32
    dwpf = None if dewp_raw is None else float(dewp_raw) * 9 / 5 + 32
    # aviationweather ``altim`` is hPa (e.g. 1028.5); ASOS archive uses inHg
    alti_inhg = hpa_to_inhg(_num(row.get("altim")))
    # Hourly precip not provided on this endpoint → explicit None (not zero)
    p01i = _num(row.get("p01i") or row.get("precip"))
    payload = {
        "station": station,
        "valid_utc": valid.isoformat(),
        "receipt_time": row.get("receiptTime"),
        "report_time": row.get("reportTime"),
        "tmpf": tmpf,
        "dwpf": dwpf,
        "tmpc_raw": temp_raw,
        "precision": precision,
        "sknt": _num(row.get("wspd")),
        "drct": _num(row.get("wdir")),
        "alti": alti_inhg,
        "alti_hpa_raw": _num(row.get("altim")),
        "p01i": p01i,
        "skyc1": row.get("cover"),
        "raw": row,
        "missing": {
            "sknt": row.get("wspd") is None,
            "drct": row.get("wdir") is None
            or (isinstance(row.get("wdir"), str) and str(row.get("wdir")).upper() == "VRB"),
            "skyc1": not row.get("cover"),
            "tmpf": tmpf is None,
            "alti": alti_inhg is None,
            "p01i": p01i is None,
        },
    }
    key = f"{station}:{valid.isoformat()}"
    return key, payload


def collect_metar(
    store: FeedStore,
    stations: tuple[str, ...] = STATIONS,
    *,
    hours: int = DEFAULT_HOURS,
) -> dict[str, Any]:
    out: dict[str, Any] = {"feed": "metar", "stations": {}, "hours_requested": hours}
    for st in stations:
        try:
            rows = fetch_metar_station(st, hours=hours)
            n_new = 0
            latest_key = None
            for row in rows:
                parsed = _payload_from_row(st, row)
                if parsed is None:
                    continue
                key, payload = parsed
                # Prefer provider receipt/report time as first_seen for new inserts so
                # post-restart backfill does not stamp every historical row with wall-clock now.
                first_seen = None
                for cand in (payload.get("receipt_time"), payload.get("report_time"), payload.get("valid_utc")):
                    if not cand:
                        continue
                    try:
                        first_seen = str(cand).replace("Z", "+00:00")
                        # normalize to iso
                        datetime.fromisoformat(first_seen)
                        break
                    except Exception:
                        first_seen = None
                res = store.upsert_sample(
                    feed=f"metar_{st}",
                    source_key=key,
                    payload=payload,
                    valid_utc=payload["valid_utc"],
                    product="aviationweather_metar",
                    first_seen_utc=first_seen,
                )
                if res["new"]:
                    n_new += 1
                latest_key = key
            store.checkpoint(
                f"metar_{st}",
                ok=True,
                source_key=latest_key,
                meta={"n_fetched": len(rows), "n_new": n_new, "hours": hours},
            )
            out["stations"][st] = {
                "ok": True,
                "n_fetched": len(rows),
                "n_new": n_new,
                "latest_key": latest_key,
                "usable": bool(latest_key),
            }
        except Exception as exc:
            logger.warning("metar %s failed: %s", st, exc)
            store.checkpoint(f"metar_{st}", ok=False, error=str(exc))
            out["stations"][st] = {"ok": False, "error": str(exc), "usable": False}
    out["ok"] = any(v.get("ok") for v in out["stations"].values())
    out["usable"] = any(v.get("usable") for v in out["stations"].values())
    return out
