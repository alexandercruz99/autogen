"""METAR adapters for KNYC + neighbors via aviationweather.gov (public, no key)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)

STATIONS = ("KNYC", "KLGA", "KJFK")


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.upper() in ("VRB", "M", "NULL"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_metar_station(station: str, hours: int = 6) -> list[dict[str, Any]]:
    url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json&hours={hours}"
    r = httpx.get(url, timeout=30.0, headers={"User-Agent": "KalshiBotFeeds/0.2"}, follow_redirects=True)
    r.raise_for_status()
    return list(r.json() or [])


def collect_metar(store: FeedStore, stations: tuple[str, ...] = STATIONS) -> dict[str, Any]:
    out: dict[str, Any] = {"feed": "metar", "stations": {}}
    for st in stations:
        try:
            rows = fetch_metar_station(st)
            n_new = 0
            latest_key = None
            for row in rows:
                ot = row.get("obsTime")
                if ot is None:
                    continue
                valid = datetime.fromtimestamp(int(ot), tz=timezone.utc)
                # Prefer finer temperature fields when present
                temp_raw = row.get("tempFloat", row.get("temp"))
                dewp_raw = row.get("dewpFloat", row.get("dewp"))
                precision = "tempFloat" if row.get("tempFloat") is not None else "temp_whole_C"
                tmpf = None if temp_raw is None else float(temp_raw) * 9 / 5 + 32
                dwpf = None if dewp_raw is None else float(dewp_raw) * 9 / 5 + 32
                payload = {
                    "station": st,
                    "valid_utc": valid.isoformat(),
                    "receipt_time": row.get("receiptTime"),
                    "tmpf": tmpf,
                    "dwpf": dwpf,
                    "tmpc_raw": temp_raw,
                    "precision": precision,
                    "sknt": _num(row.get("wspd")),
                    "drct": _num(row.get("wdir")),
                    "skyc1": row.get("cover"),
                    "raw": row,
                    "missing": {
                        "sknt": row.get("wspd") is None,
                        "drct": row.get("wdir") is None or (isinstance(row.get("wdir"), str) and str(row.get("wdir")).upper() == "VRB"),
                        "skyc1": not row.get("cover"),
                        "tmpf": tmpf is None,
                    },
                }
                key = f"{st}:{valid.isoformat()}"
                res = store.upsert_sample(
                    feed=f"metar_{st}",
                    source_key=key,
                    payload=payload,
                    valid_utc=valid.isoformat(),
                    product="aviationweather_metar",
                )
                if res["new"]:
                    n_new += 1
                latest_key = key
            store.checkpoint(f"metar_{st}", ok=True, source_key=latest_key, meta={"n_fetched": len(rows), "n_new": n_new})
            out["stations"][st] = {"ok": True, "n_fetched": len(rows), "n_new": n_new, "latest_key": latest_key}
        except Exception as exc:
            logger.warning("metar %s failed: %s", st, exc)
            store.checkpoint(f"metar_{st}", ok=False, error=str(exc))
            out["stations"][st] = {"ok": False, "error": str(exc)}
    out["ok"] = any(v.get("ok") for v in out["stations"].values())
    return out
