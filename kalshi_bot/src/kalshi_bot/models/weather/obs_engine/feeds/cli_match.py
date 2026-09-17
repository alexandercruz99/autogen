"""CLI matching helpers: station + climate-day + decision-time availability."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore


def list_cli_reports(store: FeedStore, feed: str = "cli_nyc") -> list[dict[str, Any]]:
    rows = store._conn.execute(
        "SELECT payload_json, first_seen_utc, retrieved_at_utc, valid_utc, source_key FROM feed_samples WHERE feed=? ORDER BY retrieved_at_utc",
        (feed,),
    ).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload_json"])
        p["_first_seen_utc"] = r["first_seen_utc"]
        p["_retrieved_at_utc"] = r["retrieved_at_utc"]
        p["_source_key"] = r["source_key"]
        out.append(p)
    return out


def select_cli_for_decision(
    store: FeedStore,
    *,
    target_day: date,
    decision_utc: datetime,
    station_id: str = NYC_TARGET.cli_location_id,
    feed: str | None = None,
) -> dict[str, Any]:
    """Return applied CLI constraint (if any) plus excluded wrong-day/station reports.

    Only reports whose payload ``station_id`` matches ``station_id``, climate day
    matches ``target_day``, and whose issuance and first_seen are ≤ decision_utc
    may constrain the forecast.
    """
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    feed_name = feed or f"cli_{station_id.lower()}"
    # Backward-compatible default for NYC
    if station_id == NYC_TARGET.cli_location_id and feed is None:
        feed_name = "cli_nyc"
    excluded: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for p in list_cli_reports(store, feed=feed_name):
        day_s = p.get("climate_day")
        try:
            day = date.fromisoformat(day_s) if day_s and day_s != "unknown" else None
        except ValueError:
            day = None
        report_station = (p.get("station_id") or p.get("cli_location_id") or "").upper()
        issuance_raw = p.get("issuance_utc") or p.get("_retrieved_at_utc")
        try:
            issuance = datetime.fromisoformat(issuance_raw.replace("Z", "+00:00")) if issuance_raw else None
        except Exception:
            issuance = None
        first_seen_raw = p.get("_first_seen_utc")
        try:
            first_seen = datetime.fromisoformat(first_seen_raw) if first_seen_raw else None
        except Exception:
            first_seen = None
        meta = {
            "climate_day": day_s,
            "station_id": report_station,
            "max_temp_f": p.get("max_temp_f"),
            "is_preliminary": p.get("is_preliminary"),
            "issuance_utc": issuance_raw,
            "first_seen_utc": first_seen_raw,
            "source_key": p.get("_source_key"),
            "feed": feed_name,
        }
        if day != target_day:
            excluded.append({**meta, "exclude_reason": "wrong_climate_day"})
            continue
        if report_station and report_station != station_id.upper():
            excluded.append({**meta, "exclude_reason": "wrong_station"})
            continue
        if not report_station and station_id.upper() != NYC_TARGET.cli_location_id:
            # Legacy NYC rows without station_id only match NYC
            excluded.append({**meta, "exclude_reason": "wrong_station"})
            continue
        if p.get("max_temp_f") is None:
            excluded.append({**meta, "exclude_reason": "missing_max_temp"})
            continue
        if issuance is None or issuance > decision_utc:
            excluded.append({**meta, "exclude_reason": "not_published_at_decision_time"})
            continue
        if first_seen is None or first_seen > decision_utc:
            excluded.append({**meta, "exclude_reason": "not_in_store_at_decision_time"})
            continue
        available_at = max(issuance, first_seen)
        candidates.append({**meta, "available_at_utc": available_at.isoformat(), "payload": p})

    applied = None
    if candidates:
        finals = [c for c in candidates if not c.get("is_preliminary")]
        pool = finals if finals else candidates
        pool.sort(key=lambda c: c.get("issuance_utc") or "", reverse=True)
        best = pool[0]
        max_f = best.get("max_temp_f")
        applied = {
            "climate_day": best.get("climate_day"),
            "station_id": best.get("station_id") or station_id,
            "max_temp_f": int(max_f) if max_f is not None else None,
            "is_preliminary": bool(best.get("is_preliminary")),
            "issuance_utc": best.get("issuance_utc"),
            "first_seen_utc": best.get("first_seen_utc"),
            "available_at_utc": best.get("available_at_utc"),
            "source_key": best.get("source_key"),
            "floor_policy": "whole_F_cli_value_only",
            "note": (
                "Preliminary CLI is revisable evidence; final CLI is settlement. "
                "Floor uses printed whole °F — not ceil(METAR tempFloat)."
            ),
        }
    return {
        "applied": applied,
        "excluded": excluded,
        "n_candidates": len(candidates),
        "station_id": station_id,
        "feed": feed_name,
        "target_day": target_day.isoformat(),
    }
