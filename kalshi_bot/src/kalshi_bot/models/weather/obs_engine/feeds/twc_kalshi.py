"""The Weather Company Kalshi settlement portal adapter (weather.com/kalshi).

Public JSON endpoints used by the climate portal (no API key required):

- ``GET /kalshi/api/climate/primary?date=YYYY-MM-DD`` — domestic daily max/min
  climate reports (official / preliminary / no_report) keyed by CLI station id
  (e.g. NYC → CLINYC).
- ``GET /kalshi/api/metar?primary=true&weekStart=YYYY-MM-DD`` — progressive
  hourly temps at the same ICAO stations (settlement-aligned max-so-far).
- ``GET /kalshi/api/climate/international?date=YYYY-MM-DD`` — international
  daily highs (Celsius); not wired into the domestic daily-max path yet.

Settlement evidence: Kalshi ``rules_primary`` for ``KXHIGHNY`` names CLINYC /
The Weather Company and points ``settlement_sources`` at weather.com/kalshi.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)

TWC_KALSHI_ORIGIN = "https://weather.com"
TWC_API_BASE = f"{TWC_KALSHI_ORIGIN}/kalshi/api"
DEFAULT_HEADERS = {
    "User-Agent": "KalshiBotTwcAdapter/0.1 (+research; settlement-aligned fetch)",
    "Accept": "application/json",
    "Referer": f"{TWC_KALSHI_ORIGIN}/kalshi",
}


class TwcKalshiClient:
    """Thin HTTP client for the public weather.com/kalshi JSON API."""

    def __init__(self, *, timeout: float = 45.0) -> None:
        self._client = httpx.Client(
            base_url=TWC_API_BASE,
            timeout=timeout,
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str, **params: Any) -> dict[str, Any]:
        r = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected TWC payload type for {path}: {type(data)}")
        return data

    def climate_primary(self, day: date | str) -> dict[str, Any]:
        d = day.isoformat() if isinstance(day, date) else str(day)
        return self.get_json("/climate/primary", date=d)

    def climate_international(self, day: date | str) -> dict[str, Any]:
        d = day.isoformat() if isinstance(day, date) else str(day)
        return self.get_json("/climate/international", date=d)

    def metar_primary_week(self, week_start: date | str) -> dict[str, Any]:
        d = week_start.isoformat() if isinstance(week_start, date) else str(week_start)
        return self.get_json("/metar", primary="true", weekStart=d)


def _monday_on_or_before(day: date) -> date:
    return day - timedelta(days=day.weekday())


def climate_feed_name(cli_id: str) -> str:
    return f"twc_climate_{cli_id.lower()}"


def metar_feed_name(icao: str) -> str:
    return f"twc_metar_{icao.upper()}"


def collect_twc_climate(
    store: FeedStore,
    *,
    day: date | None = None,
    cli_ids: list[str] | tuple[str, ...] | None = None,
    also_yesterday: bool = True,
    client: TwcKalshiClient | None = None,
) -> dict[str, Any]:
    """Fetch domestic TWC climate reports and upsert per-station samples."""
    own = client is None
    client = client or TwcKalshiClient()
    days: list[date] = []
    base = day or datetime.now(timezone.utc).date()
    days.append(base)
    if also_yesterday:
        days.append(base - timedelta(days=1))
    want = {c.upper() for c in (cli_ids or [])} if cli_ids is not None else None
    summary: dict[str, Any] = {"ok": True, "days": [], "n_new": 0, "stations": {}}
    try:
        for d in days:
            payload = client.climate_primary(d)
            day_info: dict[str, Any] = {
                "date": d.isoformat(),
                "officialReports": payload.get("officialReports"),
                "preliminaryReports": payload.get("preliminaryReports"),
                "noReports": payload.get("noReports"),
                "source": payload.get("source"),
            }
            n_day_new = 0
            for row in payload.get("results") or []:
                st = row.get("station") or {}
                cli = (st.get("cliId") or "").upper()
                if not cli:
                    continue
                if want is not None and cli not in want:
                    continue
                data = row.get("data")
                status = row.get("status") or ("official" if (data or {}).get("isOfficial") else "no_report")
                feed = climate_feed_name(cli)
                if data and data.get("maxTemp") is not None:
                    report_date = data.get("reportDate") or d.isoformat()
                    is_prelim = status == "preliminary" or not bool(data.get("isOfficial"))
                    key = f"TWC:{cli}:{report_date}:{'P' if is_prelim else 'O'}:{status}"
                    # Prefer issueTime when present; else report date noon UTC as availability proxy
                    issue = (data.get("issueTime") or "").strip()
                    if not issue:
                        issue = f"{report_date}T12:00:00+00:00"
                    sample = {
                        "climate_day": report_date,
                        "issuance_utc": issue,
                        "max_temp_f": int(data["maxTemp"]) if data.get("maxTemp") is not None else None,
                        "min_temp_f": int(data["minTemp"]) if data.get("minTemp") is not None else None,
                        "precipitation": data.get("precipitation"),
                        "snowfall": data.get("snowfall"),
                        "is_preliminary": is_prelim,
                        "is_official": bool(data.get("isOfficial")),
                        "status": status,
                        "station_id": cli,
                        "cli_location_id": cli,
                        "icao": st.get("icao"),
                        "twc_station_id": st.get("id"),
                        "location": data.get("location") or st.get("city"),
                        "timezone": st.get("timezone"),
                        "settlement_note": (
                            "Kalshi KXHIGH* daily markets settle on The Weather Company climate "
                            "report at this CLI station (portal weather.com/kalshi)"
                        ),
                        "source": payload.get("source") or "weather.com/kalshi",
                    }
                    res = store.upsert_sample(
                        feed=feed,
                        source_key=key,
                        payload=sample,
                        valid_utc=issue,
                        first_seen_utc=issue,
                        product="twc_climate",
                    )
                    if res["new"]:
                        n_day_new += 1
                        summary["n_new"] += 1
                    store.checkpoint(feed, ok=True, source_key=key, meta={"status": status, "day": report_date})
                    summary["stations"][cli] = {
                        "feed": feed,
                        "status": status,
                        "max_temp_f": sample["max_temp_f"],
                        "climate_day": report_date,
                        "usable": True,
                    }
                else:
                    # Still checkpoint no_report so callers know we looked
                    store.checkpoint(
                        feed,
                        ok=True,
                        source_key=f"TWC:{cli}:{d.isoformat()}:NONE",
                        meta={"status": status, "day": d.isoformat()},
                    )
                    summary["stations"].setdefault(
                        cli,
                        {"feed": feed, "status": status, "usable": False, "climate_day": d.isoformat()},
                    )
            day_info["n_new"] = n_day_new
            summary["days"].append(day_info)
        summary["usable"] = any(v.get("usable") for v in summary["stations"].values()) or summary["n_new"] > 0
        # usable if we successfully contacted the API even with no_report for today
        summary["usable"] = True
        return summary
    except Exception as exc:
        logger.warning("twc climate collect failed: %s", exc)
        summary.update({"ok": False, "error": str(exc), "usable": False})
        return summary
    finally:
        if own:
            client.close()


def collect_twc_metar(
    store: FeedStore,
    *,
    icao_ids: list[str] | tuple[str, ...],
    week_start: date | None = None,
    client: TwcKalshiClient | None = None,
) -> dict[str, Any]:
    """Fetch TWC progressive hourly temps and upsert per-ICAO samples."""
    own = client is None
    client = client or TwcKalshiClient()
    want = {i.upper() for i in icao_ids}
    # Portal weeks are Monday-based; include prior Monday so we cover ~2 local days.
    today = datetime.now(timezone.utc).date()
    monday = week_start or _monday_on_or_before(today)
    summary: dict[str, Any] = {"ok": True, "week_start": monday.isoformat(), "n_new": 0, "stations": {}}
    try:
        payload = client.metar_primary_week(monday)
        summary["source"] = payload.get("source")
        summary["totalObservations"] = payload.get("totalObservations")
        summary["fetchedAt"] = payload.get("fetchedAt")
        for st in payload.get("stations") or []:
            icao = (st.get("icaoId") or "").upper()
            if icao not in want:
                continue
            feed = metar_feed_name(icao)
            n_new = 0
            latest_key = None
            temps_today: list[float] = []
            for obs in st.get("observations") or []:
                temp_f = obs.get("tempF")
                report_utc = obs.get("reportTimeUTC")
                if temp_f is None or not report_utc:
                    continue
                try:
                    valid = datetime.fromisoformat(str(report_utc).replace("Z", "+00:00"))
                except Exception:
                    continue
                local_date = obs.get("localDate")
                key = f"TWC_METAR:{icao}:{valid.isoformat()}"
                sample = {
                    "station": icao,
                    "valid_utc": valid.isoformat(),
                    "receipt_time": payload.get("fetchedAt"),
                    "report_time": report_utc,
                    "tmpf": float(temp_f),
                    "tmpc_raw": obs.get("tempC"),
                    "precision": "twc_portal_tempF",
                    "local_date": local_date,
                    "local_hour": obs.get("localHour"),
                    "report_time_local": obs.get("reportTimeLocal"),
                    "status": obs.get("status"),
                    "dwpf": None,
                    "sknt": None,
                    "drct": None,
                    "alti": None,
                    "p01i": None,
                    "skyc1": None,
                    "missing": {
                        "tmpf": False,
                        "dwpf": True,
                        "sknt": True,
                        "drct": True,
                        "alti": True,
                        "p01i": True,
                        "skyc1": True,
                    },
                    "source": "weather.com/kalshi/metar",
                }
                res = store.upsert_sample(
                    feed=feed,
                    source_key=key,
                    payload=sample,
                    valid_utc=valid.isoformat(),
                    # Portal historical hours: treat report valid time as availability for
                    # same-cycle collect→infer (mirrors AviationWeather receipt preference).
                    first_seen_utc=valid.isoformat(),
                    product="twc_metar",
                )
                if res["new"]:
                    n_new += 1
                latest_key = key
                if local_date == today.isoformat():
                    temps_today.append(float(temp_f))
            store.checkpoint(
                feed,
                ok=True,
                source_key=latest_key,
                meta={"n_new": n_new, "n_obs": len(st.get("observations") or [])},
            )
            summary["n_new"] += n_new
            summary["stations"][icao] = {
                "feed": feed,
                "n_new": n_new,
                "n_obs": len(st.get("observations") or []),
                "max_so_far_today_f": max(temps_today) if temps_today else None,
                "usable": bool(latest_key),
            }
        summary["usable"] = any(v.get("usable") for v in summary["stations"].values())
        return summary
    except Exception as exc:
        logger.warning("twc metar collect failed: %s", exc)
        for icao in want:
            store.checkpoint(metar_feed_name(icao), ok=False, error=str(exc))
        return {"ok": False, "error": str(exc), "usable": False, "week_start": monday.isoformat()}
    finally:
        if own:
            client.close()


def select_twc_climate_for_decision(
    store: FeedStore,
    *,
    target_day: date,
    decision_utc: datetime,
    cli_id: str,
) -> dict[str, Any]:
    """Apply TWC climate whole-°F floor when a same-day report is already in the store."""
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    feed = climate_feed_name(cli_id)
    rows = store._conn.execute(
        "SELECT payload_json, first_seen_utc, retrieved_at_utc, source_key FROM feed_samples WHERE feed=? ORDER BY retrieved_at_utc",
        (feed,),
    ).fetchall()
    import json

    excluded: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for r in rows:
        p = json.loads(r["payload_json"])
        day_s = p.get("climate_day")
        try:
            day = date.fromisoformat(day_s) if day_s else None
        except ValueError:
            day = None
        station = (p.get("station_id") or p.get("cli_location_id") or "").upper()
        issuance_raw = p.get("issuance_utc") or r["retrieved_at_utc"]
        try:
            issuance = datetime.fromisoformat(str(issuance_raw).replace("Z", "+00:00")) if issuance_raw else None
        except Exception:
            issuance = None
        first_seen_raw = r["first_seen_utc"]
        try:
            first_seen = datetime.fromisoformat(first_seen_raw) if first_seen_raw else None
        except Exception:
            first_seen = None
        meta = {
            "climate_day": day_s,
            "station_id": station,
            "max_temp_f": p.get("max_temp_f"),
            "is_preliminary": p.get("is_preliminary"),
            "status": p.get("status"),
            "issuance_utc": issuance_raw,
            "first_seen_utc": first_seen_raw,
            "source_key": r["source_key"],
            "feed": feed,
        }
        if day != target_day:
            excluded.append({**meta, "exclude_reason": "wrong_climate_day"})
            continue
        if station and station != cli_id.upper():
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
        candidates.append({**meta, "available_at_utc": max(issuance, first_seen).isoformat()})

    applied = None
    if candidates:
        officials = [c for c in candidates if c.get("status") == "official" or not c.get("is_preliminary")]
        pool = officials if officials else candidates
        pool.sort(key=lambda c: c.get("issuance_utc") or "", reverse=True)
        best = pool[0]
        applied = {
            "climate_day": best.get("climate_day"),
            "station_id": best.get("station_id") or cli_id.upper(),
            "max_temp_f": int(best["max_temp_f"]),
            "is_preliminary": bool(best.get("is_preliminary")),
            "status": best.get("status"),
            "issuance_utc": best.get("issuance_utc"),
            "first_seen_utc": best.get("first_seen_utc"),
            "available_at_utc": best.get("available_at_utc"),
            "source_key": best.get("source_key"),
            "floor_policy": "whole_F_twc_climate_value_only",
            "note": (
                "TWC climate report floor (weather.com/kalshi). Preliminary is revisable; "
                "Final portal status is convenience-only per Kalshi rules."
            ),
        }
    return {
        "applied": applied,
        "excluded": excluded,
        "n_candidates": len(candidates),
        "station_id": cli_id.upper(),
        "feed": feed,
        "target_day": target_day.isoformat(),
    }


def twc_metar_max_so_far(
    store: FeedStore,
    *,
    icao: str,
    climate_day: date,
    tz_name: str,
    decision_utc: datetime,
) -> dict[str, Any]:
    """Max progressive temp (°F) on the LST climate day from TWC portal METAR samples."""
    import json

    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    feed = metar_feed_name(icao)
    rows = store._conn.execute(
        "SELECT payload_json, first_seen_utc FROM feed_samples WHERE feed=? ORDER BY valid_utc ASC",
        (feed,),
    ).fetchall()
    temps: list[float] = []
    n = 0
    for r in rows:
        p = json.loads(r["payload_json"])
        tmpf = p.get("tmpf")
        if tmpf is None:
            continue
        local_date = p.get("local_date")
        if local_date:
            try:
                if date.fromisoformat(local_date) != climate_day:
                    continue
            except ValueError:
                continue
        else:
            try:
                valid = datetime.fromisoformat(p["valid_utc"])
                local = valid.astimezone(ZoneInfo(tz_name))
                if local.date() != climate_day:
                    continue
            except Exception:
                continue
        try:
            valid = datetime.fromisoformat(p["valid_utc"])
        except Exception:
            continue
        if valid > decision_utc:
            continue
        first_seen_raw = r["first_seen_utc"]
        if first_seen_raw:
            try:
                if datetime.fromisoformat(first_seen_raw) > decision_utc:
                    continue
            except Exception:
                pass
        temps.append(float(tmpf))
        n += 1
    return {
        "feed": feed,
        "icao": icao.upper(),
        "climate_day": climate_day.isoformat(),
        "n_obs": n,
        "max_so_far": max(temps) if temps else None,
        "usable": n > 0,
    }
