"""NWS CLI preliminary/final climate report feed (location-parameterized)."""

from __future__ import annotations

import logging
from typing import Any

from kalshi_bot.models.weather.cli_reports import CliReportClient
from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)


def collect_cli(
    store: FeedStore,
    *,
    cli_location_id: str | None = None,
    feed: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    loc = (cli_location_id or NYC_TARGET.cli_location_id).upper()
    feed_name = feed or (f"cli_{loc.lower()}" if loc != NYC_TARGET.cli_location_id else "cli_nyc")
    client = CliReportClient()
    try:
        reports = client.collect_recent_with_max(loc, limit=limit)
        n_new = 0
        latest = None
        for rep in reports:
            issuance = rep.issuance_time.isoformat() if rep.issuance_time else "unknown"
            day = rep.climate_day.isoformat() if rep.climate_day else "unknown"
            key = f"CLI:{loc}:{day}:{issuance}:{'P' if rep.is_preliminary else 'F'}"
            payload = {
                "climate_day": day,
                "issuance_utc": issuance,
                "max_temp_f": rep.max_temp_f,
                "min_temp_f": getattr(rep, "min_temp_f", None),
                "is_preliminary": bool(rep.is_preliminary),
                "product_id": getattr(rep, "product_id", None),
                "station_id": loc,
                "cli_location_id": loc,
                "settlement_note": "Kalshi daily high settles on FINAL CLI; prelim is revisable evidence only",
            }
            res = store.upsert_sample(
                feed=feed_name,
                source_key=key,
                payload=payload,
                valid_utc=issuance if issuance != "unknown" else None,
                product="nws_cli",
            )
            if res["new"]:
                n_new += 1
            latest = key
        store.checkpoint(
            feed_name, ok=True, source_key=latest, meta={"n_reports": len(reports), "n_new": n_new, "cli_location_id": loc}
        )
        return {
            "ok": True,
            "n_reports": len(reports),
            "n_new": n_new,
            "latest_key": latest,
            "feed": feed_name,
            "cli_location_id": loc,
            "usable": bool(latest),
        }
    except Exception as exc:
        logger.warning("cli collect %s failed: %s", loc, exc)
        store.checkpoint(feed_name, ok=False, error=str(exc))
        return {"ok": False, "error": str(exc), "feed": feed_name, "cli_location_id": loc, "usable": False}
    finally:
        client.close()
