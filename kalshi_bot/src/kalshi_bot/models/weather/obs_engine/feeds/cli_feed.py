"""NWS CLI preliminary/final climate report feed for Central Park."""

from __future__ import annotations

import logging
from typing import Any

from kalshi_bot.models.weather.cli_reports import CliReportClient
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine import NYC_TARGET

logger = logging.getLogger(__name__)


def collect_cli(store: FeedStore, limit: int = 8) -> dict[str, Any]:
    client = CliReportClient()
    try:
        reports = client.collect_recent_with_max(NYC_TARGET.cli_location_id, limit=limit)
        n_new = 0
        latest = None
        for rep in reports:
            issuance = rep.issuance_time.isoformat() if rep.issuance_time else "unknown"
            day = rep.climate_day.isoformat() if rep.climate_day else "unknown"
            key = f"CLI:{NYC_TARGET.cli_location_id}:{day}:{issuance}:{'P' if rep.is_preliminary else 'F'}"
            payload = {
                "climate_day": day,
                "issuance_utc": issuance,
                "max_temp_f": rep.max_temp_f,
                "min_temp_f": getattr(rep, "min_temp_f", None),
                "is_preliminary": bool(rep.is_preliminary),
                "product_id": getattr(rep, "product_id", None),
                "settlement_note": "Kalshi daily high settles on FINAL CLI; prelim is revisable evidence only",
            }
            res = store.upsert_sample(
                feed="cli_nyc",
                source_key=key,
                payload=payload,
                valid_utc=issuance if issuance != "unknown" else None,
                product="nws_cli",
            )
            if res["new"]:
                n_new += 1
            latest = key
        store.checkpoint("cli_nyc", ok=True, source_key=latest, meta={"n_reports": len(reports), "n_new": n_new})
        return {"ok": True, "n_reports": len(reports), "n_new": n_new, "latest_key": latest}
    except Exception as exc:
        logger.warning("cli collect failed: %s", exc)
        store.checkpoint("cli_nyc", ok=False, error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()
