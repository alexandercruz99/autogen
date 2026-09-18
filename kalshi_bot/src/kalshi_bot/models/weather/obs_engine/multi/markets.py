"""Fetch open Kalshi markets for an arbitrary weather series ticker."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from kalshi_bot.models.weather.nws_client import parse_market_date

logger = logging.getLogger(__name__)


def fetch_open_series_markets(client, series_tickers: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for series in series_tickers:
        if not series:
            continue
        try:
            payload = client.get_markets(series_ticker=series, status="open", limit=200)
            batch = payload.get("markets") or []
        except Exception as exc:
            logger.warning("get_markets %s failed: %s", series, exc)
            continue
        for m in batch:
            ticker = (m.get("ticker") or "").upper()
            if "HOUR" in ticker or "TEMPAT" in ticker or "DIRECTION" in ticker:
                continue
            markets.append(m)
    return markets


def select_event_markets(
    markets: list[dict[str, Any]], *, prefer_day: date
) -> tuple[date | None, list[dict[str, Any]]]:
    by_day: dict[date, list[dict[str, Any]]] = {}
    for m in markets:
        d = parse_market_date(m.get("ticker") or "")
        if d is None:
            continue
        by_day.setdefault(d, []).append(m)
    if not by_day:
        return None, []
    if prefer_day in by_day:
        return prefer_day, by_day[prefer_day]
    future = sorted(d for d in by_day if d >= prefer_day)
    if future:
        return future[0], by_day[future[0]]
    past = sorted(by_day)
    return past[-1], by_day[past[-1]]
