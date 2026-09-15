from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.orderbook import ExecutableBook, parse_orderbook
from kalshi_bot.config import ScanConfig
from kalshi_bot.data.store import Store

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    market: dict[str, Any]
    category: str
    book: ExecutableBook
    captured_at: datetime
    series_ticker: str = ""
    skip_reason: str | None = None


@dataclass
class ScanResult:
    snapshots: list[MarketSnapshot] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketScanner:
    def __init__(self, client: KalshiClient, store: Store, config: ScanConfig) -> None:
        self.client = client
        self.store = store
        self.config = config

    def discover_series(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for category in self.config.categories:
            try:
                payload = self.client.get_series_list(category=category, limit=200)
                series = payload.get("series") or payload.get("series_list") or []
                for s in series:
                    s = dict(s)
                    s["_category"] = category
                    found.append(s)
            except Exception as exc:
                logger.warning("Series discovery failed for %s: %s", category, exc)
        return found

    def scan(self) -> ScanResult:
        result = ScanResult()
        tickers_seen: set[str] = set()
        series_list = list(self.config.series_tickers)

        # Enrich with category series if configured categories present.
        if self.config.categories and not series_list:
            for s in self.discover_series():
                ticker = s.get("ticker")
                if ticker:
                    series_list.append(ticker)

        for series_ticker in series_list:
            cursor = None
            while True:
                try:
                    payload = self.client.get_markets(
                        status="open",
                        series_ticker=series_ticker,
                        limit=min(200, self.config.max_markets_per_scan),
                        cursor=cursor,
                    )
                except Exception as exc:
                    msg = f"markets fetch failed for {series_ticker}: {exc}"
                    logger.warning(msg)
                    result.errors.append(msg)
                    break

                markets = payload.get("markets") or []
                for market in markets:
                    ticker = market.get("ticker")
                    if not ticker or ticker in tickers_seen:
                        continue
                    tickers_seen.add(ticker)
                    if len(result.snapshots) >= self.config.max_markets_per_scan:
                        return result
                    category = self._infer_category(series_ticker, market)
                    self.store.upsert_market(market, category=category)
                    try:
                        ob_raw = self.client.get_orderbook(
                            ticker, depth=self.config.orderbook_depth_levels
                        )
                        book = parse_orderbook(ob_raw)
                        self.store.save_snapshot(
                            ticker,
                            str(book.best_yes_bid) if book.best_yes_bid is not None else None,
                            str(book.best_yes_ask) if book.best_yes_ask is not None else None,
                            str(book.best_no_bid) if book.best_no_bid is not None else None,
                            str(book.best_no_ask) if book.best_no_ask is not None else None,
                            ob_raw,
                        )
                        result.snapshots.append(
                            MarketSnapshot(
                                market=market,
                                category=category,
                                book=book,
                                captured_at=datetime.now(timezone.utc),
                                series_ticker=series_ticker,
                            )
                        )
                    except Exception as exc:
                        result.skipped.append(
                            {"ticker": ticker, "reason": f"orderbook unavailable: {exc}"}
                        )
                cursor = payload.get("cursor") or None
                if not cursor or not markets:
                    break
        return result

    def _infer_category(self, series_ticker: str, market: dict[str, Any]) -> str:
        title = (market.get("title") or "") + " " + series_ticker
        upper = title.upper()
        if "HIGH" in upper or "TEMP" in upper or "RAIN" in upper or "SNOW" in upper:
            return "Climate and Weather"
        return market.get("category") or "unknown"
