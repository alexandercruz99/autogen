from __future__ import annotations

"""Smoke: public Kalshi markets endpoint responds for configured series."""

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.config import ApiConfig


def test_kalshi_public_markets_smoke():
    client = KalshiClient(ApiConfig(base_url="https://api.elections.kalshi.com/trade-api/v2"))
    try:
        status = client.get_exchange_status()
        assert status.get("trading_active") or status.get("exchange_active")
        markets = client.get_markets(status="open", series_ticker="KXHIGHNY", limit=5)
        assert "markets" in markets
    finally:
        client.close()
