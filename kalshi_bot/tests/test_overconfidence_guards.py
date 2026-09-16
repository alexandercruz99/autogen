from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from kalshi_bot.api.orderbook import market_implied_yes_prob, parse_orderbook
from kalshi_bot.config import TradingConfig
from kalshi_bot.ev.calculator import evaluate_binary_contract
from kalshi_bot.models.base import Prediction
from kalshi_bot.models.weather.high_temp import WeatherHighTempModel
from kalshi_bot.money import D


def test_continuity_correction_not_fifty_at_strike():
    m = WeatherHighTempModel.__new__(WeatherHighTempModel)
    # μ = 82, strict >82 must not be ~50% (integer °F needs ≥83).
    p = m._prob_yes(82.0, 3.0, {"op": "gt", "low": 82.0, "high": None})
    assert p < 0.45
    # μ = 72, strict <70 should be well below coin-flip.
    p_lt = m._prob_yes(72.0, 3.0, {"op": "lt", "low": None, "high": 70.0})
    assert p_lt < 0.30


def test_one_sided_book_uses_ask_as_implied():
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [],
                "no_dollars": [["0.9900", "100.00"]],
            }
        }
    )
    assert book.best_yes_ask == Decimal("0.0100")
    assert market_implied_yes_prob(book) == Decimal("0.0100")


def test_unvalidated_longshot_refused():
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [],
                "no_dollars": [["0.9900", "500.00"]],
            }
        }
    )
    pred = Prediction(
        market_ticker="TEST-T70",
        p_yes=D("0.25"),
        p_yes_conservative=D("0.20"),
        uncertainty=D("0.08"),
        model_version="test",
        data_sources=[],
        factors=[],
        validation_evidence="UNVALIDATED: unit test",
        as_of=datetime.now(timezone.utc),
        supported=True,
    )
    cfg = TradingConfig(
        target_trade_dollars=D("5"),
        max_loss_per_trade_dollars=D("5"),
        max_contracts_per_order=D("500"),
        min_net_edge=D("0.05"),
        uncertainty_buffer=D("0.05"),
        max_model_market_divergence=D("0.12"),
        unvalidated_market_shrink=D("0.65"),
        unvalidated_longshot_max_price=D("0.05"),
        min_cash_reserve_dollars=D("0"),
    )
    yes = next(e for e in evaluate_binary_contract(pred, book, cfg) if e.side == "yes")
    assert not yes.qualifies
    assert "longshot guard" in yes.reason
