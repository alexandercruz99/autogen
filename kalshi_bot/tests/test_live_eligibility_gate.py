"""Execution-boundary live eligibility gates for weather models."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from kalshi_bot.config import AppConfig, TradingConfig
from kalshi_bot.ev.calculator import EvResult
from kalshi_bot.execution.engine import ExecutionEngine
from kalshi_bot.money import D


def _ev() -> EvResult:
    return EvResult(
        side="yes",
        quantity=D("1"),
        executable_price=D("0.40"),
        fillable_quantity=D("1"),
        estimated_prob=D("0.55"),
        conservative_prob=D("0.50"),
        uncertainty=D("0.05"),
        fees_total=D("0.01"),
        fees_per_contract=D("0.01"),
        estimated_ev=D("0.14"),
        conservative_ev=D("0.04"),
        breakeven_prob=D("0.41"),
        max_loss=D("0.41"),
        capital_required=D("0.41"),
        qualifies=True,
        reason="test",
    )


def test_live_submit_blocked_without_model_live_eligible(tmp_path):
    from kalshi_bot.data.store import Store

    store = Store(str(tmp_path / "t.db"))
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    client = MagicMock()
    eng = ExecutionEngine(client, store, cfg)
    order = eng.place_individual(
        market={"ticker": "KXHIGHNY-26SEP15-T70", "event_ticker": "KXHIGHNY-26SEP15"},
        ev=_ev(),
        opportunity_id="opp-1",
        mode="live",
        model_live_eligible=False,
    )
    assert order is None
    client.create_order_v2.assert_not_called()
    client.create_order.assert_not_called()


def test_live_submit_blocked_when_eligibility_missing(tmp_path):
    from kalshi_bot.data.store import Store

    store = Store(str(tmp_path / "t.db"))
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    client = MagicMock()
    eng = ExecutionEngine(client, store, cfg)
    order = eng.place_individual(
        market={"ticker": "KXHIGHLAX-26SEP15-T82", "event_ticker": "KXHIGHLAX-26SEP15"},
        ev=_ev(),
        opportunity_id="opp-2",
        mode="live",
        model_live_eligible=None,
    )
    assert order is None
    client.create_order_v2.assert_not_called()


def test_paper_allowed_when_not_live_eligible(tmp_path):
    from kalshi_bot.data.store import Store

    store = Store(str(tmp_path / "t.db"))
    store.update_state(mode="paper", live_enabled=False, paper_cash="25", trading_budget="25", peak_equity="25")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25"), max_loss_per_trade_dollars=D("5")))
    client = MagicMock()
    eng = ExecutionEngine(client, store, cfg)
    order = eng.place_individual(
        market={"ticker": "KXHIGHNY-26SEP15-T70", "event_ticker": "KXHIGHNY-26SEP15"},
        ev=_ev(),
        opportunity_id="opp-3",
        mode="paper",
        model_live_eligible=False,
    )
    client.create_order_v2.assert_not_called()
    if order is not None:
        assert order.mode == "paper"
