from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from kalshi_bot.config import TradingConfig
from kalshi_bot.data.store import Store
from kalshi_bot.risk.limits import RiskManager


def test_risk_blocks_over_budget(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    store.update_state(
        mode="paper",
        paper_cash="100",
        trading_budget="20",
        pause_buying=False,
        kill_switch=False,
    )
    rm = RiskManager(store, TradingConfig(budget_dollars=Decimal("20"), min_cash_reserve_dollars=Decimal("0")))
    d = rm.check_purchase(
        capital_required=Decimal("25"),
        max_loss=Decimal("25"),
        event_ticker="EVT",
        kind="individual",
    )
    assert not d.allowed


def test_kill_switch_blocks(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    store.update_state(kill_switch=True, paper_cash="100", trading_budget="100")
    rm = RiskManager(store, TradingConfig())
    d = rm.check_purchase(
        capital_required=Decimal("1"),
        max_loss=Decimal("1"),
        event_ticker="EVT",
        kind="individual",
    )
    assert not d.allowed
    assert "kill switch" in d.reasons[0].lower()


def test_duplicate_order_prevention(tmp_path: Path):
    from kalshi_bot.data.store import OrderRecord, utcnow

    store = Store(tmp_path / "t.db")
    oid = "opp-1"
    store.save_order(
        OrderRecord(
            client_order_id="c1",
            created_at=utcnow(),
            mode="paper",
            kind="individual",
            market_ticker="M",
            event_ticker="E",
            side="yes",
            quantity="1",
            limit_price="0.5",
            status="filled",
            opportunity_id=oid,
        )
    )
    existing = [o for o in store.list_orders() if o["opportunity_id"] == oid]
    assert len(existing) == 1
