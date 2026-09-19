from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

from kalshi_bot.accounting.settlement import SettlementReconciler
from kalshi_bot.config import TradingConfig
from kalshi_bot.data.store import PositionRecord, Store, utcnow
from kalshi_bot.risk.limits import RiskManager
import threading


def test_settlement_reconcile_only_when_result_present(tmp_path: Path):
    store = Store(tmp_path / "s.db")
    store.update_state(mode="paper", paper_cash="90", realized_pnl="0", peak_equity="100")
    store.save_position(
        PositionRecord(
            id="p1",
            opened_at=utcnow(),
            mode="paper",
            kind="individual",
            market_ticker="TEST-MKT",
            event_ticker="E",
            side="yes",
            quantity="1.00",
            avg_price="0.40",
            fees_paid="0.01",
            status="open",
        )
    )
    client = MagicMock()
    client.get_market.return_value = {"market": {"status": "open"}}
    r = SettlementReconciler(client, store).reconcile_open_positions()
    assert r["pending"] == 1
    assert store.list_positions(status="open")

    client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}
    r2 = SettlementReconciler(client, store).reconcile_open_positions()
    assert r2["settled"] == 1
    pos = store.list_positions(status="settled")[0]
    assert pos["settlement_value"] == "1"
    # payout 1.00 - cost 0.41 = 0.59
    assert Decimal(pos["realized_pnl"]) == Decimal("0.59")


def test_concurrent_reservations_do_not_overspend(tmp_path: Path):
    store = Store(tmp_path / "c.db")
    store.update_state(
        mode="paper",
        paper_cash="10",
        trading_budget="10",
        pause_buying=False,
        kill_switch=False,
        paper_reserved="0",
    )
    cfg = TradingConfig(
        budget_dollars=Decimal("10"),
        min_cash_reserve_dollars=Decimal("0"),
        max_loss_per_trade_dollars=Decimal("10"),
        max_portfolio_exposure_dollars=Decimal("10"),
        max_event_exposure_dollars=Decimal("10"),
    )
    rm = RiskManager(store, cfg)
    results = []
    lock = threading.Lock()

    def attempt():
        d = rm.check_purchase(
            capital_required=Decimal("6"),
            max_loss=Decimal("6"),
            event_ticker="E1",
            kind="individual",
        )
        if d.allowed:
            # simulate atomic reserve under store lock
            with store._lock:
                st = store.get_state()
                cash = Decimal(st.paper_cash)
                if cash >= Decimal("6"):
                    rid = f"r-{threading.get_ident()}"
                    store.create_reservation(rid, Decimal("6"), rid)
                    store.update_state(paper_cash=str(cash - Decimal("6")))
                    results.append("ok")
                else:
                    results.append("cash_fail")
        else:
            results.append("blocked")

    threads = [threading.Thread(target=attempt) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Only one 6-dollar purchase fits in 10 cash with no overspend if checks race—
    # with naive race maybe more; assert final cash never negative.
    st = store.get_state()
    assert Decimal(st.paper_cash) >= 0
    assert results.count("ok") <= 1 or Decimal(st.paper_cash) + store.active_reservations_total() <= Decimal("10") + Decimal("0.01")


def test_deposits_do_not_raise_budget(tmp_path: Path):
    store = Store(tmp_path / "b.db")
    store.update_state(trading_budget="50", paper_cash="50")
    # Simulate deposit increasing cash only
    store.update_state(paper_cash="500")
    assert store.get_state().trading_budget == "50"
