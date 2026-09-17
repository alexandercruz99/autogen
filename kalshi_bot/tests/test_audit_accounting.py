"""Fill normalization, partial fills, opposing positions, settlement idempotency."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from kalshi_bot.accounting.fills import (
    NORMALIZATION_VERSION,
    acquisition_cost_dollars,
    historical_fixture_expectations,
    net_realized_from_parts,
    normalize_fill,
)
from kalshi_bot.accounting.settlement import SettlementReconciler
from kalshi_bot.config import AppConfig, TradingConfig
from kalshi_bot.data.store import PositionRecord, Store, utcnow
from kalshi_bot.ev.calculator import EvResult
from kalshi_bot.execution.engine import ExecutionEngine
from kalshi_bot.money import D


def test_no_side_acquisition_cost_uses_no_price_not_yes_book():
    # NYC NO ~$0.29; local average showed $0.71 (YES book).
    cost = acquisition_cost_dollars(
        outcome_side="no",
        yes_price_dollars="0.71",
        no_price_dollars="0.29",
    )
    assert cost == D("0.29")
    # Chicago NO ~$0.03 vs YES book 0.97
    cost2 = acquisition_cost_dollars(
        outcome_side="no",
        yes_price_dollars="0.97",
        no_price_dollars="0.03",
    )
    assert cost2 == D("0.03")


def test_normalize_fill_preserves_raw_and_version():
    n = normalize_fill(
        {
            "fill_id": "f1",
            "outcome_side": "no",
            "book_side": "ask",
            "yes_price_dollars": "0.71",
            "no_price_dollars": "0.29",
            "count_fp": "10.00",
            "fee_cost": "0.01",
        }
    )
    assert n["normalization_version"] == NORMALIZATION_VERSION
    assert n["acquisition_cost_per_contract"] == "0.29"
    assert n["raw"]["yes_price_dollars"] == "0.71"


def test_historical_fixture_expectations():
    fx = historical_fixture_expectations()
    assert fx["sep15_nyc_t70"]["approx_net_pnl"] == "-4.9000"
    assert fx["sep15_lax_t82"]["approx_net_pnl"] == "-0.6352"
    nyc = fx["sep16_nyc_b77_5"]
    net = net_realized_from_parts(
        gross_realized=nyc["gross_realized_pnl_dollars"],
        fees_paid=nyc["fees_paid_dollars"],
        fees_already_in_gross=False,
    )
    assert net == D("2.9822")


def test_engine_normalizes_no_fill_avg_price(tmp_path):
    store = Store(str(tmp_path / "n.db"))
    store.update_state(
        mode="live", live_enabled=True, paper_cash="25", trading_budget="25", peak_equity="25"
    )
    cfg = AppConfig(
        trading=TradingConfig(
            budget_dollars=D("25"),
            max_loss_per_trade_dollars=D("5"),
            min_net_edge=D("0.01"),
            uncertainty_buffer=D("0.01"),
            min_cash_reserve_dollars=D("0"),
            max_portfolio_exposure_dollars=D("25"),
            max_event_exposure_dollars=D("25"),
        )
    )
    client = MagicMock()
    client.create_order_v2.return_value = {
        "order_id": "ex-1",
        "fill_count": "10.00",
        "remaining_count": "0.00",
        "outcome_side": "no",
        "book_side": "ask",
        "yes_price_dollars": "0.71",
        "no_price_dollars": "0.29",
        "average_fill_price": "0.71",
        "average_fee_paid": "0.01",
    }
    eng = ExecutionEngine(client, store, cfg)
    ev = EvResult(
        side="no",
        quantity=D("10"),
        executable_price=D("0.29"),
        fillable_quantity=D("10"),
        estimated_prob=D("0.40"),
        conservative_prob=D("0.35"),
        uncertainty=D("0.05"),
        fees_total=D("0.01"),
        fees_per_contract=D("0.001"),
        estimated_ev=D("0.10"),
        conservative_ev=D("0.05"),
        breakeven_prob=D("0.30"),
        max_loss=D("2.90"),
        capital_required=D("2.91"),
        qualifies=True,
        reason="test",
    )
    order = eng.place_individual(
        market={"ticker": "KXHIGHNY-B77.5", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-no-norm",
        mode="live",
        model_live_eligible=True,
    )
    assert order is not None
    assert D(order.avg_fill_price) == D("0.29")
    pos = store.list_positions(status="open")[0]
    assert pos["side"] == "no"
    assert D(pos["avg_price"]) == D("0.29")


def test_partial_then_additional_fill(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    store.update_state(
        mode="live", live_enabled=True, paper_cash="25", trading_budget="25", peak_equity="25"
    )
    cfg = AppConfig(
        trading=TradingConfig(
            budget_dollars=D("25"),
            max_loss_per_trade_dollars=D("5"),
            min_net_edge=D("0.01"),
            uncertainty_buffer=D("0.01"),
            min_cash_reserve_dollars=D("0"),
            max_portfolio_exposure_dollars=D("25"),
            max_event_exposure_dollars=D("25"),
        )
    )
    client = MagicMock()
    client.create_order_v2.return_value = {
        "order_id": "ex-p",
        "fill_count": "5.00",
        "remaining_count": "5.00",
        "outcome_side": "yes",
        "yes_price_dollars": "0.40",
        "no_price_dollars": "0.60",
        "average_fee_paid": "0.01",
    }
    eng = ExecutionEngine(client, store, cfg)
    ev = EvResult(
        side="yes",
        quantity=D("10"),
        executable_price=D("0.40"),
        fillable_quantity=D("10"),
        estimated_prob=D("0.55"),
        conservative_prob=D("0.50"),
        uncertainty=D("0.05"),
        fees_total=D("0.02"),
        fees_per_contract=D("0.002"),
        estimated_ev=D("0.10"),
        conservative_ev=D("0.05"),
        breakeven_prob=D("0.41"),
        max_loss=D("4.00"),
        capital_required=D("4.02"),
        qualifies=True,
        reason="test",
    )
    order = eng.place_individual(
        market={"ticker": "T-PARTIAL", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-partial",
        mode="live",
        model_live_eligible=True,
    )
    assert order is not None
    assert order.status == "partial"
    assert D(store.list_positions(status="open")[0]["quantity"]) == D("5.00")

    client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "ex-p",
                "client_order_id": order.client_order_id,
                "fill_count_fp": "10.00",
                "remaining_count_fp": "0.00",
                "outcome_side": "yes",
                "yes_price_dollars": "0.40",
                "no_price_dollars": "0.60",
                "taker_fees_dollars": "0.02",
            }
        ]
    }
    order2 = eng.reconcile_live_order(order.client_order_id)
    assert order2 is not None
    assert order2.status == "filled"
    pos = store.list_positions(status="open")[0]
    assert D(pos["quantity"]) == D("10.00")


def test_ambiguous_submit_reconciles_without_second_create(tmp_path):
    store = Store(str(tmp_path / "a.db"))
    store.update_state(
        mode="live", live_enabled=True, paper_cash="25", trading_budget="25", peak_equity="25"
    )
    cfg = AppConfig(
        trading=TradingConfig(
            budget_dollars=D("25"),
            max_loss_per_trade_dollars=D("5"),
            min_net_edge=D("0.01"),
            uncertainty_buffer=D("0.01"),
            min_cash_reserve_dollars=D("0"),
            max_portfolio_exposure_dollars=D("25"),
            max_event_exposure_dollars=D("25"),
        )
    )
    client = MagicMock()
    client.create_order_v2.side_effect = TimeoutError("network")
    eng = ExecutionEngine(client, store, cfg)
    ev = EvResult(
        side="yes",
        quantity=D("1"),
        executable_price=D("0.40"),
        fillable_quantity=D("1"),
        estimated_prob=D("0.55"),
        conservative_prob=D("0.50"),
        uncertainty=D("0.05"),
        fees_total=D("0.01"),
        fees_per_contract=D("0.01"),
        estimated_ev=D("0.10"),
        conservative_ev=D("0.05"),
        breakeven_prob=D("0.41"),
        max_loss=D("0.41"),
        capital_required=D("0.41"),
        qualifies=True,
        reason="test",
    )
    order = eng.place_individual(
        market={"ticker": "T-AMB", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-amb",
        mode="live",
        model_live_eligible=True,
    )
    assert order is not None
    assert order.status in ("error", "ambiguous")
    assert client.create_order_v2.call_count == 1

    client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "ex-amb",
                "client_order_id": order.client_order_id,
                "fill_count_fp": "1.00",
                "remaining_count_fp": "0.00",
                "outcome_side": "yes",
                "yes_price_dollars": "0.40",
            }
        ]
    }
    again = eng.place_individual(
        market={"ticker": "T-AMB", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-amb",
        mode="live",
        model_live_eligible=True,
    )
    assert client.create_order_v2.call_count == 1
    assert again is not None


def test_opposing_positions_not_overwritten(tmp_path):
    store = Store(str(tmp_path / "o.db"))
    store.save_position(
        PositionRecord(
            id="yes-pos",
            opened_at=utcnow(),
            mode="live",
            kind="individual",
            market_ticker="T-OPP",
            event_ticker="E",
            side="yes",
            quantity="5.00",
            avg_price="0.40",
            fees_paid="0.01",
            status="open",
        )
    )
    store.update_state(
        mode="live", live_enabled=True, paper_cash="25", trading_budget="25", peak_equity="25"
    )
    cfg = AppConfig(
        trading=TradingConfig(
            budget_dollars=D("25"),
            max_loss_per_trade_dollars=D("5"),
            min_net_edge=D("0.01"),
            uncertainty_buffer=D("0.01"),
            min_cash_reserve_dollars=D("0"),
            max_portfolio_exposure_dollars=D("25"),
            max_event_exposure_dollars=D("25"),
        )
    )
    client = MagicMock()
    client.create_order_v2.return_value = {
        "order_id": "ex-no",
        "fill_count": "3.00",
        "remaining_count": "0",
        "outcome_side": "no",
        "yes_price_dollars": "0.70",
        "no_price_dollars": "0.30",
    }
    eng = ExecutionEngine(client, store, cfg)
    ev = EvResult(
        side="no",
        quantity=D("3"),
        executable_price=D("0.30"),
        fillable_quantity=D("3"),
        estimated_prob=D("0.40"),
        conservative_prob=D("0.35"),
        uncertainty=D("0.05"),
        fees_total=D("0.01"),
        fees_per_contract=D("0.003"),
        estimated_ev=D("0.05"),
        conservative_ev=D("0.02"),
        breakeven_prob=D("0.31"),
        max_loss=D("0.90"),
        capital_required=D("0.91"),
        qualifies=True,
        reason="test",
    )
    eng.place_individual(
        market={"ticker": "T-OPP", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-opp",
        mode="live",
        model_live_eligible=True,
    )
    open_pos = store.list_positions(status="open")
    sides = {p["side"] for p in open_pos}
    assert sides == {"yes", "no"}
    yes_p = next(p for p in open_pos if p["side"] == "yes")
    assert D(yes_p["quantity"]) == D("5.00")


def test_settlement_idempotent_and_portfolio_api(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    store.update_state(
        mode="live", paper_cash="0", realized_pnl="0", daily_realized_pnl="0", peak_equity="0"
    )
    store.save_position(
        PositionRecord(
            id="p-live",
            opened_at=utcnow(),
            mode="live",
            kind="individual",
            market_ticker="KXHIGHNY-26SEP15-T70",
            event_ticker="KXHIGHNY-26SEP15",
            side="yes",
            quantity="458.24",
            avg_price="0.0100",
            fees_paid="0.3176",
            status="open",
        )
    )
    client = MagicMock()
    client.get_market.return_value = {"market": {"status": "open"}}
    client.iter_settlements.return_value = [
        {
            "ticker": "KXHIGHNY-26SEP15-T70",
            "market_result": "no",
            "yes_count_fp": "458.24",
            "no_count_fp": "0",
            "revenue": 0,
            "fee_cost": "0.3176",
            "settled_time": "2026-09-16T00:00:00Z",
            "value": 0,
        }
    ]
    r = SettlementReconciler(client, store)
    out1 = r.reconcile_open_positions()
    assert out1["settled"] == 1
    pos = store.list_positions(status="settled")[0]
    assert Decimal(pos["realized_pnl"]) == Decimal("-4.9000")
    out2 = r.reconcile_open_positions()
    assert out2["settled"] == 0
    assert out2["skipped_already_settled"] >= 1
    assert r.load_checkpoint() is not None


def test_paper_purchase_through_fixture_settlement(tmp_path):
    store = Store(str(tmp_path / "paper.db"))
    store.update_state(
        mode="paper", paper_cash="10", realized_pnl="0", peak_equity="10", daily_realized_pnl="0"
    )
    store.save_position(
        PositionRecord(
            id="paper1",
            opened_at=utcnow(),
            mode="paper",
            kind="individual",
            market_ticker="PAPER-MKT",
            event_ticker="E",
            side="yes",
            quantity="1.00",
            avg_price="0.40",
            fees_paid="0.01",
            status="open",
        )
    )
    client = MagicMock()
    client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}
    client.iter_settlements.return_value = []
    out = SettlementReconciler(client, store).reconcile_open_positions()
    assert out["settled"] == 1
    assert Decimal(store.list_positions(status="settled")[0]["realized_pnl"]) == Decimal("0.59")
