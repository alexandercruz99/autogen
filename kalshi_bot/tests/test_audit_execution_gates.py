"""Behavioral tests: live entry points never submit without eligibility / EV requalify."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from kalshi_bot.api.orderbook import BookLevel, ExecutableBook
from kalshi_bot.config import AppConfig, TradingConfig
from kalshi_bot.data.store import Store
from kalshi_bot.ev.calculator import EvResult
from kalshi_bot.execution.engine import ExecutionEngine
from kalshi_bot.models.weather.obs_engine.multi.live_bet import place_capped_live_bet
from kalshi_bot.money import D


def _book(*, yes_ask: str, depth: str = "100") -> ExecutableBook:
    """Build bid-only book where best YES ask = yes_ask (via NO bid = 1 - ask)."""
    no_bid = D("1") - D(yes_ask)
    yes_bid = D("0.01")
    return ExecutableBook(
        yes_bids=[BookLevel(price=yes_bid, size=D(depth))],
        no_bids=[BookLevel(price=no_bid, size=D(depth))],
        raw={},
    )


def _forecast_result(**overrides):
    base = {
        "probabilities_available": True,
        "model_live_eligible": False,
        "model_origin": "same_icao_transfer:lax_airport",
        "point_median_f": 81.0,
        "decision_hour_local": 11,
        "callout": "LAX test callout",
        "features_summary": {"max_so_far": 78.0},
        "brackets": [
            {
                "ticker": "KXHIGHLAX-T80",
                "interval": {"op": "gt", "low": 80.0, "high": None},
            }
        ],
        "ev_evaluations": [
            {
                "ticker": "KXHIGHLAX-T80",
                "side": "yes",
                "p": "0.60",
                "ask": "0.40",
                "ev": "0.15",
                "interval": {"op": "gt", "low": 80.0, "high": None},
                "decision": "ev_evaluated",
            }
        ],
        "force_decision": False,
        "location_id": "twc_lax",
        "series_ticker": "KXHIGHLAX",
    }
    base.update(overrides)
    return base


def test_create_order_v2_only_invoked_via_execution_engine():
    import kalshi_bot.models.weather.obs_engine.multi.live_bet as live_mod
    import inspect

    src = inspect.getsource(live_mod.place_capped_live_bet)
    assert "create_order_v2" not in src
    assert "ExecutionEngine" in src


def test_twc_live_ineligible_model_never_hits_exchange(tmp_path):
    store_path = str(tmp_path / "t.db")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    cfg.mode = "live"
    cfg.live.enabled = True
    cfg.storage.sqlite_path = store_path
    store = Store(store_path)
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25")

    client = MagicMock()
    with patch("kalshi_bot.models.weather.obs_engine.multi.live_bet.KalshiClient", return_value=client), patch(
        "kalshi_bot.models.weather.obs_engine.multi.live_bet.Store", return_value=store
    ):
        out = place_capped_live_bet(_forecast_result(model_live_eligible=False), cfg, dollars=5.0)
    assert out["ok"] is False
    assert out.get("live_order_submitted") is False
    client.create_order_v2.assert_not_called()
    client.get_orderbook.assert_not_called()


def test_force_decision_refuses_live(tmp_path):
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    cfg.mode = "live"
    cfg.live.enabled = True
    out = place_capped_live_bet(
        _forecast_result(model_live_eligible=True, force_decision=True),
        cfg,
        dollars=5.0,
        dry_run=False,
    )
    assert out["ok"] is False
    assert "research-only" in out["error"]
    assert out.get("live_order_submitted") is False


def test_requalify_rejects_when_ask_moves_040_to_090(tmp_path):
    """Required fixture: p=0.60, ask 0.40→0.90 → reject, zero exchange submissions."""
    store_path = str(tmp_path / "rq.db")
    cfg = AppConfig(
        trading=TradingConfig(
            budget_dollars=D("25"),
            min_net_edge=D("0.02"),
            uncertainty_buffer=D("0.05"),
            max_loss_per_trade_dollars=D("5"),
            target_trade_dollars=D("5"),
        )
    )
    cfg.mode = "live"
    cfg.live.enabled = True
    cfg.storage.sqlite_path = store_path
    store = Store(store_path)
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25", peak_equity="25")

    client = MagicMock()
    client.get_market.return_value = {"market": {"ticker": "KXHIGHLAX-T80", "status": "active", "event_ticker": "E"}}
    client.get_orderbook.return_value = {
        "orderbook": {
            "yes": [[1, 100]],
            "no": [[10, 100]],  # YES ask ≈ 0.90 via 1 - best no bid if parser uses that
        }
    }

    # Force parse_orderbook / evaluate path with a refreshed 0.90 ask.
    refreshed = _book(yes_ask="0.90")
    with patch("kalshi_bot.models.weather.obs_engine.multi.live_bet.KalshiClient", return_value=client), patch(
        "kalshi_bot.models.weather.obs_engine.multi.live_bet.Store", return_value=store
    ), patch(
        "kalshi_bot.models.weather.obs_engine.multi.live_bet.parse_orderbook", return_value=refreshed
    ):
        out = place_capped_live_bet(
            _forecast_result(model_live_eligible=True),
            cfg,
            dollars=5.0,
            dry_run=False,
        )
    assert out["ok"] is False
    assert out.get("error") == "requalify_failed_at_refreshed_price"
    assert out.get("live_order_submitted") is False
    client.create_order_v2.assert_not_called()
    assert D(out["refreshed_ask"]) == D("0.90")


def test_engine_blocks_ineligible_before_create_order_v2(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    client = MagicMock()
    eng = ExecutionEngine(client, store, cfg)
    ev = EvResult(
        side="yes",
        quantity=D("1"),
        executable_price=D("0.40"),
        fillable_quantity=D("1"),
        estimated_prob=D("0.60"),
        conservative_prob=D("0.55"),
        uncertainty=D("0.05"),
        fees_total=D("0.01"),
        fees_per_contract=D("0.01"),
        estimated_ev=D("0.19"),
        conservative_ev=D("0.09"),
        breakeven_prob=D("0.41"),
        max_loss=D("0.41"),
        capital_required=D("0.41"),
        qualifies=True,
        reason="test",
    )
    order = eng.place_individual(
        market={"ticker": "T", "event_ticker": "E"},
        ev=ev,
        opportunity_id="opp-ineligible",
        mode="live",
        model_live_eligible=False,
    )
    assert order is None
    client.create_order_v2.assert_not_called()


def test_closed_market_rejects_before_submit(tmp_path):
    store_path = str(tmp_path / "c.db")
    cfg = AppConfig(trading=TradingConfig(budget_dollars=D("25")))
    cfg.mode = "live"
    cfg.live.enabled = True
    cfg.storage.sqlite_path = store_path
    store = Store(store_path)
    store.update_state(mode="live", live_enabled=True, paper_cash="25", trading_budget="25")
    client = MagicMock()
    client.get_market.return_value = {"market": {"ticker": "KXHIGHLAX-T80", "status": "closed"}}
    with patch("kalshi_bot.models.weather.obs_engine.multi.live_bet.KalshiClient", return_value=client), patch(
        "kalshi_bot.models.weather.obs_engine.multi.live_bet.Store", return_value=store
    ):
        out = place_capped_live_bet(_forecast_result(model_live_eligible=True), cfg, dollars=5.0)
    assert out["ok"] is False
    assert "not tradable" in out["error"]
    client.create_order_v2.assert_not_called()
