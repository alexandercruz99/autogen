from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_bot.config import TradingConfig
from kalshi_bot.data.store import Store
from kalshi_bot.money import D, ZERO


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    reserved_amount: Decimal = ZERO
    capacity: dict[str, str] = field(default_factory=dict)


class RiskManager:
    """Hard exposure caps. No martingale. Deposits do not raise trading_budget."""

    def __init__(self, store: Store, config: TradingConfig) -> None:
        self.store = store
        self.config = config

    def portfolio_snapshot(self) -> dict[str, Decimal]:
        state = self.store.get_state()
        open_positions = self.store.list_positions(status="open")
        open_orders = self.store.list_open_orders()
        reserved = self.store.active_reservations_total()

        position_exposure = ZERO
        event_exposure: dict[str, Decimal] = {}
        combo_exposure = ZERO
        for p in open_positions:
            notional = D(p["avg_price"]) * D(p["quantity"]) + D(p.get("fees_paid") or 0)
            position_exposure += notional
            event_exposure[p["event_ticker"]] = event_exposure.get(p["event_ticker"], ZERO) + notional
            if p.get("kind") == "combo":
                combo_exposure += notional

        order_exposure = ZERO
        for o in open_orders:
            order_exposure += D(o["limit_price"]) * D(o["quantity"])

        cash = D(state.paper_cash) if state.mode != "live" else D(state.paper_cash)
        # In live mode paper_cash field stores last known free cash snapshot for display.
        budget = D(state.trading_budget)
        realized = D(state.realized_pnl)
        daily = D(state.daily_realized_pnl)
        peak = D(state.peak_equity)
        equity = cash + position_exposure
        drawdown = max(ZERO, peak - equity)

        return {
            "cash": cash,
            "reserved": reserved,
            "available": cash - reserved,
            "budget": budget,
            "position_exposure": position_exposure,
            "order_exposure": order_exposure,
            "combo_exposure": combo_exposure,
            "portfolio_exposure": position_exposure + order_exposure + reserved,
            "realized_pnl": realized,
            "daily_realized_pnl": daily,
            "drawdown": drawdown,
            "equity": equity,
            **{f"event::{k}": v for k, v in event_exposure.items()},
        }

    def check_purchase(
        self,
        *,
        capital_required: Decimal,
        max_loss: Decimal,
        event_ticker: str,
        kind: str,
        correlation_keys: list[str] | None = None,
    ) -> RiskDecision:
        state = self.store.get_state()
        reasons: list[str] = []
        if state.kill_switch:
            return RiskDecision(False, ["kill switch active — new purchases paused"])
        if state.pause_buying:
            return RiskDecision(False, ["buying paused by operator"])

        snap = self.portfolio_snapshot()
        capital = D(capital_required)
        loss = D(max_loss)

        if capital > snap["budget"]:
            # Budget is the configured trading budget, not account deposits.
            # Also ensure cumulative exposure stays within budget.
            pass
        if snap["portfolio_exposure"] + capital > snap["budget"]:
            reasons.append(
                f"would exceed trading budget ({snap['portfolio_exposure']}+{capital} > {snap['budget']})"
            )
        if snap["available"] - capital < self.config.min_cash_reserve_dollars:
            reasons.append("insufficient available cash after min reserve")
        if loss > self.config.max_loss_per_trade_dollars:
            reasons.append("max loss exceeds per-trade limit")
        event_key = f"event::{event_ticker}"
        event_exp = snap.get(event_key, ZERO)
        if event_exp + capital > self.config.max_event_exposure_dollars:
            reasons.append("event exposure limit")
        if snap["portfolio_exposure"] + capital > self.config.max_portfolio_exposure_dollars:
            reasons.append("portfolio exposure limit")
        if kind == "combo" and snap["combo_exposure"] + capital > self.config.max_combo_exposure_dollars:
            reasons.append("combo exposure limit")
        if snap["daily_realized_pnl"] <= -self.config.max_daily_loss_dollars:
            reasons.append("daily loss limit reached — pausing new purchases")
        if snap["drawdown"] >= self.config.max_drawdown_dollars:
            reasons.append("drawdown limit reached — pausing new purchases")

        # Correlated group: treat shared keys as event-like caps (same limit).
        if correlation_keys:
            open_positions = self.store.list_positions(status="open")
            for key in correlation_keys:
                grouped = ZERO
                for p in open_positions:
                    import json

                    keys = json.loads(p.get("correlation_keys_json") or "[]")
                    if key in keys:
                        grouped += D(p["avg_price"]) * D(p["quantity"])
                if grouped + capital > self.config.max_event_exposure_dollars:
                    reasons.append(f"correlated group exposure limit ({key})")

        capacity = {
            "budget_remaining": str(snap["budget"] - snap["portfolio_exposure"]),
            "cash_available": str(snap["available"]),
            "event_remaining": str(
                self.config.max_event_exposure_dollars - snap.get(event_key, ZERO)
            ),
            "daily_loss_remaining": str(
                self.config.max_daily_loss_dollars + snap["daily_realized_pnl"]
            ),
        }
        if reasons:
            return RiskDecision(False, reasons, capacity=capacity)
        return RiskDecision(True, ["risk checks passed"], reserved_amount=capital, capacity=capacity)
