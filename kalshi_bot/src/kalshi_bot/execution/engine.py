from __future__ import annotations

import logging
import threading
import uuid
from decimal import Decimal
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OrderRecord, PositionRecord, Store, dumps, utcnow
from kalshi_bot.ev.calculator import EvResult, max_limit_price
from kalshi_bot.money import D, ZERO, fp_count, fp_price
from kalshi_bot.risk.limits import RiskManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, client: KalshiClient, store: Store, config: AppConfig) -> None:
        self.client = client
        self.store = store
        self.config = config
        self.risk = RiskManager(store, config.trading)
        self._lock = threading.RLock()

    def place_individual(
        self,
        *,
        market: dict[str, Any],
        ev: EvResult,
        opportunity_id: str,
        correlation_keys: list[str] | None = None,
        mode: str | None = None,
    ) -> OrderRecord | None:
        with self._lock:
            state = self.store.get_state()
            mode = mode or state.mode
            if mode == "live" and not state.live_enabled:
                self.store.audit("block", "live mode blocked: live_enabled is false")
                return None
            if mode == "research":
                self.store.audit("skip", "research mode — no orders")
                return None

            risk = self.risk.check_purchase(
                capital_required=ev.capital_required,
                max_loss=ev.max_loss,
                event_ticker=market.get("event_ticker") or "",
                kind="individual",
                correlation_keys=correlation_keys,
            )
            if not risk.allowed:
                self.store.audit("risk_block", "; ".join(risk.reasons), details={"opp": opportunity_id})
                return None

            # Recheck price bound from conservative EV
            limit = max_limit_price(
                ev.conservative_prob,
                ev.fees_per_contract,
                self.config.trading.min_net_edge,
                self.config.trading.uncertainty_buffer,
            )
            if ev.executable_price > limit:
                self.store.audit(
                    "price_block",
                    f"executable {ev.executable_price} > max limit {limit}",
                )
                return None

            client_order_id = str(uuid.uuid4())
            # Duplicate prevention: if opportunity already has an order, skip.
            for existing in self.store.list_orders(limit=500):
                if existing.get("opportunity_id") == opportunity_id and existing.get("status") not in (
                    "rejected",
                    "canceled",
                    "error",
                ):
                    self.store.audit("dup_block", f"order already exists for opportunity {opportunity_id}")
                    return None

            reservation_id = str(uuid.uuid4())
            self.store.create_reservation(reservation_id, ev.capital_required, client_order_id)

            # Deduct paper cash atomically with reservation
            if mode == "paper":
                cash = D(state.paper_cash)
                if cash < ev.capital_required:
                    self.store.release_reservation(reservation_id, "failed")
                    return None
                self.store.update_state(
                    paper_cash=str(cash - ev.capital_required),
                    paper_reserved=str(D(state.paper_reserved) + ev.capital_required),
                )

            order = OrderRecord(
                client_order_id=client_order_id,
                created_at=utcnow(),
                mode=mode,
                kind="individual",
                market_ticker=market["ticker"],
                event_ticker=market.get("event_ticker") or "",
                side=ev.side,
                quantity=str(ev.quantity),
                limit_price=str(min(ev.executable_price, limit)),
                status="pending",
                reservation_id=reservation_id,
                opportunity_id=opportunity_id,
                details_json=dumps({"risk": risk.capacity, "ev_reason": ev.reason}),
            )
            self.store.save_order(order)

            try:
                if mode == "paper":
                    return self._paper_fill(order, market, ev, correlation_keys or [])
                return self._live_submit(order, market, ev)
            except Exception as exc:
                logger.exception("order failed")
                order.status = "error"
                order.details_json = dumps({"error": str(exc)})
                self.store.save_order(order)
                self.store.release_reservation(reservation_id, "failed")
                if mode == "paper":
                    # refund
                    st = self.store.get_state()
                    self.store.update_state(
                        paper_cash=str(D(st.paper_cash) + ev.capital_required),
                        paper_reserved=str(max(ZERO, D(st.paper_reserved) - ev.capital_required)),
                    )
                self.store.audit("order_error", str(exc), level="error")
                return order

    def _paper_fill(
        self,
        order: OrderRecord,
        market: dict[str, Any],
        ev: EvResult,
        correlation_keys: list[str],
    ) -> OrderRecord:
        # Simulate immediate limit fill at executable price if depth existed at decision time.
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.limit_price
        order.fees_paid = str(ev.fees_total)
        order.exchange_order_id = f"paper-{order.client_order_id[:8]}"
        self.store.save_order(order)
        self.store.release_reservation(order.reservation_id, "consumed")
        st = self.store.get_state()
        self.store.update_state(
            paper_reserved=str(max(ZERO, D(st.paper_reserved) - ev.capital_required)),
        )
        pos = PositionRecord(
            id=str(uuid.uuid4()),
            opened_at=utcnow(),
            mode="paper",
            kind="individual",
            market_ticker=order.market_ticker,
            event_ticker=order.event_ticker,
            side=order.side,
            quantity=order.quantity,
            avg_price=order.avg_fill_price,
            fees_paid=order.fees_paid,
            status="open",
            correlation_keys_json=dumps(correlation_keys),
            details_json=dumps({"client_order_id": order.client_order_id}),
        )
        self.store.save_position(pos)
        self.store.audit("paper_fill", f"filled {order.market_ticker} {order.side} x{order.quantity}")
        return order

    def _live_submit(self, order: OrderRecord, market: dict[str, Any], ev: EvResult) -> OrderRecord:
        # V2 orders are on the YES book: bid = buy YES, ask = sell YES (= buy NO economically at 1-price).
        if order.side == "yes":
            book_side = "bid"
            price = order.limit_price
        else:
            book_side = "ask"
            # Selling YES at (1 - no_price) is equivalent to buying NO at no_price.
            price = str(fp_price(D("1") - D(order.limit_price)))

        body = {
            "ticker": order.market_ticker,
            "client_order_id": order.client_order_id,
            "side": book_side,
            "count": str(fp_count(order.quantity)),
            "price": price,
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": False,
        }
        # Before submit: reconcile duplicate client_order_id if prior uncertain submit.
        existing = self.store.get_order(order.client_order_id)
        if existing and existing.get("exchange_order_id"):
            order.status = "reconciled_existing"
            self.store.save_order(order)
            return order

        resp = self.client.create_order_v2(body)
        order.exchange_order_id = str(resp.get("order_id") or "")
        order.filled_quantity = str(resp.get("fill_count") or "0.00")
        remaining = D(resp.get("remaining_count") or order.quantity)
        if D(order.filled_quantity) > ZERO and remaining > ZERO:
            order.status = "partial"
        elif remaining <= ZERO:
            order.status = "filled"
        else:
            order.status = "resting"
        if resp.get("average_fill_price"):
            order.avg_fill_price = str(resp["average_fill_price"])
        if resp.get("average_fee_paid"):
            order.fees_paid = str(resp["average_fee_paid"])
        self.store.save_order(order)
        self.store.audit("live_order", f"submitted {order.client_order_id}", details={"resp_keys": list(resp.keys())})
        if order.status == "filled":
            self.store.release_reservation(order.reservation_id, "consumed")
            pos = PositionRecord(
                id=str(uuid.uuid4()),
                opened_at=utcnow(),
                mode="live",
                kind="individual",
                market_ticker=order.market_ticker,
                event_ticker=order.event_ticker,
                side=order.side,
                quantity=order.filled_quantity,
                avg_price=order.avg_fill_price or order.limit_price,
                fees_paid=order.fees_paid,
                status="open",
                details_json=dumps({"client_order_id": order.client_order_id}),
            )
            self.store.save_position(pos)
        return order

    def cancel_outstanding(self) -> int:
        orders = self.store.list_open_orders()
        n = 0
        for o in orders:
            if o["mode"] == "paper":
                fields = {k: o[k] for k in OrderRecord.__dataclass_fields__ if k in o}
                fields["status"] = "canceled"
                self.store.save_order(OrderRecord(**fields))
                if o.get("reservation_id"):
                    self.store.release_reservation(o["reservation_id"], "canceled")
                n += 1
            else:
                try:
                    self.client.cancel_order_v2(o.get("exchange_order_id") or o["client_order_id"], o["market_ticker"])
                    fields = {k: o[k] for k in OrderRecord.__dataclass_fields__ if k in o}
                    fields["status"] = "canceled"
                    self.store.save_order(OrderRecord(**fields))
                    n += 1
                except Exception as exc:
                    self.store.audit("cancel_error", str(exc), level="error")
        return n
