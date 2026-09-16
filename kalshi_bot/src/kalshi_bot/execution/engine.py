from __future__ import annotations

import logging
import threading
import uuid
from decimal import Decimal
from typing import Any

from kalshi_bot.accounting.fills import NORMALIZATION_VERSION, acquisition_cost_dollars
from kalshi_bot.api.client import KalshiClient
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OrderRecord, PositionRecord, Store, dumps, utcnow
from kalshi_bot.ev.calculator import EvResult, max_limit_price
from kalshi_bot.money import D, ONE, ZERO, fp_count, fp_price
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
        model_live_eligible: bool | None = None,
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

            # Final order-submission boundary for live: require explicit eligibility.
            # Default None/False blocks live (tests passing / config alone never suffice).
            if mode == "live" and model_live_eligible is not True:
                self.store.audit(
                    "model_live_block",
                    "live submit blocked at execution boundary: model_live_eligible is not True",
                    details={"opportunity_id": opportunity_id, "model_live_eligible": model_live_eligible},
                )
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
            # Duplicate prevention: durable intent + existing orders for opportunity.
            prior_intent = self.store.get_order_intent_by_opportunity(opportunity_id)
            if prior_intent and prior_intent.get("client_order_id"):
                prior_order = self.store.get_order(prior_intent["client_order_id"])
                if prior_order and prior_order.get("status") in (
                    "ambiguous",
                    "submitted",
                    "pending",
                    "partial",
                    "resting",
                    "filled",
                ):
                    self.store.audit(
                        "dup_block",
                        f"reconcile existing intent before new submit ({opportunity_id})",
                        details={"client_order_id": prior_intent["client_order_id"]},
                    )
                    if prior_order.get("status") == "ambiguous":
                        return self.reconcile_live_order(prior_intent["client_order_id"])
                    fields = {
                        k: prior_order[k]
                        for k in OrderRecord.__dataclass_fields__
                        if k in prior_order
                    }
                    return OrderRecord(**fields)

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
            self.store.save_order_intent(
                opportunity_id,
                client_order_id=client_order_id,
                opportunity_id=opportunity_id,
                market_ticker=market["ticker"],
                side=ev.side,
                status="reserved",
                details={"capital_required": str(ev.capital_required)},
            )

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
                # Preserve ambiguous submit for idempotent reconcile (do not release exposure).
                current = self.store.get_order(client_order_id)
                if current and current.get("status") == "ambiguous":
                    fields = {
                        k: current[k] for k in OrderRecord.__dataclass_fields__ if k in current
                    }
                    return OrderRecord(**fields)
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

    def _normalize_acquisition_cost(self, order: OrderRecord, resp: dict[str, Any]) -> str:
        """Outcome-side acquisition cost (NO premium), not ambiguous YES-book average."""
        outcome = (resp.get("outcome_side") or order.side or "").lower()
        cost = acquisition_cost_dollars(
            outcome_side=outcome,
            yes_price_dollars=resp.get("yes_price_dollars") or resp.get("average_yes_price_dollars"),
            no_price_dollars=resp.get("no_price_dollars") or resp.get("average_no_price_dollars"),
            legacy_avg_fill=None,
        )
        if cost is not None:
            return str(fp_price(cost))
        raw_avg = resp.get("average_fill_price")
        if raw_avg not in (None, ""):
            raw = D(str(raw_avg))
            # V2 YES-book: NO buy submits as ask; average_fill_price is often YES price.
            if outcome == "no" and raw > ZERO:
                return str(fp_price(ONE - raw))
            return str(fp_price(raw))
        return order.limit_price

    def reconcile_live_order(self, client_order_id: str) -> OrderRecord | None:
        """Idempotent reconcile after ambiguous submit (timeout / unknown). No new create_order."""
        with self._lock:
            existing = self.store.get_order(client_order_id)
            if not existing:
                return None
            fields = {k: existing[k] for k in OrderRecord.__dataclass_fields__ if k in existing}
            order = OrderRecord(**fields)
            try:
                if order.exchange_order_id:
                    payload = self.client.get_orders(order_id=order.exchange_order_id)
                else:
                    payload = self.client.get_orders(client_order_id=client_order_id)
                orders = payload.get("orders") or []
                if isinstance(payload.get("order"), dict):
                    orders = [payload["order"]] + list(orders)
                match = next(
                    (
                        o
                        for o in orders
                        if str(o.get("client_order_id") or "") == client_order_id
                        or str(o.get("order_id") or "") == str(order.exchange_order_id or "")
                    ),
                    None,
                )
            except Exception as exc:
                self.store.audit("reconcile_error", str(exc), level="error")
                return order
            if not match:
                return order
            return self._apply_live_response(order, match, source="reconcile")

    def _apply_live_response(
        self, order: OrderRecord, resp: dict[str, Any], *, source: str
    ) -> OrderRecord:
        order.exchange_order_id = str(
            resp.get("order_id") or resp.get("exchange_order_id") or order.exchange_order_id or ""
        )
        filled = resp.get("fill_count") or resp.get("fill_count_fp") or order.filled_quantity or "0"
        order.filled_quantity = str(filled)
        remaining_raw = resp.get("remaining_count") or resp.get("remaining_count_fp")
        remaining = D(remaining_raw) if remaining_raw not in (None, "") else D(order.quantity) - D(
            order.filled_quantity
        )
        if D(order.filled_quantity) > ZERO and remaining > ZERO:
            order.status = "partial"
        elif remaining <= ZERO and D(order.filled_quantity) > ZERO:
            order.status = "filled"
        elif (resp.get("status") or "").lower() in ("canceled", "cancelled"):
            order.status = "canceled"
        else:
            order.status = "resting"
        order.avg_fill_price = self._normalize_acquisition_cost(order, resp)
        fee = (
            resp.get("average_fee_paid")
            or resp.get("taker_fees_dollars")
            or resp.get("fee_cost")
            or order.fees_paid
        )
        if fee not in (None, ""):
            order.fees_paid = str(fee)
        details = {}
        try:
            import json

            details = json.loads(order.details_json or "{}")
        except Exception:
            details = {}
        details["fill_normalization_version"] = NORMALIZATION_VERSION
        details["raw_exchange_response"] = {
            k: resp.get(k)
            for k in (
                "order_id",
                "status",
                "outcome_side",
                "book_side",
                "yes_price_dollars",
                "no_price_dollars",
                "average_fill_price",
                "fill_count",
                "fill_count_fp",
                "remaining_count",
                "remaining_count_fp",
                "taker_fees_dollars",
                "maker_fees_dollars",
            )
            if k in resp
        }
        details["apply_source"] = source
        order.details_json = dumps(details)
        self.store.save_order(order)
        if order.status in ("resting", "partial") and order.reservation_id:
            self.store.release_reservation(order.reservation_id, "linked_order")
        if order.status == "canceled" and order.reservation_id:
            self.store.release_reservation(order.reservation_id, "canceled")
        if order.status in ("filled", "partial") and D(order.filled_quantity) > ZERO:
            if order.status == "filled" and order.reservation_id:
                self.store.release_reservation(order.reservation_id, "consumed")
            # Only book the incremental filled quantity since last apply.
            prev_booked = D(str(details.get("booked_fill_quantity") or "0"))
            total_filled = D(order.filled_quantity)
            delta = total_filled - prev_booked
            if delta > ZERO:
                booked_order = OrderRecord(
                    client_order_id=order.client_order_id,
                    created_at=order.created_at,
                    mode=order.mode,
                    kind=order.kind,
                    market_ticker=order.market_ticker,
                    event_ticker=order.event_ticker,
                    side=order.side,
                    quantity=order.quantity,
                    limit_price=order.limit_price,
                    status=order.status,
                    exchange_order_id=order.exchange_order_id,
                    filled_quantity=str(fp_count(delta)),
                    avg_fill_price=order.avg_fill_price,
                    fees_paid=order.fees_paid if prev_booked <= ZERO else "0",
                    reservation_id=order.reservation_id,
                    opportunity_id=order.opportunity_id,
                    details_json=order.details_json,
                )
                self._upsert_live_position(booked_order)
                details["booked_fill_quantity"] = str(fp_count(total_filled))
                order.details_json = dumps(details)
                self.store.save_order(order)
        if order.opportunity_id:
            self.store.save_order_intent(
                order.opportunity_id,
                client_order_id=order.client_order_id,
                opportunity_id=order.opportunity_id,
                market_ticker=order.market_ticker,
                side=order.side,
                status=order.status,
            )
        return order

    def _upsert_live_position(self, order: OrderRecord) -> None:
        """Open or extend a same-side position; never overwrite an opposing side."""
        existing_same = None
        for p in self.store.list_positions(status="open"):
            if (
                p.get("market_ticker") == order.market_ticker
                and p.get("mode") == "live"
                and p.get("side") == order.side
            ):
                existing_same = p
                break
            if (
                p.get("market_ticker") == order.market_ticker
                and p.get("mode") == "live"
                and p.get("side") != order.side
            ):
                # Opposing side: keep both open; mark netting note only.
                self.store.audit(
                    "opposing_position",
                    f"{order.market_ticker} has open {p.get('side')} while filling {order.side}",
                    details={"existing_id": p.get("id"), "client_order_id": order.client_order_id},
                )
        acq = D(order.avg_fill_price or order.limit_price)
        fill_qty = D(order.filled_quantity)
        fees = D(order.fees_paid or 0)
        if existing_same:
            import json

            old_qty = D(existing_same["quantity"])
            old_avg = D(existing_same["avg_price"])
            new_qty = old_qty + fill_qty
            new_avg = ((old_avg * old_qty) + (acq * fill_qty)) / new_qty if new_qty > ZERO else acq
            prev_details: dict[str, Any] = {}
            try:
                prev_details = json.loads(existing_same.get("details_json") or "{}")
            except Exception:
                prev_details = {}
            ids = list(prev_details.get("client_order_ids") or [])
            if prev_details.get("client_order_id") and prev_details["client_order_id"] not in ids:
                ids.append(prev_details["client_order_id"])
            if order.client_order_id not in ids:
                ids.append(order.client_order_id)
            from kalshi_bot.data.store import PositionRecord as PR

            updated = PR(
                id=existing_same["id"],
                opened_at=existing_same["opened_at"],
                mode="live",
                kind=existing_same.get("kind") or "individual",
                market_ticker=order.market_ticker,
                event_ticker=order.event_ticker,
                side=order.side,
                quantity=str(fp_count(new_qty)),
                avg_price=str(fp_price(new_avg)),
                fees_paid=str(D(existing_same.get("fees_paid") or 0) + fees),
                status="open",
                correlation_keys_json=existing_same.get("correlation_keys_json") or "[]",
                details_json=dumps(
                    {
                        **prev_details,
                        "client_order_ids": ids,
                        "fill_normalization_version": NORMALIZATION_VERSION,
                    }
                ),
            )
            self.store.save_position(updated)
            return
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
            details_json=dumps(
                {
                    "client_order_id": order.client_order_id,
                    "client_order_ids": [order.client_order_id],
                    "fill_normalization_version": NORMALIZATION_VERSION,
                }
            ),
        )
        self.store.save_position(pos)

    def _live_submit(self, order: OrderRecord, market: dict[str, Any], ev: EvResult) -> OrderRecord:
        # V2 orders are on the YES book: bid = buy YES, ask = sell YES (= buy NO economically at 1-price).
        if order.side == "yes":
            book_side = "bid"
            price = order.limit_price
        else:
            book_side = "ask"
            # Selling YES at (1 - no_price) is equivalent to buying NO at no_price.
            price = str(fp_price(ONE - D(order.limit_price)))

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
        # Intent already persisted as pending OrderRecord. Reconcile before re-submit.
        existing = self.store.get_order(order.client_order_id)
        if existing and existing.get("exchange_order_id"):
            order.status = "reconciled_existing"
            order.exchange_order_id = existing["exchange_order_id"]
            self.store.save_order(order)
            return order

        order.status = "submitted"
        self.store.save_order(order)
        try:
            resp = self.client.create_order_v2(body)
        except Exception as exc:
            # Ambiguous network: leave submitted for reconcile_live_order; do not invent fills.
            order.status = "ambiguous"
            order.details_json = dumps({"error": str(exc), "intent": "await_reconcile"})
            self.store.save_order(order)
            self.store.audit(
                "live_submit_ambiguous",
                str(exc),
                level="error",
                details={"client_order_id": order.client_order_id},
            )
            raise
        order = self._apply_live_response(order, resp if isinstance(resp, dict) else {}, source="submit")
        self.store.audit(
            "live_order",
            f"submitted {order.client_order_id}",
            details={"status": order.status, "exchange_order_id": order.exchange_order_id},
        )
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
