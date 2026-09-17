from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.fees import estimate_net_fee
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OrderRecord, PositionRecord, Store, dumps, utcnow
from kalshi_bot.ev.combo import ComboLeg, combo_ev, joint_probability
from kalshi_bot.money import D, ZERO, fp_price
from kalshi_bot.risk.limits import RiskManager

logger = logging.getLogger(__name__)


class ComboRFQExecutor:
    """RFQ lifecycle for combo (MVE) markets per Kalshi docs.

    Flow: create combo market → create RFQ → wait for quotes → accept if within max price.
    Paper mode simulates a quote only when explicitly provided (never invents prices).
    """

    def __init__(self, client: KalshiClient, store: Store, config: AppConfig) -> None:
        self.client = client
        self.store = store
        self.config = config
        self.risk = RiskManager(store, config.trading)

    def evaluate_candidate(
        self,
        legs: list[ComboLeg],
        *,
        quoted_yes_price: Decimal | None,
        quantity: Decimal,
        dependence_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        joint = joint_probability(
            legs,
            allow_independence=self.config.combos.allow_independence_assumption,
            dependence_model=dependence_model,
        )
        if not joint.supported or quoted_yes_price is None:
            return {
                "qualifies": False,
                "reason": joint.skip_reason or "no quote available (will not invent combo price)",
                "joint": joint,
            }
        fees = estimate_net_fee(
            quantity,
            quoted_yes_price,
            multiplier=self.config.trading.fee_multiplier,
            assume_taker=True,
            balance_precision=self.config.trading.balance_precision,
        )
        est = combo_ev(joint.p_all, quoted_yes_price, fees / quantity)
        cons = combo_ev(joint.p_all_conservative, quoted_yes_price, fees / quantity)
        cons -= self.config.trading.uncertainty_buffer
        capital = quoted_yes_price * quantity + fees
        qualifies = cons >= self.config.trading.min_net_edge
        return {
            "qualifies": qualifies,
            "reason": (
                f"conservative combo EV {cons}"
                if qualifies
                else f"combo EV {cons} below min edge / unsupported"
            ),
            "joint": joint,
            "estimated_ev": est,
            "conservative_ev": cons,
            "fees": fees,
            "capital": capital,
            "price": quoted_yes_price,
            "quantity": quantity,
        }

    def paper_execute(
        self,
        *,
        market_ticker: str,
        event_ticker: str,
        evaluation: dict[str, Any],
        opportunity_id: str,
    ) -> OrderRecord | None:
        if not evaluation.get("qualifies"):
            return None
        risk = self.risk.check_purchase(
            capital_required=D(evaluation["capital"]),
            max_loss=D(evaluation["capital"]),
            event_ticker=event_ticker,
            kind="combo",
        )
        if not risk.allowed:
            self.store.audit("combo_risk_block", "; ".join(risk.reasons))
            return None
        state = self.store.get_state()
        if state.mode != "paper":
            self.store.audit("combo_skip", "live combo RFQ requires credentials and explicit live enablement")
            return None

        client_order_id = str(uuid.uuid4())
        reservation_id = str(uuid.uuid4())
        capital = D(evaluation["capital"])
        self.store.create_reservation(reservation_id, capital, client_order_id)
        cash = D(state.paper_cash)
        if cash < capital:
            self.store.release_reservation(reservation_id, "failed")
            return None
        self.store.update_state(paper_cash=str(cash - capital))

        order = OrderRecord(
            client_order_id=client_order_id,
            created_at=utcnow(),
            mode="paper",
            kind="combo",
            market_ticker=market_ticker,
            event_ticker=event_ticker,
            side="yes",
            quantity=str(evaluation["quantity"]),
            limit_price=str(evaluation["price"]),
            status="filled",
            exchange_order_id=f"paper-rfq-{client_order_id[:8]}",
            filled_quantity=str(evaluation["quantity"]),
            avg_fill_price=str(evaluation["price"]),
            fees_paid=str(evaluation["fees"]),
            reservation_id=reservation_id,
            opportunity_id=opportunity_id,
            details_json=dumps({"note": "paper RFQ simulation using provided quote only"}),
        )
        self.store.save_order(order)
        self.store.release_reservation(reservation_id, "consumed")
        self.store.save_position(
            PositionRecord(
                id=str(uuid.uuid4()),
                opened_at=utcnow(),
                mode="paper",
                kind="combo",
                market_ticker=market_ticker,
                event_ticker=event_ticker,
                side="yes",
                quantity=str(evaluation["quantity"]),
                avg_price=str(evaluation["price"]),
                fees_paid=str(evaluation["fees"]),
                status="open",
                details_json=dumps({"client_order_id": client_order_id}),
            )
        )
        return order

    def live_rfq_buy(
        self,
        *,
        market_ticker: str,
        quantity: Decimal,
        max_yes_price: Decimal,
    ) -> dict[str, Any]:
        """Create RFQ and accept a qualifying quote. Does not auto-fallback to legs."""
        state = self.store.get_state()
        if not state.live_enabled or state.mode != "live":
            return {"ok": False, "reason": "live trading not enabled"}
        if state.kill_switch or state.pause_buying:
            return {"ok": False, "reason": "purchases paused"}

        rfq = self.client.create_rfq(
            {
                "market_ticker": market_ticker,
                "contracts_fp": str(quantity),
                "rest_remainder": False,
            }
        )
        rfq_id = rfq.get("id") or rfq.get("rfq_id")
        if not rfq_id:
            return {"ok": False, "reason": "RFQ create returned no id", "raw_keys": list(rfq.keys())}

        deadline = time.time() + self.config.combos.rfq_wait_seconds
        best: dict[str, Any] | None = None
        while time.time() < deadline:
            quotes = self.client.get_quotes(rfq_id=str(rfq_id))
            for q in quotes.get("quotes") or quotes.get("quote") or []:
                yes_bid = q.get("yes_bid") or q.get("yes_bid_dollars")
                if yes_bid in (None, "0", 0, "0.0000"):
                    continue
                price = fp_price(yes_bid)
                if price <= max_yes_price and (best is None or price < best["price"]):
                    best = {"quote": q, "price": price, "quote_id": q.get("id") or q.get("quote_id")}
            if best:
                break
            time.sleep(self.config.combos.rfq_poll_seconds)

        if not best:
            try:
                self.client.delete_rfq(str(rfq_id))
            except Exception:
                pass
            return {"ok": False, "reason": "no valid quote within wait window"}

        # Accept YES side of quote
        accept = self.client.accept_quote(
            str(rfq_id),
            str(best["quote_id"]),
            {"accepted_side": "yes"},
        )
        self.store.audit(
            "rfq_accept",
            f"accepted quote for {market_ticker} at {best['price']}",
            details={"rfq_id": rfq_id, "quote_id": best["quote_id"]},
        )
        return {"ok": True, "accept": accept, "price": str(best["price"]), "rfq_id": rfq_id}
