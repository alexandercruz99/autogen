from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, ZERO, fp_price


class RfqState(enum.Enum):
    CREATED = "created"
    QUOTED = "quoted"
    ACCEPTED = "accepted"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    EXPIRED = "expired"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    CANCELED = "canceled"


@dataclass
class RfqSession:
    rfq_id: str
    market_ticker: str
    quantity: Decimal
    state: RfqState = RfqState.CREATED
    quotes: list[dict[str, Any]] = field(default_factory=list)
    accepted_quote_id: str | None = None
    accepted_side: str | None = None
    accepted_price: Decimal | None = None
    client_intent_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class PaperRfqFsm:
    """Documented RFQ lifecycle for paper/demo without exchange makers.

    Fixture quotes are explicitly labeled — not market evidence of profitability.
    Funds should remain reserved while state is ACCEPTED/CONFIRMED/UNCERTAIN.
    """

    def __init__(self, wait_seconds: float = 15.0, hvm: bool = True) -> None:
        self.wait_seconds = wait_seconds
        self.hvm = hvm
        self.sessions: dict[str, RfqSession] = {}

    def create(self, market_ticker: str, quantity: Decimal, client_intent_id: str) -> RfqSession:
        # Duplicate open RFQ prevention (409 semantics)
        for s in self.sessions.values():
            if (
                s.market_ticker == market_ticker
                and s.state in (RfqState.CREATED, RfqState.QUOTED, RfqState.ACCEPTED, RfqState.CONFIRMED, RfqState.UNCERTAIN)
            ):
                raise RuntimeError("409 Conflict: open RFQ already exists on this market ticker")
        rfq_id = str(uuid.uuid4())
        sess = RfqSession(
            rfq_id=rfq_id,
            market_ticker=market_ticker,
            quantity=quantity,
            client_intent_id=client_intent_id or str(uuid.uuid4()),
        )
        self.sessions[rfq_id] = sess
        return sess

    def ingest_fixture_quote(
        self,
        rfq_id: str,
        yes_bid: Decimal,
        no_bid: Decimal,
        *,
        label: str = "FIXTURE_QUOTE_NOT_LIVE_EVIDENCE",
    ) -> dict[str, Any]:
        sess = self.sessions[rfq_id]
        if sess.state not in (RfqState.CREATED, RfqState.QUOTED):
            raise RuntimeError(f"cannot quote in state {sess.state}")
        if yes_bid + no_bid > D("1"):
            raise RuntimeError("invalid quote: yes_bid + no_bid > 1")
        q = {
            "id": str(uuid.uuid4()),
            "yes_bid": str(fp_price(yes_bid)),
            "no_bid": str(fp_price(no_bid)),
            "label": label,
            "received_at": time.time(),
        }
        sess.quotes = [q]  # replace prior maker quote
        sess.state = RfqState.QUOTED
        return q

    def accept(self, rfq_id: str, quote_id: str, side: str, max_price: Decimal) -> RfqSession:
        sess = self.sessions[rfq_id]
        if sess.state != RfqState.QUOTED:
            raise RuntimeError(f"accept invalid in {sess.state}")
        quote = next((q for q in sess.quotes if q["id"] == quote_id), None)
        if not quote:
            raise RuntimeError("quote not found / expired")
        price = D(quote["yes_bid"] if side == "yes" else quote["no_bid"])
        if price <= ZERO:
            raise RuntimeError("side declined by quoter")
        if price > max_price:
            sess.state = RfqState.REJECTED
            sess.details["reject"] = f"price {price} > max {max_price}"
            return sess
        sess.accepted_quote_id = quote_id
        sess.accepted_side = side
        sess.accepted_price = price
        sess.state = RfqState.ACCEPTED
        return sess

    def confirm_maker(self, rfq_id: str, within_window: bool = True) -> RfqSession:
        sess = self.sessions[rfq_id]
        if sess.state != RfqState.ACCEPTED:
            raise RuntimeError(f"confirm invalid in {sess.state}")
        window = 3.0 if self.hvm else 30.0
        if not within_window:
            sess.state = RfqState.EXPIRED
            return sess
        sess.state = RfqState.CONFIRMED
        sess.details["confirm_window_s"] = window
        return sess

    def execute(self, rfq_id: str, *, fill_fraction: Decimal = D("1"), uncertain: bool = False) -> RfqSession:
        sess = self.sessions[rfq_id]
        if sess.state != RfqState.CONFIRMED:
            raise RuntimeError(f"execute invalid in {sess.state}")
        if uncertain:
            sess.state = RfqState.UNCERTAIN
            return sess
        sess.details["filled_qty"] = str(sess.quantity * fill_fraction)
        sess.details["partial"] = fill_fraction < D("1")
        sess.state = RfqState.EXECUTED
        return sess

    def expire_if_needed(self, rfq_id: str) -> RfqSession:
        sess = self.sessions[rfq_id]
        if sess.state in (RfqState.CREATED, RfqState.QUOTED) and time.time() - sess.created_at > self.wait_seconds:
            sess.state = RfqState.EXPIRED
        return sess

    def requires_fund_reservation(self, state: RfqState) -> bool:
        return state in (
            RfqState.ACCEPTED,
            RfqState.CONFIRMED,
            RfqState.UNCERTAIN,
            RfqState.QUOTED,  # optional; we reserve on accept
        )
