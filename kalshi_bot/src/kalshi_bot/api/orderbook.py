from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, ONE, ZERO, fp_count, fp_price


@dataclass
class BookLevel:
    price: Decimal
    size: Decimal


@dataclass
class ExecutableBook:
    """Derived executable prices from Kalshi bid-only orderbook_fp."""

    yes_bids: list[BookLevel] = field(default_factory=list)
    no_bids: list[BookLevel] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def best_yes_bid(self) -> Decimal | None:
        return self.yes_bids[-1].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Decimal | None:
        return self.no_bids[-1].price if self.no_bids else None

    @property
    def best_yes_ask(self) -> Decimal | None:
        # YES ask = 1 - best NO bid
        if self.best_no_bid is None:
            return None
        return fp_price(ONE - self.best_no_bid)

    @property
    def best_no_ask(self) -> Decimal | None:
        # NO ask = 1 - best YES bid
        if self.best_yes_bid is None:
            return None
        return fp_price(ONE - self.best_yes_bid)

    def yes_ask_depth(self, max_price: Decimal | None = None) -> list[BookLevel]:
        """Liquidity available to buy YES (from NO bids, descending aggressiveness)."""
        levels: list[BookLevel] = []
        for bid in reversed(self.no_bids):
            ask = fp_price(ONE - bid.price)
            if max_price is not None and ask > max_price:
                continue
            levels.append(BookLevel(price=ask, size=bid.size))
        return levels

    def no_ask_depth(self, max_price: Decimal | None = None) -> list[BookLevel]:
        levels: list[BookLevel] = []
        for bid in reversed(self.yes_bids):
            ask = fp_price(ONE - bid.price)
            if max_price is not None and ask > max_price:
                continue
            levels.append(BookLevel(price=ask, size=bid.size))
        return levels

    def fillable_yes(self, quantity: Decimal, max_price: Decimal) -> tuple[Decimal, Decimal]:
        """Return (fillable_qty, vwap) for buying YES up to max_price."""
        return _walk_asks(self.yes_ask_depth(max_price), quantity)

    def fillable_no(self, quantity: Decimal, max_price: Decimal) -> tuple[Decimal, Decimal]:
        return _walk_asks(self.no_ask_depth(max_price), quantity)


def _walk_asks(levels: list[BookLevel], quantity: Decimal) -> tuple[Decimal, Decimal]:
    remaining = fp_count(quantity)
    filled = ZERO
    cost = ZERO
    for level in levels:
        take = min(remaining, level.size)
        if take <= ZERO:
            continue
        cost += take * level.price
        filled += take
        remaining -= take
        if remaining <= ZERO:
            break
    if filled <= ZERO:
        return ZERO, ZERO
    return fp_count(filled), fp_price(cost / filled)


def parse_orderbook(payload: dict[str, Any]) -> ExecutableBook:
    ob = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    yes_raw = ob.get("yes_dollars") or ob.get("yes") or []
    no_raw = ob.get("no_dollars") or ob.get("no") or []

    def levels(raw: list[Any]) -> list[BookLevel]:
        out: list[BookLevel] = []
        for row in raw:
            if not row or len(row) < 2:
                continue
            out.append(BookLevel(price=fp_price(row[0]), size=fp_count(row[1])))
        out.sort(key=lambda x: x.price)
        return out

    return ExecutableBook(yes_bids=levels(yes_raw), no_bids=levels(no_raw), raw=payload)


def market_implied_yes_prob(book: ExecutableBook) -> Decimal | None:
    """Midpoint of best YES bid/ask when both exist; else None (no fabrication)."""
    bid = book.best_yes_bid
    ask = book.best_yes_ask
    if bid is None or ask is None:
        return None
    return fp_price((bid + ask) / 2)
