"""Normalize Kalshi fill/order prices to outcome-side acquisition cost.

Kalshi quotes both yes_price_dollars and no_price_dollars on every fill (they sum to 1).
Buying NO is often submitted as selling YES on the book; raw average_fill_price may be the
YES book price (e.g. 0.71) while the economic acquisition cost of NO is no_price (0.29).

Prefer outcome_side / book_side when present (current API). Legacy side+action are fallbacks.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, ONE, ZERO

NORMALIZATION_VERSION = "fill_price_norm.v1"


def acquisition_cost_dollars(
    *,
    outcome_side: str,
    yes_price_dollars: Any = None,
    no_price_dollars: Any = None,
    legacy_avg_fill: Any = None,
) -> Decimal | None:
    """Return premium paid per contract for the purchased outcome side."""
    side = (outcome_side or "").lower()
    yes_p = D(str(yes_price_dollars)) if yes_price_dollars not in (None, "") else None
    no_p = D(str(no_price_dollars)) if no_price_dollars not in (None, "") else None
    if side == "yes":
        if yes_p is not None:
            return yes_p
        if no_p is not None:
            return ONE - no_p
    if side == "no":
        if no_p is not None:
            return no_p
        if yes_p is not None:
            return ONE - yes_p
    if legacy_avg_fill not in (None, ""):
        # Ambiguous — caller must not treat this as outcome cost without side context.
        return D(str(legacy_avg_fill))
    return None


def normalize_fill(fill: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized fill view; preserve raw fields under raw_*."""
    outcome = (fill.get("outcome_side") or "").lower()
    book = (fill.get("book_side") or "").lower()
    # Legacy: side=yes|no with action=buy|sell. Prefer outcome_side.
    if not outcome:
        legacy_side = (fill.get("side") or "").lower()
        action = (fill.get("action") or "").lower()
        if legacy_side in ("yes", "no"):
            # Buying NO often appears as side=no action=sell (sell YES) or similar.
            if action == "buy":
                outcome = legacy_side
            elif action == "sell" and legacy_side == "no":
                outcome = "no"
            elif action == "sell" and legacy_side == "yes":
                # Selling YES closes YES or opens NO depending on prior position — keep legacy_side
                # as book hint only; economic side must come from order intent when ambiguous.
                outcome = legacy_side
            else:
                outcome = legacy_side

    yes_p = fill.get("yes_price_dollars")
    no_p = fill.get("no_price_dollars")
    cost = acquisition_cost_dollars(
        outcome_side=outcome or "yes",
        yes_price_dollars=yes_p,
        no_price_dollars=no_p,
        legacy_avg_fill=fill.get("average_fill_price") or fill.get("avg_fill_price"),
    )
    qty = D(str(fill.get("count_fp") or fill.get("count") or "0"))
    fee = D(str(fill.get("fee_cost") or fill.get("fee_cost_dollars") or "0"))
    premium = (cost or ZERO) * qty
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "fill_id": fill.get("fill_id") or fill.get("trade_id"),
        "order_id": fill.get("order_id"),
        "ticker": fill.get("ticker") or fill.get("market_ticker"),
        "outcome_side": outcome,
        "book_side": book or None,
        "quantity": str(qty),
        "yes_price_dollars": str(yes_p) if yes_p is not None else None,
        "no_price_dollars": str(no_p) if no_p is not None else None,
        "acquisition_cost_per_contract": str(cost) if cost is not None else None,
        "premium_paid": str(premium),
        "fee_cost": str(fee),
        "is_taker": fill.get("is_taker"),
        "created_time": fill.get("created_time"),
        "raw": {k: fill.get(k) for k in fill},
    }


def net_realized_from_parts(
    *,
    gross_realized: Decimal | str,
    fees_paid: Decimal | str,
    fees_already_in_gross: bool = False,
) -> Decimal:
    """Net PnL. If gross already subtracts fees (exchange realized_pnl), do not subtract again."""
    g = D(str(gross_realized))
    f = D(str(fees_paid or 0))
    if fees_already_in_gross:
        return g
    return g - f


def historical_fixture_expectations() -> dict[str, Any]:
    """Audit snapshot expectations — fixtures only, not current account claims."""
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "sep15_nyc_t70": {
            "note": "YES @ 0.01 x 458.24 + ~0.3176 fee → ≈ -4.90 if held to loss",
            "approx_net_pnl": "-4.9000",
        },
        "sep15_lax_t82": {
            "note": "YES buy@0.01 and sell@0.01 with 0.3176 fee each side → ≈ -0.6352",
            "approx_net_pnl": "-0.6352",
        },
        "sep16_nyc_b77_5": {
            "note": "Exchange realized_pnl_dollars 3.2840 already net of some reporting; fees_paid 0.3018",
            "gross_realized_pnl_dollars": "3.2840",
            "fees_paid_dollars": "0.3018",
            "net_if_fees_not_in_gross": "2.9822",
            "fees_already_in_gross_unknown": True,
        },
    }
