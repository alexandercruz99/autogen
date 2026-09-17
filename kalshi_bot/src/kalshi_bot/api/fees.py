from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, MICRO, ONE, ZERO, ceil_to, floor_to

logger = logging.getLogger(__name__)


def quadratic_taker_fee(
    contracts: Decimal,
    price: Decimal,
    multiplier: Decimal = ONE,
) -> Decimal:
    """Kalshi general taker fee model (before rounding fee / rebate).

    fees = M * 0.07 * C * P * (1-P)
    Documented at https://kalshi.com/regulatory/fee-schedule and fee schedule PDF.
    """
    c = D(contracts)
    p = D(price)
    m = D(multiplier)
    if c <= ZERO or p <= ZERO or p >= ONE:
        return ZERO
    return m * Decimal("0.07") * c * p * (ONE - p)


def quadratic_maker_fee(
    contracts: Decimal,
    price: Decimal,
    multiplier: Decimal = ONE,
) -> Decimal:
    """Maker fee where series use quadratic_with_maker_fees (0.0175 coefficient)."""
    c = D(contracts)
    p = D(price)
    m = D(multiplier)
    if c <= ZERO or p <= ZERO or p >= ONE:
        return ZERO
    return m * Decimal("0.0175") * c * p * (ONE - p)


def estimate_net_fee(
    contracts: Decimal,
    price: Decimal,
    *,
    multiplier: Decimal = ONE,
    assume_taker: bool = True,
    balance_precision: Decimal = Decimal("0.0001"),
) -> Decimal:
    """Estimate net fee including trade-fee ceil to 6dp and a conservative rounding fee.

    Per docs: trade_fee = ceil_6dp(model_fee); rounding restores balance precision.
    We conservatively add up to (balance_precision - MICRO) of potential rounding
    without claiming rebate (rebates are opportunistic).
    """
    model = (
        quadratic_taker_fee(contracts, price, multiplier)
        if assume_taker
        else quadratic_maker_fee(contracts, price, multiplier)
    )
    trade_fee = ceil_to(model, MICRO)
    # Worst-case rounding fee before rebate for a single fill approximation.
    # Revenue for a buy is -price*contracts; aligned change floors to balance precision.
    revenue = -(D(price) * D(contracts))
    raw_change = revenue - trade_fee
    aligned = floor_to(raw_change, balance_precision)
    rounding_fee = raw_change - aligned
    # Net fee before rebate; rebate cannot make net fee negative.
    net = trade_fee + rounding_fee
    return max(net, ZERO)


def fee_per_contract(total_fee: Decimal, contracts: Decimal) -> Decimal:
    c = D(contracts)
    if c <= ZERO:
        return ZERO
    return D(total_fee) / c
