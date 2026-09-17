from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from kalshi_bot.money import D, ZERO


def fixed_size(
    capital_per_contract: Decimal,
    max_loss_limit: Decimal,
    max_contracts: Decimal,
) -> Decimal:
    """Small fixed risk sizing. Kelly is intentionally not used until calibration is validated."""
    if capital_per_contract <= ZERO:
        return ZERO
    by_loss = (max_loss_limit / capital_per_contract).to_integral_value(rounding=ROUND_DOWN)
    qty = min(D(by_loss), D(max_contracts))
    if qty < 1:
        # Allow fractional down to 0.01 if capital permits single contract fraction
        frac = min(D(max_contracts), D("1.00"))
        if capital_per_contract * frac <= max_loss_limit:
            return frac
        return ZERO
    return D(qty).quantize(D("0.01"))
