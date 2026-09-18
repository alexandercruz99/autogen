from __future__ import annotations

from decimal import Decimal

from kalshi_bot.api.fees import estimate_net_fee, quadratic_taker_fee
from kalshi_bot.money import D


def test_quadratic_taker_fee_at_50c_100_contracts():
    # From published schedule examples: ~$1.75 per 100 contracts at $0.50 before rounding nuances
    fee = quadratic_taker_fee(D("100"), D("0.50"), D("1"))
    assert fee == D("1.75")


def test_fee_estimate_nonnegative_and_covers_model():
    net = estimate_net_fee(D("10"), D("0.40"), assume_taker=True, balance_precision=D("0.0001"))
    model = quadratic_taker_fee(D("10"), D("0.40"))
    assert net >= model
    assert net >= 0


def test_extreme_prices_near_zero_fee_small():
    fee = quadratic_taker_fee(D("1"), D("0.01"))
    assert fee == D("0.000693")
