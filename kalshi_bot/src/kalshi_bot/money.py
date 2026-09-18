from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Iterable


ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")
CENTICENT = Decimal("0.0001")
MICRO = Decimal("0.000001")


def D(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        raise ValueError("Cannot convert None to Decimal")
    return Decimal(str(value))


def money(value: object, places: str = "0.0001") -> Decimal:
    return D(value).quantize(D(places), rounding=ROUND_HALF_UP)


def fp_count(value: object) -> Decimal:
    return D(value).quantize(Decimal("0.01"), rounding=ROUND_DOWN)


def fp_price(value: object) -> Decimal:
    return D(value).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def ceil_to(value: Decimal, quantum: Decimal) -> Decimal:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    # Round up to nearest quantum.
    units = (value / quantum).to_integral_value(rounding=ROUND_CEILING)
    return units * quantum


def floor_to(value: Decimal, quantum: Decimal) -> Decimal:
    units = (value / quantum).to_integral_value(rounding=ROUND_DOWN)
    return units * quantum


def clamp01(p: Decimal) -> Decimal:
    if p < ZERO:
        return ZERO
    if p > ONE:
        return ONE
    return p


def sum_decimal(values: Iterable[Decimal]) -> Decimal:
    total = ZERO
    for v in values:
        total += v
    return total
