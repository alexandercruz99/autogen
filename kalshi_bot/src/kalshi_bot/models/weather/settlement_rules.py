from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TempInterval:
    """YES if observed max temp (°F) falls in this interval.

    For Kalshi CLINYC / Weather Company markets:
    - greater: temp > floor (yes_sub_title often 'N or above' for floor N-1)
    - less: temp < cap
    - between: floor <= temp <= cap (inclusive integer degrees per rules_primary)
    """

    op: str
    low: float | None
    high: float | None
    source: str
    rules_primary: str = ""


def interval_from_market(market: dict[str, Any]) -> TempInterval | None:
    """Prefer exchange strike fields over ticker heuristics."""
    rules = market.get("rules_primary") or ""
    strike_type = (market.get("strike_type") or "").lower()
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")

    if strike_type == "greater" and floor is not None:
        return TempInterval("gt", float(floor), None, "market.strike_type", rules)
    if strike_type == "less" and cap is not None:
        return TempInterval("lt", None, float(cap), "market.strike_type", rules)
    if strike_type == "between" and floor is not None and cap is not None:
        # Inclusive integer band per Kalshi rules_primary ("between 79 and 80").
        return TempInterval(
            "range_inclusive",
            float(floor),
            float(cap),
            "market.strike_type",
            rules,
        )
    return None


def yes_from_observation(temp_f: float, interval: TempInterval) -> bool:
    if interval.op == "gt":
        return temp_f > float(interval.low)
    if interval.op == "lt":
        return temp_f < float(interval.high)
    if interval.op == "range_inclusive":
        return float(interval.low) <= temp_f <= float(interval.high)
    if interval.op == "range":
        return float(interval.low) <= temp_f < float(interval.high)
    raise ValueError(interval.op)
