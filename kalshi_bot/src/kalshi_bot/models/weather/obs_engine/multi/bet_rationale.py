"""Bet selection with explicit rejection classes — not marketing claims.

A low model probability can still have positive EV at a low price. A high
probability can be overpriced. Minimum-probability filters are **user preferences**,
not proof that a probability is undefended or that returns will compound.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from kalshi_bot.money import D, ZERO

# Default preference filter (configurable by callers). Not a scientific cutoff.
DEFAULT_MIN_MODEL_P = D("0.55")
MAX_STRIKE_DISTANCE_F = 4.0
MIN_EV = D("0.02")

# Back-compat alias used by live_bet imports
MIN_MODEL_P = DEFAULT_MIN_MODEL_P

RejectClass = Literal[
    "preference_min_probability",
    "inadequate_ev",
    "unsupported_probability",
    "weather_alignment",
    "liquidity_or_price",
    "other",
]


def interval_label(interval: dict[str, Any] | None) -> str:
    if not interval:
        return "this bracket"
    op = interval.get("op")
    low, high = interval.get("low"), interval.get("high")
    if op in ("gt", "greater") and low is not None:
        return f"above {float(low):g}°F"
    if op in ("lt", "less") and high is not None:
        return f"below {float(high):g}°F"
    if op in ("range_inclusive", "range") and low is not None and high is not None:
        lo, hi = float(low), float(high)
        if lo == hi:
            return f"{lo:.0f}°F"
        return f"{lo:.0f}–{hi:.0f}°F"
    return "this bracket"


def format_bet_rationale(
    *,
    point_median_f: float | None,
    max_so_far: float | None,
    ticker: str,
    side: str,
    p: Decimal | float | str,
    ask: Decimal | float | str,
    interval: dict[str, Any] | None = None,
    conservative_ev: Decimal | float | str | None = None,
) -> str:
    """State forecast, contract, model p, ask, and EV — with uncertainty, no growth claims."""
    p_d, ask_d = D(str(p)), D(str(ask))
    claim = interval_label(interval)
    pred = f"{round(float(point_median_f))}°F" if point_median_f is not None else "unknown"
    seen = f"{float(max_so_far):.0f}°F" if max_so_far is not None else "unknown"
    side_u = side.upper()
    if side_u == "YES":
        weather = (
            f"Point forecast ~{pred} (already {seen}); candidate YES {ticker} ({claim})."
        )
    else:
        weather = (
            f"Point forecast ~{pred} (already {seen}); candidate NO {ticker} "
            f"(against {claim})."
        )
    ev_bit = ""
    if conservative_ev is not None:
        ev_bit = f" Conservative EV≈{float(D(str(conservative_ev))):.3f} after fees/buffer."
    edge = (
        f" Model P({side_u})={float(p_d)*100:.1f}% vs ask {float(ask_d)*100:.1f}¢.{ev_bit} "
        f"Point forecast ≠ calibrated probability; transfer/settlement limits still apply."
    )
    return f"{weather}{edge}"


def strike_distance_f(interval: dict[str, Any] | None, *, side: str, median_f: float) -> float:
    if not interval:
        return 999.0
    op = interval.get("op")
    low, high = interval.get("low"), interval.get("high")
    side_u = side.lower()
    if op in ("gt", "greater") and low is not None:
        thr = float(low)
        if side_u == "yes":
            return 0.0 if median_f > thr else abs(median_f - thr)
        return 0.0 if median_f <= thr else abs(median_f - thr)
    if op in ("lt", "less") and high is not None:
        thr = float(high)
        if side_u == "yes":
            return 0.0 if median_f < thr else abs(median_f - thr)
        return 0.0 if median_f >= thr else abs(median_f - thr)
    if op in ("range_inclusive", "range") and low is not None and high is not None:
        lo, hi = float(low), float(high)
        mid = 0.5 * (lo + hi)
        inside = lo <= median_f <= hi
        if side_u == "yes":
            return 0.0 if inside else abs(median_f - mid)
        if not inside:
            return 0.0
        half = max(0.5 * (hi - lo), 0.5)
        return max(abs(median_f - mid), half)
    return 999.0


def interval_for_ticker(brackets: list[dict[str, Any]] | None, ticker: str) -> dict[str, Any] | None:
    for b in brackets or []:
        if b.get("ticker") == ticker:
            return b.get("interval")
    return None


def selection_rule_text(
    *,
    min_model_p: Decimal = DEFAULT_MIN_MODEL_P,
    max_strike_distance_f: float = MAX_STRIKE_DISTANCE_F,
    min_ev: Decimal = MIN_EV,
) -> str:
    return (
        f"select among +EV contracts within {max_strike_distance_f}°F of median; "
        f"preference filter model_p≥{min_model_p}; require EV≥{min_ev}; "
        f"rank by strike distance then p then EV. Preference ≠ proven edge."
    )


def select_forecast_consistent(
    evaluations: list[dict[str, Any]],
    *,
    median_f: float,
    brackets: list[dict[str, Any]] | None = None,
    min_model_p: Decimal = DEFAULT_MIN_MODEL_P,
    max_strike_distance_f: float = MAX_STRIKE_DISTANCE_F,
    min_ev: Decimal = MIN_EV,
    apply_min_probability_preference: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Filter and rank candidates; tag rejection class on each rejected row."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for e in evaluations:
        if e.get("p") is None or e.get("ask") is None or e.get("ev") is None:
            rejected.append(
                {
                    **e,
                    "reject_class": "unsupported_probability",
                    "reject_reason": "missing p/ask/ev",
                }
            )
            continue
        ev = D(e["ev"])
        p = D(e["p"])
        ask = D(e["ask"])
        interval = e.get("interval") or interval_for_ticker(brackets, str(e.get("ticker") or ""))
        dist = strike_distance_f(interval, side=str(e["side"]), median_f=median_f)
        row = {**e, "interval": interval, "strike_distance_f": dist}
        if ask <= ZERO or ask >= D("1"):
            rejected.append(
                {**row, "reject_class": "liquidity_or_price", "reject_reason": "ask not executable"}
            )
            continue
        if dist > max_strike_distance_f:
            rejected.append(
                {
                    **row,
                    "reject_class": "weather_alignment",
                    "reject_reason": (
                        f"strike {dist:.1f}°F from forecast {median_f:.0f}°F "
                        f"(max {max_strike_distance_f}°F)"
                    ),
                }
            )
            continue
        if ev <= ZERO or ev < min_ev:
            rejected.append(
                {
                    **row,
                    "reject_class": "inadequate_ev",
                    "reject_reason": f"EV={ev} below min_ev={min_ev} (or ≤0)",
                }
            )
            continue
        if apply_min_probability_preference and p < min_model_p:
            rejected.append(
                {
                    **row,
                    "reject_class": "preference_min_probability",
                    "reject_reason": (
                        f"model_p {p} < preference min {min_model_p} "
                        "(not labeled 'no edge' — filtered by preference)"
                    ),
                }
            )
            continue
        kept.append(row)

    if not kept:
        return None, rejected
    best = max(
        kept,
        key=lambda e: (-float(e["strike_distance_f"]), D(e["p"]), D(e["ev"])),
    )
    return best, rejected
