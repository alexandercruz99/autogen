"""Weather-first bet selection: explain why we buy; do not chase cheapest asks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, ZERO

# Live/paper picks must match the forecast story; cheap asks alone are not a reason.
MIN_MODEL_P = D("0.35")
MAX_STRIKE_DISTANCE_F = 4.0


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
) -> str:
    """Explain the weather story first; price is only the mispricing check."""
    p_d, ask_d = D(str(p)), D(str(ask))
    claim = interval_label(interval)
    pred = f"{round(float(point_median_f))}°F" if point_median_f is not None else "unknown"
    seen = f"{float(max_so_far):.0f}°F" if max_so_far is not None else "unknown"
    side_u = side.upper()
    if side_u == "YES":
        weather = (
            f"Forecast high ~{pred} (already {seen}), so YES on {ticker} "
            f"({claim}) matches the weather story."
        )
    else:
        weather = (
            f"Forecast high ~{pred} (already {seen}), so NO on {ticker} "
            f"(rejecting {claim}) matches the weather story."
        )
    edge = (
        f"Model puts {float(p_d)*100:.0f}% on that side vs market ask "
        f"{float(ask_d)*100:.0f}¢ — buy because of that gap, not because the ticket is cheap."
    )
    return f"{weather} {edge}"


def strike_distance_f(interval: dict[str, Any] | None, *, side: str, median_f: float) -> float:
    """How far the bet's weather claim sits from the forecast median (°F). Lower is better."""
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
        # NO: consistent when forecast is outside the band
        if not inside:
            return 0.0
        # Forecast sits in the band we're rejecting → weather mismatch
        half = max(0.5 * (hi - lo), 0.5)
        return max(abs(median_f - mid), half)
    return 999.0


def interval_for_ticker(brackets: list[dict[str, Any]] | None, ticker: str) -> dict[str, Any] | None:
    for b in brackets or []:
        if b.get("ticker") == ticker:
            return b.get("interval")
    return None


def select_forecast_consistent(
    evaluations: list[dict[str, Any]],
    *,
    median_f: float,
    brackets: list[dict[str, Any]] | None = None,
    min_model_p: Decimal = MIN_MODEL_P,
    max_strike_distance_f: float = MAX_STRIKE_DISTANCE_F,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Among +EV sides, keep only forecast-aligned / min-p rows; rank by distance then EV."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for e in evaluations:
        if e.get("p") is None or e.get("ask") is None or e.get("ev") is None:
            continue
        if D(e["ev"]) <= ZERO:
            continue
        p = D(e["p"])
        ask = D(e["ask"])
        interval = e.get("interval") or interval_for_ticker(brackets, str(e.get("ticker") or ""))
        dist = strike_distance_f(interval, side=str(e["side"]), median_f=median_f)
        row = {**e, "interval": interval, "strike_distance_f": dist}
        if p < min_model_p:
            rejected.append(
                {**row, "reject_reason": f"model_p {p} < {min_model_p} (blocks cheap longshots)"}
            )
            continue
        if dist > max_strike_distance_f:
            rejected.append(
                {
                    **row,
                    "reject_reason": (
                        f"strike {dist:.1f}°F from forecast {median_f:.0f}°F "
                        f"(max {max_strike_distance_f}°F) — weather story mismatch"
                    ),
                }
            )
            continue
        if ask <= ZERO or ask >= D("1"):
            continue
        kept.append(row)

    if not kept:
        return None, rejected
    best = max(kept, key=lambda e: (-float(e["strike_distance_f"]), D(e["ev"])))
    return best, rejected
