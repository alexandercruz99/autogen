"""High-confidence paper selection — YES-only, p≥0.95, late lock.

Goal: raise *paper* hit rate by refusing the losing live pattern (cheap NO / tails).
This does **not** magically make the raw forecast 95% accurate. Trades only fire when
model mass on the YES bracket is ≥ ``min_model_p`` (default 0.95) on the modal
interval at hour 14 with a lock check on max_so_far / remain.

Replay note: open-forecast YES on the modal 2° bracket is ~55–77% by city at 14:00.
p≥0.95 mass almost never appears in historical PMFs — so this filter trades rarely.
When remain is tiny and max_so_far already sits in the YES bracket, hit rate approaches
locked-in certainty on proxy labels (not TWC settlement proof).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
    interval_for_ticker,
    select_forecast_consistent,
    strike_distance_f,
)
from kalshi_bot.money import D

POLICY_VERSION = "paper.high_confidence.v1"
DEFAULT_MIN_MODEL_P = D("0.95")


@dataclass(frozen=True)
class HighConfidencePaperPolicy:
    version: str = POLICY_VERSION
    yes_only: bool = True
    min_model_p: Decimal = DEFAULT_MIN_MODEL_P
    require_decision_hour_local: int = 14
    require_median_inside_yes_interval: bool = True
    # Lock: max_so_far must already lie in the YES interval (high is "in the bucket").
    require_max_so_far_in_interval: bool = True
    # Optional remain gate (None = skip). q90 °F remaining rise.
    max_remain_q90_f: float | None = 2.0
    max_strike_distance_f: float = 0.0  # YES must cover the median
    min_ev: Decimal = D("0.02")
    live_eligible: bool = False


DEFAULT_PAPER_POLICY = HighConfidencePaperPolicy()


def _interval_contains(interval: dict[str, Any] | None, temp_f: float) -> bool:
    if not interval:
        return False
    op = interval.get("op")
    low, high = interval.get("low"), interval.get("high")
    if op in ("range_inclusive", "range") and low is not None and high is not None:
        return float(low) <= temp_f <= float(high)
    if op in ("gt", "greater") and low is not None:
        return temp_f > float(low)
    if op in ("lt", "less") and high is not None:
        return temp_f < float(high)
    return False


def select_high_confidence_paper(
    evaluations: list[dict[str, Any]],
    *,
    median_f: float,
    brackets: list[dict[str, Any]] | None = None,
    decision_hour_local: int | None = None,
    max_so_far: float | None = None,
    remain_q10_q50_q90: list[float] | tuple[float, ...] | None = None,
    policy: HighConfidencePaperPolicy | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select a single high-confidence YES paper candidate or reject all."""
    pol = policy or DEFAULT_PAPER_POLICY
    rejected: list[dict[str, Any]] = []

    if (
        pol.require_decision_hour_local is not None
        and decision_hour_local is not None
        and int(decision_hour_local) != int(pol.require_decision_hour_local)
    ):
        return None, [
            {
                "reject_class": "off_hour",
                "reject_reason": (
                    f"high-confidence paper only at local hour "
                    f"{pol.require_decision_hour_local}; got {decision_hour_local}"
                ),
            }
        ]

    if pol.max_remain_q90_f is not None and remain_q10_q50_q90 is not None:
        try:
            q90 = float(remain_q10_q50_q90[2])
        except (TypeError, ValueError, IndexError):
            q90 = None
        if q90 is not None and q90 > float(pol.max_remain_q90_f):
            return None, [
                {
                    "reject_class": "not_locked",
                    "reject_reason": (
                        f"remain q90={q90:.2f}°F > max {pol.max_remain_q90_f}°F "
                        "(day not locked enough for high-confidence paper)"
                    ),
                }
            ]

    # Base filters at high min_p; then YES-only + lock geometry.
    best, base_rejected = select_forecast_consistent(
        evaluations,
        median_f=median_f,
        brackets=brackets,
        min_model_p=pol.min_model_p,
        max_strike_distance_f=pol.max_strike_distance_f,
        min_ev=pol.min_ev,
        apply_min_probability_preference=True,
    )
    rejected.extend(base_rejected)

    kept: list[dict[str, Any]] = []
    # Re-scan evaluations with high-confidence extras (base may return one; we want full kept set)
    for e in evaluations:
        if e.get("p") is None or e.get("ask") is None or e.get("ev") is None:
            continue
        side = str(e.get("side") or "").lower()
        interval = e.get("interval") or interval_for_ticker(brackets, str(e.get("ticker") or ""))
        row = {
            **e,
            "interval": interval,
            "strike_distance_f": strike_distance_f(interval, side=side, median_f=median_f),
        }
        if pol.yes_only and side != "yes":
            rejected.append(
                {
                    **row,
                    "reject_class": "yes_only_policy",
                    "reject_reason": "high-confidence paper bans NO (losing live pattern)",
                }
            )
            continue
        p = D(e["p"])
        if p < pol.min_model_p:
            # already tagged in base_rejected usually
            continue
        if pol.require_median_inside_yes_interval and side == "yes":
            if not _interval_contains(interval, median_f):
                rejected.append(
                    {
                        **row,
                        "reject_class": "not_modal_bracket",
                        "reject_reason": (
                            f"median {median_f:.1f}°F not inside YES interval "
                            f"(high-confidence requires modal YES)"
                        ),
                    }
                )
                continue
        if pol.require_max_so_far_in_interval and max_so_far is not None and side == "yes":
            if not _interval_contains(interval, float(max_so_far)):
                rejected.append(
                    {
                        **row,
                        "reject_class": "max_so_far_outside",
                        "reject_reason": (
                            f"max_so_far {float(max_so_far):.1f}°F not in YES interval "
                            "(lock rule)"
                        ),
                    }
                )
                continue
        # Must also pass EV / alignment from base logic
        if D(e["ev"]) < pol.min_ev:
            continue
        if float(row["strike_distance_f"]) > float(pol.max_strike_distance_f) + 1e-9:
            continue
        kept.append(row)

    if not kept:
        return None, rejected

    best = max(kept, key=lambda e: (D(e["p"]), D(e["ev"]), -float(e["strike_distance_f"])))
    best = {
        **best,
        "selection_policy": pol.version,
        "selection_note": (
            f"{pol.version}: YES-only model_p≥{pol.min_model_p} hour="
            f"{pol.require_decision_hour_local}; not a guarantee of 95% live wins"
        ),
    }
    return best, rejected


def selection_rule_text(policy: HighConfidencePaperPolicy | None = None) -> str:
    pol = policy or DEFAULT_PAPER_POLICY
    return (
        f"{pol.version}: YES-only; model_p≥{pol.min_model_p}; "
        f"hour={pol.require_decision_hour_local}; median inside bracket; "
        f"max_so_far in bracket; remain q90≤{pol.max_remain_q90_f}; "
        f"EV≥{pol.min_ev}. Paper only — live_eligible=false."
    )
