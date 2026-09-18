from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from kalshi_bot.money import D, ONE, ZERO


@dataclass
class ComboLeg:
    market_ticker: str
    event_ticker: str
    side: str  # yes|no
    p_marginal: Decimal
    settlement_rules_note: str = ""


@dataclass
class JointResult:
    p_all: Decimal
    p_all_conservative: Decimal
    method: str
    supported: bool
    skip_reason: str | None = None
    details: dict[str, Any] | None = None


def joint_probability(
    legs: list[ComboLeg],
    *,
    allow_independence: bool = False,
    dependence_model: dict[str, Any] | None = None,
) -> JointResult:
    """Compute P(all legs win) only when dependence is justified.

    Pairwise correlations alone are insufficient for multi-leg distributions.
    """
    if len(legs) < 2:
        return JointResult(
            p_all=ZERO,
            p_all_conservative=ZERO,
            method="none",
            supported=False,
            skip_reason="combo requires at least 2 legs",
        )

    # Contradictory: same market both sides
    tickers = [l.market_ticker for l in legs]
    if len(tickers) != len(set(tickers)):
        return JointResult(
            ZERO, ZERO, "reject", False, "duplicate market legs are not allowed"
        )

    if dependence_model and dependence_model.get("type") == "independent_weather_cities":
        # Distinct cities' daily highs — approximate independence only if model says so
        # AND allow_independence is true. Still shrink jointly.
        if not allow_independence:
            return JointResult(
                ZERO,
                ZERO,
                "blocked",
                False,
                "independence assumption disabled in config",
            )
        p = ONE
        for leg in legs:
            p *= D(leg.p_marginal)
        # Extra conservative haircut for multi-leg compounding of model error
        haircut = Decimal("0.85") ** (len(legs) - 1)
        p_cons = p * haircut
        return JointResult(
            p_all=p,
            p_all_conservative=p_cons,
            method="independent_product_with_haircut",
            supported=True,
            details={"legs": len(legs), "haircut": str(haircut)},
        )

    if dependence_model and dependence_model.get("type") == "simulation":
        # Placeholder for validated simulators — require explicit samples.
        samples = dependence_model.get("p_all_samples")
        if not samples:
            return JointResult(
                ZERO, ZERO, "simulation", False, "simulation missing p_all_samples"
            )
        vals = [D(x) for x in samples]
        mean = sum(vals) / D(len(vals))
        # conservative = lower 20th percentile-ish simple order statistic
        ordered = sorted(vals)
        idx = max(0, len(ordered) // 5)
        return JointResult(
            p_all=mean,
            p_all_conservative=ordered[idx],
            method="simulation_samples",
            supported=True,
            details={"n": len(vals)},
        )

    return JointResult(
        ZERO,
        ZERO,
        "unsupported",
        False,
        "dependence cannot be estimated reliably; skipping combo "
        "(will not multiply marginals without justification)",
    )


def combo_ev(
    p_all: Decimal,
    executable_price: Decimal,
    additional_costs: Decimal,
) -> Decimal:
    """EV_combo = P(all) - price - costs for $1 all-or-nothing payout."""
    return D(p_all) - D(executable_price) - D(additional_costs)


def combo_settlement_expectation(
    leg_settlement_values: list[Decimal],
) -> Decimal:
    """Product of leg settlement values (Kalshi combo rule, including DNP scalars)."""
    prod = ONE
    for v in leg_settlement_values:
        prod *= D(v)
    return prod
