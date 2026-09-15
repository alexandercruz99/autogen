from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from kalshi_bot.api.fees import estimate_net_fee, fee_per_contract
from kalshi_bot.api.orderbook import ExecutableBook, market_implied_yes_prob
from kalshi_bot.config import TradingConfig
from kalshi_bot.models.base import Prediction
from kalshi_bot.money import D, ONE, ZERO, clamp01, fp_count, fp_price


Side = Literal["yes", "no"]


@dataclass
class EvResult:
    side: Side
    quantity: Decimal
    executable_price: Decimal
    fillable_quantity: Decimal
    estimated_prob: Decimal
    conservative_prob: Decimal
    uncertainty: Decimal
    fees_total: Decimal
    fees_per_contract: Decimal
    estimated_ev: Decimal
    conservative_ev: Decimal
    breakeven_prob: Decimal
    max_loss: Decimal
    capital_required: Decimal
    qualifies: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_binary_contract(
    prediction: Prediction,
    book: ExecutableBook,
    config: TradingConfig,
    quantity: Decimal | None = None,
) -> list[EvResult]:
    """Evaluate YES and NO purchases; return both with qualify flags.

    EV_yes = p - a_yes - c_yes
    EV_no  = (1-p) - a_no - c_no
    Uses conservative probability and uncertainty buffer for qualification.
    """
    qty = fp_count(quantity or config.default_contract_quantity)
    qty = min(qty, fp_count(config.max_contracts_per_order))
    results: list[EvResult] = []

    for side in ("yes", "no"):
        results.append(_eval_side(side, prediction, book, config, qty))
    return results


def _eval_side(
    side: Side,
    prediction: Prediction,
    book: ExecutableBook,
    config: TradingConfig,
    qty: Decimal,
) -> EvResult:
    if side == "yes":
        ask = book.best_yes_ask
        fillable, vwap = (ZERO, ZERO)
        if ask is not None:
            # Cap at ask + small slip only within available depth at/under a max limit later.
            fillable, vwap = book.fillable_yes(qty, max_price=ask)
        p = prediction.p_yes
        # One-sided haircut against the purchase (never inflate longshot probabilities).
        p_cons = min(prediction.p_yes_conservative, p) - prediction.uncertainty
        p_cons = clamp01(p_cons)
    else:
        ask = book.best_no_ask
        fillable, vwap = (ZERO, ZERO)
        if ask is not None:
            fillable, vwap = book.fillable_no(qty, max_price=ask)
        p = prediction.p_no
        p_cons = min(prediction.p_no_conservative, p) - prediction.uncertainty
        p_cons = clamp01(p_cons)

    if ask is None or fillable <= ZERO:
        return EvResult(
            side=side,
            quantity=qty,
            executable_price=ZERO,
            fillable_quantity=ZERO,
            estimated_prob=p,
            conservative_prob=p_cons,
            uncertainty=prediction.uncertainty,
            fees_total=ZERO,
            fees_per_contract=ZERO,
            estimated_ev=ZERO,
            conservative_ev=ZERO,
            breakeven_prob=ZERO,
            max_loss=ZERO,
            capital_required=ZERO,
            qualifies=False,
            reason="no executable ask / insufficient depth",
        )

    price = fp_price(vwap if vwap > ZERO else ask)
    use_qty = fp_count(fillable)
    fees = estimate_net_fee(
        use_qty,
        price,
        multiplier=config.fee_multiplier,
        assume_taker=config.assume_taker,
        balance_precision=config.balance_precision,
    )
    c = fee_per_contract(fees, use_qty)
    est_ev = p - price - c
    # Conservative EV uses shrunk probability and extra uncertainty buffer.
    cons_ev = p_cons - price - c - config.uncertainty_buffer
    breakeven = price + c
    capital = price * use_qty + fees
    max_loss = capital  # hold to settlement; lose premium + fees if wrong (binary)

    qualifies = True
    reasons: list[str] = []
    if not prediction.supported:
        qualifies = False
        reasons.append(prediction.skip_reason or "model unsupported")
    if cons_ev < config.min_net_edge:
        qualifies = False
        reasons.append(
            f"conservative EV {cons_ev} < min_net_edge {config.min_net_edge}"
        )
    if max_loss > config.max_loss_per_trade_dollars:
        qualifies = False
        reasons.append("max loss exceeds per-trade limit")
    if use_qty < qty:
        reasons.append(f"partial depth only fillable={use_qty}")

    implied = market_implied_yes_prob(book)
    max_div = D(config.max_model_market_divergence)
    if implied is not None:
        model_yes = prediction.p_yes
        if abs(model_yes - implied) > max_div and "UNVALIDATED" in (prediction.validation_evidence or ""):
            qualifies = False
            reasons.append(
                f"model vs market mid divergence {abs(model_yes - implied)} > {max_div} "
                f"while model unvalidated (mid={implied}, model={model_yes})"
            )

    if qualifies:
        reasons.append(
            f"conservative EV {cons_ev} meets min edge; model={prediction.model_version}"
        )

    return EvResult(
        side=side,
        quantity=use_qty,
        executable_price=price,
        fillable_quantity=use_qty,
        estimated_prob=p,
        conservative_prob=p_cons,
        uncertainty=prediction.uncertainty,
        fees_total=fees,
        fees_per_contract=c,
        estimated_ev=est_ev,
        conservative_ev=cons_ev,
        breakeven_prob=clamp01(breakeven),
        max_loss=max_loss,
        capital_required=capital,
        qualifies=qualifies,
        reason="; ".join(reasons),
        details={
            "best_ask": str(ask),
            "market_implied_note": "Disagreement with market is not proof of edge",
        },
    )


def max_limit_price(
    conservative_prob: Decimal,
    fee_per_ct: Decimal,
    min_edge: Decimal,
    uncertainty_buffer: Decimal,
) -> Decimal:
    """Highest purchase price that still preserves required conservative edge."""
    # cons_ev = p_cons - price - fee - buffer >= min_edge
    # price <= p_cons - fee - buffer - min_edge
    limit = D(conservative_prob) - D(fee_per_ct) - D(uncertainty_buffer) - D(min_edge)
    return fp_price(max(limit, ZERO))
