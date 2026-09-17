"""Deterministic contract picker — pure decision function (no I/O, no clock, no LLM).

pick_contract(...) -> Decision (TRADE or NO_TRADE)

Identical inputs + policy version ⇒ identical outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Literal

from kalshi_bot.api.fees import estimate_net_fee
from kalshi_bot.models.weather.distribution import PredictiveDistribution
from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import strike_distance_f
from kalshi_bot.models.weather.obs_engine.multi.replay.policy import DEFAULT_PICKER_POLICY, PickerPolicy
from kalshi_bot.models.weather.obs_engine.multi.replay.scoring import modal_degree, yes_probability
from kalshi_bot.models.weather.settlement_rules import TempInterval, interval_from_market
from kalshi_bot.money import D, ONE, ZERO


DecisionKind = Literal["TRADE", "NO_TRADE"]


@dataclass
class Decision:
    kind: DecisionKind
    policy_version: str
    decision_id: str
    reasons: list[str] = field(default_factory=list)
    # Most-likely bracket (separate from purchased contract)
    most_likely_degree_f: int | None = None
    most_likely_bracket_ticker: str | None = None
    # Trade fields
    ticker: str | None = None
    side: str | None = None
    quantity: int | None = None
    limit_price: str | None = None
    total_executable_cost: str | None = None
    applicable_total_fees: str | None = None
    probability_purchased_side: str | None = None
    expected_net_profit: str | None = None
    expected_net_profit_per_contract: str | None = None
    candidates_evaluated: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _q(d: Decimal, places: int) -> Decimal:
    quant = Decimal(10) ** -places
    return d.quantize(quant, rounding=ROUND_HALF_EVEN)


def _interval_from_contract(meta: dict[str, Any]) -> TempInterval | None:
    if meta.get("interval") and isinstance(meta["interval"], TempInterval):
        return meta["interval"]
    if isinstance(meta.get("interval"), dict):
        iv = meta["interval"]
        return TempInterval(
            str(iv.get("op") or "range_inclusive"),
            float(iv["low"]) if iv.get("low") is not None else None,
            float(iv["high"]) if iv.get("high") is not None else None,
            str(iv.get("source") or "replay"),
            str(iv.get("rules_primary") or ""),
        )
    # market-shaped
    return interval_from_market(meta)


def _executable_ask(book: dict[str, Any], side: str, quantity: int) -> tuple[Decimal | None, str]:
    """Depth-aware ask for YES or NO purchases from a snapshot.

    Expected snapshot shape (research):
      {"yes_asks": [{"price": "0.42", "qty": 10}, ...], "no_asks": [...], ...}
    Falls back to yes_ask / no_ask top-of-book fields.
    """
    side_u = side.upper()
    levels_key = "yes_asks" if side_u == "YES" else "no_asks"
    levels = book.get(levels_key)
    if isinstance(levels, list) and levels:
        need = int(quantity)
        cost = ZERO
        filled = 0
        for lvl in levels:
            px = D(str(lvl.get("price")))
            qty = int(lvl.get("qty") or lvl.get("quantity") or 0)
            if qty <= 0 or px <= ZERO:
                continue
            take = min(need - filled, qty)
            cost += px * D(take)
            filled += take
            if filled >= need:
                break
        if filled < need:
            return None, f"insufficient_depth_need_{need}_have_{filled}"
        return cost, "depth_aware"
    # top of book
    key = "yes_ask" if side_u == "YES" else "no_ask"
    if book.get(key) is None:
        return None, "missing_ask"
    px = D(str(book[key]))
    if px <= ZERO or px >= ONE:
        return None, "ask_not_executable"
    return px * D(quantity), "top_of_book"


def _rank_key(row: dict[str, Any], places: int) -> tuple:
    enp = _q(D(row["expected_net_profit_per_contract"]), places)
    p = _q(D(row["probability_purchased_side"]), places)
    ticker = str(row["ticker"])
    side = str(row["side"]).upper()
    side_ord = 0 if side == "YES" else 1
    # max ENP, then max p, then ticker, then YES before NO
    return (-enp, -p, ticker, side_ord)


def pick_contract(
    forecast_distribution: PredictiveDistribution | None,
    contract_metadata: list[dict[str, Any]],
    orderbook_snapshot: dict[str, dict[str, Any]] | None,
    account_and_open_order_snapshot: dict[str, Any] | None,
    decision_time: str,
    versioned_policy: PickerPolicy | dict[str, Any] | None = None,
    *,
    decision_id: str = "decision",
    location_id: str | None = None,
    climate_day: str | None = None,
    settlement_source_family: str | None = None,
    target_metric: str = "daily_max_temp_f",
    point_median_f: float | None = None,
) -> Decision:
    """Pure picker. Does not fetch data, read clocks, place orders, or call an LLM."""
    policy = versioned_policy
    if policy is None:
        policy = DEFAULT_PICKER_POLICY
    elif isinstance(policy, dict):
        policy = PickerPolicy(**{k: v for k, v in policy.items() if k in PickerPolicy.__dataclass_fields__})

    reasons: list[str] = []
    rejected: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []

    if forecast_distribution is None or not forecast_distribution.temps_f:
        return Decision(
            kind="NO_TRADE",
            policy_version=policy.version,
            decision_id=decision_id,
            reasons=["missing_or_empty_forecast_distribution"],
        )
    if not contract_metadata:
        return Decision(
            kind="NO_TRADE",
            policy_version=policy.version,
            decision_id=decision_id,
            reasons=["no_contract_metadata"],
            most_likely_degree_f=modal_degree(forecast_distribution),
        )
    if orderbook_snapshot is None:
        return Decision(
            kind="NO_TRADE",
            policy_version=policy.version,
            decision_id=decision_id,
            reasons=["missing_orderbook_snapshot"],
            most_likely_degree_f=modal_degree(forecast_distribution),
            extras={"trading_replay": "blocked_no_historical_books"},
        )

    modal = modal_degree(forecast_distribution)
    median = float(point_median_f) if point_median_f is not None else float(forecast_distribution.quantile(0.5))

    # Identify most-likely bracket (report only)
    ml_ticker = None
    best_mass = -1.0
    for meta in contract_metadata:
        iv = _interval_from_contract(meta)
        if iv is None:
            continue
        mass = yes_probability(forecast_distribution, iv)
        tkr = str(meta.get("ticker") or "")
        if mass > best_mass or (mass == best_mass and tkr < str(ml_ticker or "~~~")):
            best_mass = mass
            ml_ticker = tkr

    qty = int(policy.evaluation_quantity)
    acct = account_and_open_order_snapshot or {}
    cash = D(str(acct.get("cash", "0")))
    open_notional = D(str(acct.get("open_notional", "0")))
    max_total = D(str(acct.get("max_total_exposure", "999999")))
    max_per = D(str(acct.get("max_per_market_exposure", "999999")))
    reserved = D(str(acct.get("reserved", "0")))
    existing_tickers = set(acct.get("open_position_tickers") or [])

    for meta in contract_metadata:
        ticker = str(meta.get("ticker") or "")
        # Compatibility checks
        if location_id and meta.get("location_id") and meta["location_id"] != location_id:
            rejected.append({"ticker": ticker, "reject_reason": "location_mismatch"})
            continue
        if climate_day and meta.get("climate_day") and str(meta["climate_day"]) != str(climate_day):
            rejected.append({"ticker": ticker, "reject_reason": "climate_day_mismatch"})
            continue
        if meta.get("metric") and meta["metric"] != target_metric:
            rejected.append({"ticker": ticker, "reject_reason": "metric_mismatch"})
            continue
        if (
            settlement_source_family
            and meta.get("settlement_source_family")
            and meta["settlement_source_family"] != settlement_source_family
        ):
            rejected.append({"ticker": ticker, "reject_reason": "settlement_source_mismatch"})
            continue
        status = str(meta.get("status") or "open").lower()
        if status not in ("open", "active", ""):
            rejected.append({"ticker": ticker, "reject_reason": f"market_status_{status}"})
            continue

        iv = _interval_from_contract(meta)
        if iv is None:
            rejected.append({"ticker": ticker, "reject_reason": "unsupported_contract_boundaries"})
            continue

        book = orderbook_snapshot.get(ticker) or {}
        # freshness
        if book.get("stale") is True:
            rejected.append({"ticker": ticker, "reject_reason": "stale_orderbook"})
            continue

        p_yes = D(f"{yes_probability(forecast_distribution, iv):.10f}")
        p_no = ONE - p_yes

        for side, p_side in (("YES", p_yes), ("NO", p_no)):
            cost, cost_mode = _executable_ask(book, side, qty)
            if cost is None:
                rejected.append(
                    {
                        "ticker": ticker,
                        "side": side,
                        "reject_reason": cost_mode,
                    }
                )
                continue
            limit_px = cost / D(qty)
            if limit_px < D(policy.min_ask) or limit_px > D(policy.max_ask):
                rejected.append(
                    {
                        "ticker": ticker,
                        "side": side,
                        "reject_reason": "ask_outside_policy_bounds",
                        "limit_price": str(limit_px),
                    }
                )
                continue

            fees = estimate_net_fee(
                D(qty),
                limit_px,
                multiplier=D(policy.fee_multiplier),
                assume_taker=policy.assume_taker_fees,
            )
            # expected_net_profit = qty * p - cost - fees  (for $1 payoff)
            enp = D(qty) * p_side - cost - fees
            enp_pc = enp / D(qty)

            dist_f = strike_distance_f(
                {"op": iv.op, "low": iv.low, "high": iv.high},
                side=side,
                median_f=median,
            )
            row = {
                "ticker": ticker,
                "side": side,
                "quantity": qty,
                "limit_price": str(limit_px),
                "total_executable_cost": str(cost),
                "applicable_total_fees": str(fees),
                "probability_purchased_side": str(p_side),
                "expected_net_profit": str(enp),
                "expected_net_profit_per_contract": str(enp_pc),
                "cost_mode": cost_mode,
                "strike_distance_f": dist_f,
                "interval": {"op": iv.op, "low": iv.low, "high": iv.high},
                "decision_time": decision_time,
            }
            evaluated.append(row)

            if dist_f > policy.max_strike_distance_f:
                rejected.append({**row, "reject_reason": "weather_alignment"})
                continue
            if enp_pc < policy.min_enp():
                rejected.append({**row, "reject_reason": "inadequate_expected_net_profit"})
                continue
            if policy.apply_min_probability_preference and p_side < policy.min_p():
                rejected.append({**row, "reject_reason": "preference_min_probability"})
                continue
            # account limits
            if cost + fees + reserved > cash:
                rejected.append({**row, "reject_reason": "insufficient_cash"})
                continue
            if open_notional + cost > max_total:
                rejected.append({**row, "reject_reason": "max_total_exposure"})
                continue
            if cost > max_per:
                rejected.append({**row, "reject_reason": "max_per_market_exposure"})
                continue
            if ticker in existing_tickers:
                rejected.append({**row, "reject_reason": "existing_position_or_reservation"})
                continue

            row["eligible"] = True

    eligible = [r for r in evaluated if r.get("eligible")]
    if not eligible:
        return Decision(
            kind="NO_TRADE",
            policy_version=policy.version,
            decision_id=decision_id,
            reasons=["no_eligible_candidates"] + reasons,
            most_likely_degree_f=modal,
            most_likely_bracket_ticker=ml_ticker,
            candidates_evaluated=evaluated,
            rejected=rejected,
        )

    # Stable sort: shuffle-invariant
    eligible_sorted = sorted(eligible, key=lambda r: _rank_key(r, policy.score_quantize_places))
    best = eligible_sorted[0]

    return Decision(
        kind="TRADE",
        policy_version=policy.version,
        decision_id=decision_id,
        reasons=["selected_by_expected_net_profit_rank"],
        most_likely_degree_f=modal,
        most_likely_bracket_ticker=ml_ticker,
        ticker=str(best["ticker"]),
        side=str(best["side"]),
        quantity=int(best["quantity"]),
        limit_price=str(best["limit_price"]),
        total_executable_cost=str(best["total_executable_cost"]),
        applicable_total_fees=str(best["applicable_total_fees"]),
        probability_purchased_side=str(best["probability_purchased_side"]),
        expected_net_profit=str(best["expected_net_profit"]),
        expected_net_profit_per_contract=str(best["expected_net_profit_per_contract"]),
        candidates_evaluated=evaluated,
        rejected=rejected,
        extras={
            "rank_policy": "enp_per_contract_then_p_then_ticker_then_side",
            "most_likely_bracket_separate_from_purchase": True,
        },
    )
