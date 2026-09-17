"""Human-readable daily-max callout + optional capped live bet for TWC markets.

Shows: \"Today's high will be about XX°F (already seen YY°F).\"
Live submits only when explicitly requested with a dollar cap (default $5).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.fees import estimate_net_fee
from kalshi_bot.api.orderbook import parse_orderbook
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OrderRecord, PositionRecord, Store, dumps, utcnow
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
    MAX_STRIKE_DISTANCE_F,
    MIN_MODEL_P,
    format_bet_rationale,
    select_forecast_consistent,
    selection_rule_text,
)
from kalshi_bot.models.weather.obs_engine.multi.pipeline import process_location
from kalshi_bot.models.weather.obs_engine.multi.registry import VERIFIED_TWC_DAILY_MAX
from kalshi_bot.models.weather.obs_engine.multi.session_budget import (
    DEFAULT_PER_BET_MAX,
    LIVE_SERIES_ALLOWLIST,
    SessionBudgetPolicy,
    commit as budget_commit,
    default_session_id,
    release as budget_release,
    remaining as budget_remaining,
    reserve as budget_reserve,
    snapshot as budget_snapshot,
)
from kalshi_bot.money import D, ONE, ZERO, fp_count, fp_price

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path("data/obs_engine/multi")


def arm_store_for_live(config: AppConfig, store: Store) -> dict[str, Any]:
    """Sync sqlite bot state to live when config is already armed.

    weather-twc-bet does not go through CLI build_runtime; without this the
    store can remain paper/live_enabled=false and block submits.
    """
    if config.mode != "live" or not config.live.enabled:
        return {
            "ok": False,
            "error": f"config not armed (mode={config.mode}, live.enabled={config.live.enabled})",
        }
    state = store.get_state()
    updates: dict[str, Any] = {
        "mode": "live",
        "live_enabled": True,
        "kill_switch": False,
        "pause_buying": False,
        "trading_budget": str(config.trading.budget_dollars),
    }
    if not state.live_ack_at:
        updates["live_ack_at"] = utcnow()
    store.update_state(**updates)
    store.audit(
        "twc_live_arm",
        f"armed for capped TWC live; budget={config.trading.budget_dollars}",
    )
    return {"ok": True, "updates": updates}


def format_max_callout(
    *,
    display_name: str,
    climate_day: str | date,
    point_median_f: float | None,
    max_so_far: float | None,
    decision_hour_local: int | None,
    tz_name: str,
) -> str:
    """One-line human forecast the user asked for."""
    day = climate_day.isoformat() if isinstance(climate_day, date) else str(climate_day)
    hour_bit = (
        f" (as of {decision_hour_local:02d}:00 {tz_name.split('/')[-1]})"
        if decision_hour_local is not None
        else ""
    )
    if point_median_f is None:
        seen = f"{max_so_far:.0f}°F" if max_so_far is not None else "unknown"
        return f"{display_name}: no max forecast yet{hour_bit}; already seen {seen} on {day}."
    pred = int(round(float(point_median_f)))
    if max_so_far is not None:
        return (
            f"{display_name}: today's high will be about {pred}°F"
            f"{hour_bit} (already seen {float(max_so_far):.0f}°F) — climate day {day}."
        )
    return f"{display_name}: today's high will be about {pred}°F{hour_bit} — climate day {day}."


def _twc_target(series_ticker: str) -> dict[str, Any]:
    tick = series_ticker.upper()
    base = VERIFIED_TWC_DAILY_MAX.get(tick)
    if not base:
        raise ValueError(f"{tick} is not in VERIFIED_TWC_DAILY_MAX")
    details = {}
    if base.get("same_station_model_location_id"):
        details["same_station_model_location_id"] = base["same_station_model_location_id"]
    return {
        **{k: v for k, v in base.items() if k != "same_station_model_location_id"},
        "series_ticker": tick,
        "measurement": "daily_max_temp_f",
        "settlement_source_family": "weather_company",
        "details": details,
    }


def run_twc_forecast(
    series_ticker: str = "KXHIGHNY",
    *,
    now: datetime | None = None,
    do_paper: bool = True,
    collect: bool = True,
    force_decision: bool = False,
) -> dict[str, Any]:
    """Collect → predict → paper for one TWC series; attach human callout."""
    store = FeedStore()
    now = now or datetime.now(timezone.utc)
    target = _twc_target(series_ticker)
    try:
        result = process_location(
            target,
            store,
            now=now,
            do_paper=do_paper,
            collect=collect,
            force_decision=force_decision,
        )
    finally:
        store.close()

    features = result.get("features_summary") or {}
    callout = format_max_callout(
        display_name=target.get("display_name") or series_ticker,
        climate_day=(result.get("forecast_context") or {}).get("climate_day")
        or features.get("climate_day")
        or "",
        point_median_f=result.get("point_median_f"),
        max_so_far=features.get("max_so_far"),
        decision_hour_local=result.get("decision_hour_local"),
        tz_name=target.get("timezone") or "America/New_York",
    )
    result["human_forecast"] = callout
    result["callout"] = callout
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_ROOT / f"twc_bet_{series_ticker.lower()}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    result["report"] = str(out_path)
    return result


def _best_live_candidate(result: dict[str, Any], *, dollars: Decimal) -> dict[str, Any] | None:
    """Pick a forecast-consistent +EV side sized to ~dollars — not the cheapest ask."""
    median = result.get("point_median_f")
    if median is None:
        return None
    median_f = float(median)
    max_so_far = (result.get("features_summary") or {}).get("max_so_far")

    best, rejected = select_forecast_consistent(
        list(result.get("ev_evaluations") or []),
        median_f=median_f,
        brackets=list(result.get("brackets") or []),
    )
    result["live_selection_rejected"] = rejected
    if best is None:
        return None

    ask = D(best["ask"])
    qty = fp_count(dollars / ask)
    if qty < D("0.01"):
        qty = D("0.01")
    fees = estimate_net_fee(qty, ask, multiplier=1.0, assume_taker=True, balance_precision=D("0.0001"))
    capital = ask * qty + fees
    while capital > dollars and qty > D("0.01"):
        qty = fp_count(qty - D("0.01"))
        fees = estimate_net_fee(qty, ask, multiplier=1.0, assume_taker=True, balance_precision=D("0.0001"))
        capital = ask * qty + fees
    why = format_bet_rationale(
        point_median_f=median_f,
        max_so_far=max_so_far,
        ticker=str(best["ticker"]),
        side=str(best["side"]),
        p=best["p"],
        ask=best["ask"],
        interval=best.get("interval"),
    )
    return {
        **best,
        "qty": str(qty),
        "fees": str(fees),
        "capital_required": str(fp_price(capital)),
        "dollars_cap": str(dollars),
        "why_buy": why,
        "selection_rule": selection_rule_text(),
    }


def place_capped_live_bet(
    result: dict[str, Any],
    config: AppConfig,
    *,
    dollars: float = 5.0,
    dry_run: bool = False,
    session_id: str | None = None,
    session_max: float | None = 20.0,
    budget_root: Path | None = None,
) -> dict[str, Any]:
    """Size a candidate and optionally submit via ExecutionEngine (never direct create_order).

    Live requires: mode+live_enabled, model_live_eligible=True (from config promotion
    flag), RiskManager, EV requalify at refreshed ask, and session spend ≤ session_max
    (default $20) with per-bet ≤ $5. force_decision cannot invent a live hour.
    """
    from datetime import datetime as _dt

    from kalshi_bot.ev.calculator import EvResult, evaluate_binary_contract
    from kalshi_bot.execution.engine import ExecutionEngine
    from kalshi_bot.models.base import Prediction

    dollars_d = D(str(dollars))
    if dollars_d <= ZERO or dollars_d > DEFAULT_PER_BET_MAX:
        return {"ok": False, "error": f"dollars must be in (0, {DEFAULT_PER_BET_MAX}]"}

    series = str(result.get("series_ticker") or "").upper()
    callout = result.get("callout") or result.get("human_forecast")
    sid = session_id or default_session_id()
    pol = SessionBudgetPolicy(
        session_id=sid,
        max_spend=D(str(session_max if session_max is not None else 20)),
        per_bet_max=DEFAULT_PER_BET_MAX,
    )

    if not result.get("probabilities_available"):
        return {
            "ok": False,
            "error": "probabilities unavailable — refusing live bet",
            "callout": callout,
            "status": result.get("status"),
            "reason": result.get("reason"),
        }

    # Live path: unsupported / forced decision hours are research-only.
    if result.get("force_decision") or result.get("status") == "unsupported_decision_time":
        if not dry_run:
            return {
                "ok": False,
                "error": (
                    "force_decision / unsupported_decision_time is research-only; "
                    "refusing live execution"
                ),
                "callout": callout,
                "live_order_submitted": False,
            }

    if series and series not in LIVE_SERIES_ALLOWLIST:
        return {
            "ok": False,
            "error": f"series {series} not in live allowlist {sorted(LIVE_SERIES_ALLOWLIST)}",
            "live_order_submitted": False,
        }

    left = budget_remaining(sid, policy=pol, root=budget_root)
    if dollars_d > left:
        return {
            "ok": False,
            "error": f"session budget remaining {left} < requested {dollars_d}",
            "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            "live_order_submitted": False,
        }
    dollars_d = min(dollars_d, left, DEFAULT_PER_BET_MAX)

    candidate = _best_live_candidate(result, dollars=dollars_d)
    if candidate is None:
        return {
            "ok": False,
            "error": "no candidate after selection filters (see selection_rejected)",
            "callout": callout,
            "paper_decision": result.get("paper_decision"),
            "selection_rejected": result.get("live_selection_rejected"),
            "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            "live_order_submitted": False,
        }

    out: dict[str, Any] = {
        "ok": True,
        "callout": callout,
        "candidate": candidate,
        "live_order_submitted": False,
        "dry_run": dry_run,
        "execution_path": "ExecutionEngine.place_individual",
        "session_id": sid,
        "dollars_cap": str(dollars_d),
        "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
    }
    if dry_run:
        out["decision"] = "dry_run_would_submit_via_execution_engine"
        out["note"] = (
            "Dry-run only. Live still requires model_live_eligible=True, "
            "config/store armed, session budget, and RiskManager."
        )
        return out

    # Live only when config promotion flag is on AND forecast stamped eligible.
    model_live_eligible = bool(result.get("model_live_eligible") is True)
    cfg_live = bool(getattr(config.models.weather, "obs_engine_live_eligible", False))
    if not model_live_eligible or not cfg_live:
        return {
            **out,
            "ok": False,
            "error": (
                "live blocked: need model_live_eligible=True and "
                "models.weather.obs_engine_live_eligible=true "
                f"(got model={model_live_eligible}, config={cfg_live})"
            ),
            "model_origin": result.get("model_origin"),
            "model_live_eligible": model_live_eligible,
            "obs_engine_live_eligible": cfg_live,
        }

    if config.mode != "live" or not config.live.enabled:
        return {
            **out,
            "ok": False,
            "error": f"config not armed for live (mode={config.mode}, live.enabled={config.live.enabled})",
        }

    store = Store(config.storage.sqlite_path)
    arm = arm_store_for_live(config, store)
    if not arm.get("ok"):
        return {**out, "ok": False, "error": arm.get("error"), "live_order_submitted": False}

    reservation = budget_reserve(
        dollars_d,
        series_ticker=series or "UNKNOWN",
        session_id=sid,
        policy=pol,
        root=budget_root,
        note=f"candidate={candidate.get('ticker')}:{candidate.get('side')}",
    )
    if not reservation.get("ok"):
        return {
            **out,
            "ok": False,
            "error": reservation.get("error"),
            "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            "live_order_submitted": False,
        }
    res_id = str(reservation["reservation_id"])
    out["session_reservation"] = reservation

    client = KalshiClient(config.api)
    try:
        state = store.get_state()
        if state.mode != "live" or not state.live_enabled:
            budget_release(
                res_id, session_id=sid, policy=pol, root=budget_root, reason="store_not_armed"
            )
            return {
                **out,
                "ok": False,
                "error": "store not armed for live (mode/live_enabled)",
            }

        ticker = str(candidate["ticker"])
        side = str(candidate["side"])
        mraw = client.get_market(ticker)
        market = mraw.get("market") if isinstance(mraw, dict) else None
        if not isinstance(market, dict):
            market = {"ticker": ticker}
        status = (market.get("status") or "").lower()
        if status and status not in ("active", "open", "initialized", ""):
            budget_release(
                res_id, session_id=sid, policy=pol, root=budget_root, reason=f"status={status}"
            )
            return {**out, "ok": False, "error": f"market status={status} not tradable"}

        raw_book = client.get_orderbook(ticker, depth=10)
        book = parse_orderbook(raw_book)
        ask = book.best_yes_ask if side == "yes" else book.best_no_ask
        if ask is None or ask <= ZERO or ask >= ONE:
            budget_release(res_id, session_id=sid, policy=pol, root=budget_root, reason="no_ask")
            return {**out, "ok": False, "error": f"no executable {side} ask at submit time"}

        # Immutable decision context → Prediction for EV requalify at refreshed ask.
        p_side = D(str(candidate["p"]))
        if side == "yes":
            p_yes = p_side
        else:
            p_yes = ONE - p_side
        # Conservative haircut: reuse uncertainty buffer as prediction uncertainty floor.
        unc = D(config.trading.uncertainty_buffer)
        pred = Prediction(
            market_ticker=ticker,
            p_yes=p_yes,
            p_yes_conservative=max(ZERO, p_yes - unc),
            uncertainty=unc,
            model_version=str(result.get("model_version") or result.get("model_origin") or "twc"),
            data_sources=[{"name": "twc_forecast_result"}],
            factors=[callout or ""],
            validation_evidence=str(
                result.get("validation_evidence")
                or "RESEARCH/TWC transfer — not settlement-calibrated for live"
            ),
            as_of=_dt.now(timezone.utc),
            details={
                "model_live_eligible": True,  # only reached if outer gate passed
                "point_median_f": result.get("point_median_f"),
                "model_origin": result.get("model_origin"),
                "twc_capped_intent": True,
                "dollars_cap": str(dollars_d),
            },
        )
        # Size toward dollar cap at refreshed ask.
        qty = min(D(candidate["qty"]), fp_count(dollars_d / ask))
        evs = evaluate_binary_contract(pred, book, config.trading, quantity=qty)
        ev = next((e for e in evs if e.side == side), None)
        if ev is None or not ev.qualifies:
            budget_release(res_id, session_id=sid, policy=pol, root=budget_root, reason="requalify_failed")
            return {
                **out,
                "ok": False,
                "error": "requalify_failed_at_refreshed_price",
                "refreshed_ask": str(ask),
                "ev": None if ev is None else {
                    "conservative_ev": str(ev.conservative_ev),
                    "executable_price": str(ev.executable_price),
                    "reason": ev.reason,
                    "qualifies": ev.qualifies,
                },
                "live_order_submitted": False,
                "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            }
        if ev.capital_required > dollars_d:
            budget_release(res_id, session_id=sid, policy=pol, root=budget_root, reason="over_dollars_cap")
            return {
                **out,
                "ok": False,
                "error": f"capital_required {ev.capital_required} exceeds dollars cap {dollars_d}",
                "live_order_submitted": False,
                "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            }

        # Cap quantity to dollar limit after EV sizing.
        if ev.executable_price > ZERO:
            max_qty = fp_count(dollars_d / ev.executable_price)
            if ev.quantity > max_qty:
                ev = EvResult(
                    side=ev.side,
                    quantity=max_qty,
                    executable_price=ev.executable_price,
                    fillable_quantity=min(ev.fillable_quantity, max_qty),
                    estimated_prob=ev.estimated_prob,
                    conservative_prob=ev.conservative_prob,
                    uncertainty=ev.uncertainty,
                    fees_total=ev.fees_total,
                    fees_per_contract=ev.fees_per_contract,
                    estimated_ev=ev.estimated_ev,
                    conservative_ev=ev.conservative_ev,
                    breakeven_prob=ev.breakeven_prob,
                    max_loss=min(ev.max_loss, dollars_d),
                    capital_required=min(ev.capital_required, dollars_d),
                    qualifies=ev.qualifies,
                    reason=ev.reason + f"; capped_qty={max_qty}",
                    details=dict(ev.details or {}),
                )

        opp_id = f"twc-gated-{ticker}-{side}-{result.get('decision_hour_local')}"
        engine = ExecutionEngine(client, store, config)
        order = engine.place_individual(
            market=market,
            ev=ev,
            opportunity_id=opp_id,
            correlation_keys=[
                f"series:{result.get('series_ticker') or market.get('event_ticker')}",
                f"twc:{result.get('location_id')}",
                f"session:{sid}",
            ],
            mode="live",
            model_live_eligible=True,
        )
        if order is None:
            budget_release(res_id, session_id=sid, policy=pol, root=budget_root, reason="engine_rejected")
            return {
                **out,
                "ok": False,
                "error": "ExecutionEngine rejected order (risk/eligibility/duplicate/price)",
                "live_order_submitted": False,
                "refreshed_ask": str(ask),
                "conservative_ev": str(ev.conservative_ev),
                "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
            }
        submitted = order.status not in ("rejected", "error", "canceled")
        if submitted:
            budget_commit(
                res_id,
                actual_spend=ev.capital_required,
                session_id=sid,
                policy=pol,
                root=budget_root,
                order_id=order.client_order_id or order.exchange_order_id,
            )
        else:
            budget_release(
                res_id, session_id=sid, policy=pol, root=budget_root, reason=f"order_status={order.status}"
            )
        out["live_order_submitted"] = submitted
        out["order"] = {
            "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id,
            "status": order.status,
            "ticker": ticker,
            "side": side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "fees_paid": order.fees_paid,
            "capital_required": str(ev.capital_required),
        }
        out["requalified_ev"] = {
            "conservative_ev": str(ev.conservative_ev),
            "executable_price": str(ev.executable_price),
            "quantity": str(ev.quantity),
            "reason": ev.reason,
        }
        out["session_budget"] = budget_snapshot(sid, policy=pol, root=budget_root)
        out["decision"] = f"live_{order.status}"
        return out
    except Exception as exc:
        logger.exception("live bet failed")
        try:
            budget_release(res_id, session_id=sid, policy=pol, root=budget_root, reason=str(exc)[:200])
        except Exception:
            pass
        try:
            store.audit("twc_live_error", str(exc), level="error")
        except Exception:
            pass
        return {
            **out,
            "ok": False,
            "error": str(exc),
            "live_order_submitted": False,
            "session_budget": budget_snapshot(sid, policy=pol, root=budget_root),
        }
    finally:
        client.close()


def weather_twc_bet(
    config: AppConfig,
    *,
    series_ticker: str = "KXHIGHNY",
    dollars: float = 5.0,
    live: bool = False,
    dry_run: bool = False,
    session_id: str | None = None,
    session_max: float = 20.0,
) -> dict[str, Any]:
    """CLI entry: forecast callout, then optional gated live bet via ExecutionEngine."""
    # force_decision is research/paper labeling only — never for live eligibility.
    forecast = run_twc_forecast(
        series_ticker,
        do_paper=True,
        collect=True,
        force_decision=bool(dry_run) and not live,
    )
    forecast["series_ticker"] = str(series_ticker).upper()

    # Eligibility: only when config flag is explicitly true (user-armed session).
    cfg_live = bool(getattr(config.models.weather, "obs_engine_live_eligible", False))
    tick = str(series_ticker).upper()
    if cfg_live and tick in LIVE_SERIES_ALLOWLIST:
        forecast["model_live_eligible"] = True
        forecast["validation_evidence"] = (
            forecast.get("validation_evidence")
            or "USER_AUTHORIZED_SESSION: obs_engine_live_eligible=true; "
            "PROMOTION_CRITERIA not fully met — capped session risk only."
        )
    elif forecast.get("model_live_eligible") is None:
        origin = str(forecast.get("model_origin") or "")
        forecast["model_live_eligible"] = False
        if origin.startswith("same_icao_transfer") or "transfer" in origin:
            forecast["validation_evidence"] = (
                forecast.get("validation_evidence")
                or "RESEARCH/TWC same-ICAO transfer — not live_eligible"
            )

    sid = session_id or default_session_id()
    payload: dict[str, Any] = {
        "callout": forecast.get("callout"),
        "human_forecast": forecast.get("human_forecast"),
        "status": forecast.get("status"),
        "point_median_f": forecast.get("point_median_f"),
        "max_so_far": (forecast.get("features_summary") or {}).get("max_so_far"),
        "decision_hour_local": forecast.get("decision_hour_local"),
        "probabilities_available": forecast.get("probabilities_available"),
        "paper_decision": forecast.get("paper_decision"),
        "why_buy": (forecast.get("paper_decision") or {}).get("why_buy"),
        "model_origin": forecast.get("model_origin"),
        "model_live_eligible": forecast.get("model_live_eligible"),
        "obs_engine_live_eligible": cfg_live,
        "live_requested": live,
        "dollars": dollars,
        "session_id": sid,
        "session_max": session_max,
        "session_budget": budget_snapshot(
            sid,
            policy=SessionBudgetPolicy(session_id=sid, max_spend=D(str(session_max))),
        ),
        "report": forecast.get("report"),
        "force_decision_used": bool(dry_run) and not live,
    }
    if not live:
        payload["live_order_submitted"] = False
        payload["note"] = (
            "Pass --live with obs_engine_live_eligible=true for capped submits. "
            "Session max $20 / $5 per bet. force_decision is research-only."
        )
        return payload

    bet = place_capped_live_bet(
        forecast,
        config,
        dollars=dollars,
        dry_run=dry_run,
        session_id=sid,
        session_max=session_max,
    )
    payload["bet"] = bet
    payload["why_buy"] = (bet.get("candidate") or {}).get("why_buy") or payload.get("why_buy")
    payload["live_order_submitted"] = bool(bet.get("live_order_submitted"))
    payload["session_budget"] = bet.get("session_budget") or payload["session_budget"]
    return payload
