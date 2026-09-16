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
)
from kalshi_bot.models.weather.obs_engine.multi.pipeline import process_location
from kalshi_bot.models.weather.obs_engine.multi.registry import VERIFIED_TWC_DAILY_MAX
from kalshi_bot.money import D, ONE, ZERO, fp_count, fp_price

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path("data/obs_engine/multi")


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
        "selection_rule": (
            f"forecast-consistent (≤{MAX_STRIKE_DISTANCE_F}°F from median), "
            f"model_p≥{MIN_MODEL_P}, then max EV — not cheapest ask"
        ),
    }


def place_capped_live_bet(
    result: dict[str, Any],
    config: AppConfig,
    *,
    dollars: float = 5.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit one live order sized to ``dollars`` (default $5).

    Explicit user-requested capped bet off the TWC point forecast. Enforces dollar
    cap + live config arming; does not require obs_engine_live_eligible promotion.
    """
    dollars_d = D(str(dollars))
    if dollars_d <= ZERO or dollars_d > D("25"):
        return {"ok": False, "error": "dollars must be in (0, 25]"}

    callout = result.get("callout") or result.get("human_forecast")
    if not result.get("probabilities_available"):
        return {
            "ok": False,
            "error": "probabilities unavailable — refusing live bet",
            "callout": callout,
            "status": result.get("status"),
            "reason": result.get("reason"),
        }

    candidate = _best_live_candidate(result, dollars=dollars_d)
    if candidate is None:
        return {
            "ok": False,
            "error": "no positive-EV bracket after fees/buffer",
            "callout": callout,
            "paper_decision": result.get("paper_decision"),
        }

    out: dict[str, Any] = {
        "ok": True,
        "callout": callout,
        "candidate": candidate,
        "live_order_submitted": False,
        "dry_run": dry_run,
    }
    if dry_run:
        out["decision"] = "dry_run_would_submit"
        return out

    if config.mode != "live" or not config.live.enabled:
        return {
            **out,
            "ok": False,
            "error": f"config not armed for live (mode={config.mode}, live.enabled={config.live.enabled})",
        }

    store = Store(config.storage.sqlite_path)
    client = KalshiClient(config.api)
    try:
        store.update_state(mode="live", live_enabled=True)
        bal = client.get_balance()
        bal_d = D(str(bal.get("balance_dollars") or "0"))
        if bal_d < dollars_d:
            return {**out, "ok": False, "error": f"balance ${bal_d} < ${dollars_d} cap", "balance": bal}

        ticker = candidate["ticker"]
        side = candidate["side"]
        mraw = client.get_market(ticker)
        market = mraw.get("market") if isinstance(mraw, dict) else None
        if not isinstance(market, dict):
            market = {"ticker": ticker}

        raw_book = client.get_orderbook(ticker, depth=10)
        book = parse_orderbook(raw_book)
        ask = book.best_yes_ask if side == "yes" else book.best_no_ask
        if ask is None or ask <= ZERO or ask >= ONE:
            return {**out, "ok": False, "error": f"no executable {side} ask at submit time"}

        qty = min(D(candidate["qty"]), fp_count(dollars_d / ask))
        if qty < D("0.01"):
            return {**out, "ok": False, "error": "qty below minimum after live ask resize"}
        fees = estimate_net_fee(
            qty,
            ask,
            multiplier=config.trading.fee_multiplier,
            assume_taker=True,
            balance_precision=D(config.trading.balance_precision),
        )
        capital = ask * qty + fees
        while capital > dollars_d and qty > D("0.01"):
            qty = fp_count(qty - D("0.01"))
            fees = estimate_net_fee(
                qty,
                ask,
                multiplier=config.trading.fee_multiplier,
                assume_taker=True,
                balance_precision=D(config.trading.balance_precision),
            )
            capital = ask * qty + fees
        if capital > dollars_d:
            return {**out, "ok": False, "error": f"cannot fit under ${dollars} at ask {ask}"}

        client_order_id = f"twc-live-{uuid.uuid4()}"
        if side == "yes":
            book_side = "bid"
            price = str(fp_price(ask))
        else:
            book_side = "ask"
            price = str(fp_price(ONE - ask))

        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": str(fp_count(qty)),
            "price": price,
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": False,
        }
        order_rec = OrderRecord(
            client_order_id=client_order_id,
            created_at=utcnow(),
            mode="live",
            kind="individual",
            market_ticker=ticker,
            event_ticker=str(market.get("event_ticker") or ""),
            side=side,
            quantity=str(fp_count(qty)),
            limit_price=str(fp_price(ask)),
            status="pending",
            reservation_id="",
            opportunity_id=f"twc-user-live-{ticker}-{side}",
            details_json=dumps(
                {
                    "callout": callout,
                    "dollars_cap": str(dollars_d),
                    "capital_required": str(fp_price(capital)),
                    "model_origin": result.get("model_origin"),
                    "point_median_f": result.get("point_median_f"),
                    "user_requested": True,
                }
            ),
        )
        store.save_order(order_rec)
        store.audit(
            "twc_live_submit",
            f"submitting {side} {ticker} x{qty} @ {ask} (~${fp_price(capital)})",
            details={"callout": callout, "client_order_id": client_order_id},
        )

        resp = client.create_order_v2(body)
        order_payload = resp.get("order") if isinstance(resp.get("order"), dict) else resp
        order_rec.exchange_order_id = str(
            order_payload.get("order_id") or order_payload.get("id") or ""
        )
        filled = str(
            order_payload.get("fill_count_fp")
            or order_payload.get("fill_count")
            or order_payload.get("filled_quantity")
            or "0"
        )
        remaining = D(
            str(
                order_payload.get("remaining_count_fp")
                or order_payload.get("remaining_count")
                or qty
            )
        )
        order_rec.filled_quantity = filled
        if D(filled) > ZERO and remaining > ZERO:
            order_rec.status = "partial"
        elif remaining <= ZERO or D(filled) >= qty:
            order_rec.status = "filled"
        else:
            order_rec.status = "resting"
        if order_payload.get("avg_fill_price") or order_payload.get("average_fill_price"):
            order_rec.avg_fill_price = str(
                order_payload.get("avg_fill_price") or order_payload.get("average_fill_price")
            )
        if order_payload.get("fees_paid") or order_payload.get("average_fee_paid"):
            order_rec.fees_paid = str(
                order_payload.get("fees_paid") or order_payload.get("average_fee_paid")
            )
        prev = json.loads(order_rec.details_json or "{}")
        prev["exchange_response_keys"] = list(resp.keys()) if isinstance(resp, dict) else []
        prev["order_status_raw"] = order_payload.get("status")
        order_rec.details_json = dumps(prev)
        store.save_order(order_rec)

        if order_rec.status in ("filled", "partial") and D(order_rec.filled_quantity or "0") > ZERO:
            store.save_position(
                PositionRecord(
                    id=str(uuid.uuid4()),
                    opened_at=utcnow(),
                    mode="live",
                    kind="individual",
                    market_ticker=ticker,
                    event_ticker=str(market.get("event_ticker") or ""),
                    side=side,
                    quantity=order_rec.filled_quantity,
                    avg_price=order_rec.avg_fill_price or order_rec.limit_price,
                    fees_paid=order_rec.fees_paid,
                    status="open",
                    details_json=dumps({"client_order_id": client_order_id, "callout": callout}),
                )
            )
        store.audit(
            "twc_live_result",
            f"{order_rec.status} {ticker} {side}",
            details={
                "exchange_order_id": order_rec.exchange_order_id,
                "filled": order_rec.filled_quantity,
                "status": order_rec.status,
            },
        )
        out["live_order_submitted"] = True
        out["order"] = {
            "client_order_id": order_rec.client_order_id,
            "exchange_order_id": order_rec.exchange_order_id,
            "status": order_rec.status,
            "ticker": ticker,
            "side": side,
            "quantity": str(fp_count(qty)),
            "limit_price": str(fp_price(ask)),
            "filled_quantity": order_rec.filled_quantity,
            "avg_fill_price": order_rec.avg_fill_price,
            "fees_paid": order_rec.fees_paid,
            "capital_required": str(fp_price(capital)),
        }
        out["balance_before"] = bal
        out["decision"] = f"live_{order_rec.status}"
        return out
    except Exception as exc:
        logger.exception("live bet failed")
        try:
            store.audit("twc_live_error", str(exc), level="error")
        except Exception:
            pass
        return {**out, "ok": False, "error": str(exc)}
    finally:
        client.close()


def weather_twc_bet(
    config: AppConfig,
    *,
    series_ticker: str = "KXHIGHNY",
    dollars: float = 5.0,
    live: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CLI entry: forecast callout, then optional capped live bet."""
    forecast = run_twc_forecast(
        series_ticker,
        do_paper=True,
        collect=True,
        force_decision=bool(live or dry_run),
    )
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
        "live_requested": live,
        "dollars": dollars,
        "report": forecast.get("report"),
    }
    if not live:
        payload["live_order_submitted"] = False
        payload["note"] = (
            "Pass --live to submit a capped live order (requires auth). "
            "why_buy explains the weather story; we do not buy just because a ticket is cheap."
        )
        return payload

    bet = place_capped_live_bet(forecast, config, dollars=dollars, dry_run=dry_run)
    payload["bet"] = bet
    payload["why_buy"] = (bet.get("candidate") or {}).get("why_buy") or payload.get("why_buy")
    payload["live_order_submitted"] = bool(bet.get("live_order_submitted"))
    return payload
