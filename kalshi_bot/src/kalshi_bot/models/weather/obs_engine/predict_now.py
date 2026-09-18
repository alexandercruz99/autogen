"""Fresh research prediction for the next open NYC Central Park daily-max market."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.fees import estimate_net_fee
from kalshi_bot.api.orderbook import ExecutableBook, parse_orderbook
from kalshi_bot.config import AppConfig, load_config
from kalshi_bot.models.weather.nws_client import parse_market_date
from kalshi_bot.models.weather.obs_engine import MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.forecaster import ObsDrivenNycForecaster
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)


def _is_nyc_daily_high(ticker: str) -> bool:
    t = ticker.upper()
    if "HOUR" in t or "TEMPAT" in t:
        return False
    return any(t.startswith(p) for p in NYC_TARGET.series_prefixes)


def _pick_target_day(now: datetime) -> date:
    """Prefer remaining current local climate day; else next calendar day."""
    local = now.astimezone(ZoneInfo(NYC_TARGET.timezone))
    # After ~20:00 local, afternoon peak usually done; still allow same-day if markets open.
    return local.date()


def fetch_open_nyc_markets(client: KalshiClient) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for series in NYC_TARGET.series_prefixes:
            try:
                payload = client.get_markets(series_ticker=series, status="open", limit=200)
                batch = payload.get("markets") or []
            except Exception as exc:
                logger.warning("get_markets %s failed: %s", series, exc)
                continue
            for m in batch:
                ticker = m.get("ticker") or ""
                if _is_nyc_daily_high(ticker):
                    markets.append(m)
        return markets


def select_event_markets(
    markets: list[dict[str, Any]], *, prefer_day: date
) -> tuple[date | None, list[dict[str, Any]]]:
    """Group by target date; prefer prefer_day if any open markets exist, else earliest future."""
    by_day: dict[date, list[dict[str, Any]]] = {}
    for m in markets:
        d = parse_market_date(m.get("ticker") or "")
        if d is None:
            continue
        by_day.setdefault(d, []).append(m)
    if not by_day:
        return None, []
    if prefer_day in by_day:
        return prefer_day, by_day[prefer_day]
    future = sorted(d for d in by_day if d >= prefer_day)
    if future:
        return future[0], by_day[future[0]]
    past = sorted(by_day)
    return past[-1], by_day[past[-1]]


def research_predict_now(config: AppConfig | None = None, config_path: str | None = None) -> dict[str, Any]:
    """Generate a research forecast for open NYC daily-max brackets. Does NOT place orders."""
    config = config or load_config(config_path)
    now = datetime.now(timezone.utc)
    local_now = now.astimezone(ZoneInfo(NYC_TARGET.timezone))
    out: dict[str, Any] = {
        "status": "starting",
        "generation_time_utc": now.isoformat(),
        "generation_time_local": local_now.isoformat(),
        "timezone": NYC_TARGET.timezone,
        "station": {
            "display": NYC_TARGET.display_name,
            "ghcnd": NYC_TARGET.ghcnd_id,
            "cli": NYC_TARGET.cli_location_id,
            "metar": NYC_TARGET.metar_id,
            "lat": NYC_TARGET.lat,
            "lon": NYC_TARGET.lon,
            "settlement_source": NYC_TARGET.settlement_source,
        },
        "model_version": MODEL_VERSION,
        "live_eligible": False,
        "mode": "RESEARCH",
        "orders_placed": False,
        "note": "Research forecast only — not a validated trading signal; no live orders submitted.",
    }

    client = KalshiClient(config.api)
    model = ObsDrivenNycForecaster(config.models.weather)
    try:
        prefer = _pick_target_day(now)
        markets = fetch_open_nyc_markets(client)
        target_day, event_markets = select_event_markets(markets, prefer_day=prefer)
        out["prefer_day_local"] = prefer.isoformat()
        out["n_open_nyc_daily_markets_found"] = len(markets)
        if target_day is None or not event_markets:
            out["status"] = "unavailable"
            out["reason"] = "No open Central Park daily-max markets found via Kalshi API"
            return out

        horizon = "same_day" if target_day == prefer else "other_day"
        out["target_date"] = target_day.isoformat()
        out["horizon"] = horizon
        if horizon != "same_day":
            out["horizon_warning"] = (
                "Selected market date differs from local today. Same-day remaining-rise model "
                "is validated for intraday decisions; day-ahead use is research-only with weaker evidence."
            )

        # One prediction pass for features/distribution, then apply to each bracket
        # Sort markets for stable bracket table
        def _sort_key(m: dict[str, Any]):
            st = (m.get("strike_type") or "").lower()
            floor = m.get("floor_strike")
            cap = m.get("cap_strike")
            return (st, floor if floor is not None else -999, cap if cap is not None else 999)

        event_markets = sorted(event_markets, key=_sort_key)
        anchor = event_markets[0]
        pred0 = model.predict(anchor, "Climate and Weather")
        if not pred0.supported:
            out["status"] = "unavailable"
            out["reason"] = pred0.skip_reason or "model unsupported"
            out["validation_evidence"] = pred0.validation_evidence
            return out

        dist = (pred0.details or {}).get("distribution") or {}
        feats = (pred0.details or {}).get("features") or {}
        out["training_cutoff"] = (model._artifact or {}).get("train_end_day")
        out["latest_observation"] = (pred0.details or {}).get("feature_provenance")
        out["observed_max_so_far_f"] = feats.get("max_so_far")
        out["predicted_median_f"] = dist.get("q50")
        out["prediction_intervals"] = {"q10": dist.get("q10"), "q50": dist.get("q50"), "q90": dist.get("q90")}
        out["distribution_mean_f"] = dist.get("mean_f")
        out["distribution_method"] = dist.get("method")
        out["factors"] = pred0.factors
        out["nws_benchmark"] = (pred0.details or {}).get("nws_benchmark")
        out["validation_evidence"] = pred0.validation_evidence
        out["model_live_eligible"] = False

        brackets = []
        for m in event_markets:
            ticker = m.get("ticker") or ""
            interval = interval_from_market(m)
            pred = model.predict(m, "Climate and Weather")
            book_raw = None
            book: ExecutableBook | None = None
            retrieved_at = datetime.now(timezone.utc).isoformat()
            try:
                book_raw = client.get_orderbook(ticker, depth=10)
                book = parse_orderbook(book_raw) if book_raw else None
            except Exception as exc:
                logger.warning("orderbook %s: %s", ticker, exc)

            yes_ask = book.best_yes_ask if book else None
            no_ask = book.best_no_ask if book else None
            p = pred.p_yes
            p_cons = pred.p_yes_conservative
            fees = None
            ev_yes = None
            if yes_ask is not None and yes_ask > ZERO:
                qty = D("1")
                fees = estimate_net_fee(
                    qty,
                    yes_ask,
                    multiplier=config.trading.fee_multiplier,
                    assume_taker=config.trading.assume_taker,
                    balance_precision=config.trading.balance_precision,
                )
                c = fees / qty if qty else ZERO
                ev_yes = str(p_cons - yes_ask - c - D(config.trading.uncertainty_buffer))

            brackets.append(
                {
                    "ticker": ticker,
                    "title": m.get("title") or m.get("yes_sub_title") or "",
                    "strike_type": m.get("strike_type"),
                    "floor_strike": m.get("floor_strike"),
                    "cap_strike": m.get("cap_strike"),
                    "interval": {
                        "op": interval.op if interval else None,
                        "low": interval.low if interval else None,
                        "high": interval.high if interval else None,
                    },
                    "p_yes": str(p),
                    "p_yes_conservative": str(p_cons),
                    "uncertainty": str(pred.uncertainty),
                    "yes_ask": str(yes_ask) if yes_ask is not None else None,
                    "no_ask": str(no_ask) if no_ask is not None else None,
                    "orderbook_retrieved_at_utc": retrieved_at,
                    "estimated_fee_1_contract": str(fees) if fees is not None else None,
                    "conservative_ev_yes_approx": ev_yes,
                    "qualifies_as_validated_signal": False,
                }
            )

        # Bracket sum check
        try:
            s = sum(Decimal(b["p_yes"]) for b in brackets)
            out["bracket_prob_sum"] = str(s)
            out["bracket_prob_sum_note"] = (
                "Sum ≈ 1 only if the listed markets form an exhaustive partition of outcomes"
            )
        except Exception:
            pass

        out["brackets"] = brackets
        out["status"] = "ok"
        out["research_vs_trading"] = (
            "RESEARCH forecast with executable prices for context. "
            "Not live_eligible; do not treat EV figures as validated edge."
        )

        # Persist
        data_dir = Path(getattr(config.models.weather, "obs_engine_data_dir", None) or "data/obs_engine")
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "latest_research_prediction.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        out["saved_path"] = str(path)
        return out
    finally:
        model.close()
        client.close()
