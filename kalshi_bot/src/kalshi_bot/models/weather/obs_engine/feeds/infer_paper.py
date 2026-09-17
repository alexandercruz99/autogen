"""Inference + paper simulation for station_v2 (never live orders).

Uses ``multi.predict.predict_station_v2`` so operating inference matches
calibration / evaluation / replay for identical inputs and context.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.models.weather.obs_engine.feeds.climate_day import (
    SUPPORTED_DECISION_HOURS_LOCAL,
    civil_local,
    is_supported_decision_time,
    lst_climate_day,
    next_supported_decision_utc,
)
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    STATION_V2_FEATURES,
)
from kalshi_bot.models.weather.obs_engine.feeds.paper_sim import PaperLedger
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.markets import fetch_open_series_markets, select_event_markets
from kalshi_bot.models.weather.obs_engine.multi.predict import (
    DEFAULT_CALIB_PATH,
    DEFAULT_MODEL_PATH,
    load_operating_model,
    predict_station_v2,
)
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)

MODEL_PATH = DEFAULT_MODEL_PATH
CALIB_PATH = DEFAULT_CALIB_PATH
LEGACY_MODEL_PATH = Path("data/obs_engine/feeds/models/station_corrected_v1.joblib")


def _load_operating_model() -> tuple[dict[str, Any] | None, str, str]:
    return load_operating_model(MODEL_PATH)


def run_infer_and_paper(store: FeedStore, feature_bundle: dict[str, Any]) -> dict[str, Any]:
    from kalshi_bot.api.client import KalshiClient
    from kalshi_bot.api.fees import estimate_net_fee
    from kalshi_bot.api.orderbook import parse_orderbook
    from kalshi_bot.config import load_config

    now = datetime.now(timezone.utc)
    tz_name = feature_bundle.get("tz_name") or "America/New_York"
    local = civil_local(now, tz_name)
    climate_day = (
        date.fromisoformat(feature_bundle["climate_day"])
        if feature_bundle.get("climate_day")
        else lst_climate_day(now, tz_name)
    )
    metar_id = feature_bundle.get("metar_id") or "KNYC"
    location_id = feature_bundle.get("location_id") or "nyc_central_park"
    series_ticker = feature_bundle.get("series_ticker") or "HIGHNY"

    out: dict[str, Any] = {
        "ok": False,
        "mode": "RESEARCH",
        "live_orders": False,
        "live_order_submitted": False,
        "generated_at_utc": now.isoformat(),
        "generated_at_local": local.isoformat(),
        "target_climate_day": climate_day.isoformat(),
        "feature_schema_version": feature_bundle.get("feature_schema_version"),
        "coverage": feature_bundle.get("coverage"),
        "attribution": feature_bundle.get("attribution"),
        "calibration_status": None,
        "probabilities_available": False,
        "paper_decision": None,
        "location_id": location_id,
    }

    supported, decision_hour = is_supported_decision_time(now, tz_name=tz_name)
    out["decision_hour_local"] = decision_hour
    out["supported_decision_hours_local"] = list(SUPPORTED_DECISION_HOURS_LOCAL)
    if not supported:
        nxt = next_supported_decision_utc(now, tz_name=tz_name)
        out.update(
            {
                "status": "unsupported_decision_time",
                "reason": (
                    f"Actionable forecasts restricted to local hours {SUPPORTED_DECISION_HOURS_LOCAL}; "
                    f"now={local.strftime('%H:%M %Z')}"
                ),
                "next_supported_run_utc": nxt.isoformat(),
                "next_supported_run_local": civil_local(nxt, tz_name).isoformat(),
                "historical_replay_ok": True,
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_unsupported_decision_time",
            "reason": out["reason"],
            "live_blocked": True,
        }
        store.save_paper_decision(
            ticker=None,
            side=None,
            decision=out["paper_decision"]["decision"],
            reason=out["paper_decision"]["reason"],
            details={"live_blocked": True, "next_supported_run_utc": out["next_supported_run_utc"]},
        )
        _mirror(out)
        return out

    cov = feature_bundle.get("coverage") or {}
    if feature_bundle.get("max_so_far") is None or not cov.get("adequate"):
        out.update(
            {
                "status": "insufficient_data",
                "reason": "Essential station coverage inadequate for verified daytime max_so_far",
                "coverage_notes": (cov.get("notes") if isinstance(cov, dict) else None),
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_insufficient_data",
            "reason": out["reason"],
            "live_blocked": True,
        }
        store.save_paper_decision(
            ticker=None,
            side=None,
            decision="blocked_insufficient_data",
            reason=out["reason"],
            details={"live_blocked": True, "coverage": cov},
        )
        _mirror(out)
        return out

    feats = feature_bundle.get("features") or {}
    cli = feature_bundle.get("cli_applied")
    attribution = dict(feature_bundle.get("attribution") or {})
    attribution["features_consumed_by_model"] = list(STATION_V2_FEATURES)
    attribution["feeds_contributing_to_prediction"] = [f"metar_{metar_id}"]
    attribution["feeds_collected_not_consumed"] = [
        f for f in (attribution.get("feeds_collected") or []) if f != f"metar_{metar_id}"
    ]
    attribution["settlement_constraints_applied"] = []
    if cli:
        attribution["settlement_constraints_applied"].append(
            f"cli_{'prelim' if cli.get('is_preliminary') else 'final'}_floor_{cli.get('max_temp_f')}"
        )
    out["attribution"] = attribution

    ctx = ForecastContext(
        location_id=location_id,
        series_ticker=series_ticker,
        measurement="daily_max_temp_f",
        settlement_source_family="nws_cli",
        climate_day=climate_day,
        decision_time_utc=now,
        horizon="same_day",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        decision_hour_local=decision_hour,
        timezone=tz_name,
        metar_id=metar_id,
        cli_location_id=feature_bundle.get("cli_location_id"),
        mode="RESEARCH",
    )
    unified = predict_station_v2(
        context=ctx,
        features=feats,
        max_so_far=float(feature_bundle["max_so_far"]),
        coverage_adequate=True,
        decision_hour_local=decision_hour,
        cli_applied=cli,
        model_path=MODEL_PATH,
        calib_path=CALIB_PATH,
    )
    out["calibration_status"] = unified.calibration_status
    out["observation_constraint"] = unified.observation_constraint
    out["forecast_context"] = ctx.as_dict()

    if unified.status in ("model_unavailable", "schema_mismatch"):
        out.update(
            {
                "status": unified.status,
                "reason": unified.reason,
                "model_artifact": unified.model_artifact,
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_unsupported_model",
            "reason": unified.reason,
            "live_blocked": True,
        }
        store.save_paper_decision(
            ticker=None,
            side=None,
            decision="blocked_unsupported_model",
            reason=unified.reason or "",
            details={"live_blocked": True},
        )
        _mirror(out)
        return out

    if not unified.probabilities_available:
        out.update(
            {
                "status": "probabilities_unavailable",
                "reason": unified.reason,
                "remain_q10_q50_q90": unified.remain_q10_q50_q90,
                "point_median_f": unified.point_median_f,
                "quantiles_crossed": unified.quantiles_crossed,
                "model_artifact": unified.model_artifact,
                "probabilities_available": False,
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_calibration_unavailable",
            "reason": unified.reason,
            "live_blocked": True,
            "research_point_forecast": unified.point_median_f,
        }
        store.save_paper_decision(
            ticker=None,
            side=None,
            decision="blocked_calibration_unavailable",
            reason=unified.reason or "",
            details={"live_blocked": True},
        )
        _mirror(out)
        return out

    dist = unified.distribution
    assert dist is not None
    out.update(
        {
            "ok": True,
            "status": "ok",
            "model_artifact": unified.model_artifact,
            "feature_set": FEATURE_SCHEMA_VERSION,
            "model_version": unified.model_version,
            "max_so_far": float(feature_bundle["max_so_far"]),
            "remain_q10_q50_q90": unified.remain_q10_q50_q90,
            "quantiles_crossed_before_sort": unified.quantiles_crossed,
            "point_median_f": unified.point_median_f,
            "distribution": dist.as_dict(),
            "probabilities_available": True,
            "cli_applied": cli,
            "missing": feature_bundle.get("missing"),
        }
    )

    cfg = load_config("config.yaml" if Path("config.yaml").exists() else "config.example.yaml")
    client = KalshiClient(cfg.api)
    ledger = PaperLedger()
    try:
        prefer = climate_day
        # Only the configured settlement series — never mix NWS CLI models with
        # Weather Company tickers (e.g. HIGHNY ≠ KXHIGHNY).
        markets = fetch_open_series_markets(client, [series_ticker])
        target_day, event_markets = select_event_markets(markets, prefer_day=prefer)
        out["target_date"] = target_day.isoformat() if target_day else None
        if target_day is None:
            out["status"] = "no_open_markets"
            out["paper_decision"] = {
                "decision": "blocked_no_markets",
                "reason": "no open markets",
                "live_blocked": True,
            }
            _mirror(out)
            return out
        if target_day != prefer:
            out["status"] = "unsupported_horizon"
            out["horizon"] = "other_day"
            out["reason"] = (
                f"Open markets are for {target_day.isoformat()} but forecast climate day is "
                f"{prefer.isoformat()}; refusing to apply same-day distribution to another day"
            )
            out["probabilities_available"] = False
            out["paper_decision"] = {
                "decision": "blocked_unsupported_horizon",
                "reason": out["reason"],
                "live_blocked": True,
            }
            store.save_paper_decision(
                ticker=None,
                side=None,
                decision="blocked_unsupported_horizon",
                reason=out["reason"],
                details={
                    "live_blocked": True,
                    "prefer": prefer.isoformat(),
                    "market_day": target_day.isoformat(),
                },
            )
            out["brackets"] = []
            store.save_prediction(
                model_version=str(unified.model_version or "station_v2"),
                feature_set=FEATURE_SCHEMA_VERSION,
                target_day=prefer.isoformat(),
                ticker=None,
                prediction=out,
            )
            _mirror(out)
            return out

        out["horizon"] = "same_day"
        brackets = []
        evaluations: list[dict[str, Any]] = []
        quote_ts = datetime.now(timezone.utc).isoformat()
        for m in sorted(event_markets, key=lambda x: (x.get("strike_type") or "", x.get("floor_strike") or 0)):
            iv = interval_from_market(m)
            if iv is None:
                continue
            p_yes = dist.p_interval(iv)
            p_no = ONE - p_yes
            ticker = m.get("ticker")
            yes_ask = no_ask = None
            yes_depth = no_depth = None
            try:
                raw = client.get_orderbook(ticker, depth=5)
                book = parse_orderbook(raw)
                yes_ask = book.best_yes_ask
                no_ask = book.best_no_ask
                yd = book.yes_ask_depth()
                nd = book.no_ask_depth()
                yes_depth = str(yd[0].size) if yd else None
                no_depth = str(nd[0].size) if nd else None
            except Exception as exc:
                logger.info("orderbook %s: %s", ticker, exc)

            brackets.append(
                {
                    "ticker": ticker,
                    "p_yes": str(p_yes),
                    "p_no": str(p_no),
                    "yes_ask": str(yes_ask) if yes_ask is not None else None,
                    "no_ask": str(no_ask) if no_ask is not None else None,
                    "yes_ask_size": yes_depth,
                    "no_ask_size": no_depth,
                    "quote_ts_utc": quote_ts,
                    "interval": {"op": iv.op, "low": iv.low, "high": iv.high},
                }
            )

            for side, p, ask, depth_s in (
                ("yes", p_yes, yes_ask, yes_depth),
                ("no", p_no, no_ask, no_depth),
            ):
                if ask is None or ask <= ZERO or ask >= ONE:
                    evaluations.append(
                        {
                            "ticker": ticker,
                            "side": side,
                            "decision": "blocked_unavailable_prices",
                            "reason": f"no executable {side} ask",
                            "p": str(p),
                            "ev": None,
                        }
                    )
                    continue
                if depth_s is None or D(depth_s) < D("1"):
                    evaluations.append(
                        {
                            "ticker": ticker,
                            "side": side,
                            "decision": "blocked_insufficient_depth",
                            "reason": f"{side} ask depth < 1",
                            "p": str(p),
                            "ask": str(ask),
                            "ev": None,
                        }
                    )
                    continue
                qty = D("1")
                fees = estimate_net_fee(
                    qty,
                    ask,
                    multiplier=cfg.trading.fee_multiplier,
                    assume_taker=True,
                    balance_precision=cfg.trading.balance_precision,
                )
                buffer = D(cfg.trading.uncertainty_buffer)
                ev = p - ask - (fees / qty) - buffer
                evaluations.append(
                    {
                        "ticker": ticker,
                        "side": side,
                        "decision": "ev_evaluated",
                        "reason": f"EV={ev}",
                        "p": str(p),
                        "ask": str(ask),
                        "fees": str(fees),
                        "uncertainty_buffer": str(buffer),
                        "ev": str(ev),
                        "qty": str(qty),
                        "quote_ts_utc": quote_ts,
                    }
                )

        paper_decision = None
        positive = [e for e in evaluations if e.get("ev") is not None and D(e["ev"]) > ZERO]
        negative = [e for e in evaluations if e.get("ev") is not None and D(e["ev"]) <= ZERO]
        for e in evaluations:
            if e.get("ev") is not None and D(e["ev"]) <= ZERO:
                e["decision"] = "paper_skip_negative_ev"
                e["reason"] = f"EV={e['ev']} ≤ 0 after fees/buffer — no simulated entry"
            store.save_paper_decision(
                ticker=e.get("ticker"),
                side=e.get("side"),
                decision=e["decision"],
                reason=e["reason"],
                details={**e, "live_blocked": True, "live_order_submitted": False},
            )

        if positive:
            best = max(positive, key=lambda e: D(e["ev"]))
            oid = f"paper-{best['ticker']}-{best['side']}-{climate_day.isoformat()}-{decision_hour}"
            sim = ledger.try_simulate_fill(
                client_order_id=oid,
                ticker=best["ticker"],
                side=best["side"],
                qty=D(best["qty"]),
                price=D(best["ask"]),
                fees=D(best["fees"]),
                decision_reason="paper_sim_fill_unvalidated",
                details=best,
                location_id=location_id,
                series_ticker=series_ticker,
                quote_ts_utc=quote_ts,
                market_ts_utc=quote_ts,
            )
            if sim.get("ok") and sim.get("filled"):
                paper_decision = {
                    **best,
                    "decision": "paper_sim_fill_unvalidated",
                    "reason": (
                        f"Simulated taker fill at ask (EV={best['ev']}); unvalidated exploratory; "
                        "live order NOT submitted"
                    ),
                    "sim": sim,
                    "live_blocked": True,
                }
            elif sim.get("reason") == "duplicate_client_order_id":
                paper_decision = {
                    **best,
                    "decision": "paper_skip_duplicate",
                    "reason": "duplicate simulated entry prevented",
                    "sim": sim,
                    "live_blocked": True,
                }
            else:
                paper_decision = {
                    **best,
                    "decision": "paper_skip_risk_or_cash",
                    "reason": sim.get("reason") or "sim fill rejected",
                    "sim": sim,
                    "live_blocked": True,
                }
            store.save_paper_decision(
                ticker=paper_decision.get("ticker"),
                side=paper_decision.get("side"),
                decision=paper_decision["decision"],
                reason=paper_decision["reason"],
                details={**paper_decision, "live_order_submitted": False},
            )
        elif negative:
            paper_decision = {
                **negative[0],
                "decision": "paper_skip_negative_ev",
                "reason": negative[0]["reason"],
                "live_blocked": True,
            }
        elif evaluations:
            paper_decision = {**evaluations[0], "live_blocked": True}
        else:
            paper_decision = {
                "decision": "paper_skip_no_brackets",
                "reason": "no evaluable brackets",
                "live_blocked": True,
            }

        out["brackets"] = brackets
        out["ev_evaluations"] = evaluations
        out["paper_decision"] = paper_decision
        out["paper_ledger"] = ledger.snapshot()
        store.save_prediction(
            model_version=str(unified.model_version or "station_v2"),
            feature_set=FEATURE_SCHEMA_VERSION,
            target_day=target_day.isoformat(),
            ticker=(brackets[0]["ticker"] if brackets else None),
            prediction=out,
        )
        _mirror(out)
    finally:
        client.close()
        ledger.close()
    return out


def _mirror(out: dict[str, Any]) -> None:
    Path("data/obs_engine/feeds").mkdir(parents=True, exist_ok=True)
    Path("data/obs_engine/feeds/latest_prediction.json").write_text(json.dumps(out, indent=2, default=str))
    Path("data/obs_engine/latest_research_prediction.json").write_text(json.dumps(out, indent=2, default=str))
