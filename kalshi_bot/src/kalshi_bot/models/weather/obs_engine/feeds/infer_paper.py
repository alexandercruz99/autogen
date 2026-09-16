"""Inference + paper simulation for station_v2 (never live orders)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from kalshi_bot.models.weather.distribution import from_empirical_residuals, truncate_below
from kalshi_bot.models.weather.obs_engine.feeds.calibration import load_calibration, pick_residuals
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
    vector_for_model,
)
from kalshi_bot.models.weather.obs_engine.feeds.paper_sim import PaperLedger
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/obs_engine/feeds/models/station_corrected_v2.joblib")
CALIB_PATH = Path("data/obs_engine/feeds/models/station_corrected_v2_calibration.json")
# Fallback to v1 only if schema matches — otherwise refuse
LEGACY_MODEL_PATH = Path("data/obs_engine/feeds/models/station_corrected_v1.joblib")


def _load_operating_model() -> tuple[dict[str, Any] | None, str, str]:
    import joblib

    if MODEL_PATH.exists():
        blob = joblib.load(MODEL_PATH)
        schema = blob.get("feature_schema_version") or blob.get("feature_set")
        if schema not in (FEATURE_SCHEMA_VERSION, "station_v2.1", "local_v2"):
            # allow station_v2.1
            pass
        if blob.get("feature_names") != STATION_V2_FEATURES and set(blob.get("feature_names") or []) != set(
            STATION_V2_FEATURES
        ):
            # strict: names must match exactly in order
            if list(blob.get("feature_names") or []) != STATION_V2_FEATURES:
                return None, str(MODEL_PATH), "schema_mismatch"
        return blob, str(MODEL_PATH), "ok"
    return None, str(MODEL_PATH), "model_missing"


def _soft_floor_from_obs(max_so_far: float) -> float | None:
    """METAR tempFloat is not an official whole-°F CLI floor.

    We do **not** ceil 69.08 → 70. Observation constraint uses the raw max_so_far
    as a continuous lower bound on the continuous forecast before integer binning
    in the residual distribution; truncate_below ceils for integer support — so we
    only apply truncate_below when we have a whole-°F CLI value.
    For METAR-only, encode constraint via distribution details without claiming
    official minimum of ceil(max_so_far).
    """
    return None  # CLI path applies whole °F; METAR alone → no integer floor claim


def run_infer_and_paper(store: FeedStore, feature_bundle: dict[str, Any]) -> dict[str, Any]:
    from kalshi_bot.api.client import KalshiClient
    from kalshi_bot.api.fees import estimate_net_fee
    from kalshi_bot.api.orderbook import parse_orderbook
    from kalshi_bot.config import load_config
    from kalshi_bot.models.weather.obs_engine.predict_now import (
        fetch_open_nyc_markets,
        select_event_markets,
    )

    now = datetime.now(timezone.utc)
    local = civil_local(now)
    climate_day = date.fromisoformat(feature_bundle["climate_day"]) if feature_bundle.get("climate_day") else lst_climate_day(now)

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
    }

    # --- Gate: supported decision time ---
    supported, decision_hour = is_supported_decision_time(now)
    out["decision_hour_local"] = decision_hour
    out["supported_decision_hours_local"] = list(SUPPORTED_DECISION_HOURS_LOCAL)
    if not supported:
        nxt = next_supported_decision_utc(now)
        out.update(
            {
                "status": "unsupported_decision_time",
                "reason": (
                    f"Actionable forecasts restricted to local hours {SUPPORTED_DECISION_HOURS_LOCAL}; "
                    f"now={local.strftime('%H:%M %Z')}"
                ),
                "next_supported_run_utc": nxt.isoformat(),
                "next_supported_run_local": civil_local(nxt).isoformat(),
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

    # --- Gate: coverage ---
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

    # --- Model + schema ---
    blob, artifact_path, load_status = _load_operating_model()
    if blob is None:
        out.update({"status": "model_unavailable", "reason": load_status, "model_artifact": artifact_path})
        out["paper_decision"] = {
            "decision": "blocked_unsupported_model",
            "reason": f"operating model unavailable: {load_status}",
            "live_blocked": True,
        }
        store.save_paper_decision(
            ticker=None, side=None, decision="blocked_unsupported_model", reason=out["paper_decision"]["reason"], details={"live_blocked": True}
        )
        _mirror(out)
        return out

    feature_names = list(blob.get("feature_names") or STATION_V2_FEATURES)
    if feature_names != STATION_V2_FEATURES:
        out.update(
            {
                "status": "schema_mismatch",
                "reason": "Artifact feature_names do not match station_v2.1",
                "model_artifact": artifact_path,
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_unsupported_model",
            "reason": out["reason"],
            "live_blocked": True,
        }
        _mirror(out)
        return out

    feats = feature_bundle.get("features") or {}
    # Attribution: only station features consumed
    consumed = [n for n in feature_names if feats.get(n) is not None or n.startswith("missing_")]
    attribution = dict(feature_bundle.get("attribution") or {})
    attribution["features_consumed_by_model"] = feature_names
    attribution["feeds_contributing_to_prediction"] = ["metar_KNYC"]
    attribution["feeds_collected_not_consumed"] = [
        f
        for f in (attribution.get("feeds_collected") or [])
        if f not in ("metar_KNYC",)
    ]
    # CLI constraint separate from model features
    cli = feature_bundle.get("cli_applied")
    attribution["settlement_constraints_applied"] = []
    if cli:
        attribution["settlement_constraints_applied"].append(
            f"cli_{'prelim' if cli.get('is_preliminary') else 'final'}_floor_{cli.get('max_temp_f')}"
        )
    out["attribution"] = attribution

    vec = vector_for_model(feats, feature_names)
    X = np.nan_to_num(np.asarray([vec], dtype=float), nan=-999.0)
    models = blob["models"]
    q10 = float(models["q10"].predict(X)[0])
    q50 = float(models["q50"].predict(X)[0])
    q90 = float(models["q90"].predict(X)[0])
    # Handle crossing quantiles explicitly
    q_sorted = sorted([q10, q50, q90])
    crossed = (q10, q50, q90) != tuple(q_sorted)
    q10, q50, q90 = q_sorted
    max_so_far = float(feature_bundle["max_so_far"])
    point = max_so_far + q50

    calib = load_calibration(CALIB_PATH)
    residuals, calib_status = pick_residuals(calib, decision_hour)
    out["calibration_status"] = calib_status
    if residuals is None:
        out.update(
            {
                "status": "probabilities_unavailable",
                "reason": calib_status,
                "remain_q10_q50_q90": [q10, q50, q90],
                "point_median_f": point,
                "quantiles_crossed": crossed,
                "model_artifact": artifact_path,
                "probabilities_available": False,
            }
        )
        out["paper_decision"] = {
            "decision": "blocked_calibration_unavailable",
            "reason": calib_status,
            "live_blocked": True,
        }
        store.save_paper_decision(
            ticker=None, side=None, decision="blocked_calibration_unavailable", reason=calib_status, details={"live_blocked": True}
        )
        _mirror(out)
        return out

    dist = from_empirical_residuals(
        point,
        residuals,
        method=f"calibrated_residual_hour_{decision_hour}",
    )
    # METAR max_so_far: do not claim ceil official floor; shift support conservatively via details only
    out["observation_constraint"] = {
        "max_so_far_f": max_so_far,
        "max_so_far_status": (feature_bundle.get("coverage") or {}).get("max_so_far_status"),
        "integer_floor_applied": False,
        "note": "Fractional METAR max_so_far is not an official whole-°F CLI minimum",
    }
    if cli and cli.get("max_temp_f") is not None:
        # whole °F CLI value only
        dist = truncate_below(dist, float(int(cli["max_temp_f"])), reason="CLI same-day whole °F floor")
        out["observation_constraint"]["integer_floor_applied"] = True
        out["observation_constraint"]["cli_floor_f"] = int(cli["max_temp_f"])
        out["observation_constraint"]["cli_is_preliminary"] = cli.get("is_preliminary")

    out.update(
        {
            "ok": True,
            "status": "ok",
            "model_artifact": artifact_path,
            "feature_set": FEATURE_SCHEMA_VERSION,
            "model_version": blob.get("model_version"),
            "max_so_far": max_so_far,
            "remain_q10_q50_q90": [q10, q50, q90],
            "quantiles_crossed_before_sort": crossed,
            "point_median_f": point,
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
        prefer = climate_day  # settlement day for this forecast
        markets = fetch_open_nyc_markets(client)
        target_day, event_markets = select_event_markets(markets, prefer_day=prefer)
        out["target_date"] = target_day.isoformat() if target_day else None
        if target_day is None:
            out["status"] = "no_open_markets"
            out["paper_decision"] = {"decision": "blocked_no_markets", "reason": "no open NYC markets", "live_blocked": True}
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
                details={"live_blocked": True, "prefer": prefer.isoformat(), "market_day": target_day.isoformat()},
            )
            # Do not attach brackets from wrong day
            out["brackets"] = []
            store.save_prediction(
                model_version=str(blob.get("model_version") or "station_v2"),
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
                    }
                )

        # Persist all evaluations; sim-fill at most the best positive EV once
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
            worst = negative[0]
            paper_decision = {
                **worst,
                "decision": "paper_skip_negative_ev",
                "reason": worst["reason"],
                "live_blocked": True,
                "note": "Probabilities below 55% still reached EV evaluation",
            }
        elif evaluations:
            paper_decision = {
                **evaluations[0],
                "live_blocked": True,
            }
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
            model_version=str(blob.get("model_version") or "station_v2"),
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
