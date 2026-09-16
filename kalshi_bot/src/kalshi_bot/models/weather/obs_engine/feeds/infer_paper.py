"""Inference from operating features + paper decision (never live)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.distribution import from_empirical_residuals, truncate_below
from kalshi_bot.models.weather.settlement_rules import TempInterval, interval_from_market

logger = logging.getLogger(__name__)


def _load_station_corrected_model():
    path = Path("data/obs_engine/feeds/models/station_corrected_v1.joblib")
    if not path.exists():
        # fallback to frozen baseline (compatible local features only — sat/radar ignored)
        path = Path("data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib")
    if not path.exists():
        path = Path("data/obs_engine/models/obs_nyc_q50.joblib")
    import joblib

    return joblib.load(path), str(path)


def run_infer_and_paper(store: FeedStore, feature_bundle: dict[str, Any]) -> dict[str, Any]:
    """Produce research prediction; optionally record paper skip/buy using live books. No live orders."""
    from kalshi_bot.config import load_config
    from kalshi_bot.api.client import KalshiClient
    from kalshi_bot.api.orderbook import parse_orderbook
    from kalshi_bot.api.fees import estimate_net_fee
    from kalshi_bot.models.weather.obs_engine.predict_now import (
        fetch_open_nyc_markets,
        select_event_markets,
        _pick_target_day,
    )
    from kalshi_bot.models.weather.obs_engine.research.features_v2 import LOCAL_V2_FEATURES
    from kalshi_bot.money import D, ZERO

    now = datetime.now(timezone.utc)
    missing = feature_bundle.get("missing") or {}
    feats = feature_bundle.get("features") or {}
    max_so_far = feature_bundle.get("max_so_far")
    if max_so_far is None:
        return {"ok": False, "reason": "insufficient station observations for max_so_far", "missing": missing}

    blob, artifact_path = _load_station_corrected_model()
    models = blob.get("models") or blob
    feature_names = blob.get("feature_names") or LOCAL_V2_FEATURES
    # Build vector in artifact order; NaN→sentinel only if model trained with that; else require station features
    vec = []
    for name in feature_names:
        v = feats.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            # Do not invent; use NaN and let model path refuse if critical
            vec.append(np.nan)
        else:
            vec.append(float(v))
    X = np.nan_to_num(np.asarray([vec], dtype=float), nan=-999.0)

    q10 = float(models["q10"].predict(X)[0])
    q50 = float(models["q50"].predict(X)[0])
    q90 = float(models["q90"].predict(X)[0])
    q10, q50, q90 = sorted([q10, q50, q90])
    resid = blob.get("remain_residuals") or [0.0, -1.0, 1.0, 2.0, -2.0]
    point = float(max_so_far) + q50
    dist = from_empirical_residuals(point, resid, method="station_corrected_remain")
    dist = truncate_below(dist, float(max_so_far), reason="observed max_so_far")

    # CLI prelim soft floor
    cli = (feature_bundle.get("provenance") or {}).get("cli") or {}
    if cli.get("max_temp_f") is not None and cli.get("is_preliminary"):
        dist = truncate_below(dist, float(cli["max_temp_f"]), reason="CLI preliminary soft floor")

    cfg = load_config("config.yaml" if Path("config.yaml").exists() else "config.example.yaml")
    # Force paper-safe: never enable live from this path
    client = KalshiClient(cfg.api)
    out: dict[str, Any] = {
        "ok": True,
        "mode": "RESEARCH",
        "live_orders": False,
        "model_artifact": artifact_path,
        "feature_set": blob.get("feature_set") or "station_corrected_or_baseline",
        "missing": missing,
        "feeds_contributed": feature_bundle.get("provenance", {}).get("feeds_used"),
        "max_so_far": max_so_far,
        "remain_q10_q50_q90": [q10, q50, q90],
        "distribution": dist.as_dict(),
        "generated_at_utc": now.isoformat(),
    }
    try:
        prefer = _pick_target_day(now)
        markets = fetch_open_nyc_markets(client)
        target_day, event_markets = select_event_markets(markets, prefer_day=prefer)
        out["target_date"] = target_day.isoformat() if target_day else None
        out["horizon"] = "same_day" if target_day == prefer else "other_day"
        if target_day != prefer:
            out["horizon_warning"] = "Day-ahead/other day — same-day model validation limits apply"
        brackets = []
        paper_decision = None
        for m in sorted(event_markets, key=lambda x: (x.get("strike_type") or "", x.get("floor_strike") or 0)):
            iv = interval_from_market(m)
            if iv is None:
                continue
            p = dist.p_interval(iv)
            ticker = m.get("ticker")
            book = None
            yes_ask = None
            try:
                raw = client.get_orderbook(ticker, depth=5)
                book = parse_orderbook(raw)
                yes_ask = book.best_yes_ask
            except Exception:
                pass
            row = {
                "ticker": ticker,
                "p_yes": str(p),
                "yes_ask": str(yes_ask) if yes_ask is not None else None,
                "interval": {"op": iv.op, "low": iv.low, "high": iv.high},
            }
            brackets.append(row)
            # Paper decision: record research signal; never live
            if yes_ask is not None and yes_ask > ZERO and p > D("0.55"):
                fees = estimate_net_fee(D("1"), yes_ask, multiplier=cfg.trading.fee_multiplier, assume_taker=True, balance_precision=cfg.trading.balance_precision)
                ev = p - yes_ask - (fees / D("1")) - D(cfg.trading.uncertainty_buffer)
                decision = "paper_skip_unvalidated"
                reason = (
                    f"research EV≈{ev} but model not live_eligible / not promotion-validated — "
                    "paper record only, no order submitted"
                )
                store.save_paper_decision(
                    ticker=ticker,
                    side="yes",
                    decision=decision,
                    reason=reason,
                    details={"p_yes": str(p), "yes_ask": str(yes_ask), "fees": str(fees), "live_blocked": True},
                )
                if paper_decision is None:
                    paper_decision = {"ticker": ticker, "decision": decision, "reason": reason}
        out["brackets"] = brackets
        out["paper_decision"] = paper_decision or {"decision": "no_signal", "reason": "no bracket met research paper criteria"}
        store.save_prediction(
            model_version=str(blob.get("model_version") or "station_corrected_or_baseline"),
            feature_set=out["feature_set"],
            target_day=out.get("target_date"),
            ticker=(brackets[0]["ticker"] if brackets else None),
            prediction=out,
        )
        # Also mirror to classic path for dashboard
        Path("data/obs_engine/latest_research_prediction.json").write_text(json.dumps(out, indent=2, default=str))
        Path("data/obs_engine/feeds/latest_prediction.json").write_text(json.dumps(out, indent=2, default=str))
    finally:
        client.close()
    return out
