from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.bot.loop import BotLoop
from kalshi_bot.bot.pipeline import TradingPipeline
from kalshi_bot.config import load_config
from kalshi_bot.dashboard.app import create_app
from kalshi_bot.data.store import Store


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_runtime(config_path: str | None = None):
    config = load_config(config_path)
    store = Store(config.storage.sqlite_path)
    # Sync mode/budget into state on boot. Live requires config.live.enabled or prior ack.
    state = store.get_state()
    from kalshi_bot.data.store import PositionRecord, utcnow

    live_ok = bool(config.live.enabled or state.live_enabled)
    if config.mode == "live" and live_ok:
        mode = "live"
        live_enabled = True
    elif config.mode == "live" and not live_ok:
        mode = "paper"
        live_enabled = False
    else:
        mode = config.mode
        live_enabled = False

    updates: dict = {
        "mode": mode,
        "live_enabled": live_enabled,
        "trading_budget": str(config.trading.budget_dollars),
    }
    if live_enabled and mode == "live":
        updates["kill_switch"] = False
        updates["pause_buying"] = False
        if not state.live_ack_at:
            updates["live_ack_at"] = utcnow()
        # Archive paper positions so they do not affect live risk accounting.
        for p in store.list_positions(status="open"):
            if p.get("mode") != "paper":
                continue
            store.save_position(
                PositionRecord(
                    id=p["id"],
                    opened_at=p["opened_at"],
                    mode="paper",
                    kind=p.get("kind") or "individual",
                    market_ticker=p["market_ticker"],
                    event_ticker=p.get("event_ticker") or "",
                    side=p["side"],
                    quantity=p["quantity"],
                    avg_price=p["avg_price"],
                    fees_paid=p.get("fees_paid") or "0",
                    status="closed",
                    settlement_value=p.get("settlement_value") or "",
                    realized_pnl=p.get("realized_pnl") or "",
                    correlation_keys_json=p.get("correlation_keys_json") or "[]",
                    details_json=p.get("details_json") or "{}",
                )
            )
        store.audit(
            "live_boot",
            f"live trading armed; budget={config.trading.budget_dollars} "
            f"max_loss/trade={config.trading.max_loss_per_trade_dollars} "
            f"target_trade={config.trading.target_trade_dollars}",
            level="warning",
        )
    if not state.daily_pnl_date:
        updates["daily_pnl_date"] = utcnow()[:10]
        updates["paper_cash"] = str(config.trading.budget_dollars)
        updates["peak_equity"] = str(config.trading.budget_dollars)
    store.update_state(**updates)
    client = KalshiClient(config.api)
    pipeline = TradingPipeline(config, store, client)
    if live_enabled and mode == "live":
        pipeline.sync_live_cash()
        # Reset peak to live equity so prior paper peak cannot trip drawdown.
        st = store.get_state()
        store.update_state(peak_equity=str(st.paper_cash))
    loop = BotLoop(pipeline, store, config.scan.scan_interval_seconds)
    return config, store, client, pipeline, loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalshi research & trading bot")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="Run one research/paper scan cycle")
    sub.add_parser("reconcile", help="Reconcile settlements for open positions (no invented results)")
    sub.add_parser("dashboard", help="Start the dashboard (and optional loop)")
    p_run = sub.add_parser("run", help="Start autonomous loop + dashboard")
    p_run.add_argument("--no-dashboard", action="store_true")
    sub.add_parser("weather-collect", help="Collect NWS CLI outcomes + forecast snapshots into archive")
    sub.add_parser("weather-train", help="Train station empirical/quantile models from archive")
    sub.add_parser("weather-validate", help="Walk-forward style validation report (honest gaps)")
    sub.add_parser("weather-obs-train", help="Train observation-driven NYC remaining-rise model")
    sub.add_parser("weather-obs-validate", help="Summarize obs-engine holdout metrics (research only)")
    sub.add_parser("weather-obs-backtest", help="Chronological backtest by decision hour vs baselines")
    sub.add_parser("weather-obs-reconcile", help="Reconcile CLI MAXIMUM vs GHCND TMAX labels")
    sub.add_parser("weather-obs-predict", help="Fresh research prediction for open NYC markets (no live orders)")
    sub.add_parser("weather-obs-freeze", help="Freeze baseline artifacts and reproduce backtest metrics")
    sub.add_parser("weather-obs-collect-once", help="One prospective collection cycle (append-only; no live orders)")
    sub.add_parser("weather-obs-audit", help="Measurement/settlement data-quality audit")
    sub.add_parser("weather-obs-diagnose", help="Export per decision-time error diagnostics")
    sub.add_parser("weather-obs-experiment", help="Compare feature candidates chronologically")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Prefer local config.yaml if present when starting from kalshi_bot dir
    cfg_path = args.config
    if cfg_path is None and Path("config.yaml").exists():
        cfg_path = "config.yaml"
    elif cfg_path is None and Path("config.example.yaml").exists():
        cfg_path = "config.example.yaml"

    # Weather ops can run without starting the trading loop / live boot side effects.
    weather_ops = (
        "weather-collect",
        "weather-train",
        "weather-validate",
        "weather-obs-train",
        "weather-obs-validate",
        "weather-obs-backtest",
        "weather-obs-reconcile",
        "weather-obs-predict",
        "weather-obs-freeze",
        "weather-obs-collect-once",
        "weather-obs-audit",
        "weather-obs-diagnose",
        "weather-obs-experiment",
    )
    if args.cmd in weather_ops:
        config = load_config(cfg_path)
        from kalshi_bot.models.weather.ops import (
            weather_collect,
            weather_obs_audit,
            weather_obs_backtest,
            weather_obs_collect_once,
            weather_obs_diagnose,
            weather_obs_experiment,
            weather_obs_freeze,
            weather_obs_predict_now,
            weather_obs_reconcile_labels,
            weather_obs_train,
            weather_obs_validate,
            weather_train,
            weather_validate,
        )

        dispatch = {
            "weather-collect": weather_collect,
            "weather-train": weather_train,
            "weather-validate": weather_validate,
            "weather-obs-train": weather_obs_train,
            "weather-obs-validate": weather_obs_validate,
            "weather-obs-backtest": weather_obs_backtest,
            "weather-obs-reconcile": weather_obs_reconcile_labels,
            "weather-obs-predict": weather_obs_predict_now,
            "weather-obs-freeze": weather_obs_freeze,
            "weather-obs-collect-once": weather_obs_collect_once,
            "weather-obs-audit": weather_obs_audit,
            "weather-obs-diagnose": weather_obs_diagnose,
            "weather-obs-experiment": weather_obs_experiment,
        }
        print(json.dumps(dispatch[args.cmd](config), indent=2, default=str))
        return 0

    config, store, client, pipeline, loop = build_runtime(cfg_path)

    try:
        if args.cmd == "scan":
            result = pipeline.run_scan_once()
            print(result)
            return 0 if result.get("ok") else 1

        if args.cmd == "reconcile":
            result = pipeline.settlement.reconcile_open_positions()
            print(result)
            return 0

        if args.cmd == "dashboard":
            app = create_app(config, store, pipeline, loop)
            uvicorn.run(app, host=config.dashboard.host, port=config.dashboard.port)
            return 0

        if args.cmd == "run":
            loop.start()
            if args.no_dashboard:
                logging.getLogger(__name__).info("Loop running without dashboard; Ctrl+C to stop")
                try:
                    while True:
                        import time

                        time.sleep(3600)
                except KeyboardInterrupt:
                    loop.stop()
                return 0
            app = create_app(config, store, pipeline, loop)
            uvicorn.run(app, host=config.dashboard.host, port=config.dashboard.port)
            return 0
    finally:
        pipeline.close()
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
