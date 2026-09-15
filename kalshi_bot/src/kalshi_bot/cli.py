from __future__ import annotations

import argparse
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
    # Sync mode/budget into state on boot without enabling live.
    state = store.get_state()
    updates = {
        "mode": "paper" if (config.mode == "live" and not state.live_enabled) else config.mode,
        "trading_budget": str(config.trading.budget_dollars),
    }
    if not state.daily_pnl_date:
        from kalshi_bot.data.store import utcnow

        updates["daily_pnl_date"] = utcnow()[:10]
        updates["paper_cash"] = str(config.trading.budget_dollars)
        updates["peak_equity"] = str(config.trading.budget_dollars)
    store.update_state(**updates)
    client = KalshiClient(config.api)
    pipeline = TradingPipeline(config, store, client)
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

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Prefer local config.yaml if present when starting from kalshi_bot dir
    cfg_path = args.config
    if cfg_path is None and Path("config.yaml").exists():
        cfg_path = "config.yaml"
    elif cfg_path is None and Path("config.example.yaml").exists():
        cfg_path = "config.example.yaml"

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
