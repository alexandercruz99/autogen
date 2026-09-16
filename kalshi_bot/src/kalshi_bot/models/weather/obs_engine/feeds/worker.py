"""Persistent weather feed collector worker (research/paper). No live orders."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kalshi_bot.models.weather.obs_engine.feeds.cli_feed import collect_cli
from kalshi_bot.models.weather.obs_engine.feeds.features_live import build_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.goes import collect_goes
from kalshi_bot.models.weather.obs_engine.feeds.metar import collect_metar
from kalshi_bot.models.weather.obs_engine.feeds.nexrad import collect_nexrad
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)

# Source-appropriate polling (seconds). Freshness limits enforced in features_live.
DEFAULT_INTERVALS = {
    "metar": 300,
    "cli": 900,
    "goes": 600,
    "nexrad": 300,
    "features_infer": 300,
}


def _with_retry(fn: Callable[[], dict[str, Any]], *, name: str, attempts: int = 3) -> dict[str, Any]:
    delay = 2.0
    last: dict[str, Any] = {"ok": False, "error": "not_run"}
    for i in range(attempts):
        try:
            last = fn()
            if last.get("ok"):
                return last
            logger.warning("%s attempt %s failed: %s", name, i + 1, last.get("error"))
        except Exception as exc:
            last = {"ok": False, "error": str(exc)}
            logger.warning("%s attempt %s exception: %s", name, i + 1, exc)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    return last


def run_collect_cycle(store: FeedStore | None = None, *, do_infer: bool = True) -> dict[str, Any]:
    store = store or FeedStore()
    summary: dict[str, Any] = {"cycle_started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        summary["metar"] = _with_retry(lambda: collect_metar(store), name="metar")
        summary["cli"] = _with_retry(lambda: collect_cli(store), name="cli")
        summary["goes"] = _with_retry(lambda: collect_goes(store), name="goes")
        summary["nexrad"] = _with_retry(lambda: collect_nexrad(store), name="nexrad")
        if do_infer:
            summary["features"] = build_operating_features(store)
            try:
                from kalshi_bot.models.weather.obs_engine.feeds.infer_paper import run_infer_and_paper

                summary["infer_paper"] = run_infer_and_paper(store, summary["features"])
            except Exception as exc:
                summary["infer_paper"] = {"ok": False, "error": str(exc)}
        summary["checkpoints"] = store.list_checkpoints()
        summary["ok"] = True
        store.heartbeat(
            "ok",
            {
                "last_cycle": summary["cycle_started_utc"],
                "feeds": {k: (summary.get(k) or {}).get("ok") for k in ("metar", "cli", "goes", "nexrad")},
            },
        )
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)
        store.heartbeat("error", {"error": str(exc)})
        logger.exception("collect cycle failed")
    finally:
        path = Path("data/obs_engine/feeds/last_cycle.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def worker_loop(interval_seconds: int = 300, pidfile: Path | None = None) -> None:
    """Blocking loop with heartbeat + restart recovery via FeedStore checkpoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = FeedStore()
    stop = {"flag": False}
    last_run: dict[str, float] = {}

    def _stop(*_args):
        stop["flag"] = True
        store.heartbeat("stopping", {})

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    pidfile = pidfile or Path("data/obs_engine/feeds/collector.pid")
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    # Restart recovery: reload checkpoints so we do not re-fetch duplicates (UNIQUE source_key)
    cps = store.list_checkpoints()
    store.heartbeat(
        "starting",
        {"pid": os.getpid(), "interval_seconds": interval_seconds, "recovered_checkpoints": len(cps)},
    )
    logger.info(
        "collector worker started pid=%s interval=%ss recovered_checkpoints=%s",
        os.getpid(),
        interval_seconds,
        len(cps),
    )

    while not stop["flag"]:
        t0 = time.time()
        now = time.time()
        # Per-source cadence inside the master loop
        due = {
            "metar": now - last_run.get("metar", 0) >= DEFAULT_INTERVALS["metar"],
            "cli": now - last_run.get("cli", 0) >= DEFAULT_INTERVALS["cli"],
            "goes": now - last_run.get("goes", 0) >= DEFAULT_INTERVALS["goes"],
            "nexrad": now - last_run.get("nexrad", 0) >= DEFAULT_INTERVALS["nexrad"],
        }
        if any(due.values()) or not last_run:
            # Full cycle when any source is due (simpler checkpoint + infer coherence)
            run_collect_cycle(store, do_infer=True)
            for k in ("metar", "cli", "goes", "nexrad", "features_infer"):
                last_run[k] = time.time()
        elapsed = time.time() - t0
        sleep_for = max(5.0, min(interval_seconds, 60) - elapsed)
        end = time.time() + sleep_for
        while time.time() < end and not stop["flag"]:
            time.sleep(min(1.0, end - time.time()))

    store.heartbeat("stopped", {"pid": os.getpid()})
    try:
        pidfile.unlink(missing_ok=True)
    except Exception:
        pass
    store.close()
    logger.info("collector worker stopped")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Weather feed collector worker")
    p.add_argument("command", choices=["once", "run", "status", "stop"])
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--pidfile", default="data/obs_engine/feeds/collector.pid")
    args = p.parse_args(argv)
    pidfile = Path(args.pidfile)

    if args.command == "once":
        print(json.dumps(run_collect_cycle(), indent=2, default=str))
        return 0
    if args.command == "status":
        store = FeedStore()
        hb = store.get_heartbeat()
        cps = store.list_checkpoints()
        running = False
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text().strip()), 0)
                running = True
            except Exception:
                running = False
        print(
            json.dumps(
                {"running": running, "pidfile": str(pidfile), "heartbeat": hb, "checkpoints": cps},
                indent=2,
                default=str,
            )
        )
        store.close()
        return 0
    if args.command == "stop":
        if pidfile.exists():
            pid = int(pidfile.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(json.dumps({"ok": True, "signaled": pid}))
            return 0
        print(json.dumps({"ok": False, "error": "pidfile missing — worker not running"}))
        return 1
    if args.command == "run":
        worker_loop(interval_seconds=args.interval, pidfile=pidfile)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
