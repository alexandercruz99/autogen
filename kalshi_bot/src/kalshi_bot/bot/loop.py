from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from kalshi_bot.bot.pipeline import TradingPipeline
from kalshi_bot.data.store import Store

logger = logging.getLogger(__name__)


class BotLoop:
    def __init__(self, pipeline: TradingPipeline, store: Store, interval_seconds: int) -> None:
        self.pipeline = pipeline
        self.store = store
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kalshi-bot-loop", daemon=True)
        self._thread.start()
        self.store.audit("loop_start", f"interval={self.interval}s")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.store.audit("loop_stop", "stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            state = self.store.get_state()
            if state.kill_switch:
                logger.warning("kill switch on — skipping scan cycle")
            else:
                try:
                    result = self.pipeline.run_scan_once()
                    logger.info("scan result: %s", result)
                except Exception as exc:
                    logger.exception("scan failed")
                    self.store.update_state(last_scan_ok=False, last_error=str(exc))
                    self.store.audit("scan_error", str(exc), level="error")
            # Wait without weakening criteria due to inactivity
            self._stop.wait(self.interval)
