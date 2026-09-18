from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def send_alert(webhook_url: str, text: str, details: dict[str, Any] | None = None) -> None:
    if not webhook_url:
        return
    payload = {"text": text, "details": details or {}}
    try:
        # Slack-compatible and generic JSON webhook
        httpx.post(webhook_url, json={"text": text, **(details or {})}, timeout=10)
    except Exception as exc:
        logger.warning("alert webhook failed: %s", exc)
