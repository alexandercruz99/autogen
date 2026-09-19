"""Simulated execution adapter for historical replay / paper (never live)."""

from __future__ import annotations

from typing import Any

from kalshi_bot.models.weather.obs_engine.multi.replay.picker import Decision


def execute_decision_simulated(
    decision: Decision,
    *,
    refreshed_book: dict[str, Any] | None = None,
    refreshed_account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapter boundary: does not place real orders.

    Historical replay calls this only with simulated state. Live submission is unreachable.
    """
    if decision.kind != "TRADE":
        return {
            "submitted": False,
            "live_order_submitted": False,
            "reason": "no_trade",
            "decision_id": decision.decision_id,
        }
    # Re-check would happen here with refreshed inputs; for sim we record intent only.
    return {
        "submitted": False,
        "live_order_submitted": False,
        "simulated_only": True,
        "reason": "research_simulated_adapter_does_not_submit",
        "decision_id": decision.decision_id,
        "would_have_submitted": {
            "ticker": decision.ticker,
            "side": decision.side,
            "quantity": decision.quantity,
            "limit_price": decision.limit_price,
        },
        "refreshed_book_provided": refreshed_book is not None,
        "refreshed_account_provided": refreshed_account is not None,
    }
