from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.data.store import Store, dumps, utcnow
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)


class SettlementReconciler:
    """Reconcile open positions only when the exchange has settled the market.

    Never invents settlement values. Paper and live both require market result fields.
    """

    def __init__(self, client: KalshiClient, store: Store) -> None:
        self.client = client
        self.store = store

    def reconcile_open_positions(self) -> dict[str, Any]:
        settled = 0
        pending = 0
        errors: list[str] = []
        for pos in self.store.list_positions(status="open"):
            ticker = pos["market_ticker"]
            # Combos may use synthetic tickers with '+'; skip exchange lookup.
            if "+" in ticker:
                pending += 1
                continue
            try:
                payload = self.client.get_market(ticker)
                market = payload.get("market") or payload
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
                continue

            status = (market.get("status") or "").lower()
            result = market.get("result")  # often 'yes'/'no'
            settlement_value = market.get("settlement_value") or market.get("settlement_value_dollars")

            if status not in ("settled", "finalized") and settlement_value is None and not result:
                pending += 1
                continue

            value = self._settlement_dollars(pos["side"], result, settlement_value)
            if value is None:
                pending += 1
                continue

            qty = D(pos["quantity"])
            avg = D(pos["avg_price"])
            fees = D(pos.get("fees_paid") or 0)
            # Binary hold-to-settlement PnL for long YES/NO premium paid.
            payout = value * qty
            cost = avg * qty + fees
            realized = payout - cost

            from kalshi_bot.data.store import PositionRecord

            updated = PositionRecord(
                id=pos["id"],
                opened_at=pos["opened_at"],
                mode=pos["mode"],
                kind=pos["kind"],
                market_ticker=pos["market_ticker"],
                event_ticker=pos["event_ticker"],
                side=pos["side"],
                quantity=pos["quantity"],
                avg_price=pos["avg_price"],
                fees_paid=pos.get("fees_paid") or "0",
                status="settled",
                settlement_value=str(value),
                realized_pnl=str(realized),
                correlation_keys_json=pos.get("correlation_keys_json") or "[]",
                details_json=dumps(
                    {
                        "reconciled_at": utcnow(),
                        "exchange_status": status,
                        "exchange_result": result,
                    }
                ),
            )
            self.store.save_position(updated)

            state = self.store.get_state()
            if pos["mode"] == "paper":
                self.store.update_state(
                    paper_cash=str(D(state.paper_cash) + payout),
                    realized_pnl=str(D(state.realized_pnl) + realized),
                    daily_realized_pnl=str(D(state.daily_realized_pnl) + realized),
                )
                state = self.store.get_state()
                equity = D(state.paper_cash)  # approx after payout credit
                if equity > D(state.peak_equity):
                    self.store.update_state(peak_equity=str(equity))
            else:
                self.store.update_state(
                    realized_pnl=str(D(state.realized_pnl) + realized),
                    daily_realized_pnl=str(D(state.daily_realized_pnl) + realized),
                )

            self.store.audit(
                "settlement",
                f"{ticker} settled value={value} pnl={realized}",
                details={"position_id": pos["id"]},
            )
            settled += 1

        return {"settled": settled, "pending": pending, "errors": errors}

    def _settlement_dollars(
        self,
        side: str,
        result: str | None,
        settlement_value: Any,
    ) -> Decimal | None:
        if settlement_value is not None and settlement_value != "":
            # Exchange may report YES settlement dollars; map to our side.
            yes_val = D(settlement_value)
            # If value looks like cents integer (>1), convert — Kalshi usually dollars string.
            if yes_val > ONE:
                yes_val = yes_val / D("100")
            return yes_val if side == "yes" else (ONE - yes_val)

        if not result:
            return None
        r = str(result).lower()
        if r in ("yes", "true", "1"):
            return ONE if side == "yes" else ZERO
        if r in ("no", "false", "0"):
            return ZERO if side == "yes" else ONE
        return None
