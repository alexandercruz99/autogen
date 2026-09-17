from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from kalshi_bot.accounting.fills import net_realized_from_parts
from kalshi_bot.api.client import KalshiClient
from kalshi_bot.data.store import Store, dumps, utcnow
from kalshi_bot.money import D, ONE, ZERO

logger = logging.getLogger(__name__)

CHECKPOINT_KEY = "settlement_reconcile_checkpoint"


class SettlementReconciler:
    """Reconcile open positions only when the exchange has settled the market.

    Never invents settlement values. Paper and live both require market result fields
    or an official portfolio settlement record. Idempotent: applies once per position.
    """

    def __init__(self, client: KalshiClient, store: Store) -> None:
        self.client = client
        self.store = store

    def reconcile_open_positions(self) -> dict[str, Any]:
        """Primary schedule hook (scan loop + CLI reconcile)."""
        market_result = self._reconcile_via_markets()
        portfolio_result = self._reconcile_via_portfolio_settlements()
        checkpoint = {
            "updated_at": utcnow(),
            "market": market_result,
            "portfolio": portfolio_result,
        }
        self._save_checkpoint(checkpoint)
        return {
            "settled": market_result["settled"] + portfolio_result["settled"],
            "pending": market_result["pending"],
            "errors": market_result["errors"] + portfolio_result["errors"],
            "portfolio_settlements_seen": portfolio_result["seen"],
            "skipped_already_settled": (
                market_result["skipped_already_settled"] + portfolio_result["skipped_already_settled"]
            ),
            "checkpoint_at": checkpoint["updated_at"],
        }

    def _save_checkpoint(self, payload: dict[str, Any]) -> None:
        # Persist under forecast_cache-style key via audit + bot_state detail if available.
        self.store.audit("settlement_checkpoint", CHECKPOINT_KEY, details=payload)
        try:
            with self.store._conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reconcile_checkpoints (
                        key TEXT PRIMARY KEY,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO reconcile_checkpoints (key, updated_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (CHECKPOINT_KEY, payload["updated_at"], dumps(payload)),
                )
        except Exception as exc:
            logger.warning("checkpoint persist failed: %s", exc)

    def load_checkpoint(self) -> dict[str, Any] | None:
        try:
            with self.store._conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reconcile_checkpoints (
                        key TEXT PRIMARY KEY,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                row = conn.execute(
                    "SELECT payload_json FROM reconcile_checkpoints WHERE key=?",
                    (CHECKPOINT_KEY,),
                ).fetchone()
                if not row:
                    return None
                return json.loads(row["payload_json"])
        except Exception:
            return None

    def _reconcile_via_markets(self) -> dict[str, Any]:
        settled = 0
        pending = 0
        skipped = 0
        errors: list[str] = []
        for pos in self.store.list_positions(status="open"):
            ticker = pos["market_ticker"]
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
            result = market.get("result")
            settlement_value = market.get("settlement_value") or market.get("settlement_value_dollars")

            if status not in ("settled", "finalized") and settlement_value is None and not result:
                pending += 1
                continue

            value = self._settlement_dollars(pos["side"], result, settlement_value)
            if value is None:
                pending += 1
                continue

            if self._apply_settlement(
                pos,
                value=value,
                source="market",
                exchange_status=status,
                exchange_result=result,
                fee_cost=None,
                revenue_cents=None,
            ):
                settled += 1
            else:
                skipped += 1

        return {
            "settled": settled,
            "pending": pending,
            "errors": errors,
            "skipped_already_settled": skipped,
        }

    def _reconcile_via_portfolio_settlements(self) -> dict[str, Any]:
        """Use GET /portfolio/settlements (paginated) as authoritative for live positions."""
        settled = 0
        skipped = 0
        errors: list[str] = []
        seen = 0
        if not hasattr(self.client, "iter_settlements") and not hasattr(self.client, "get_settlements"):
            errors.append("client missing get_settlements — cannot sync portfolio settlements")
            return {
                "settled": 0,
                "seen": 0,
                "errors": errors,
                "skipped_already_settled": 0,
            }
        try:
            if hasattr(self.client, "iter_settlements"):
                rows = self.client.iter_settlements()
            else:
                payload = self.client.get_settlements()
                rows = payload.get("settlements") or []
        except Exception as exc:
            errors.append(f"portfolio_settlements: {exc}")
            return {
                "settled": 0,
                "seen": 0,
                "errors": errors,
                "skipped_already_settled": 0,
            }

        by_ticker: dict[str, dict[str, Any]] = {}
        for row in rows:
            seen += 1
            tick = row.get("ticker")
            if tick:
                by_ticker[str(tick)] = row

        for pos in list(self.store.list_positions(status="open")) + list(
            self.store.list_positions(status="settled")
        ):
            if pos.get("mode") != "live":
                continue
            row = by_ticker.get(pos["market_ticker"])
            if not row:
                continue
            if pos.get("status") == "settled":
                skipped += 1
                continue
            market_result = row.get("market_result")
            value_cents = row.get("value")
            settlement_value = None
            if value_cents is not None:
                settlement_value = D(str(value_cents)) / D("100")
            value = self._settlement_dollars(pos["side"], market_result, settlement_value)
            if value is None and market_result:
                value = self._settlement_dollars(pos["side"], market_result, None)
            if value is None:
                continue
            fee_cost = row.get("fee_cost")
            if self._apply_settlement(
                pos,
                value=value,
                source="portfolio_settlements",
                exchange_status="settled",
                exchange_result=market_result,
                fee_cost=str(fee_cost) if fee_cost is not None else None,
                revenue_cents=row.get("revenue"),
                raw_settlement=row,
            ):
                settled += 1
            else:
                skipped += 1

        return {
            "settled": settled,
            "seen": seen,
            "errors": errors,
            "skipped_already_settled": skipped,
        }

    def _apply_settlement(
        self,
        pos: dict[str, Any],
        *,
        value: Decimal,
        source: str,
        exchange_status: str,
        exchange_result: Any,
        fee_cost: str | None,
        revenue_cents: Any,
        raw_settlement: dict[str, Any] | None = None,
    ) -> bool:
        """Apply once. Returns False if already settled / closed."""
        # Re-read to survive restarts / races.
        current = None
        for p in self.store.list_positions():
            if p["id"] == pos["id"]:
                current = p
                break
        if current is None:
            return False
        if current.get("status") in ("settled", "closed"):
            return False
        # Detect already-applied settlement marker.
        try:
            details = json.loads(current.get("details_json") or "{}")
        except Exception:
            details = {}
        if details.get("settlement_applied") is True:
            return False

        qty = D(current["quantity"])
        avg = D(current["avg_price"])
        fees_local = D(current.get("fees_paid") or 0)
        fees_exchange = D(fee_cost) if fee_cost not in (None, "") else fees_local
        # Prefer local fill fees when exchange fee_cost covers whole account activity.
        fees = fees_local if fees_local > ZERO else fees_exchange
        payout = value * qty
        cost = avg * qty
        # Gross = payout - premium; net subtracts fees once (fees not already in payout).
        gross = payout - cost
        realized = net_realized_from_parts(
            gross_realized=gross,
            fees_paid=fees,
            fees_already_in_gross=False,
        )

        from kalshi_bot.data.store import PositionRecord

        details = {
            **details,
            "reconciled_at": utcnow(),
            "exchange_status": exchange_status,
            "exchange_result": exchange_result,
            "settlement_source": source,
            "settlement_applied": True,
            "gross_realized_pnl": str(gross),
            "fees_used": str(fees),
            "net_realized_pnl": str(realized),
            "premium_paid": str(cost),
            "payout": str(payout),
        }
        if raw_settlement is not None:
            details["raw_portfolio_settlement"] = {
                k: raw_settlement.get(k)
                for k in (
                    "ticker",
                    "market_result",
                    "yes_count_fp",
                    "no_count_fp",
                    "revenue",
                    "fee_cost",
                    "settled_time",
                    "value",
                )
            }
        if revenue_cents is not None:
            details["revenue_cents"] = revenue_cents

        updated = PositionRecord(
            id=current["id"],
            opened_at=current["opened_at"],
            mode=current["mode"],
            kind=current["kind"],
            market_ticker=current["market_ticker"],
            event_ticker=current["event_ticker"],
            side=current["side"],
            quantity=current["quantity"],
            avg_price=current["avg_price"],
            fees_paid=str(fees),
            status="settled",
            settlement_value=str(value),
            realized_pnl=str(realized),
            correlation_keys_json=current.get("correlation_keys_json") or "[]",
            details_json=dumps(details),
        )
        self.store.save_position(updated)

        state = self.store.get_state()
        if current["mode"] == "paper":
            self.store.update_state(
                paper_cash=str(D(state.paper_cash) + payout),
                realized_pnl=str(D(state.realized_pnl) + realized),
                daily_realized_pnl=str(D(state.daily_realized_pnl) + realized),
            )
            state = self.store.get_state()
            equity = D(state.paper_cash)
            if equity > D(state.peak_equity):
                self.store.update_state(peak_equity=str(equity))
        else:
            self.store.update_state(
                realized_pnl=str(D(state.realized_pnl) + realized),
                daily_realized_pnl=str(D(state.daily_realized_pnl) + realized),
            )

        self.store.audit(
            "settlement",
            f"{current['market_ticker']} settled value={value} pnl={realized} via {source}",
            details={"position_id": current["id"], "source": source},
        )
        return True

    def _settlement_dollars(
        self,
        side: str,
        result: str | None,
        settlement_value: Any,
    ) -> Decimal | None:
        if settlement_value is not None and settlement_value != "":
            yes_val = D(settlement_value)
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
