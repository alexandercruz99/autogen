"""Simulated paper ledger for weather research (never submits live orders)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_bot.money import D, ZERO

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_ledger_meta (
  id INTEGER PRIMARY KEY CHECK (id=1),
  cash TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  client_order_id TEXT NOT NULL UNIQUE,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  qty TEXT NOT NULL,
  price TEXT NOT NULL,
  fees TEXT NOT NULL,
  decision_reason TEXT,
  details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_positions (
  ticker TEXT PRIMARY KEY,
  side TEXT NOT NULL,
  qty TEXT NOT NULL,
  avg_price TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_settlements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  pnl TEXT NOT NULL,
  details_json TEXT NOT NULL
);
"""


class PaperLedger:
    """Persistent paper cash / fills / positions. Restart-safe; duplicate client_order_id rejected."""

    def __init__(self, path: Path | None = None, *, starting_cash: str = "100.00") -> None:
        root = Path("data/obs_engine/feeds")
        root.mkdir(parents=True, exist_ok=True)
        self.path = path or (root / "paper_ledger.db")
        self._conn = sqlite3.connect(str(self.path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(LEDGER_SCHEMA)
        row = self._conn.execute("SELECT cash FROM paper_ledger_meta WHERE id=1").fetchone()
        if not row:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO paper_ledger_meta(id, cash, updated_at_utc) VALUES (1, ?, ?)",
                (starting_cash, now),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def cash(self) -> Decimal:
        return D(self._conn.execute("SELECT cash FROM paper_ledger_meta WHERE id=1").fetchone()["cash"])

    def snapshot(self) -> dict[str, Any]:
        fills = [dict(r) for r in self._conn.execute("SELECT * FROM paper_fills ORDER BY id DESC LIMIT 20")]
        pos = [dict(r) for r in self._conn.execute("SELECT * FROM paper_positions")]
        return {"cash": str(self.cash()), "positions": pos, "recent_fills": fills, "path": str(self.path)}

    def try_simulate_fill(
        self,
        *,
        client_order_id: str,
        ticker: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        fees: Decimal,
        decision_reason: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """Conservative fill at executable ask; never calls live APIs."""
        existing = self._conn.execute(
            "SELECT id FROM paper_fills WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        if existing:
            return {"ok": False, "reason": "duplicate_client_order_id", "client_order_id": client_order_id}

        cost = D(qty) * D(price) + D(fees)
        cash = self.cash()
        if cost > cash:
            return {"ok": False, "reason": "insufficient_paper_cash", "cash": str(cash), "need": str(cost)}

        now = datetime.now(timezone.utc).isoformat()
        new_cash = cash - cost
        self._conn.execute(
            "UPDATE paper_ledger_meta SET cash=?, updated_at_utc=? WHERE id=1",
            (str(new_cash), now),
        )
        self._conn.execute(
            """INSERT INTO paper_fills(created_at_utc, client_order_id, ticker, side, qty, price, fees, decision_reason, details_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                now,
                client_order_id,
                ticker,
                side,
                str(qty),
                str(price),
                str(fees),
                decision_reason,
                json.dumps({**details, "live_blocked": True, "sim_label": "unvalidated_exploratory_paper"}),
            ),
        )
        prev = self._conn.execute("SELECT * FROM paper_positions WHERE ticker=?", (ticker,)).fetchone()
        if prev:
            pq, ap = D(prev["qty"]), D(prev["avg_price"])
            nq = pq + D(qty)
            navg = (ap * pq + D(price) * D(qty)) / nq if nq > ZERO else D(price)
            self._conn.execute(
                "UPDATE paper_positions SET qty=?, avg_price=?, side=?, updated_at_utc=? WHERE ticker=?",
                (str(nq), str(navg), side, now, ticker),
            )
        else:
            self._conn.execute(
                "INSERT INTO paper_positions(ticker, side, qty, avg_price, updated_at_utc) VALUES (?,?,?,?,?)",
                (ticker, side, str(qty), str(price), now),
            )
        self._conn.commit()
        return {
            "ok": True,
            "filled": True,
            "ticker": ticker,
            "side": side,
            "qty": str(qty),
            "price": str(price),
            "fees": str(fees),
            "cash_after": str(new_cash),
            "client_order_id": client_order_id,
            "live_order_submitted": False,
        }
