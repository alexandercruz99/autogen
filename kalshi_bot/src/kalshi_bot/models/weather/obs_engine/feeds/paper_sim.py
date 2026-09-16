"""Simulated paper ledger for weather research (never submits live orders).

Positions are keyed by (ticker, side) so YES and NO are held separately.
Settlement is idempotent. Cash/fills/positions update atomically.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_bot.money import D, ONE, ZERO

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_ledger_meta (
  id INTEGER PRIMARY KEY CHECK (id=1),
  cash TEXT NOT NULL,
  starting_cash TEXT NOT NULL,
  max_total_exposure TEXT NOT NULL,
  max_per_market_exposure TEXT NOT NULL,
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
  location_id TEXT,
  series_ticker TEXT,
  quote_ts_utc TEXT,
  market_ts_utc TEXT,
  details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_positions (
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  qty TEXT NOT NULL,
  avg_price TEXT NOT NULL,
  fees_paid TEXT NOT NULL DEFAULT '0',
  location_id TEXT,
  series_ticker TEXT,
  updated_at_utc TEXT NOT NULL,
  PRIMARY KEY (ticker, side)
);
CREATE TABLE IF NOT EXISTS paper_settlements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  qty TEXT NOT NULL,
  result TEXT NOT NULL,
  payout TEXT NOT NULL,
  cost_basis TEXT NOT NULL,
  fees TEXT NOT NULL,
  pnl TEXT NOT NULL,
  settlement_key TEXT NOT NULL UNIQUE,
  details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_exposure (
  id INTEGER PRIMARY KEY CHECK (id=1),
  open_notional TEXT NOT NULL
);
"""

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


class PaperLedger:
    """Persistent paper cash / fills / positions. Restart-safe; duplicate client_order_id rejected."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        starting_cash: str = "100.00",
        max_total_exposure: str = "80.00",
        max_per_market_exposure: str = "25.00",
    ) -> None:
        root = Path("data/obs_engine/feeds")
        root.mkdir(parents=True, exist_ok=True)
        self.path = path or (root / "paper_ledger.db")
        self._lock = _path_lock(self.path)
        self._conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(LEDGER_SCHEMA)
        self._migrate()
        row = self._conn.execute("SELECT cash FROM paper_ledger_meta WHERE id=1").fetchone()
        if not row:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """INSERT INTO paper_ledger_meta(
                     id, cash, starting_cash, max_total_exposure, max_per_market_exposure, updated_at_utc
                   ) VALUES (1, ?, ?, ?, ?, ?)""",
                (starting_cash, starting_cash, max_total_exposure, max_per_market_exposure, now),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO paper_exposure(id, open_notional) VALUES (1, '0')"
            )
            self._conn.commit()

    def _migrate(self) -> None:
        """Upgrade legacy ticker-only PK positions to (ticker, side)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(paper_positions)").fetchall()}
        if "fees_paid" not in cols:
            try:
                self._conn.execute(
                    "ALTER TABLE paper_positions ADD COLUMN fees_paid TEXT NOT NULL DEFAULT '0'"
                )
            except sqlite3.OperationalError:
                pass
        for col in ("location_id", "series_ticker"):
            if col not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass
        # Detect legacy schema: PRIMARY KEY on ticker only (no composite). Rebuild if needed.
        pk = [
            r
            for r in self._conn.execute("PRAGMA table_info(paper_positions)").fetchall()
            if r[5]  # pk ordinal
        ]
        if len(pk) == 1 and pk[0][1] == "ticker":
            rows = [dict(r) for r in self._conn.execute("SELECT * FROM paper_positions").fetchall()]
            self._conn.execute("ALTER TABLE paper_positions RENAME TO paper_positions_legacy")
            self._conn.execute(
                """CREATE TABLE paper_positions (
                     ticker TEXT NOT NULL,
                     side TEXT NOT NULL,
                     qty TEXT NOT NULL,
                     avg_price TEXT NOT NULL,
                     fees_paid TEXT NOT NULL DEFAULT '0',
                     location_id TEXT,
                     series_ticker TEXT,
                     updated_at_utc TEXT NOT NULL,
                     PRIMARY KEY (ticker, side)
                   )"""
            )
            for r in rows:
                self._conn.execute(
                    """INSERT INTO paper_positions(
                         ticker, side, qty, avg_price, fees_paid, updated_at_utc
                       ) VALUES (?,?,?,?,?,?)""",
                    (
                        r["ticker"],
                        r.get("side") or "yes",
                        r["qty"],
                        r["avg_price"],
                        r.get("fees_paid") or "0",
                        r.get("updated_at_utc") or datetime.now(timezone.utc).isoformat(),
                    ),
                )
            self._conn.execute("DROP TABLE paper_positions_legacy")
        fill_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(paper_fills)").fetchall()}
        for col in ("location_id", "series_ticker", "quote_ts_utc", "market_ts_utc"):
            if col not in fill_cols:
                try:
                    self._conn.execute(f"ALTER TABLE paper_fills ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass
        meta_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(paper_ledger_meta)").fetchall()
        }
        for col, default in (
            ("starting_cash", "100.00"),
            ("max_total_exposure", "80.00"),
            ("max_per_market_exposure", "25.00"),
        ):
            if col not in meta_cols:
                try:
                    self._conn.execute(
                        f"ALTER TABLE paper_ledger_meta ADD COLUMN {col} TEXT NOT NULL DEFAULT '{default}'"
                    )
                except sqlite3.OperationalError:
                    pass
        # Ensure settlements have unique settlement_key
        sett_cols = {
            r[1] for r in self._conn.execute("PRAGMA table_info(paper_settlements)").fetchall()
        }
        if "settlement_key" not in sett_cols:
            # Rebuild settlements table for new schema
            try:
                self._conn.execute("ALTER TABLE paper_settlements RENAME TO paper_settlements_legacy")
                self._conn.execute(
                    """CREATE TABLE paper_settlements (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         created_at_utc TEXT NOT NULL,
                         ticker TEXT NOT NULL,
                         side TEXT NOT NULL,
                         qty TEXT NOT NULL,
                         result TEXT NOT NULL,
                         payout TEXT NOT NULL,
                         cost_basis TEXT NOT NULL,
                         fees TEXT NOT NULL,
                         pnl TEXT NOT NULL,
                         settlement_key TEXT NOT NULL UNIQUE,
                         details_json TEXT NOT NULL
                       )"""
                )
                for r in self._conn.execute("SELECT * FROM paper_settlements_legacy"):
                    d = dict(r)
                    key = f"legacy-{d['id']}-{d['ticker']}"
                    self._conn.execute(
                        """INSERT INTO paper_settlements(
                             created_at_utc, ticker, side, qty, result, payout, cost_basis, fees, pnl,
                             settlement_key, details_json
                           ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            d["created_at_utc"],
                            d["ticker"],
                            "unknown",
                            "0",
                            "legacy",
                            "0",
                            "0",
                            "0",
                            d.get("pnl") or "0",
                            key,
                            d.get("details_json") or "{}",
                        ),
                    )
                self._conn.execute("DROP TABLE paper_settlements_legacy")
            except sqlite3.OperationalError:
                pass
        self._conn.execute(
            "INSERT OR IGNORE INTO paper_exposure(id, open_notional) VALUES (1, '0')"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def cash(self) -> Decimal:
        return D(self._conn.execute("SELECT cash FROM paper_ledger_meta WHERE id=1").fetchone()["cash"])

    def limits(self) -> dict[str, str]:
        row = self._conn.execute(
            "SELECT max_total_exposure, max_per_market_exposure, starting_cash FROM paper_ledger_meta WHERE id=1"
        ).fetchone()
        return {
            "max_total_exposure": row["max_total_exposure"],
            "max_per_market_exposure": row["max_per_market_exposure"],
            "starting_cash": row["starting_cash"],
        }

    def positions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM paper_positions ORDER BY ticker, side")]

    def open_notional(self) -> Decimal:
        total = ZERO
        for p in self.positions():
            total += D(p["qty"]) * D(p["avg_price"])
        return total

    def market_notional(self, ticker: str) -> Decimal:
        total = ZERO
        for p in self.positions():
            if p["ticker"] == ticker:
                total += D(p["qty"]) * D(p["avg_price"])
        return total

    def snapshot(self) -> dict[str, Any]:
        fills = [dict(r) for r in self._conn.execute("SELECT * FROM paper_fills ORDER BY id DESC LIMIT 40")]
        settlements = [
            dict(r) for r in self._conn.execute("SELECT * FROM paper_settlements ORDER BY id DESC LIMIT 40")
        ]
        return {
            "cash": str(self.cash()),
            "open_notional": str(self.open_notional()),
            "limits": self.limits(),
            "positions": self.positions(),
            "recent_fills": fills,
            "recent_settlements": settlements,
            "path": str(self.path),
            "live_order_submitted": False,
        }

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
        location_id: str | None = None,
        series_ticker: str | None = None,
        quote_ts_utc: str | None = None,
        market_ts_utc: str | None = None,
    ) -> dict[str, Any]:
        """Conservative fill at executable ask; never calls live APIs.

        YES and NO for the same ticker are stored as separate positions.
        """
        side = (side or "").lower()
        if side not in ("yes", "no"):
            return {"ok": False, "reason": "invalid_side", "side": side}
        if D(qty) <= ZERO or D(price) <= ZERO or D(price) >= ONE:
            return {"ok": False, "reason": "invalid_qty_or_price"}
        if D(fees) < ZERO:
            return {"ok": False, "reason": "invalid_fees"}

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    "SELECT id FROM paper_fills WHERE client_order_id=?", (client_order_id,)
                ).fetchone()
                if existing:
                    self._conn.rollback()
                    return {
                        "ok": False,
                        "reason": "duplicate_client_order_id",
                        "client_order_id": client_order_id,
                    }

                cost = D(qty) * D(price) + D(fees)
                cash = self.cash()
                if cost > cash:
                    self._conn.rollback()
                    return {
                        "ok": False,
                        "reason": "insufficient_paper_cash",
                        "cash": str(cash),
                        "need": str(cost),
                    }

                lim = self.limits()
                add_notional = D(qty) * D(price)
                if self.open_notional() + add_notional > D(lim["max_total_exposure"]):
                    self._conn.rollback()
                    return {
                        "ok": False,
                        "reason": "blocked_total_exposure",
                        "open_notional": str(self.open_notional()),
                        "limit": lim["max_total_exposure"],
                    }
                if self.market_notional(ticker) + add_notional > D(lim["max_per_market_exposure"]):
                    self._conn.rollback()
                    return {
                        "ok": False,
                        "reason": "blocked_per_market_exposure",
                        "market_notional": str(self.market_notional(ticker)),
                        "limit": lim["max_per_market_exposure"],
                    }

                now = datetime.now(timezone.utc).isoformat()
                new_cash = cash - cost
                self._conn.execute(
                    "UPDATE paper_ledger_meta SET cash=?, updated_at_utc=? WHERE id=1",
                    (str(new_cash), now),
                )
                self._conn.execute(
                    """INSERT INTO paper_fills(
                         created_at_utc, client_order_id, ticker, side, qty, price, fees,
                         decision_reason, location_id, series_ticker, quote_ts_utc, market_ts_utc, details_json
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now,
                        client_order_id,
                        ticker,
                        side,
                        str(qty),
                        str(price),
                        str(fees),
                        decision_reason,
                        location_id,
                        series_ticker,
                        quote_ts_utc,
                        market_ts_utc,
                        json.dumps(
                            {
                                **details,
                                "live_blocked": True,
                                "live_order_submitted": False,
                                "sim_label": "unvalidated_exploratory_paper",
                            }
                        ),
                    ),
                )
                prev = self._conn.execute(
                    "SELECT * FROM paper_positions WHERE ticker=? AND side=?", (ticker, side)
                ).fetchone()
                if prev:
                    pq, ap = D(prev["qty"]), D(prev["avg_price"])
                    pf = D(prev["fees_paid"] or "0")
                    nq = pq + D(qty)
                    navg = (ap * pq + D(price) * D(qty)) / nq if nq > ZERO else D(price)
                    self._conn.execute(
                        """UPDATE paper_positions SET qty=?, avg_price=?, fees_paid=?,
                           location_id=COALESCE(?, location_id),
                           series_ticker=COALESCE(?, series_ticker),
                           updated_at_utc=?
                           WHERE ticker=? AND side=?""",
                        (
                            str(nq),
                            str(navg),
                            str(pf + D(fees)),
                            location_id,
                            series_ticker,
                            now,
                            ticker,
                            side,
                        ),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO paper_positions(
                             ticker, side, qty, avg_price, fees_paid, location_id, series_ticker, updated_at_utc
                           ) VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            ticker,
                            side,
                            str(qty),
                            str(price),
                            str(fees),
                            location_id,
                            series_ticker,
                            now,
                        ),
                    )
                self._conn.execute(
                    "UPDATE paper_exposure SET open_notional=?",
                    (str(self.open_notional()),),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

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
            "positions": self.positions(),
        }

    def settle_market(
        self,
        *,
        ticker: str,
        result: str,
        settlement_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply payouts for open YES/NO positions. Idempotent via settlement_key.

        ``result`` is ``yes`` or ``no`` (winning contract side).
        Each contract pays $1 if that side wins, else $0. Fees already paid at fill.
        """
        result = (result or "").lower()
        if result not in ("yes", "no"):
            return {"ok": False, "reason": "invalid_result", "result": result}
        key_base = settlement_key or f"settle:{ticker}:{result}"

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                positions = [
                    dict(r)
                    for r in self._conn.execute(
                        "SELECT * FROM paper_positions WHERE ticker=?", (ticker,)
                    )
                ]
                if not positions:
                    self._conn.rollback()
                    return {"ok": True, "settled": False, "reason": "no_open_positions", "ticker": ticker}

                applied: list[dict[str, Any]] = []
                now = datetime.now(timezone.utc).isoformat()
                cash = self.cash()
                for pos in positions:
                    side = pos["side"]
                    sk = f"{key_base}:{side}"
                    exists = self._conn.execute(
                        "SELECT id FROM paper_settlements WHERE settlement_key=?", (sk,)
                    ).fetchone()
                    if exists:
                        applied.append({"side": side, "status": "already_settled", "settlement_key": sk})
                        continue
                    qty = D(pos["qty"])
                    avg = D(pos["avg_price"])
                    fees = D(pos.get("fees_paid") or "0")
                    cost = qty * avg
                    payout = qty * ONE if side == result else ZERO
                    pnl = payout - cost - fees
                    cash = cash + payout
                    self._conn.execute(
                        """INSERT INTO paper_settlements(
                             created_at_utc, ticker, side, qty, result, payout, cost_basis, fees, pnl,
                             settlement_key, details_json
                           ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            now,
                            ticker,
                            side,
                            str(qty),
                            result,
                            str(payout),
                            str(cost),
                            str(fees),
                            str(pnl),
                            sk,
                            json.dumps(
                                {
                                    **(details or {}),
                                    "winning_side": result,
                                    "live_blocked": True,
                                    "sim_label": "fixture_or_official_settlement",
                                }
                            ),
                        ),
                    )
                    self._conn.execute(
                        "DELETE FROM paper_positions WHERE ticker=? AND side=?", (ticker, side)
                    )
                    applied.append(
                        {
                            "side": side,
                            "status": "settled",
                            "qty": str(qty),
                            "payout": str(payout),
                            "pnl": str(pnl),
                            "settlement_key": sk,
                        }
                    )

                self._conn.execute(
                    "UPDATE paper_ledger_meta SET cash=?, updated_at_utc=? WHERE id=1",
                    (str(cash), now),
                )
                self._conn.execute(
                    "UPDATE paper_exposure SET open_notional=?",
                    (str(self.open_notional()),),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return {
            "ok": True,
            "settled": True,
            "ticker": ticker,
            "result": result,
            "applied": applied,
            "cash_after": str(self.cash()),
            "live_order_submitted": False,
        }
