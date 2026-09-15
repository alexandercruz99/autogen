from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True)


@dataclass
class OpportunityRecord:
    id: str
    scanned_at: str
    kind: str  # individual | combo
    market_ticker: str
    event_ticker: str
    category: str
    side: str  # yes | no
    quantity: str
    executable_price: str
    estimated_prob: str
    conservative_prob: str
    uncertainty: str
    breakeven_prob: str
    estimated_ev: str
    conservative_ev: str
    max_loss: str
    fees: str
    data_freshness: str
    decision: str  # buy | skip
    reason: str
    model_version: str
    factors_json: str
    validation_note: str
    details_json: str = "{}"


@dataclass
class OrderRecord:
    client_order_id: str
    created_at: str
    mode: str
    kind: str
    market_ticker: str
    event_ticker: str
    side: str
    quantity: str
    limit_price: str
    status: str
    exchange_order_id: str = ""
    filled_quantity: str = "0.00"
    avg_fill_price: str = ""
    fees_paid: str = "0"
    reservation_id: str = ""
    opportunity_id: str = ""
    details_json: str = "{}"


@dataclass
class PositionRecord:
    id: str
    opened_at: str
    mode: str
    kind: str
    market_ticker: str
    event_ticker: str
    side: str
    quantity: str
    avg_price: str
    fees_paid: str
    status: str  # open | settled | closed
    settlement_value: str = ""
    realized_pnl: str = ""
    correlation_keys_json: str = "[]"
    details_json: str = "{}"


@dataclass
class BotState:
    mode: str = "paper"
    live_enabled: bool = False
    pause_buying: bool = False
    kill_switch: bool = False
    last_scan_at: str = ""
    last_scan_ok: bool = False
    last_error: str = ""
    connection_ok: bool = False
    paper_cash: str = "100.00"
    paper_reserved: str = "0"
    trading_budget: str = "100.00"
    realized_pnl: str = "0"
    peak_equity: str = "100.00"
    daily_realized_pnl: str = "0"
    daily_pnl_date: str = ""
    live_ack_at: str = ""
    extra_json: str = "{}"


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS markets (
                    ticker TEXT PRIMARY KEY,
                    event_ticker TEXT,
                    series_ticker TEXT,
                    category TEXT,
                    title TEXT,
                    status TEXT,
                    close_time TEXT,
                    expected_expiration_time TEXT,
                    raw_json TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    captured_at TEXT,
                    yes_bid TEXT,
                    yes_ask TEXT,
                    no_bid TEXT,
                    no_ask TEXT,
                    book_json TEXT
                );
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    scanned_at TEXT,
                    kind TEXT,
                    market_ticker TEXT,
                    event_ticker TEXT,
                    category TEXT,
                    side TEXT,
                    quantity TEXT,
                    executable_price TEXT,
                    estimated_prob TEXT,
                    conservative_prob TEXT,
                    uncertainty TEXT,
                    breakeven_prob TEXT,
                    estimated_ev TEXT,
                    conservative_ev TEXT,
                    max_loss TEXT,
                    fees TEXT,
                    data_freshness TEXT,
                    decision TEXT,
                    reason TEXT,
                    model_version TEXT,
                    factors_json TEXT,
                    validation_note TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    mode TEXT,
                    kind TEXT,
                    market_ticker TEXT,
                    event_ticker TEXT,
                    side TEXT,
                    quantity TEXT,
                    limit_price TEXT,
                    status TEXT,
                    exchange_order_id TEXT,
                    filled_quantity TEXT,
                    avg_fill_price TEXT,
                    fees_paid TEXT,
                    reservation_id TEXT,
                    opportunity_id TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    opened_at TEXT,
                    mode TEXT,
                    kind TEXT,
                    market_ticker TEXT,
                    event_ticker TEXT,
                    side TEXT,
                    quantity TEXT,
                    avg_price TEXT,
                    fees_paid TEXT,
                    status TEXT,
                    settlement_value TEXT,
                    realized_pnl TEXT,
                    correlation_keys_json TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    amount TEXT,
                    status TEXT,
                    order_client_id TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    level TEXT,
                    event TEXT,
                    message TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS forecast_cache (
                    cache_key TEXT PRIMARY KEY,
                    fetched_at TEXT,
                    source TEXT,
                    payload_json TEXT
                );
                """
            )
            row = conn.execute("SELECT payload FROM bot_state WHERE id=1").fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO bot_state (id, payload) VALUES (1, ?)",
                    (dumps(asdict(BotState())),),
                )

    def get_state(self) -> BotState:
        with self._conn() as conn:
            row = conn.execute("SELECT payload FROM bot_state WHERE id=1").fetchone()
            data = json.loads(row["payload"])
            return BotState(**data)

    def update_state(self, **kwargs: Any) -> BotState:
        with self._conn() as conn:
            row = conn.execute("SELECT payload FROM bot_state WHERE id=1").fetchone()
            data = json.loads(row["payload"])
            data.update(kwargs)
            conn.execute("UPDATE bot_state SET payload=? WHERE id=1", (dumps(data),))
            return BotState(**data)

    def upsert_market(self, market: dict[str, Any], category: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO markets (ticker, event_ticker, series_ticker, category, title, status,
                    close_time, expected_expiration_time, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    event_ticker=excluded.event_ticker,
                    series_ticker=excluded.series_ticker,
                    category=excluded.category,
                    title=excluded.title,
                    status=excluded.status,
                    close_time=excluded.close_time,
                    expected_expiration_time=excluded.expected_expiration_time,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                (
                    market.get("ticker"),
                    market.get("event_ticker"),
                    market.get("series_ticker") or "",
                    category,
                    market.get("title") or market.get("subtitle") or "",
                    market.get("status"),
                    market.get("close_time"),
                    market.get("expected_expiration_time"),
                    dumps(market),
                    utcnow(),
                ),
            )

    def save_snapshot(
        self,
        ticker: str,
        yes_bid: str | None,
        yes_ask: str | None,
        no_bid: str | None,
        no_ask: str | None,
        book: dict[str, Any],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (ticker, captured_at, yes_bid, yes_ask, no_bid, no_ask, book_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, utcnow(), yes_bid, yes_ask, no_bid, no_ask, dumps(book)),
            )

    def save_opportunity(self, opp: OpportunityRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO opportunities (
                    id, scanned_at, kind, market_ticker, event_ticker, category, side, quantity,
                    executable_price, estimated_prob, conservative_prob, uncertainty, breakeven_prob,
                    estimated_ev, conservative_ev, max_loss, fees, data_freshness, decision, reason,
                    model_version, factors_json, validation_note, details_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    opp.id, opp.scanned_at, opp.kind, opp.market_ticker, opp.event_ticker,
                    opp.category, opp.side, opp.quantity, opp.executable_price, opp.estimated_prob,
                    opp.conservative_prob, opp.uncertainty, opp.breakeven_prob, opp.estimated_ev,
                    opp.conservative_ev, opp.max_loss, opp.fees, opp.data_freshness, opp.decision,
                    opp.reason, opp.model_version, opp.factors_json, opp.validation_note, opp.details_json,
                ),
            )

    def list_opportunities(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM opportunities ORDER BY scanned_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_order(self, order: OrderRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders (
                    client_order_id, created_at, mode, kind, market_ticker, event_ticker, side,
                    quantity, limit_price, status, exchange_order_id, filled_quantity, avg_fill_price,
                    fees_paid, reservation_id, opportunity_id, details_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    order.client_order_id, order.created_at, order.mode, order.kind, order.market_ticker,
                    order.event_ticker, order.side, order.quantity, order.limit_price, order.status,
                    order.exchange_order_id, order.filled_quantity, order.avg_fill_price, order.fees_paid,
                    order.reservation_id, order.opportunity_id, order.details_json,
                ),
            )

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_open_orders(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE status IN ('pending','submitted','partial','resting') ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_position(self, pos: PositionRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions (
                    id, opened_at, mode, kind, market_ticker, event_ticker, side, quantity, avg_price,
                    fees_paid, status, settlement_value, realized_pnl, correlation_keys_json, details_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pos.id, pos.opened_at, pos.mode, pos.kind, pos.market_ticker, pos.event_ticker,
                    pos.side, pos.quantity, pos.avg_price, pos.fees_paid, pos.status,
                    pos.settlement_value, pos.realized_pnl, pos.correlation_keys_json, pos.details_json,
                ),
            )

    def list_positions(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE status=? ORDER BY opened_at DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM positions ORDER BY opened_at DESC").fetchall()
            return [dict(r) for r in rows]

    def create_reservation(self, reservation_id: str, amount: Decimal, order_client_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO reservations (id, created_at, amount, status, order_client_id, details_json)
                VALUES (?, ?, ?, 'active', ?, '{}')
                """,
                (reservation_id, utcnow(), str(amount), order_client_id),
            )

    def release_reservation(self, reservation_id: str, status: str = "released") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE reservations SET status=? WHERE id=?",
                (status, reservation_id),
            )

    def active_reservations_total(self) -> Decimal:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(CAST(amount AS REAL)), 0) AS total FROM reservations WHERE status='active'"
            ).fetchone()
            return Decimal(str(row["total"] or 0))

    def audit(self, event: str, message: str, level: str = "info", details: dict[str, Any] | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, level, event, message, details_json) VALUES (?,?,?,?,?)",
                (utcnow(), level, event, message, dumps(details or {})),
            )

    def list_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def cache_forecast(self, key: str, source: str, payload: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_cache (cache_key, fetched_at, source, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (key, utcnow(), source, dumps(payload)),
            )

    def get_forecast_cache(self, key: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM forecast_cache WHERE cache_key=?", (key,)
            ).fetchone()
            return dict(row) if row else None
