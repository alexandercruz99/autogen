"""Persistent forecast/observation/CLI archive with decision-time metadata."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class WeatherArchive:
    """SQLite archive separate from trading bot_state but colocated under data/."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecast_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model_name TEXT,
                    init_time TEXT,
                    available_at TEXT,
                    valid_date TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    temp_max_f REAL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(station_key, source, model_name, init_time, valid_date)
                );
                CREATE TABLE IF NOT EXISTS cli_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_key TEXT NOT NULL,
                    climate_day TEXT NOT NULL,
                    max_temp_f INTEGER,
                    min_temp_f INTEGER,
                    issuance_time TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    product_id TEXT,
                    is_preliminary INTEGER,
                    raw_text TEXT,
                    UNIQUE(station_key, climate_day, product_id)
                );
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_key TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    train_end_day TEXT,
                    metrics_json TEXT,
                    artifact_json TEXT NOT NULL,
                    UNIQUE(station_key, model_name)
                );
                CREATE TABLE IF NOT EXISTS validation_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    station_key TEXT,
                    report_json TEXT NOT NULL
                );
                """
            )

    def save_forecast(
        self,
        *,
        station_key: str,
        source: str,
        valid_date: date,
        temp_max_f: float | None,
        payload: dict[str, Any],
        model_name: str | None = None,
        init_time: str | None = None,
        available_at: str | None = None,
        retrieved_at: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_snapshots
                (station_key, source, model_name, init_time, available_at, valid_date, retrieved_at, temp_max_f, payload_json)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    station_key,
                    source,
                    model_name,
                    init_time,
                    available_at or retrieved_at or utcnow(),
                    valid_date.isoformat(),
                    retrieved_at or utcnow(),
                    temp_max_f,
                    json.dumps(payload),
                ),
            )

    def save_cli(self, *, station_key: str, report: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cli_outcomes
                (station_key, climate_day, max_temp_f, min_temp_f, issuance_time, available_at,
                 retrieved_at, product_id, is_preliminary, raw_text)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    station_key,
                    report.climate_day.isoformat(),
                    report.max_temp_f,
                    report.min_temp_f,
                    report.issuance_time.isoformat(),
                    report.available_at.isoformat(),
                    report.retrieved_at.isoformat(),
                    report.product_id,
                    1 if report.is_preliminary else 0,
                    report.raw_text,
                ),
            )

    def list_cli_finals(self, station_key: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cli_outcomes
                WHERE station_key=? AND max_temp_f IS NOT NULL AND is_preliminary=0
                ORDER BY climate_day
                """,
                (station_key,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_cli_all(self, station_key: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cli_outcomes
                WHERE station_key=? AND max_temp_f IS NOT NULL
                ORDER BY climate_day, issuance_time
                """,
                (station_key,),
            ).fetchall()
            return [dict(r) for r in rows]

    def forecasts_for(
        self, station_key: str, source: str, valid_date: date
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM forecast_snapshots
                WHERE station_key=? AND source=? AND valid_date=?
                ORDER BY available_at
                """,
                (station_key, source, valid_date.isoformat()),
            ).fetchall()
            return [dict(r) for r in rows]

    def paired_forecast_outcomes(
        self, station_key: str, source: str = "nws_grid"
    ) -> list[dict[str, Any]]:
        """Join forecasts to CLI outcomes for residual learning (best-effort vintage)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT f.valid_date, f.temp_max_f AS forecast_f, f.available_at, f.init_time,
                       c.max_temp_f AS outcome_f, c.issuance_time, c.is_preliminary
                FROM forecast_snapshots f
                JOIN cli_outcomes c
                  ON c.station_key=f.station_key AND c.climate_day=f.valid_date
                WHERE f.station_key=? AND f.source=? AND f.temp_max_f IS NOT NULL
                  AND c.max_temp_f IS NOT NULL
                ORDER BY f.valid_date, f.available_at
                """,
                (station_key, source),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_artifact(self, station_key: str, model_name: str, artifact: dict[str, Any], metrics: dict[str, Any] | None = None, train_end_day: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_artifacts
                (station_key, model_name, trained_at, train_end_day, metrics_json, artifact_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    station_key,
                    model_name,
                    utcnow(),
                    train_end_day,
                    json.dumps(metrics or {}),
                    json.dumps(artifact),
                ),
            )

    def load_artifact(self, station_key: str, model_name: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM model_artifacts WHERE station_key=? AND model_name=?
                """,
                (station_key, model_name),
            ).fetchone()
            return dict(row) if row else None

    def save_validation_report(self, report: dict[str, Any], station_key: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO validation_reports (created_at, station_key, report_json) VALUES (?,?,?)",
                (utcnow(), station_key, json.dumps(report)),
            )

    def latest_validation(self, station_key: str | None = None) -> dict[str, Any] | None:
        with self._conn() as conn:
            if station_key:
                row = conn.execute(
                    "SELECT * FROM validation_reports WHERE station_key=? ORDER BY id DESC LIMIT 1",
                    (station_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM validation_reports ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["report"] = json.loads(d["report_json"])
            return d
