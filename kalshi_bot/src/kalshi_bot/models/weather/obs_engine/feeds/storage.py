"""Shared SQLite store for feed samples, features, predictions, checkpoints."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS feed_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  feed TEXT NOT NULL,
  retrieved_at_utc TEXT NOT NULL,
  valid_utc TEXT,
  first_seen_utc TEXT,
  source_key TEXT,
  product TEXT,
  payload_json TEXT NOT NULL,
  local_path TEXT,
  UNIQUE(feed, source_key)
);
CREATE TABLE IF NOT EXISTS feature_rows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  decision_time_utc TEXT NOT NULL,
  climate_day TEXT,
  feature_set TEXT NOT NULL,
  model_version TEXT,
  features_json TEXT NOT NULL,
  missing_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  model_version TEXT NOT NULL,
  feature_set TEXT,
  target_day TEXT,
  ticker TEXT,
  prediction_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  ticker TEXT,
  side TEXT,
  decision TEXT NOT NULL,
  reason TEXT,
  details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  feed TEXT PRIMARY KEY,
  last_success_utc TEXT,
  last_attempt_utc TEXT,
  last_error TEXT,
  last_source_key TEXT,
  consecutive_failures INTEGER DEFAULT 0,
  meta_json TEXT
);
CREATE TABLE IF NOT EXISTS heartbeat (
  id INTEGER PRIMARY KEY CHECK (id=1),
  updated_at_utc TEXT NOT NULL,
  status TEXT NOT NULL,
  detail_json TEXT
);
"""


class FeedStore:
    def __init__(self, path: Path | None = None) -> None:
        root = Path("data/obs_engine/feeds")
        root.mkdir(parents=True, exist_ok=True)
        self.path = path or (root / "feeds.db")
        self._conn = sqlite3.connect(str(self.path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert_sample(
        self,
        *,
        feed: str,
        source_key: str,
        payload: dict[str, Any],
        valid_utc: str | None = None,
        product: str | None = None,
        local_path: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        row = self._conn.execute(
            "SELECT first_seen_utc FROM feed_samples WHERE feed=? AND source_key=?",
            (feed, source_key),
        ).fetchone()
        first = row["first_seen_utc"] if row else now
        self._conn.execute(
            """INSERT INTO feed_samples(feed, retrieved_at_utc, valid_utc, first_seen_utc, source_key, product, payload_json, local_path)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(feed, source_key) DO UPDATE SET
                 retrieved_at_utc=excluded.retrieved_at_utc,
                 payload_json=excluded.payload_json,
                 local_path=COALESCE(excluded.local_path, feed_samples.local_path)""",
            (feed, now, valid_utc, first, source_key, product, json.dumps(payload), local_path),
        )
        self._conn.commit()
        return {"feed": feed, "source_key": source_key, "first_seen_utc": first, "retrieved_at_utc": now, "new": row is None}

    def checkpoint(self, feed: str, *, ok: bool, source_key: str | None = None, error: str | None = None, meta: dict | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        prev = self._conn.execute("SELECT consecutive_failures FROM checkpoints WHERE feed=?", (feed,)).fetchone()
        fails = 0 if ok else int(prev["consecutive_failures"] if prev else 0) + 1
        self._conn.execute(
            """INSERT INTO checkpoints(feed, last_success_utc, last_attempt_utc, last_error, last_source_key, consecutive_failures, meta_json)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(feed) DO UPDATE SET
                 last_success_utc=CASE WHEN ? THEN excluded.last_success_utc ELSE checkpoints.last_success_utc END,
                 last_attempt_utc=excluded.last_attempt_utc,
                 last_error=excluded.last_error,
                 last_source_key=COALESCE(excluded.last_source_key, checkpoints.last_source_key),
                 consecutive_failures=excluded.consecutive_failures,
                 meta_json=excluded.meta_json""",
            (
                feed,
                now if ok else None,
                now,
                error,
                source_key,
                fails,
                json.dumps(meta or {}),
                ok,
            ),
        )
        self._conn.commit()

    def heartbeat(self, status: str, detail: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO heartbeat(id, updated_at_utc, status, detail_json) VALUES (1,?,?,?)
               ON CONFLICT(id) DO UPDATE SET updated_at_utc=excluded.updated_at_utc, status=excluded.status, detail_json=excluded.detail_json""",
            (now, status, json.dumps(detail or {})),
        )
        self._conn.commit()

    def get_heartbeat(self) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM heartbeat WHERE id=1").fetchone()
        return dict(row) if row else None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM checkpoints ORDER BY feed").fetchall()]

    def latest_sample(self, feed: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM feed_samples WHERE feed=? ORDER BY retrieved_at_utc DESC LIMIT 1",
            (feed,),
        ).fetchone()
        return dict(row) if row else None

    def save_features(self, *, decision_time_utc: str, climate_day: str | None, feature_set: str, model_version: str | None, features: dict, missing: dict, provenance: dict) -> None:
        self._conn.execute(
            """INSERT INTO feature_rows(created_at_utc, decision_time_utc, climate_day, feature_set, model_version, features_json, missing_json, provenance_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                decision_time_utc,
                climate_day,
                feature_set,
                model_version,
                json.dumps(features),
                json.dumps(missing),
                json.dumps(provenance),
            ),
        )
        self._conn.commit()

    def save_prediction(self, *, model_version: str, feature_set: str | None, target_day: str | None, ticker: str | None, prediction: dict) -> None:
        self._conn.execute(
            """INSERT INTO predictions(created_at_utc, model_version, feature_set, target_day, ticker, prediction_json)
               VALUES (?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                model_version,
                feature_set,
                target_day,
                ticker,
                json.dumps(prediction),
            ),
        )
        self._conn.commit()

    def save_paper_decision(self, *, ticker: str | None, side: str | None, decision: str, reason: str, details: dict) -> None:
        self._conn.execute(
            """INSERT INTO paper_decisions(created_at_utc, ticker, side, decision, reason, details_json)
               VALUES (?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(), ticker, side, decision, reason, json.dumps(details)),
        )
        self._conn.commit()

    def latest_prediction(self) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
