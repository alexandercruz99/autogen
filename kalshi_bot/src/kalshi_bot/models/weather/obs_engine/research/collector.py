"""Append-only prospective collector for observations, CLI, forecasts, books, predictions.

Persists to SQLite under data/obs_engine/research/prospective.db.
Schedule: run via `weather-obs-collect-once` or wire into BotLoop as an optional hook.
This session does NOT claim a daemon continues after exit unless the user keeps `kalshi-bot run` alive.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.config import AppConfig
from kalshi_bot.models.weather.obs_engine import MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import default_data_dir

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS collector_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  retrieved_at_utc TEXT NOT NULL,
  station TEXT NOT NULL,
  source TEXT NOT NULL,
  valid_utc TEXT,
  receipt_time TEXT,
  first_seen_utc TEXT,
  tmpf REAL,
  tmpc_raw TEXT,
  precision_note TEXT,
  dwpf REAL,
  sknt REAL,
  drct REAL,
  skyc1 TEXT,
  raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cli_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  retrieved_at_utc TEXT NOT NULL,
  climate_day TEXT,
  issuance_utc TEXT,
  max_temp_f REAL,
  is_preliminary INTEGER,
  product_id TEXT,
  raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark_forecasts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  retrieved_at_utc TEXT NOT NULL,
  target_day TEXT,
  source TEXT NOT NULL,
  temp_f REAL,
  available_at TEXT,
  issue_time TEXT,
  raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS market_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  retrieved_at_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  market_json TEXT NOT NULL,
  orderbook_json TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  retrieved_at_utc TEXT NOT NULL,
  model_version TEXT NOT NULL,
  ticker TEXT,
  target_day TEXT,
  prediction_json TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_obs_first ON observations(station, valid_utc, source);
"""


class ProspectiveCollector:
    def __init__(self, db_path: Path | None = None) -> None:
        data_dir = default_data_dir()
        self.db_path = db_path or (data_dir / "research" / "prospective.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO collector_meta(key,value) VALUES(?,?)",
            ("created_or_opened_utc", datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _first_seen(self, station: str, valid_utc: str | None, source: str) -> str:
        now = datetime.now(timezone.utc).isoformat()
        if not valid_utc:
            return now
        row = self._conn.execute(
            "SELECT first_seen_utc FROM observations WHERE station=? AND valid_utc=? AND source=?",
            (station, valid_utc, source),
        ).fetchone()
        return row[0] if row and row[0] else now

    def collect_metar(self) -> dict[str, Any]:
        import httpx

        now = datetime.now(timezone.utc)
        url = f"https://aviationweather.gov/api/data/metar?ids={NYC_TARGET.metar_id}&format=json&hours=6"
        r = httpx.get(url, timeout=30.0, headers={"User-Agent": "KalshiBotResearchCollector/0.1"}, follow_redirects=True)
        r.raise_for_status()
        rows = r.json() or []
        n_new = 0
        for row in rows:
            ot = row.get("obsTime")
            valid = datetime.fromtimestamp(int(ot), tz=timezone.utc).isoformat() if ot is not None else None
            receipt = row.get("receiptTime")
            first = self._first_seen(NYC_TARGET.metar_id, valid, "aviationweather_metar")
            temp_c = row.get("temp")
            # Prefer more precise fields when present
            temp_precise = row.get("tempFloat") or row.get("temp")
            tmpf = None
            precision = "whole_C_to_F" if temp_c is not None and row.get("tempFloat") is None else "as_reported"
            if temp_precise is not None:
                try:
                    tmpf = float(temp_precise) * 9 / 5 + 32
                except (TypeError, ValueError):
                    tmpf = None
            # Insert or ignore duplicates on (station, valid, source)
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO observations(
                    retrieved_at_utc, station, source, valid_utc, receipt_time, first_seen_utc,
                    tmpf, tmpc_raw, precision_note, dwpf, sknt, drct, skyc1, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    now.isoformat(),
                    NYC_TARGET.metar_id,
                    "aviationweather_metar",
                    valid,
                    str(receipt) if receipt is not None else None,
                    first,
                    tmpf,
                    str(temp_c) if temp_c is not None else None,
                    precision,
                    None,
                    float(row["wspd"]) if isinstance(row.get("wspd"), (int, float)) else None,
                    None,
                    row.get("cover"),
                    json.dumps(row),
                ),
            )
            n_new += cur.rowcount
        self._conn.commit()
        return {"ok": True, "n_rows_fetched": len(rows), "n_new_inserted": n_new}

    def collect_cli(self) -> dict[str, Any]:
        from kalshi_bot.models.weather.cli_reports import CliReportClient

        now = datetime.now(timezone.utc)
        client = CliReportClient()
        n = 0
        try:
            reports = client.collect_recent_with_max(NYC_TARGET.cli_location_id, limit=5)
            for rep in reports:
                self._conn.execute(
                    """INSERT INTO cli_reports(retrieved_at_utc, climate_day, issuance_utc, max_temp_f,
                       is_preliminary, product_id, raw_json) VALUES (?,?,?,?,?,?,?)""",
                    (
                        now.isoformat(),
                        rep.climate_day.isoformat() if rep.climate_day else None,
                        rep.issuance_time.isoformat() if rep.issuance_time else None,
                        float(rep.max_temp_f) if rep.max_temp_f is not None else None,
                        1 if rep.is_preliminary else 0,
                        getattr(rep, "product_id", None),
                        json.dumps(
                            {
                                "climate_day": rep.climate_day.isoformat() if rep.climate_day else None,
                                "max_temp_f": rep.max_temp_f,
                                "is_preliminary": rep.is_preliminary,
                                "issuance_time": rep.issuance_time.isoformat() if rep.issuance_time else None,
                            }
                        ),
                    ),
                )
                n += 1
            self._conn.commit()
            return {"ok": True, "n_reports": n}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            client.close()

    def collect_nws_benchmark(self, target_day) -> dict[str, Any]:
        from kalshi_bot.models.weather.nws_client import NWSClient

        now = datetime.now(timezone.utc)
        nws = NWSClient()
        try:
            fc = nws.daily_high_forecast(NYC_TARGET.lat, NYC_TARGET.lon, target_day)
            payload = fc or {"status": "unavailable"}
            self._conn.execute(
                """INSERT INTO benchmark_forecasts(retrieved_at_utc, target_day, source, temp_f,
                   available_at, issue_time, raw_json) VALUES (?,?,?,?,?,?,?)""",
                (
                    now.isoformat(),
                    target_day.isoformat(),
                    "nws_grid_daytime",
                    (fc or {}).get("temp_f"),
                    (fc or {}).get("start_time"),
                    (fc or {}).get("fetched_at"),
                    json.dumps(payload),
                ),
            )
            self._conn.commit()
            return {"ok": True, "forecast": fc}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            nws.close()

    def collect_markets_and_books(self, client) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        n_m = 0
        for series in NYC_TARGET.series_prefixes:
            try:
                payload = client.get_markets(series_ticker=series, status="open", limit=50)
                for m in payload.get("markets") or []:
                    ticker = m.get("ticker") or ""
                    if "HOUR" in ticker.upper():
                        continue
                    book = None
                    try:
                        book = client.get_orderbook(ticker, depth=5)
                    except Exception:
                        book = None
                    self._conn.execute(
                        """INSERT INTO market_snapshots(retrieved_at_utc, ticker, market_json, orderbook_json)
                           VALUES (?,?,?,?)""",
                        (now.isoformat(), ticker, json.dumps(m), json.dumps(book) if book else None),
                    )
                    n_m += 1
            except Exception as exc:
                logger.warning("market collect %s: %s", series, exc)
        self._conn.commit()
        return {"ok": True, "n_markets": n_m}

    def collect_prediction(self, config: AppConfig) -> dict[str, Any]:
        from kalshi_bot.models.weather.obs_engine.predict_now import research_predict_now

        now = datetime.now(timezone.utc)
        pred = research_predict_now(config)
        self._conn.execute(
            """INSERT INTO predictions(retrieved_at_utc, model_version, ticker, target_day, prediction_json)
               VALUES (?,?,?,?,?)""",
            (
                now.isoformat(),
                pred.get("model_version") or MODEL_VERSION,
                (pred.get("brackets") or [{}])[0].get("ticker") if pred.get("brackets") else None,
                pred.get("target_date"),
                json.dumps(pred),
            ),
        )
        self._conn.commit()
        return {"ok": pred.get("status") == "ok", "prediction_status": pred.get("status"), "target_date": pred.get("target_date")}


def run_collect_once(config: AppConfig) -> dict[str, Any]:
    """One prospective collection cycle. Safe for cron / BotLoop hook. No live orders."""
    from datetime import date
    from zoneinfo import ZoneInfo

    from kalshi_bot.api.client import KalshiClient

    coll = ProspectiveCollector()
    client = KalshiClient(config.api)
    summary: dict[str, Any] = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(coll.db_path),
        "scheduling": (
            "Invoke weather-obs-collect-once periodically, or keep `kalshi-bot run` with "
            "research_collect_hook enabled. No background daemon is started by this function alone."
        ),
        "live_orders": False,
    }
    try:
        summary["metar"] = coll.collect_metar()
        summary["cli"] = coll.collect_cli()
        today = datetime.now(timezone.utc).astimezone(ZoneInfo(NYC_TARGET.timezone)).date()
        summary["nws"] = coll.collect_nws_benchmark(today)
        summary["markets"] = coll.collect_markets_and_books(client)
        summary["prediction"] = coll.collect_prediction(config)
        summary["ok"] = True
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)
    finally:
        coll.close()
        client.close()
    out = default_data_dir() / "research" / "last_collect_once.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    return summary
