"""Tests for feed adapters: missingness hygiene and storage duplicate protection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kalshi_bot.models.weather.obs_engine.feeds.features_live import SATRAD_EXTRA, build_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import STATION_V2_FEATURES
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

SATRAD_FEATURES = list(STATION_V2_FEATURES) + list(SATRAD_EXTRA)


def test_feed_store_duplicate_protection(tmp_path: Path):
    store = FeedStore(path=tmp_path / "feeds.db")
    a = store.upsert_sample(feed="metar_KNYC", source_key="k1", payload={"tmpf": 70.0}, valid_utc="2026-01-01T12:00:00+00:00")
    b = store.upsert_sample(feed="metar_KNYC", source_key="k1", payload={"tmpf": 71.0}, valid_utc="2026-01-01T12:00:00+00:00")
    assert a["new"] is True
    assert b["new"] is False
    assert a["first_seen_utc"] == b["first_seen_utc"]
    n = store._conn.execute("SELECT COUNT(*) AS c FROM feed_samples").fetchone()["c"]
    assert n == 1
    store.close()


def test_missing_radar_not_interpreted_as_dry(tmp_path: Path, monkeypatch):
    store = FeedStore(path=tmp_path / "feeds.db")
    # Station obs only — no radar/goes
    now = datetime.now(timezone.utc)
    store.upsert_sample(
        feed="metar_KNYC",
        source_key="s1",
        payload={
            "station": "KNYC",
            "valid_utc": now.isoformat(),
            "tmpf": 72.0,
            "dwpf": 55.0,
            "sknt": None,
            "drct": None,
            "skyc1": None,
            "missing": {"sknt": True, "drct": True, "skyc1": True, "tmpf": False},
        },
        valid_utc=now.isoformat(),
    )
    # Point store path used by build_operating_features via default — inject via monkeypatch of FeedStore methods already on instance
    bundle = build_operating_features(store, now=now)
    assert bundle["features"]["radar_available"] == 0.0
    assert bundle["features"]["radar_precip_frac"] is None
    assert bundle["missing"].get("radar") is True
    assert "NOT interpreted as no precipitation" in (bundle["provenance"].get("radar_note") or "")
    assert bundle["features"]["goes_available"] == 0.0
    assert bundle["feature_names"] == list(STATION_V2_FEATURES)
    store.close()


def test_stale_goes_flagged(tmp_path: Path):
    store = FeedStore(path=tmp_path / "feeds.db")
    old = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    store.upsert_sample(
        feed="goes19_acmc",
        source_key="g1",
        payload={
            "features": {
                "cloud_frac_bcm": 0.4,
                "cloudy_or_probably_frac_acm": 0.5,
                "valid_utc": old.isoformat(),
                "provenance": {"includes_rtm_model_assist": True},
            }
        },
        valid_utc=old.isoformat(),
    )
    now = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)
    bundle = build_operating_features(store, now=now)
    assert bundle["features"]["goes_available"] == 1.0
    assert bundle["missing"].get("goes_stale") is True
    store.close()
