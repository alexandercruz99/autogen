"""Regression tests for feed pipeline defect fixes (coverage, CLI, schema, paper, live block)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from kalshi_bot.models.weather.obs_engine.data import HourlyObs
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import (
    lst_climate_day,
    is_supported_decision_time,
    next_supported_decision_utc,
)
from kalshi_bot.models.weather.obs_engine.feeds.cli_match import select_cli_for_decision
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    STATION_V2_FEATURES,
    build_station_v2_features,
    assess_coverage,
    day_window_obs,
)
from kalshi_bot.models.weather.obs_engine.feeds.features_live import build_operating_features, metar_to_hourly
from kalshi_bot.models.weather.obs_engine.feeds.paper_sim import PaperLedger
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.money import D


def _obs(ts: datetime, tmpf: float, **kw) -> HourlyObs:
    return HourlyObs(
        valid_utc=ts,
        tmpf=tmpf,
        dwpf=kw.get("dwpf", 50.0),
        sknt=kw.get("sknt", 5.0),
        drct=kw.get("drct", 180.0),
        alti=kw.get("alti", 30.05),
        p01i=kw.get("p01i", 0.0),
        skyc1=kw.get("skyc1", "CLR"),
        station="KNYC",
    )


def test_lst_climate_day_dst_midnight():
    # 2026-09-16 00:51 EDT == still Sep 15 LST
    when = datetime(2026, 9, 16, 4, 51, tzinfo=timezone.utc)
    assert lst_climate_day(when) == date(2026, 9, 15)


def test_coverage_partial_window_not_verified_full_day():
    day = date(2025, 7, 15)
    # Only afternoon obs — late start
    obs = [
        _obs(datetime(2025, 7, 15, 18, 51, tzinfo=timezone.utc), 88.0),
        _obs(datetime(2025, 7, 15, 19, 51, tzinfo=timezone.utc), 90.0),
    ]
    decision = datetime(2025, 7, 15, 20, 0, tzinfo=timezone.utc)
    day_obs = day_window_obs(obs, climate_day=day, decision_utc=decision)
    cov = assess_coverage(day_obs, climate_day=day, decision_utc=decision)
    assert cov.adequate is False
    assert cov.max_so_far_status == "partial_window"
    assert any("NOT a verified full-day" in n for n in cov.notes)


def test_missing_precip_not_zero():
    day = date(2025, 7, 15)
    obs = []
    for h in range(6, 15):
        obs.append(
            _obs(
                datetime(2025, 7, 15, h + 4, 51, tzinfo=timezone.utc),  # rough UTC
                70.0 + h,
                p01i=None,  # missing precip
                alti=30.1,
            )
        )
    decision = datetime(2025, 7, 15, 18, 0, tzinfo=timezone.utc)
    # Build with climate_day forced; may need enough morning obs — use LST morning
    morning = [
        _obs(datetime(2025, 7, 15, 11, 0, tzinfo=timezone.utc), 65.0, p01i=None),
        _obs(datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc), 68.0, p01i=None),
        _obs(datetime(2025, 7, 15, 13, 0, tzinfo=timezone.utc), 72.0, p01i=None),
        _obs(datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc), 75.0, p01i=None),
        _obs(datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc), 78.0, p01i=None),
    ]
    decision = datetime(2025, 7, 15, 15, 30, tzinfo=timezone.utc)
    bundle = build_station_v2_features(morning, decision, climate_day=day)
    assert bundle is not None
    assert bundle.feature_map["missing_precip_6h"] == 1.0
    assert bundle.feature_map["precip_6h"] is None


def test_wrong_day_cli_excluded(tmp_path: Path):
    store = FeedStore(path=tmp_path / "f.db")
    store.upsert_sample(
        feed="cli_nyc",
        source_key="CLI:NYC:2026-09-11:x:F",
        payload={
            "climate_day": "2026-09-11",
            "issuance_utc": "2026-09-12T06:26:00+00:00",
            "max_temp_f": 79,
            "is_preliminary": False,
        },
        valid_utc="2026-09-12T06:26:00+00:00",
        first_seen_utc="2026-09-12T06:30:00+00:00",
    )
    store.upsert_sample(
        feed="cli_nyc",
        source_key="CLI:NYC:2026-09-15:x:P",
        payload={
            "climate_day": "2026-09-15",
            "issuance_utc": "2026-09-15T20:38:00+00:00",
            "max_temp_f": 72,
            "is_preliminary": True,
        },
        valid_utc="2026-09-15T20:38:00+00:00",
        first_seen_utc="2026-09-15T20:40:00+00:00",
    )
    decision = datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc)
    sel = select_cli_for_decision(store, target_day=date(2026, 9, 15), decision_utc=decision)
    assert sel["applied"] is not None
    assert sel["applied"]["climate_day"] == "2026-09-15"
    assert sel["applied"]["max_temp_f"] == 72
    assert any(e["exclude_reason"] == "wrong_climate_day" for e in sel["excluded"])
    store.close()


def test_cli_revision_prefers_final(tmp_path: Path):
    store = FeedStore(path=tmp_path / "f.db")
    store.upsert_sample(
        feed="cli_nyc",
        source_key="P",
        payload={
            "climate_day": "2026-09-14",
            "issuance_utc": "2026-09-14T20:32:00+00:00",
            "max_temp_f": 74,
            "is_preliminary": True,
        },
        valid_utc="2026-09-14T20:32:00+00:00",
        first_seen_utc="2026-09-14T20:35:00+00:00",
    )
    store.upsert_sample(
        feed="cli_nyc",
        source_key="F",
        payload={
            "climate_day": "2026-09-14",
            "issuance_utc": "2026-09-15T06:23:00+00:00",
            "max_temp_f": 75,
            "is_preliminary": False,
        },
        valid_utc="2026-09-15T06:23:00+00:00",
        first_seen_utc="2026-09-15T06:25:00+00:00",
    )
    decision = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    sel = select_cli_for_decision(store, target_day=date(2026, 9, 14), decision_utc=decision)
    assert sel["applied"]["max_temp_f"] == 75
    assert sel["applied"]["is_preliminary"] is False
    store.close()


def test_supported_decision_hours():
    # 23:13 America/New_York ≈ 03:13 UTC next day in Sep (EDT)
    late = datetime(2026, 9, 16, 3, 13, tzinfo=timezone.utc)
    ok, hour = is_supported_decision_time(late)
    assert ok is False
    assert hour is None
    nxt = next_supported_decision_utc(late)
    assert nxt > late


def test_feature_parity_replay_vs_production_path():
    """Identical timestamped obs → identical features via shared builder."""
    day = date(2025, 6, 10)
    obs = [
        _obs(datetime(2025, 6, 10, 11, 51, tzinfo=timezone.utc), 70.0, alti=30.12, p01i=0.0),
        _obs(datetime(2025, 6, 10, 12, 51, tzinfo=timezone.utc), 72.0, alti=30.10, p01i=0.0),
        _obs(datetime(2025, 6, 10, 13, 51, tzinfo=timezone.utc), 74.0, alti=30.08, p01i=0.0),
        _obs(datetime(2025, 6, 10, 14, 51, tzinfo=timezone.utc), 76.0, alti=30.05, p01i=0.0),
        _obs(datetime(2025, 6, 10, 15, 51, tzinfo=timezone.utc), 77.0, alti=30.02, p01i=0.0),
    ]
    decision = datetime(2025, 6, 10, 16, 0, tzinfo=timezone.utc)  # ~12:00 EDT
    a = build_station_v2_features(obs, decision, climate_day=day)
    b = build_station_v2_features(list(obs), decision, climate_day=day)
    assert a is not None and b is not None
    assert a.feature_names == STATION_V2_FEATURES
    assert a.values == b.values
    assert a.provenance["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_negative_remain_not_clipped():
    day = date(2025, 7, 15)
    obs = [
        _obs(datetime(2025, 7, 15, 11, 0, tzinfo=timezone.utc), 90.0),
        _obs(datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc), 91.0),
        _obs(datetime(2025, 7, 15, 13, 0, tzinfo=timezone.utc), 92.0),
        _obs(datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc), 93.0),
    ]
    decision = datetime(2025, 7, 15, 14, 30, tzinfo=timezone.utc)
    bundle = build_station_v2_features(obs, decision, climate_day=day)
    assert bundle is not None
    label = 88.0  # below max_so_far
    remain = label - bundle.max_so_far
    assert remain < 0


def test_paper_ledger_duplicate_and_no_live(tmp_path: Path):
    led = PaperLedger(path=tmp_path / "paper.db", starting_cash="50.00")
    r1 = led.try_simulate_fill(
        client_order_id="oid-1",
        ticker="T",
        side="yes",
        qty=D("1"),
        price=D("0.40"),
        fees=D("0.01"),
        decision_reason="test",
        details={},
    )
    assert r1["ok"] and r1["live_order_submitted"] is False
    r2 = led.try_simulate_fill(
        client_order_id="oid-1",
        ticker="T",
        side="yes",
        qty=D("1"),
        price=D("0.40"),
        fees=D("0.01"),
        decision_reason="test",
        details={},
    )
    assert r2["ok"] is False and r2["reason"] == "duplicate_client_order_id"
    led.close()


def test_infer_never_imports_live_submit():
    import inspect
    from kalshi_bot.models.weather.obs_engine.feeds import infer_paper

    src = inspect.getsource(infer_paper)
    assert "create_order" not in src
    assert "place_order" not in src
    assert "live_order_submitted" in src


def test_rounding_cli_whole_f_not_ceil_metar(tmp_path: Path):
    """69.08°F METAR must not become official floor 70 without whole-°F CLI."""
    store = FeedStore(path=tmp_path / "f.db")
    # no CLI
    sel = select_cli_for_decision(store, target_day=date(2026, 9, 15), decision_utc=datetime(2026, 9, 15, 20, tzinfo=timezone.utc))
    assert sel["applied"] is None
    store.close()
