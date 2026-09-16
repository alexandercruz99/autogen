"""Multi-location architecture, coverage, unified predict, paper netting/settlement tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from kalshi_bot.models.weather.obs_engine.data import HourlyObs
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import lst_climate_day
from kalshi_bot.models.weather.obs_engine.feeds.cli_match import select_cli_for_decision
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    assess_coverage,
    build_station_v2_features,
    day_window_obs,
)
from kalshi_bot.models.weather.obs_engine.feeds.features_live import metar_to_hourly
from kalshi_bot.models.weather.obs_engine.feeds.paper_sim import PaperLedger
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.pipeline import process_location
from kalshi_bot.models.weather.obs_engine.multi.predict import predict_station_v2
from kalshi_bot.models.weather.obs_engine.multi.registry import VERIFIED_NWS_CLI_DAILY_MAX, LocationRegistry
from kalshi_bot.money import D


def _obs(ts: datetime, tmpf: float, *, station: str = "KNYC", first_seen: datetime | None = None, **kw) -> HourlyObs:
    return HourlyObs(
        valid_utc=ts,
        tmpf=tmpf,
        dwpf=kw.get("dwpf", 50.0),
        sknt=kw.get("sknt", 5.0),
        drct=kw.get("drct", 180.0),
        alti=kw.get("alti", 30.05),
        p01i=kw.get("p01i", 0.0),
        skyc1=kw.get("skyc1", "CLR"),
        station=station,
        first_seen_utc=first_seen,
    )


def _morning_through(decision: datetime, *, station: str = "KNYC", start_h: int = 5) -> list[HourlyObs]:
    """Hourly temps from ~start_h UTC to decision on a summer day."""
    day = decision.date()
    out = []
    t = datetime(day.year, day.month, day.day, start_h, 0, tzinfo=timezone.utc)
    val = 60.0
    while t <= decision:
        out.append(_obs(t, val, station=station))
        val += 1.0
        t += timedelta(hours=1)
    return out


def test_coverage_rejects_stale_ten_hour_gap():
    day = date(2025, 7, 15)
    # Dense morning then last obs 10h before decision
    obs = []
    for h in range(5, 11):
        obs.append(_obs(datetime(2025, 7, 15, h, 0, tzinfo=timezone.utc), 65.0 + h))
    decision = datetime(2025, 7, 15, 20, 0, tzinfo=timezone.utc)
    day_obs = day_window_obs(obs, climate_day=day, decision_utc=decision)
    cov = assess_coverage(day_obs, climate_day=day, decision_utc=decision)
    assert cov.adequate is False
    assert cov.stale_exceeded is True
    assert cov.stale_hours is not None and cov.stale_hours >= 9.0


def test_coverage_rejects_missing_first_nine_hours():
    day = date(2025, 7, 15)
    # Window starts 00:00 LST = 05:00 UTC in July (EDT offset -4 civil but LST is -5)
    # First obs at 14:00 UTC ≈ 09:00 LST → start gap > 4h
    obs = [
        _obs(datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc), 80.0),
        _obs(datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc), 82.0),
        _obs(datetime(2025, 7, 15, 16, 0, tzinfo=timezone.utc), 84.0),
        _obs(datetime(2025, 7, 15, 17, 0, tzinfo=timezone.utc), 85.0),
    ]
    decision = datetime(2025, 7, 15, 18, 0, tzinfo=timezone.utc)
    day_obs = day_window_obs(obs, climate_day=day, decision_utc=decision)
    cov = assess_coverage(day_obs, climate_day=day, decision_utc=decision)
    assert cov.adequate is False
    assert cov.start_gap_exceeded is True


def test_first_seen_after_decision_excluded():
    day = date(2025, 7, 15)
    decision = datetime(2025, 7, 15, 16, 0, tzinfo=timezone.utc)
    base = _morning_through(decision - timedelta(hours=1))
    late = _obs(
        datetime(2025, 7, 15, 15, 30, tzinfo=timezone.utc),
        99.0,
        first_seen=datetime(2025, 7, 15, 17, 0, tzinfo=timezone.utc),  # after decision
    )
    day_obs = day_window_obs(base + [late], climate_day=day, decision_utc=decision)
    assert all(o.tmpf != 99.0 for o in day_obs)
    bundle = build_station_v2_features(base + [late], decision, climate_day=day)
    assert bundle is not None
    assert bundle.max_so_far < 99.0


def test_chicago_timezone_climate_day():
    # LST America/Chicago = UTC−6; America/New_York = UTC−5
    when = datetime(2026, 9, 16, 5, 30, tzinfo=timezone.utc)
    assert lst_climate_day(when, "America/Chicago") == date(2026, 9, 15)
    assert lst_climate_day(when, "America/New_York") == date(2026, 9, 16)


def test_wrong_station_cli_excluded(tmp_path: Path):
    store = FeedStore(path=tmp_path / "f.db")
    store.upsert_sample(
        feed="cli_mdw",
        source_key="wrong",
        payload={
            "climate_day": "2026-09-15",
            "issuance_utc": "2026-09-15T20:00:00+00:00",
            "max_temp_f": 88,
            "is_preliminary": False,
            "station_id": "ORD",  # wrong vs Midway
        },
        valid_utc="2026-09-15T20:00:00+00:00",
        first_seen_utc="2026-09-15T20:05:00+00:00",
    )
    sel = select_cli_for_decision(
        store,
        target_day=date(2026, 9, 15),
        decision_utc=datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc),
        station_id="MDW",
        feed="cli_mdw",
    )
    assert sel["applied"] is None
    assert any(e["exclude_reason"] == "wrong_station" for e in sel["excluded"])
    store.close()


def test_metar_adapter_preserves_first_seen(tmp_path: Path):
    store = FeedStore(path=tmp_path / "f.db")
    store.upsert_sample(
        feed="metar_KNYC",
        source_key="k1",
        payload={
            "station": "KNYC",
            "valid_utc": "2026-09-15T15:51:00+00:00",
            "tmpf": 72.5,
            "dwpf": 55.0,
            "sknt": 4.0,
            "drct": 180.0,
            "alti": 30.1,
            "p01i": None,
            "skyc1": "FEW",
            "receipt_time": "2026-09-15T15:52:00Z",
        },
        valid_utc="2026-09-15T15:51:00+00:00",
        first_seen_utc="2026-09-15T15:53:00+00:00",
    )
    hourly = metar_to_hourly(store, "metar_KNYC")
    assert len(hourly) == 1
    assert hourly[0].first_seen_utc == datetime(2026, 9, 15, 15, 53, tzinfo=timezone.utc)
    assert hourly[0].tmpf == 72.5
    store.close()


def test_cross_location_features_not_mixed():
    decision = datetime(2025, 7, 15, 16, 0, tzinfo=timezone.utc)
    day = date(2025, 7, 15)
    nyc = _morning_through(decision, station="KNYC")
    chi = [_obs(o.valid_utc, (o.tmpf or 0) + 20, station="KMDW") for o in nyc]
    b_nyc = build_station_v2_features(nyc, decision, climate_day=day, station_id="KNYC", lat=40.78, lon=-73.97)
    b_chi = build_station_v2_features(
        chi, decision, climate_day=day, tz_name="America/Chicago", station_id="KMDW", lat=41.79, lon=-87.75
    )
    assert b_nyc is not None and b_chi is not None
    assert b_nyc.max_so_far != b_chi.max_so_far
    assert b_nyc.provenance["station"] == "KNYC"
    assert b_chi.provenance["station"] == "KMDW"


def test_unified_predict_parity_eval_and_operating():
    """Same features/context → identical point + residual path."""
    from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import STATION_V2_FEATURES

    class _M:
        def predict(self, X):
            return np.array([2.0] * len(X))

    models = {"q10": _M(), "q50": _M(), "q90": _M()}
    feats = {n: 1.0 for n in STATION_V2_FEATURES}
    feats["tmpf"] = 70.0
    feats["max_so_far"] = 72.0
    residuals = [-1.0, 0.0, 1.0] * 20  # 60 >= MIN
    calib = {"residuals_by_hour": {"11": residuals, "all": residuals}}
    ctx = ForecastContext(
        location_id="nyc_central_park",
        series_ticker="HIGHNY",
        measurement="daily_max_temp_f",
        settlement_source_family="nws_cli",
        climate_day=date(2025, 7, 15),
        decision_time_utc=datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc),
        horizon="same_day",
        decision_hour_local=11,
        mode="HISTORICAL_REPLAY",
    )
    a = predict_station_v2(
        context=ctx,
        features=feats,
        max_so_far=72.0,
        coverage_adequate=True,
        decision_hour_local=11,
        model_blob={"models": models, "feature_names": STATION_V2_FEATURES, "model_version": "t"},
        calibration=calib,
        require_supported_hour=True,
    )
    b = predict_station_v2(
        context=ctx,
        features=dict(feats),
        max_so_far=72.0,
        coverage_adequate=True,
        decision_hour_local=11,
        model_blob={"models": models, "feature_names": STATION_V2_FEATURES, "model_version": "t"},
        calibration=calib,
        require_supported_hour=True,
    )
    assert a.ok and b.ok
    assert a.probabilities_available and b.probabilities_available
    assert a.point_median_f == b.point_median_f == 74.0  # 72 + 2, clamped
    assert a.distribution is not None and b.distribution is not None
    assert a.distribution.as_dict() == b.distribution.as_dict()


def test_paper_yes_then_no_separate_positions(tmp_path: Path):
    led = PaperLedger(path=tmp_path / "p.db", starting_cash="100.00")
    r1 = led.try_simulate_fill(
        client_order_id="y1",
        ticker="MKT-1",
        side="yes",
        qty=D("1"),
        price=D("0.40"),
        fees=D("0.01"),
        decision_reason="t",
        details={},
    )
    r2 = led.try_simulate_fill(
        client_order_id="n1",
        ticker="MKT-1",
        side="no",
        qty=D("1"),
        price=D("0.55"),
        fees=D("0.01"),
        decision_reason="t",
        details={},
    )
    assert r1["ok"] and r2["ok"]
    pos = led.positions()
    assert len(pos) == 2
    sides = {(p["ticker"], p["side"], p["qty"]) for p in pos}
    assert ("MKT-1", "yes", "1") in sides
    assert ("MKT-1", "no", "1") in sides
    # Not collapsed into two NO contracts
    assert not (len(pos) == 1 and pos[0]["side"] == "no" and pos[0]["qty"] == "2")
    led.close()


def test_paper_settlement_idempotent(tmp_path: Path):
    led = PaperLedger(path=tmp_path / "p.db", starting_cash="100.00")
    led.try_simulate_fill(
        client_order_id="y1",
        ticker="MKT-1",
        side="yes",
        qty=D("1"),
        price=D("0.40"),
        fees=D("0.02"),
        decision_reason="t",
        details={},
    )
    led.try_simulate_fill(
        client_order_id="n1",
        ticker="MKT-1",
        side="no",
        qty=D("1"),
        price=D("0.50"),
        fees=D("0.02"),
        decision_reason="t",
        details={},
    )
    cash_before_settle = led.cash()
    s1 = led.settle_market(ticker="MKT-1", result="yes")
    assert s1["settled"] is True
    cash_after = led.cash()
    # YES wins: +1.00 payout for yes; NO gets 0
    assert cash_after == cash_before_settle + D("1.00")
    assert led.positions() == []
    s2 = led.settle_market(ticker="MKT-1", result="yes")
    assert s2["ok"] is True
    # Second call: no open positions OR already_settled — cash unchanged
    assert led.cash() == cash_after
    led.close()


def test_paper_insufficient_cash_and_exposure(tmp_path: Path):
    led = PaperLedger(
        path=tmp_path / "p.db",
        starting_cash="1.00",
        max_total_exposure="0.50",
        max_per_market_exposure="0.50",
    )
    r = led.try_simulate_fill(
        client_order_id="x",
        ticker="M",
        side="yes",
        qty=D("1"),
        price=D("0.80"),
        fees=D("0.01"),
        decision_reason="t",
        details={},
    )
    assert r["ok"] is False
    assert r["reason"] in ("insufficient_paper_cash", "blocked_total_exposure")
    led.close()


def test_paper_concurrent_duplicate_protection(tmp_path: Path):
    led = PaperLedger(path=tmp_path / "p.db", starting_cash="50.00")

    def _fill():
        return led.try_simulate_fill(
            client_order_id="same-oid",
            ticker="M",
            side="yes",
            qty=D("1"),
            price=D("0.30"),
            fees=D("0.01"),
            decision_reason="t",
            details={},
        )

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: _fill(), range(8)))
    oks = [r for r in results if r.get("ok")]
    dups = [r for r in results if r.get("reason") == "duplicate_client_order_id"]
    assert len(oks) == 1
    assert len(dups) == 7
    led.close()


def test_paper_restart_persistence(tmp_path: Path):
    path = tmp_path / "p.db"
    led = PaperLedger(path=path, starting_cash="100.00")
    led.try_simulate_fill(
        client_order_id="persist",
        ticker="M",
        side="yes",
        qty=D("1"),
        price=D("0.25"),
        fees=D("0.01"),
        decision_reason="t",
        details={},
    )
    cash = led.cash()
    led.close()
    led2 = PaperLedger(path=path)
    assert led2.cash() == cash
    assert len(led2.positions()) == 1
    r = led2.try_simulate_fill(
        client_order_id="persist",
        ticker="M",
        side="yes",
        qty=D("1"),
        price=D("0.25"),
        fees=D("0.01"),
        decision_reason="t",
        details={},
    )
    assert r["reason"] == "duplicate_client_order_id"
    led2.close()


def test_registry_seed_and_isolation(tmp_path: Path):
    reg = LocationRegistry(path=tmp_path / "reg.db")
    for tick, base in VERIFIED_NWS_CLI_DAILY_MAX.items():
        reg.upsert_target(
            {
                **base,
                "series_ticker": tick,
                "measurement": "daily_max_temp_f",
                "settlement_source_family": "nws_cli",
                "unit": "F",
                "uses_lst_climate_day": True,
            }
        )
    ops = reg.operating_daily_max()
    assert any(t["location_id"] == "nyc_central_park" for t in ops)
    assert any(t["location_id"] == "chi_midway" for t in ops)
    # TWC must not be operating
    reg.upsert_target(
        {
            "location_id": "twc_x",
            "series_ticker": "KXHIGHNY",
            "measurement": "daily_max_temp_f",
            "settlement_source_family": "weather_company",
            "mapping_status": "discovered",
            "validation_status": "blocked",
        }
    )
    assert all(t["settlement_source_family"] == "nws_cli" for t in reg.operating_daily_max())
    reg.close()


def test_unsupported_location_fails_independently(tmp_path: Path):
    store = FeedStore(path=tmp_path / "f.db")
    twc = {
        "location_id": "twc_foo",
        "series_ticker": "KXFOO",
        "measurement": "daily_max_temp_f",
        "settlement_source_family": "weather_company",
        "mapping_status": "discovered",
        "validation_status": "blocked_unsupported_settlement_source",
        "metar_ids": [],
        "timezone": "UTC",
        "notes": "TWC — do not apply NWS pipeline",
    }
    chi = {
        **VERIFIED_NWS_CLI_DAILY_MAX["HIGHCHI"],
        "series_ticker": "HIGHCHI",
        "measurement": "daily_max_temp_f",
        "settlement_source_family": "nws_cli",
    }
    r1 = process_location(twc, store, collect=False, do_paper=False)
    r2 = process_location(chi, store, collect=False, do_paper=False)
    assert r1["status"] == "blocked_unsupported_settlement_source"
    assert r2["status"] in (
        "blocked_no_usable_metar",
        "blocked_no_location_model",
        "insufficient_data",
        "unsupported_decision_time",
        "blocked_no_metar",
    )
    # Independent: TWC block does not prevent CHI from producing its own status
    assert r2["location_id"] == "chi_midway"
    store.close()


def test_no_live_order_in_multi_modules():
    import inspect
    from kalshi_bot.models.weather.obs_engine.multi import pipeline, predict
    from kalshi_bot.models.weather.obs_engine.feeds import infer_paper, paper_sim

    for mod in (pipeline, predict, infer_paper, paper_sim):
        src = inspect.getsource(mod)
        assert "create_order" not in src
        assert "place_order" not in src


def test_fill_to_settlement_fixture_integration(tmp_path: Path):
    """Automated paper fill→settlement with clearly labeled fixtures (not prospective perf)."""
    led = PaperLedger(path=tmp_path / "fixture_paper.db", starting_cash="100.00")
    # Fixture market — not a live Kalshi fill
    fill = led.try_simulate_fill(
        client_order_id="fixture-yes-1",
        ticker="FIXTURE-HIGHNY-T70",
        side="yes",
        qty=D("2"),
        price=D("0.35"),
        fees=D("0.04"),
        decision_reason="fixture_integration_test",
        details={"sim_label": "fixture_not_prospective_performance", "live_blocked": True},
        location_id="nyc_central_park",
        series_ticker="HIGHNY",
        quote_ts_utc="2026-09-15T15:00:00+00:00",
        market_ts_utc="2026-09-15T15:00:00+00:00",
    )
    assert fill["ok"] and fill["live_order_submitted"] is False
    settle = led.settle_market(
        ticker="FIXTURE-HIGHNY-T70",
        result="yes",
        details={"fixture": True, "label": "simulated_settlement_not_official"},
    )
    assert settle["settled"] is True
    # cost 2*0.35+0.04=0.74; payout 2.00; pnl = 2 - 0.70 - 0.04 = 1.26
    applied = settle["applied"][0]
    assert D(applied["payout"]) == D("2")
    assert D(applied["pnl"]) == D("1.26")
    snap = led.snapshot()
    assert snap["live_order_submitted"] is False
    assert snap["positions"] == []
    led.close()
