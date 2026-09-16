"""Tests for research pipeline gates and feature-set hygiene."""

from __future__ import annotations

from datetime import date, datetime, timezone

from kalshi_bot.models.weather.obs_engine.data import HourlyObs
from kalshi_bot.models.weather.obs_engine.research.features_v2 import (
    build_baseline,
    build_local_v2,
    _sky_code,
)


def test_sky_code_missing_not_clear():
    assert _sky_code(None) == -1.0
    assert _sky_code("") == -1.0
    assert _sky_code("CLR") == 0.0


def test_local_v2_adds_missing_indicators():
    obs = [
        HourlyObs(
            valid_utc=datetime(2025, 7, 15, 12, 51, tzinfo=timezone.utc),
            tmpf=80.0,
            dwpf=None,
            sknt=None,
            drct=None,
            alti=None,
            p01i=0.0,
            skyc1=None,
            station="NYC",
        ),
        HourlyObs(
            valid_utc=datetime(2025, 7, 15, 14, 51, tzinfo=timezone.utc),
            tmpf=84.0,
            dwpf=None,
            sknt=None,
            drct=None,
            alti=None,
            p01i=0.0,
            skyc1=None,
            station="NYC",
        ),
    ]
    decision = datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc)
    base = build_baseline(obs, decision, climate_day=date(2025, 7, 15))
    loc = build_local_v2(obs, decision, climate_day=date(2025, 7, 15))
    assert base is not None and loc is not None
    assert len(loc.values) > len(base.values)
    # missing_sknt, missing_alti, missing_dwpf should be 1
    assert loc.values[-4] == 1.0  # missing_sknt near end before sky
    assert loc.values[-3] == 1.0
    assert loc.values[-2] == 1.0
    assert loc.values[-1] == -1.0  # sky missing


def test_negative_remain_not_silently_dropped_in_audit_logic():
    # Remain can be negative; audit reports rather than clipping in feature build
    obs = [
        HourlyObs(
            valid_utc=datetime(2025, 7, 15, 14, 51, tzinfo=timezone.utc),
            tmpf=90.0,
            dwpf=60.0,
            sknt=5.0,
            drct=180.0,
            alti=30.0,
            p01i=0.0,
            skyc1="CLR",
            station="NYC",
        )
    ]
    decision = datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc)
    feats = build_baseline(obs, decision, climate_day=date(2025, 7, 15))
    assert feats is not None
    label = 88  # below max_so_far
    remain = label - feats.max_so_far
    assert remain < 0
