"""Regression tests for obs-driven engine audit cases and distribution floors."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from kalshi_bot.config import WeatherModelConfig
from kalshi_bot.models.weather.distribution import (
    from_empirical_residuals,
    from_normal,
    truncate_below,
)
from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs
from kalshi_bot.models.weather.obs_engine.features import build_features_at
from kalshi_bot.models.weather.obs_engine.forecaster import ObsDrivenNycForecaster
from kalshi_bot.models.weather.settlement_rules import TempInterval


def test_truncate_below_zeros_band_under_observed_72():
    """Audit case: CLI/METAR already at 72 → P(70–71) must be ~0, not ~50%."""
    dist = from_normal(71.0, 2.0, support_low=65, support_high=80)
    floored = truncate_below(dist, 72.0, reason="observed 72")
    iv = TempInterval("range_inclusive", 70.0, 71.0, "t")
    assert floored.p_interval(iv) == Decimal("0")
    assert floored.details["same_day_floor_int"] == 72
    assert min(floored.temps_f) >= 72


def test_variance_floor_prevents_collapse_on_tiny_residual_n():
    """Audit case: ~6–7 residuals must not yield ~99.7% on a single bin."""
    # Tiny sample centered so point=83 maps residuals to near-zero
    dist = from_empirical_residuals(83.0, [0.0, 0.0, 0.1, -0.1, 0.0, 0.2, -0.2])
    assert "var_floor" in dist.method or dist.details.get("variance_floor_std") is not None
    iv = TempInterval("gt", 82.0, None, "t")  # ≥83
    p = dist.p_interval(iv)
    # With variance floor, mass spreads — must not be near-certainty
    assert p < Decimal("0.97")
    assert p > Decimal("0.3")


def test_features_exclude_future_observations():
    """Timing leakage: obs after decision_utc must not enter features."""
    day = date(2025, 7, 15)
    obs = [
        HourlyObs(
            valid_utc=datetime(2025, 7, 15, 14, 51, tzinfo=timezone.utc),
            tmpf=78.0,
            dwpf=60.0,
            sknt=5.0,
            drct=180.0,
            alti=30.0,
            p01i=0.0,
            skyc1="CLR",
            station="NYC",
        ),
        HourlyObs(
            valid_utc=datetime(2025, 7, 15, 18, 51, tzinfo=timezone.utc),
            tmpf=90.0,  # future relative to decision
            dwpf=62.0,
            sknt=5.0,
            drct=180.0,
            alti=30.0,
            p01i=0.0,
            skyc1="CLR",
            station="NYC",
        ),
    ]
    decision = datetime(2025, 7, 15, 16, 0, tzinfo=timezone.utc)
    feats = build_features_at(obs, decision, climate_day=day)
    assert feats is not None
    assert feats.max_so_far == 78.0
    assert feats.provenance["n_obs_used"] == 1


def test_nyc_target_is_central_park_not_lga():
    assert NYC_TARGET.ghcnd_id == "USW00094728"
    assert NYC_TARGET.metar_id == "KNYC"
    assert abs(NYC_TARGET.lat - 40.77898) < 0.01
    assert "Central Park" in NYC_TARGET.display_name


def test_obs_forecaster_supports_nyc_only_and_not_live_eligible():
    cfg = WeatherModelConfig(obs_engine_enabled=True, obs_engine_live_eligible=False)
    m = ObsDrivenNycForecaster(cfg)
    assert m.supports({"ticker": "KXHIGHNY-26SEP15-T70"}, "Climate and Weather")
    assert not m.supports({"ticker": "KXHIGHLAX-26SEP15-T82"}, "Climate and Weather")
    assert not m.supports({"ticker": "KXHIGHNY-26SEP15-T70-HOUR14"}, "Climate and Weather")
    # Hard live default
    assert getattr(cfg, "obs_engine_live_eligible") is False
    m.close()


def test_obs_forecaster_details_block_live():
    cfg = WeatherModelConfig(obs_engine_enabled=True, obs_engine_live_eligible=True)
    m = ObsDrivenNycForecaster(cfg)
    # Even if config flag is true, predict path keeps live_eligible False until promotion
    # (implementation hard-codes False; config alone is insufficient).
    pred = m.predict(
        {
            "ticker": "KXHIGHNY-26SEP15-T70",
            "strike_type": "greater",
            "floor_strike": 70,
            "cap_strike": None,
            "title": "Will the high temp in NYC be >70°",
        },
        "Climate and Weather",
    )
    # May be unsupported if no live METAR — still must expose live flag false when supported
    if pred.supported:
        assert pred.details.get("model_live_eligible") is False
        assert "RESEARCH" in pred.validation_evidence
    else:
        assert pred.details.get("model_live_eligible") is False
    m.close()
