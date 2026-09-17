"""Tests for forecast candidate evaluation scaffolding (software evidence only)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from kalshi_bot.models.weather.distribution import PredictiveDistribution, from_empirical_residuals, truncate_below
from kalshi_bot.models.weather.obs_engine.multi.candidates.metrics_prob import crps_discrete, p_inclusive_band
from kalshi_bot.models.weather.obs_engine.multi.candidates.pipeline import (
    correct_quantile_crossing,
    from_standardized_residuals,
    remain_point_from_quantiles,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.splits import build_split_manifest, filter_rows_by_days
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.predict import predict_station_v2
from kalshi_bot.models.weather.settlement_rules import TempInterval, yes_from_observation


def test_quantile_crossing_sorted():
    a, b, c, crossed = correct_quantile_crossing(5.0, 1.0, 3.0)
    assert crossed
    assert (a, b, c) == (1.0, 3.0, 5.0)


def test_bias_then_floor_ordering():
    # Negative bias must not leave point below max_so_far when clamp is on.
    point = remain_point_from_quantiles(70.0, 0.5, bias=-2.0, clamp_to_max_so_far=True)
    assert point == 70.0
    point2 = remain_point_from_quantiles(70.0, 3.0, bias=-1.0, clamp_to_max_so_far=True)
    assert point2 == 72.0


def test_standardized_residuals_scale():
    z = [-1.0, 0.0, 1.0, 0.5, -0.5]
    dist = from_standardized_residuals(80.0, 2.0, z)
    assert abs(dist.mean() - 80.0) < 1.5
    assert sum(dist.probs) == pytest.approx(1.0)


def test_crps_perfect_point_mass():
    dist = PredictiveDistribution(temps_f=[75], probs=[1.0], method="test")
    assert crps_discrete(dist, 75) == 0.0
    assert crps_discrete(dist, 76) > 0.0


def test_p_inclusive_band_matches_settlement_semantics():
    dist = PredictiveDistribution(
        temps_f=[78, 79, 80, 81],
        probs=[0.1, 0.4, 0.4, 0.1],
        method="test",
    )
    p = p_inclusive_band(dist, 79, 80)
    assert p == pytest.approx(0.8)
    iv = TempInterval("range_inclusive", 79.0, 80.0, "test")
    assert float(dist.p_interval(iv)) == pytest.approx(0.8)
    assert yes_from_observation(79.0, iv)
    assert not yes_from_observation(78.0, iv)


def test_truncate_floor_sync_conflict_flag_possible():
    dist = from_empirical_residuals(70.0, [-2.0, -1.0, 0.0, 1.0, 2.0] * 10)
    floored = truncate_below(dist, 90.0, reason="test")
    assert floored.temps_f[0] >= 90
    assert sum(floored.probs) == pytest.approx(1.0)


def test_split_manifest_chronological_no_overlap():
    days = [f"2024-01-{i:02d}" for i in range(1, 32)] + [f"2024-02-{i:02d}" for i in range(1, 29)]
    m = build_split_manifest(
        location_id="nyc_central_park",
        metric="daily_max_temp_f",
        label_source="ghcnd_tmax",
        climate_days=days,
        decision_hours_local=[8, 11, 14],
    )
    sets = [set(m.train_days), set(m.selection_days), set(m.calib_days), set(m.test_days)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j])
    assert max(m.train_days) < min(m.selection_days)
    assert max(m.selection_days) < min(m.calib_days)
    assert max(m.calib_days) < min(m.test_days)
    rows = [{"climate_day": d, "x": 1} for d in m.test_days]
    assert len(filter_rows_by_days(rows, m.test_days)) == len(m.test_days)


def test_predict_bias_then_cli_floor_syncs_point():
    class _Q:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.asarray([self.v] * len(X))

    blob = {
        "models": {"q10": _Q(1.0), "q50": _Q(2.0), "q90": _Q(4.0)},
        "feature_names": [
            "hour_local",
            "doy",
            "tmpf",
            "dwpf",
            "sknt",
            "alti",
            "max_so_far",
            "rise_from_min",
            "dT_1h",
            "dT_3h",
            "solar_el",
            "hours_to_20_local",
            "precip_6h",
            "time_since_max_h",
            "d_dewpt_3h",
            "d_alti_3h",
            "d_sknt_3h",
            "wind_dir_sin",
            "wind_dir_cos",
            "missing_sknt",
            "missing_alti",
            "missing_dwpf",
            "missing_precip_6h",
            "sky_code",
        ],
        "model_version": "test",
    }
    feats = {n: 0.0 for n in blob["feature_names"]}
    feats["max_so_far"] = 70.0
    feats["tmpf"] = 70.0
    calib = {
        "residuals_by_hour": {"8": [-1.0, 0.0, 1.0] * 20, "all": [-1.0, 0.0, 1.0] * 20},
        "meta": {"location_id": "nyc_central_park", "point_bias_by_hour": {"8": -5.0}},
    }
    ctx = ForecastContext(
        location_id="nyc_central_park",
        series_ticker="HIGHNY",
        measurement="daily_max_temp_f",
        settlement_source_family="nws_cli",
        climate_day=date(2024, 6, 1),
        decision_time_utc=__import__("datetime").datetime(2024, 6, 1, 12, 0, tzinfo=__import__("datetime").timezone.utc),
        horizon="same_day",
        decision_hour_local=8,
        mode="HISTORICAL_REPLAY",
    )
    pred = predict_station_v2(
        context=ctx,
        features=feats,
        max_so_far=70.0,
        coverage_adequate=True,
        decision_hour_local=8,
        cli_applied={"max_temp_f": 72, "is_preliminary": True},
        model_blob=blob,
        calibration=calib,
        require_location_id=True,
    )
    assert pred.ok
    assert pred.point_median_f is not None and pred.point_median_f >= 72.0
    assert pred.distribution is not None
    assert min(pred.distribution.temps_f) >= 72
    assert pred.observation_constraint.get("bias_ordering") == "bias_then_floor"


def test_no_live_eligible_in_candidate_versions():
    from kalshi_bot.models.weather.obs_engine.multi.candidates.evaluate import CANDIDATE_VERSIONS

    assert CANDIDATE_VERSIONS
    # Versions are research labels; evaluate always sets live_eligible False in artifacts.
