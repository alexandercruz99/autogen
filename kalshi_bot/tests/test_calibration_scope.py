"""Calibration scope: no silent NYC fallback for other locations."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import STATION_V2_FEATURES
from kalshi_bot.models.weather.obs_engine.multi.context import ForecastContext
from kalshi_bot.models.weather.obs_engine.multi.predict import (
    DEFAULT_CALIB_PATH,
    predict_station_v2,
)


class _ConstModel:
    def __init__(self, value: float):
        self._value = value

    def predict(self, X):
        return np.array([self._value] * len(X))


def _mock_model_blob():
    models = {"q10": _ConstModel(1.0), "q50": _ConstModel(2.0), "q90": _ConstModel(3.0)}
    return {"models": models, "feature_names": STATION_V2_FEATURES, "model_version": "test"}


def _fake_context(location_id: str = "fake_city") -> ForecastContext:
    return ForecastContext(
        location_id=location_id,
        series_ticker="HIGHFAKE",
        measurement="daily_max_temp_f",
        settlement_source_family="nws_cli",
        climate_day=date(2025, 7, 15),
        decision_time_utc=datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc),
        horizon="same_day",
        decision_hour_local=11,
        mode="HISTORICAL_REPLAY",
    )


def _predict(*, calib_path=None, calibration=None, require_location_id=True, location_id="fake_city"):
    feats = {n: 1.0 for n in STATION_V2_FEATURES}
    return predict_station_v2(
        context=_fake_context(location_id),
        features=feats,
        max_so_far=72.0,
        coverage_adequate=True,
        decision_hour_local=11,
        model_blob=_mock_model_blob(),
        calib_path=calib_path,
        calibration=calibration,
        require_location_id=require_location_id,
    )


def test_missing_calib_path_fails_closed_without_nyc_fallback():
    pred = _predict(calib_path=None)
    assert pred.status == "probabilities_unavailable"
    assert pred.probabilities_available is False
    assert pred.reason == "missing_location_calibration"
    assert pred.point_median_f == 74.0  # point forecast still computed


def test_missing_calib_file_fails_closed(tmp_path: Path):
    missing = tmp_path / "no_such_calibration.json"
    pred = _predict(calib_path=missing)
    assert pred.status == "probabilities_unavailable"
    assert pred.reason == "missing_location_calibration"
    assert pred.probabilities_available is False


def test_fake_location_does_not_use_nyc_calib_when_path_omitted():
    """Even when NYC calibration exists on disk, calib_path=None must not load it."""
    assert DEFAULT_CALIB_PATH.exists(), "fixture expects NYC calib artifact in repo"
    pred = _predict(calib_path=None, location_id="chi_midway")
    assert pred.probabilities_available is False
    assert pred.reason == "missing_location_calibration"


def test_nyc_calib_rejected_for_mismatched_location_when_required():
    nyc_calib = json.loads(DEFAULT_CALIB_PATH.read_text())
    nyc_calib.setdefault("meta", {})["location_id"] = "nyc_central_park"
    pred = _predict(calibration=nyc_calib, location_id="chi_midway", require_location_id=True)
    assert pred.status == "probabilities_unavailable"
    assert pred.reason.startswith("calibration_location_mismatch:")
    assert pred.probabilities_available is False


def test_nyc_calib_allowed_when_path_explicit_and_location_matches():
    pred = _predict(calib_path=DEFAULT_CALIB_PATH, location_id="nyc_central_park")
    assert pred.probabilities_available is True
    assert pred.status == "ok"


def test_resolve_calibration_path_sibling_without_calib_fails_closed(tmp_path: Path, monkeypatch):
    from kalshi_bot.models.weather.obs_engine.multi import pipeline as pl

    art = tmp_path / "artifacts"
    monkeypatch.setattr(pl, "ARTIFACT_ROOT", art)

    sibling = "chi_midway"
    measurement = "daily_max_temp_f"
    sib_dir = art / "artifacts" / f"{sibling}__{measurement}"
    sib_dir.mkdir(parents=True)
    model_path = sib_dir / "station_corrected_v2.joblib"
    model_path.write_bytes(b"stub")

    calib_path, require_loc = pl._resolve_calibration_path(
        "twc_chi_midway",
        measurement,
        target={"same_station_model_location_id": sibling},
        model_path=model_path,
        model_origin=f"same_icao_transfer:{sibling}",
    )
    assert calib_path == sib_dir / "station_corrected_v2_calibration.json"
    assert not calib_path.exists()
    assert require_loc is False

    pred = predict_station_v2(
        context=_fake_context("twc_chi_midway"),
        features={n: 1.0 for n in STATION_V2_FEATURES},
        max_so_far=70.0,
        coverage_adequate=True,
        decision_hour_local=11,
        model_blob=_mock_model_blob(),
        calib_path=calib_path,
        require_location_id=require_loc,
    )
    assert pred.probabilities_available is False
    assert pred.reason == "missing_location_calibration"
