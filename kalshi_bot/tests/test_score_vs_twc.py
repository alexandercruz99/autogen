"""Production median vs official TWC settlement highs."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from kalshi_bot.models.weather.obs_engine.multi.score_vs_twc import (
    ensure_nyc_multi_artifacts,
    fetch_twc_actuals,
    score_location_vs_twc,
)


def test_ensure_nyc_artifacts(tmp_path, monkeypatch):
    data = tmp_path / "obs"
    feeds = data / "feeds" / "models"
    feeds.mkdir(parents=True)
    (feeds / "station_corrected_v2.joblib").write_bytes(b"x")
    (feeds / "station_corrected_v2_calibration.json").write_text("{}")
    (feeds / "station_corrected_v2_report.json").write_text("{}")
    out = ensure_nyc_multi_artifacts(data)
    assert out["station_corrected_v2.joblib"] == "copied"
    dest = data / "multi" / "artifacts" / "nyc_central_park__daily_max_temp_f"
    assert (dest / "station_corrected_v2_calibration.json").exists()


def test_fetch_twc_actuals_structure():
    day = date(2026, 9, 15)
    actuals = fetch_twc_actuals([day])
    assert day.isoformat() in actuals
    nyc = actuals[day.isoformat()].get("NYC")
    assert nyc is not None
    assert nyc.get("status") == "official"
    assert nyc.get("max_temp_f") == 72
