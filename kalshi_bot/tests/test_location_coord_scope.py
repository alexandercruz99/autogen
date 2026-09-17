"""Fail-closed location coords — no silent NYC fallback for other stations."""

from __future__ import annotations

import pytest

from kalshi_bot.models.weather.obs_engine.feeds.features_live import build_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore


def test_features_live_refuses_nyc_fallback_for_other_location(tmp_path):
    store = FeedStore(str(tmp_path / "f.db"))
    try:
        with pytest.raises(ValueError, match="refusing NYC coordinate fallback"):
            build_operating_features(
                store,
                location_id="mia_cli",
                metar_id=None,
                lat=None,
                lon=None,
            )
    finally:
        store.close()


def test_features_live_allows_explicit_non_nyc_coords(tmp_path):
    store = FeedStore(str(tmp_path / "f2.db"))
    try:
        # May return incomplete features without samples, but must not raise on coords.
        bundle = build_operating_features(
            store,
            location_id="mia_cli",
            metar_id="KMIA",
            lat=25.7959,
            lon=-80.2870,
            tz_name="America/New_York",
            cli_location_id="MIA",
        )
        assert isinstance(bundle, dict)
    finally:
        store.close()
