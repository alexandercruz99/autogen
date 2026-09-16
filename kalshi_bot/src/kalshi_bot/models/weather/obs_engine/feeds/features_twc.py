"""Operating features for Weather Company (TWC) daily-max settlement markets.

Uses:
- AviationWeather METAR for rich station_v2 predictors (same physical ICAO station).
- TWC portal METAR max-so-far as settlement-aligned progressive constraint.
- TWC climate report (not NWS CLI) as whole-°F floor when published.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kalshi_bot.models.weather.obs_engine.feeds.climate_day import lst_climate_day
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import FEATURE_SCHEMA_VERSION, STATION_V2_FEATURES
from kalshi_bot.models.weather.obs_engine.feeds.features_live import build_operating_features
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.feeds.twc_kalshi import (
    select_twc_climate_for_decision,
    twc_metar_max_so_far,
)


def build_twc_operating_features(
    store: FeedStore,
    *,
    now: datetime | None = None,
    metar_id: str,
    twc_cli_id: str,
    tz_name: str,
    lat: float | None = None,
    lon: float | None = None,
    location_id: str | None = None,
    include_satrad: bool = False,
) -> dict[str, Any]:
    """Build features for a TWC-settled daily-max market at a verified ICAO/CLI station."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Rich predictors from public ASOS/METAR (same ICAO as TWC portal station).
    base = build_operating_features(
        store,
        now=now,
        metar_id=metar_id,
        tz_name=tz_name,
        lat=lat,
        lon=lon,
        cli_location_id=twc_cli_id,  # NWS CLI feed may be empty; ignored for floor below
        location_id=location_id,
        include_satrad=include_satrad,
    )

    climate_day = lst_climate_day(now, tz_name)
    # Override climate day from base if present
    try:
        from datetime import date as date_cls

        climate_day = date_cls.fromisoformat(base["climate_day"])
    except Exception:
        pass

    twc_prog = twc_metar_max_so_far(
        store,
        icao=metar_id,
        climate_day=climate_day,
        tz_name=tz_name,
        decision_utc=now,
    )
    twc_cli = select_twc_climate_for_decision(
        store,
        target_day=climate_day,
        decision_utc=now,
        cli_id=twc_cli_id,
    )

    # Settlement-aligned progressive max: prefer TWC portal METAR when present.
    av_max = base.get("max_so_far")
    twc_max = twc_prog.get("max_so_far")
    max_so_far = twc_max if twc_max is not None else av_max
    if av_max is not None and twc_max is not None:
        max_so_far = max(float(av_max), float(twc_max))

    features = dict(base.get("features") or {})
    if max_so_far is not None and "max_so_far" in features:
        features["max_so_far"] = float(max_so_far)

    provenance = dict(base.get("provenance") or {})
    provenance["settlement_source_family"] = "weather_company"
    provenance["twc_metar_progressive"] = twc_prog
    provenance["twc_climate_selection"] = {
        "applied": twc_cli.get("applied"),
        "excluded": twc_cli.get("excluded"),
        "n_candidates": twc_cli.get("n_candidates"),
        "feed": twc_cli.get("feed"),
    }
    # Strip NWS CLI floor — TWC markets must not use NWS CLI constraint.
    provenance["nws_cli_ignored_for_twc"] = provenance.get("cli_selection")
    provenance["cli"] = None
    provenance["cli_selection"] = None

    attribution = dict(base.get("attribution") or {})
    feeds_collected = list(attribution.get("feeds_collected") or [])
    feeds_quality_ok = list(attribution.get("feeds_quality_ok") or [])
    if twc_prog.get("usable"):
        feeds_collected.append(twc_prog["feed"])
        feeds_quality_ok.append(twc_prog["feed"])
    if twc_cli.get("applied"):
        feeds_collected.append(twc_cli["feed"])
        feeds_quality_ok.append(twc_cli["feed"])
    constraints = []
    if twc_cli.get("applied"):
        constraints.append("twc_climate_prelim_or_official_floor")
    if twc_max is not None:
        constraints.append("twc_metar_max_so_far")
    attribution.update(
        {
            "feeds_collected": list(dict.fromkeys(feeds_collected)),
            "feeds_quality_ok": list(dict.fromkeys(feeds_quality_ok)),
            "settlement_constraints_applied": constraints,
            "features_consumed_by_model": list(STATION_V2_FEATURES),
            "settlement_source_family": "weather_company",
        }
    )
    provenance["attribution"] = attribution

    coverage = base.get("coverage")
    # If aviationweather coverage failed but TWC metar has temps, mark partial usable.
    if (not coverage or not (coverage or {}).get("adequate")) and twc_prog.get("usable"):
        coverage = {
            "adequate": twc_prog.get("n_obs", 0) >= 4,
            "max_so_far_status": "twc_metar_only",
            "notes": [
                "AviationWeather coverage inadequate; using TWC portal METAR progressive temps",
                f"twc_n_obs={twc_prog.get('n_obs')}",
            ],
            "n_temp_obs": twc_prog.get("n_obs"),
        }

    return {
        "feature_names": list(STATION_V2_FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
        "missing": base.get("missing") or {},
        "provenance": provenance,
        "max_so_far": float(max_so_far) if max_so_far is not None else None,
        "coverage": coverage,
        "cli_applied": twc_cli.get("applied"),  # TWC climate floor (same key for predict truncate)
        "attribution": attribution,
        "climate_day": climate_day.isoformat(),
        "decision_time_utc": now.isoformat(),
        "location_id": location_id or metar_id,
        "metar_id": metar_id.upper(),
        "tz_name": tz_name,
        "cli_location_id": twc_cli_id.upper(),
        "twc_metar": twc_prog,
        "settlement_source_family": "weather_company",
    }
