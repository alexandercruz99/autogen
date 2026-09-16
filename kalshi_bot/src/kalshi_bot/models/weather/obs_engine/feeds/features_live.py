"""Live operating features via shared station_v2 schema + feed attribution.

Location-parameterized: pass metar_id / tz / lat / lon / cli_location_id.
Defaults preserve NYC Central Park behavior for the existing worker.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import lst_climate_day
from kalshi_bot.models.weather.obs_engine.feeds.cli_match import select_cli_for_decision
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    STATION_V2_FEATURES,
    build_station_v2_features,
)
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

SATRAD_EXTRA = [
    "goes_cloud_frac_bcm",
    "goes_cloudyish_acm",
    "goes_available",
    "radar_precip_frac",
    "radar_mean_level",
    "radar_available",
]


def _rows_for_feed(store: FeedStore, feed: str) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        "SELECT payload_json, valid_utc, first_seen_utc FROM feed_samples WHERE feed=? ORDER BY valid_utc ASC",
        (feed,),
    ).fetchall()
    return [dict(r) for r in rows]


def metar_to_hourly(store: FeedStore, feed: str = "metar_KNYC") -> list[HourlyObs]:
    out: list[HourlyObs] = []
    default_station = feed.replace("metar_", "") if feed.startswith("metar_") else "KNYC"
    for r in _rows_for_feed(store, feed):
        p = json.loads(r["payload_json"])
        if not p.get("valid_utc"):
            continue
        first_seen = None
        if r.get("first_seen_utc"):
            try:
                first_seen = datetime.fromisoformat(r["first_seen_utc"])
            except Exception:
                first_seen = None
        receipt = None
        if p.get("receipt_time"):
            try:
                receipt = datetime.fromisoformat(str(p["receipt_time"]).replace("Z", "+00:00"))
            except Exception:
                receipt = None
        out.append(
            HourlyObs(
                valid_utc=datetime.fromisoformat(p["valid_utc"]),
                tmpf=p.get("tmpf"),
                dwpf=p.get("dwpf"),
                sknt=p.get("sknt"),
                drct=p.get("drct"),
                alti=p.get("alti"),
                p01i=p.get("p01i"),
                skyc1=p.get("skyc1"),
                station=p.get("station") or default_station,
                source="feed_store",
                first_seen_utc=first_seen,
                receipt_utc=receipt,
            )
        )
    return out


def _latest_payload(store: FeedStore, feed: str) -> dict[str, Any] | None:
    row = store.latest_sample(feed)
    if not row:
        return None
    return json.loads(row["payload_json"])


def build_operating_features(
    store: FeedStore,
    *,
    now: datetime | None = None,
    target_day=None,
    metar_id: str | None = None,
    tz_name: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    cli_location_id: str | None = None,
    location_id: str | None = None,
    goes_feed: str = "goes19_acmc",
    radar_feed: str = "nexrad_okx_n0b",
    include_satrad: bool = True,
) -> dict[str, Any]:
    """Build station_v2 features for inference; sat/radar collected separately for attribution."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    metar = (metar_id or NYC_TARGET.metar_id).upper()
    tz = tz_name or NYC_TARGET.timezone
    use_lat = lat if lat is not None else NYC_TARGET.lat
    use_lon = lon if lon is not None else NYC_TARGET.lon
    cli_id = (cli_location_id or NYC_TARGET.cli_location_id).upper()
    climate_day = target_day or lst_climate_day(now, tz)
    metar_feed = f"metar_{metar}"

    feeds_collected: list[str] = []
    feeds_quality_ok: list[str] = []
    features_consumed: list[str] = []
    missing: dict[str, bool] = {}

    hourly = metar_to_hourly(store, metar_feed)
    if hourly:
        feeds_collected.append(metar_feed)

    # Availability: live rows carry first_seen; enforce at day_window via schema.
    has_first_seen = any(o.first_seen_utc is not None for o in hourly)
    availability = (
        "feed_store_first_seen_enforced"
        if has_first_seen
        else "valid_utc_as_availability_archive_assumption"
    )

    bundle = build_station_v2_features(
        hourly,
        now,
        climate_day=climate_day,
        tz_name=tz,
        lat=use_lat,
        lon=use_lon,
        station_id=metar,
        availability_assumption=availability,
    )

    features: dict[str, float | None] = {k: None for k in STATION_V2_FEATURES + SATRAD_EXTRA}
    provenance: dict[str, Any] = {
        "built_at_utc": now.isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "climate_day_basis": f"LST_{tz}",
        "target_climate_day": climate_day.isoformat(),
        "location_id": location_id or metar,
        "metar_id": metar,
        "tz_name": tz,
        "availability_assumption": availability,
    }

    coverage = None
    max_so_far = None
    if bundle is None:
        missing["station_obs"] = True
        missing["essential_coverage"] = True
        provenance["coverage"] = {
            "adequate": False,
            "max_so_far_status": "insufficient",
            "notes": [f"No {metar} temperature observations in LST climate-day window"],
        }
    else:
        feeds_quality_ok.append(metar_feed)
        for name, val in bundle.feature_map.items():
            features[name] = val
        coverage = bundle.coverage.as_dict()
        provenance["coverage"] = coverage
        provenance["local_v2"] = bundle.provenance
        max_so_far = bundle.max_so_far
        provenance["max_so_far"] = max_so_far
        provenance["max_so_far_status"] = bundle.coverage.max_so_far_status
        if not bundle.coverage.adequate:
            missing["essential_coverage"] = True

    if include_satrad:
        goes = _latest_payload(store, goes_feed)
        if goes:
            feeds_collected.append(goes_feed)
        if goes and goes.get("features"):
            g = goes["features"]
            features["goes_cloud_frac_bcm"] = g.get("cloud_frac_bcm")
            features["goes_cloudyish_acm"] = g.get("cloudy_or_probably_frac_acm")
            features["goes_available"] = 1.0
            provenance["goes"] = g.get("provenance")
            try:
                age_h = (now - datetime.fromisoformat(g["valid_utc"])).total_seconds() / 3600.0
                provenance["goes_age_hours"] = age_h
                if age_h > 3:
                    missing["goes_stale"] = True
                else:
                    feeds_quality_ok.append(goes_feed)
            except Exception:
                feeds_quality_ok.append(goes_feed)
        else:
            features["goes_cloud_frac_bcm"] = None
            features["goes_cloudyish_acm"] = None
            features["goes_available"] = 0.0
            missing["goes"] = True

        radar = _latest_payload(store, radar_feed)
        if radar:
            feeds_collected.append(radar_feed)
        if (
            radar
            and radar.get("features")
            and not radar.get("features", {}).get("missing_scan")
            and not radar.get("features", {}).get("geometry_miss")
            and radar.get("features", {}).get("precip_gate_frac") is not None
        ):
            r = radar["features"]
            features["radar_precip_frac"] = r.get("precip_gate_frac")
            features["radar_mean_level"] = r.get("mean_level_if_any")
            features["radar_available"] = 1.0
            provenance["radar"] = r.get("provenance")
            try:
                age_h = (now - datetime.fromisoformat(r["valid_utc"])).total_seconds() / 3600.0
                provenance["radar_age_hours"] = age_h
                if age_h > 2:
                    missing["radar_stale"] = True
                else:
                    feeds_quality_ok.append(radar_feed)
            except Exception:
                feeds_quality_ok.append(radar_feed)
        else:
            features["radar_precip_frac"] = None
            features["radar_mean_level"] = None
            features["radar_available"] = 0.0
            missing["radar"] = True
            provenance["radar_note"] = "Unavailable/missing scan — NOT interpreted as no precipitation"

    cli_feed = "cli_nyc" if cli_id == NYC_TARGET.cli_location_id else f"cli_{cli_id.lower()}"
    cli_sel = select_cli_for_decision(
        store, target_day=climate_day, decision_utc=now, station_id=cli_id, feed=cli_feed
    )
    if any(True for _ in store._conn.execute("SELECT 1 FROM feed_samples WHERE feed=? LIMIT 1", (cli_feed,))):
        feeds_collected.append(cli_feed)
    provenance["cli_selection"] = {
        "applied": cli_sel.get("applied"),
        "excluded": cli_sel.get("excluded"),
        "n_candidates": cli_sel.get("n_candidates"),
        "feed": cli_feed,
        "station_id": cli_id,
    }
    if cli_sel.get("applied"):
        feeds_quality_ok.append(cli_feed)
        provenance["cli"] = cli_sel["applied"]
    else:
        provenance["cli"] = None
        if cli_sel.get("excluded"):
            provenance["cli_note"] = "No same-day CLI available at decision time; wrong-day/station excluded"

    attribution = {
        "feeds_collected": feeds_collected,
        "feeds_quality_ok": feeds_quality_ok,
        "features_consumed_by_model": features_consumed,
        "settlement_constraints_applied": (
            ["cli_prelim_or_final_floor"] if cli_sel.get("applied") else []
        ),
        "location_id": location_id or metar,
    }
    provenance["attribution"] = attribution

    store.save_features(
        decision_time_utc=now.isoformat(),
        climate_day=climate_day.isoformat(),
        feature_set=FEATURE_SCHEMA_VERSION,
        model_version=None,
        features={k: features[k] for k in STATION_V2_FEATURES + SATRAD_EXTRA},
        missing=missing,
        provenance=provenance,
    )
    return {
        "feature_names": list(STATION_V2_FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
        "missing": missing,
        "provenance": provenance,
        "max_so_far": max_so_far,
        "coverage": coverage,
        "cli_applied": cli_sel.get("applied"),
        "attribution": attribution,
        "climate_day": climate_day.isoformat(),
        "decision_time_utc": now.isoformat(),
        "location_id": location_id or metar,
        "metar_id": metar,
        "tz_name": tz,
        "cli_location_id": cli_id,
    }
