"""Unified live/historical feature builder with explicit missingness (no calm/clear defaults)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore
from kalshi_bot.models.weather.obs_engine.research.features_v2 import LOCAL_V2_FEATURES


# Operating station-corrected + optional sat/radar schema
STATION_CORRECTED_FEATURES = list(LOCAL_V2_FEATURES)  # already has missing_* indicators

SATRAD_FEATURES = STATION_CORRECTED_FEATURES + [
    "goes_cloud_frac_bcm",
    "goes_cloudyish_acm",
    "goes_available",
    "radar_precip_frac",
    "radar_mean_level",
    "radar_available",
]


def _latest_payload(store: FeedStore, feed: str) -> dict[str, Any] | None:
    import json

    row = store.latest_sample(feed)
    if not row:
        return None
    return json.loads(row["payload_json"])


def build_operating_features(store: FeedStore, *, now: datetime | None = None) -> dict[str, Any]:
    """Build feature dict for current inference from feed store + optional ASOS-derived locals.

    Missing feeds → NaN feature values + missing flags. Never invent clear/calm/dry.
    """
    now = now or datetime.now(timezone.utc)
    import json

    from kalshi_bot.models.weather.obs_engine.data import HourlyObs
    from kalshi_bot.models.weather.obs_engine.research.features_v2 import build_local_v2

    # Reconstruct recent METAR as HourlyObs for local_v2
    knyc = []
    for feed in ("metar_KNYC",):
        # pull last samples from DB
        rows = store._conn.execute(
            "SELECT payload_json, valid_utc FROM feed_samples WHERE feed=? ORDER BY valid_utc DESC LIMIT 24",
            (feed,),
        ).fetchall()
        for r in reversed(list(rows)):
            p = json.loads(r["payload_json"])
            if not p.get("valid_utc"):
                continue
            knyc.append(
                HourlyObs(
                    valid_utc=datetime.fromisoformat(p["valid_utc"]),
                    tmpf=p.get("tmpf"),
                    dwpf=p.get("dwpf"),
                    sknt=p.get("sknt"),  # may be None — local_v2 marks missing
                    drct=p.get("drct"),
                    alti=None,
                    p01i=None,
                    skyc1=p.get("skyc1"),
                    station="KNYC",
                    source="feed_store",
                )
            )

    missing: dict[str, bool] = {}
    provenance: dict[str, Any] = {"built_at_utc": now.isoformat(), "feeds_used": []}
    features: dict[str, float | None] = {k: None for k in SATRAD_FEATURES}

    if knyc:
        feats = build_local_v2(knyc, now)
        if feats is not None:
            for name, val in zip(LOCAL_V2_FEATURES, feats.values):
                features[name] = float(val)
            provenance["feeds_used"].append("metar_KNYC")
            provenance["local_v2"] = feats.provenance
            provenance["max_so_far"] = feats.max_so_far
            provenance["climate_day"] = feats.climate_day.isoformat()
    else:
        missing["station_obs"] = True

    # GOES
    goes = _latest_payload(store, "goes19_acmc")
    if goes and goes.get("features"):
        g = goes["features"]
        features["goes_cloud_frac_bcm"] = g.get("cloud_frac_bcm")
        features["goes_cloudyish_acm"] = g.get("cloudy_or_probably_frac_acm")
        features["goes_available"] = 1.0
        provenance["feeds_used"].append("goes19_acmc")
        provenance["goes"] = g.get("provenance")
        # age
        try:
            age_h = (now - datetime.fromisoformat(g["valid_utc"])).total_seconds() / 3600.0
            provenance["goes_age_hours"] = age_h
            if age_h > 3:
                missing["goes_stale"] = True
        except Exception:
            pass
    else:
        features["goes_cloud_frac_bcm"] = None
        features["goes_cloudyish_acm"] = None
        features["goes_available"] = 0.0
        missing["goes"] = True

    # Radar — missing scan ≠ no precip; geometry miss also explicit missing
    radar = _latest_payload(store, "nexrad_okx_n0b")
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
        provenance["feeds_used"].append("nexrad_okx_n0b")
        provenance["radar"] = r.get("provenance")
        try:
            age_h = (now - datetime.fromisoformat(r["valid_utc"])).total_seconds() / 3600.0
            provenance["radar_age_hours"] = age_h
            if age_h > 2:
                missing["radar_stale"] = True
        except Exception:
            pass
    else:
        features["radar_precip_frac"] = None
        features["radar_mean_level"] = None
        features["radar_available"] = 0.0
        missing["radar"] = True
        provenance["radar_note"] = "Unavailable/missing scan — NOT interpreted as no precipitation"

    # CLI prelim
    cli = _latest_payload(store, "cli_nyc")
    if cli:
        provenance["cli"] = {
            "max_temp_f": cli.get("max_temp_f"),
            "is_preliminary": cli.get("is_preliminary"),
            "climate_day": cli.get("climate_day"),
            "issuance_utc": cli.get("issuance_utc"),
        }
        provenance["feeds_used"].append("cli_nyc")

    vector = [features[k] for k in SATRAD_FEATURES]
    # For models that need dense arrays: keep None as NaN explicitly later
    store.save_features(
        decision_time_utc=now.isoformat(),
        climate_day=provenance.get("climate_day"),
        feature_set="satrad_v1",
        model_version=None,
        features={k: features[k] for k in SATRAD_FEATURES},
        missing=missing,
        provenance=provenance,
    )
    return {
        "feature_names": SATRAD_FEATURES,
        "features": features,
        "vector": vector,
        "missing": missing,
        "provenance": provenance,
        "max_so_far": provenance.get("max_so_far"),
    }
