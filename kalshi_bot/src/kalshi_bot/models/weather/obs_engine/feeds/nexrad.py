"""NEXRAD Level III base reflectivity (N0B) for OKX — precipitation features near NYC.

Bucket: s3://unidata-nexrad-level3 (current; deprecated noaa-nexrad-level2 not used).
Site: OKX (Upton, NY). Product N0B = digital base reflectivity.

Level II also available at s3://unidata-nexrad-level2 with keys YYYY/MM/DD/KOKX/...
We use Level III for compact precipitation coverage features without full volume decode.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.feeds import NEXRAD_L3_BUCKET, NYC_LAT, NYC_LON, RADAR_L3_PREFIX
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)


def _unsigned_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def latest_n0b_key(s3, when: datetime | None = None) -> str | None:
    when = when or datetime.now(timezone.utc)
    for days_ago in range(0, 3):
        t = when - timedelta(days=days_ago)
        prefix = f"{RADAR_L3_PREFIX}_N0B_{t.year}_{t.month:02d}_{t.day:02d}"
        last = None
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=NEXRAD_L3_BUCKET, Prefix=prefix):
            for o in page.get("Contents") or []:
                last = o["Key"]
        if last:
            return last
    return None


def n0b_key_near(s3, when: datetime) -> str | None:
    """Pick Level-III N0B object nearest to `when` on that UTC calendar day (scoped, not full archive)."""
    prefix = f"{RADAR_L3_PREFIX}_N0B_{when.year}_{when.month:02d}_{when.day:02d}_"
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NEXRAD_L3_BUCKET, Prefix=prefix):
        for o in page.get("Contents") or []:
            keys.append(o["Key"])
    if not keys:
        return None

    def _key_time(k: str) -> datetime | None:
        # OKX_N0B_YYYY_MM_DD_HH_MM_SS
        parts = k.split("_")
        if len(parts) < 8:
            return None
        try:
            return datetime(
                int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7][:2]),
                tzinfo=timezone.utc,
            )
        except Exception:
            return None

    best = None
    best_dt = None
    for k in keys:
        kt = _key_time(k)
        if kt is None:
            continue
        if best_dt is None or abs((kt - when).total_seconds()) < abs((best_dt - when).total_seconds()):
            best, best_dt = k, kt
    return best or keys[-1]


def _dest_point(lat: float, lon: float, az_deg: float, dist_km: float) -> tuple[float, float]:
    """Approximate destination lat/lon given start, azimuth, distance."""
    R = 6371.0
    brng = math.radians(az_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(dist_km / R) + math.cos(lat1) * math.sin(dist_km / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dist_km / R) * math.cos(lat1),
        math.cos(dist_km / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def extract_precip_features(path: Path, *, center_lat: float = NYC_LAT, center_lon: float = NYC_LON, radius_km: float = 100.0) -> dict[str, Any]:
    from metpy.io import Level3File

    f = Level3File(str(path))
    radar_lat = float(f.lat)
    radar_lon = float(f.lon)
    block = f.sym_block[0][0]
    raw = block["data"]
    # Map digital codes → dBZ via MetPy product thresholds (missing stays NaN)
    mapped = f.map_data(raw)
    data = np.ma.filled(np.asarray(mapped, dtype=float), np.nan)
    if data.ndim != 2:
        raise ValueError(f"unexpected N0B data ndim={data.ndim}")
    naz, nr = data.shape
    start_az_raw = block["start_az"]
    if isinstance(start_az_raw, (list, tuple, np.ndarray)):
        az = np.asarray(start_az_raw, dtype=float) % 360.0
        if len(az) != naz:
            az = (float(az[0]) + np.arange(naz) * (360.0 / max(naz, 1))) % 360.0
    else:
        az = (float(start_az_raw) + np.arange(naz) * (360.0 / max(naz, 1))) % 360.0
    gate_raw = float(block.get("gate_scale") or 1.0)
    # MetPy N0B often reports ~1.0 (km-scale). Values >> 10 are meters.
    if gate_raw > 10:
        gate_scale = gate_raw / 1000.0
    else:
        gate_scale = gate_raw
    first = float(block.get("first") or 0.0)
    if first > 100:
        first = first / 1000.0
    step = gate_scale if gate_scale >= 0.1 else 0.25
    ranges = first + np.arange(nr) * step

    # Collect ALL finite gates in NYC radius. Empty ≠ dry if geometry misses.
    vals = []
    for i, a in enumerate(az):
        for j in range(0, nr, 4):
            plat, plon = _dest_point(radar_lat, radar_lon, float(a), float(ranges[j]))
            dlat = (plat - center_lat) * 111.0
            dlon = (plon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
            if dlat * dlat + dlon * dlon <= radius_km * radius_km:
                v = data[i, j]
                if np.isfinite(v):
                    vals.append(float(v))
    vals_a = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
    vol_time = f.metadata.get("vol_time")
    valid_utc = vol_time.replace(tzinfo=timezone.utc).isoformat() if isinstance(vol_time, datetime) else None
    geo_ok = len(vals_a) > 0
    return {
        "n_gates_in_nyc_radius": int(len(vals_a)),
        # frac of gates with reflectivity >= 20 dBZ (light precip threshold)
        "precip_gate_frac": float(np.mean(vals_a >= 20.0)) if geo_ok else None,
        "mean_dbz_if_any": float(np.mean(vals_a)) if geo_ok else None,
        "max_dbz_if_any": float(np.max(vals_a)) if geo_ok else None,
        # keep aliases used by feature schema
        "mean_level_if_any": float(np.mean(vals_a)) if geo_ok else None,
        "max_level_if_any": float(np.max(vals_a)) if geo_ok else None,
        "radar_lat": radar_lat,
        "radar_lon": radar_lon,
        "radius_km": radius_km,
        "valid_utc": valid_utc,
        "product_max_code": f.metadata.get("max"),
        "missing_scan": False,
        "geometry_miss": not geo_ok,
        "provenance": {
            "product": "N0B",
            "bucket": NEXRAD_L3_BUCKET,
            "site": RADAR_L3_PREFIX,
            "units_note": "MetPy map_data reflectivity (dBZ); precip_gate_frac uses >=20 dBZ",
            "forecast_model_inputs": False,
        },
    }


def collect_nexrad(store: FeedStore, sample_dir: Path | None = None) -> dict[str, Any]:
    sample_dir = sample_dir or Path("data/obs_engine/feeds/nexrad_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        s3 = _unsigned_s3()
        key = latest_n0b_key(s3)
        if not key:
            store.checkpoint("nexrad_okx_n0b", ok=False, error="no N0B object found")
            return {"ok": False, "error": "no N0B object found", "missing_scan": True}
        local = sample_dir / key
        if not local.exists():
            s3.download_file(NEXRAD_L3_BUCKET, key, str(local))
        feats = extract_precip_features(local)
        payload = {"s3_key": key, "local_path": str(local), "features": feats, "bytes": local.stat().st_size}
        store.upsert_sample(
            feed="nexrad_okx_n0b",
            source_key=key,
            payload=payload,
            valid_utc=feats.get("valid_utc"),
            product="N0B",
            local_path=str(local),
        )
        store.checkpoint(
            "nexrad_okx_n0b",
            ok=True,
            source_key=key,
            meta={"precip_gate_frac": feats.get("precip_gate_frac"), "mean_level": feats.get("mean_level_if_any")},
        )
        return {"ok": True, "source_key": key, "features": feats, "local_path": str(local)}
    except Exception as exc:
        logger.warning("NEXRAD collect failed: %s", exc)
        store.checkpoint("nexrad_okx_n0b", ok=False, error=str(exc))
        return {"ok": False, "error": str(exc), "missing_scan": True}
