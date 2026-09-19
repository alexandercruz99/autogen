"""GOES-19 ABI Clear Sky Mask (ACM/BCM) adapter — geographically scoped to NYC.

Bucket: s3://noaa-goes19 (GOES-East operational). Public NOAA NODD license.
Product: ABI-L2-ACMC (CONUS Clear Sky Mask).

PROVENANCE DISCLOSURE: ABI L2 Clear Sky Mask includes radiative-transfer
model comparison fields (obs_modeled_diff_RTM_*). It is a satellite
observation-derived cloud product that may incorporate model-assist.
It is NOT an NWS point temperature forecast. Documented here; not silent.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.feeds import GOES_EAST_BUCKET, NYC_LAT, NYC_LON
from kalshi_bot.models.weather.obs_engine.feeds.storage import FeedStore

logger = logging.getLogger(__name__)


def _unsigned_s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def _goes_xy_for_latlon(lat_deg: float, lon_deg: float, proj) -> tuple[float, float]:
    """GOES-R ABI fixed-grid scan angles (radians) for geodetic lat/lon."""
    req = float(proj.semi_major_axis)
    rpol = float(proj.semi_minor_axis)
    H = float(proj.perspective_point_height) + req
    lon0 = math.radians(float(proj.longitude_of_projection_origin))
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = (req**2 - rpol**2) / req**2
    lam = lon - lon0
    phi_c = math.atan((rpol**2) / (req**2) * math.tan(lat))
    r_c = rpol / math.sqrt(1 - e2 * math.cos(phi_c) ** 2)
    sx = H - r_c * math.cos(phi_c) * math.cos(lam)
    sy = -r_c * math.cos(phi_c) * math.sin(lam)
    sz = r_c * math.sin(phi_c)
    yy = math.atan(sz / sx)
    xx = math.asin(-sy / math.sqrt(sx**2 + sy**2 + sz**2))
    return xx, yy


def latest_acmc_key(s3, when: datetime | None = None, bucket: str | None = None) -> str | None:
    when = when or datetime.now(timezone.utc)
    bucket = bucket or GOES_EAST_BUCKET
    for hours_ago in range(0, 8):
        t = when - timedelta(hours=hours_ago)
        ddd = t.timetuple().tm_yday
        prefix = f"ABI-L2-ACMC/{t.year}/{ddd:03d}/{t.hour:02d}/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        keys = sorted(o["Key"] for o in (resp.get("Contents") or []))
        if keys:
            return keys[-1]
    return None


def acmc_key_near(s3, when: datetime, bucket: str | None = None) -> tuple[str, str] | None:
    """Return (bucket, key) for ACMC nearest to `when` hour. Prefer GOES-19, else GOES-16."""
    buckets = [bucket] if bucket else [GOES_EAST_BUCKET, "noaa-goes16"]
    ddd = when.timetuple().tm_yday
    for b in buckets:
        for hour_off in (0, -1, 1, -2, 2):
            t = when + timedelta(hours=hour_off)
            ddd_t = t.timetuple().tm_yday
            prefix = f"ABI-L2-ACMC/{t.year}/{ddd_t:03d}/{t.hour:02d}/"
            resp = s3.list_objects_v2(Bucket=b, Prefix=prefix, MaxKeys=1000)
            keys = sorted(o["Key"] for o in (resp.get("Contents") or []))
            if keys:
                return b, keys[len(keys) // 2]
    return None


def extract_nyc_cloud_features(nc_path: Path, lat: float = NYC_LAT, lon: float = NYC_LON, half_win: int = 8) -> dict[str, Any]:
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path))
    try:
        proj = ds.variables["goes_imager_projection"]
        x = np.asarray(ds.variables["x"][:], dtype=float)
        y = np.asarray(ds.variables["y"][:], dtype=float)
        xx, yy = _goes_xy_for_latlon(lat, lon, proj)
        ix = int(np.argmin(np.abs(x - xx)))
        iy = int(np.argmin(np.abs(y - yy)))
        sl_y = slice(max(0, iy - half_win), iy + half_win + 1)
        sl_x = slice(max(0, ix - half_win), ix + half_win + 1)
        bcm = np.asarray(ds.variables["BCM"][sl_y, sl_x])
        acm = np.asarray(ds.variables["ACM"][sl_y, sl_x])
        dqf = np.asarray(ds.variables["DQF"][sl_y, sl_x]) if "DQF" in ds.variables else None
        # BCM: 0 clear, 1 cloudy (masked values excluded)
        bcm_f = bcm.astype(float)
        if np.ma.isMaskedArray(bcm):
            bcm_f = np.ma.filled(bcm.astype(float), np.nan)
        valid = np.isfinite(bcm_f)
        cloud_frac = float(np.nanmean(bcm_f == 1)) if valid.any() else None
        # ACM 0 clear, 1 probably clear, 2 probably cloudy, 3 cloudy
        acm_f = acm.astype(float)
        if np.ma.isMaskedArray(acm):
            acm_f = np.ma.filled(acm.astype(float), np.nan)
        cloudyish = float(np.nanmean(acm_f >= 2)) if np.isfinite(acm_f).any() else None
        t = ds.variables["t"][:]
        # time is seconds since 2000-01-01 12:00 UTC for GOES-R
        epoch = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
        valid_utc = (epoch + timedelta(seconds=float(np.asarray(t).ravel()[0]))).isoformat()
        return {
            "cloud_frac_bcm": cloud_frac,
            "cloudy_or_probably_frac_acm": cloudyish,
            "acm_center": float(acm_f[half_win, half_win]) if acm_f.shape[0] > half_win and acm_f.shape[1] > half_win else None,
            "pixel_iy_ix": [iy, ix],
            "window": half_win,
            "valid_utc": valid_utc,
            "dqf_nonzero_frac": float(np.nanmean(dqf != 0)) if dqf is not None else None,
            "provenance": {
                "product": "ABI-L2-ACMC",
                "bucket": GOES_EAST_BUCKET,
                "includes_rtm_model_assist": True,
                "note": "Clear Sky Mask uses spectral tests + RTM BT comparisons; not an NWS temp forecast",
            },
        }
    finally:
        ds.close()


def collect_goes(store: FeedStore, sample_dir: Path | None = None) -> dict[str, Any]:
    sample_dir = sample_dir or Path("data/obs_engine/feeds/goes_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        s3 = _unsigned_s3()
        key = latest_acmc_key(s3)
        if not key:
            store.checkpoint("goes19_acmc", ok=False, error="no ACMC object found")
            return {"ok": False, "error": "no ACMC object found"}
        local = sample_dir / Path(key).name
        if not local.exists():
            s3.download_file(GOES_EAST_BUCKET, key, str(local))
        feats = extract_nyc_cloud_features(local)
        payload = {
            "s3_key": key,
            "local_path": str(local),
            "features": feats,
            "bytes": local.stat().st_size,
        }
        store.upsert_sample(
            feed="goes19_acmc",
            source_key=key,
            payload=payload,
            valid_utc=feats.get("valid_utc"),
            product="ABI-L2-ACMC",
            local_path=str(local),
        )
        store.checkpoint("goes19_acmc", ok=True, source_key=key, meta={"cloud_frac_bcm": feats.get("cloud_frac_bcm")})
        return {"ok": True, "source_key": key, "features": feats, "local_path": str(local)}
    except Exception as exc:
        logger.warning("GOES collect failed: %s", exc)
        store.checkpoint("goes19_acmc", ok=False, error=str(exc))
        return {"ok": False, "error": str(exc)}
