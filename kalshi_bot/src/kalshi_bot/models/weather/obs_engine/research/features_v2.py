"""Candidate feature sets for controlled experiments (observations only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs, local_date_for, solar_elevation_approx
from kalshi_bot.models.weather.obs_engine.features import FEATURE_NAMES, DecisionFeatures


BASELINE_FEATURES = list(FEATURE_NAMES)

LOCAL_V2_FEATURES = BASELINE_FEATURES + [
    "time_since_max_h",
    "d_dewpt_3h",
    "d_alti_3h",
    "d_sknt_3h",
    "wind_dir_sin",
    "wind_dir_cos",
    "missing_sknt",
    "missing_alti",
    "missing_dwpf",
    "sky_code",
]

NEIGHBOR_FEATURES = LOCAL_V2_FEATURES + [
    "lga_tmpf",
    "lga_minus_nyc",
    "lga_dT_3h",
    "upwind_from_lga",  # 1 if wind suggests flow from LGA azimuth
]

# Cloud/precip without external satellite/radar archives:
CLOUD_PRECIP_FEATURES = NEIGHBOR_FEATURES + [
    "precip_1h",
    "precip_flag",
    "sky_overcast",
]


def _sky_code(skyc1: str | None) -> float:
    if not skyc1:
        return -1.0  # explicit missing — not clear
    m = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 5}
    return float(m.get(skyc1.upper()[:3], -1.0))


def _obs_day(obs: list[HourlyObs], decision_utc: datetime, day: date) -> list[HourlyObs]:
    return [o for o in obs if o.valid_utc <= decision_utc and local_date_for(o.valid_utc) == day and o.tmpf is not None]


def build_baseline(obs: list[HourlyObs], decision_utc: datetime, climate_day: date | None = None) -> DecisionFeatures | None:
    from kalshi_bot.models.weather.obs_engine.features import build_features_at

    return build_features_at(obs, decision_utc, climate_day=climate_day)


def build_local_v2(obs: list[HourlyObs], decision_utc: datetime, climate_day: date | None = None) -> DecisionFeatures | None:
    base = build_baseline(obs, decision_utc, climate_day=climate_day)
    if base is None:
        return None
    tz = ZoneInfo(NYC_TARGET.timezone)
    local = decision_utc.astimezone(tz)
    day = climate_day or local.date()
    day_obs = _obs_day(obs, decision_utc, day)
    latest = day_obs[-1]
    temps = [o.tmpf for o in day_obs if o.tmpf is not None]
    max_so_far = max(temps)
    # time since max
    tmax_time = max((o for o in day_obs if o.tmpf == max_so_far), key=lambda o: o.valid_utc).valid_utc
    time_since_max = (decision_utc - tmax_time).total_seconds() / 3600.0

    def _prev(hours: float, attr: str):
        target = decision_utc - timedelta(hours=hours)
        cands = [o for o in day_obs if o.valid_utc <= target]
        if not cands:
            return None
        return getattr(cands[-1], attr)

    dw0 = latest.dwpf
    dw3 = _prev(3.0, "dwpf")
    al0 = latest.alti
    al3 = _prev(3.0, "alti")
    sk0 = latest.sknt
    sk3 = _prev(3.0, "sknt")
    d_dew = (dw0 - dw3) if dw0 is not None and dw3 is not None else 0.0
    d_alti = (al0 - al3) if al0 is not None and al3 is not None else 0.0
    d_sknt = (sk0 - sk3) if sk0 is not None and sk3 is not None else 0.0
    drct = latest.drct
    if drct is None:
        wsin, wcos = 0.0, 0.0
    else:
        rad = math.radians(float(drct))
        wsin, wcos = math.sin(rad), math.cos(rad)

    extra = [
        float(time_since_max),
        float(d_dew),
        float(d_alti),
        float(d_sknt),
        float(wsin),
        float(wcos),
        1.0 if latest.sknt is None else 0.0,
        1.0 if latest.alti is None else 0.0,
        1.0 if latest.dwpf is None else 0.0,
        _sky_code(latest.skyc1),
    ]
    return DecisionFeatures(
        climate_day=day,
        decision_time_utc=decision_utc,
        values=list(base.values) + extra,
        max_so_far=base.max_so_far,
        provenance={**base.provenance, "feature_set": "local_v2"},
    )


def build_neighbor(
    obs_nyc: list[HourlyObs],
    obs_lga: list[HourlyObs],
    decision_utc: datetime,
    climate_day: date | None = None,
) -> DecisionFeatures | None:
    local = build_local_v2(obs_nyc, decision_utc, climate_day=climate_day)
    if local is None:
        return None
    tz = ZoneInfo(NYC_TARGET.timezone)
    day = climate_day or decision_utc.astimezone(tz).date()
    lga_day = _obs_day(obs_lga, decision_utc, day)
    if not lga_day:
        # Keep aligned dates: mark missing neighbor explicitly
        extra = [float("nan"), float("nan"), 0.0, 0.0]
        # Replace nan with sentinel for GBM
        extra = [-999.0, -999.0, 0.0, 0.0]
        return DecisionFeatures(
            climate_day=day,
            decision_time_utc=decision_utc,
            values=list(local.values) + extra,
            max_so_far=local.max_so_far,
            provenance={**local.provenance, "feature_set": "neighbor", "lga_missing": True},
        )
    lga_latest = lga_day[-1]
    lga_tmp = float(lga_latest.tmpf or 0.0)
    nyc_tmp = float(local.values[2])  # tmpf index in baseline
    # LGA dT 3h
    target = decision_utc - timedelta(hours=3)
    cands = [o for o in lga_day if o.valid_utc <= target]
    lga_dT = (lga_tmp - float(cands[-1].tmpf)) if cands and cands[-1].tmpf is not None else 0.0
    # LGA is roughly ENE of Central Park (~40.78,-73.87). Wind FROM LGA ≈ from ~70-100°
    drct = lga_latest.drct if lga_latest.drct is not None else (obs_nyc and _obs_day(obs_nyc, decision_utc, day)[-1].drct)
    upwind = 0.0
    if drct is not None:
        # If wind is from the east (45-135), air arriving from LGA direction
        if 45.0 <= float(drct) <= 135.0:
            upwind = 1.0
    extra = [lga_tmp, lga_tmp - nyc_tmp, float(lga_dT), upwind]
    return DecisionFeatures(
        climate_day=day,
        decision_time_utc=decision_utc,
        values=list(local.values) + extra,
        max_so_far=local.max_so_far,
        provenance={**local.provenance, "feature_set": "neighbor", "lga_missing": False},
    )


def build_cloud_precip(
    obs_nyc: list[HourlyObs],
    obs_lga: list[HourlyObs],
    decision_utc: datetime,
    climate_day: date | None = None,
) -> DecisionFeatures | None:
    neigh = build_neighbor(obs_nyc, obs_lga, decision_utc, climate_day=climate_day)
    if neigh is None:
        return None
    tz = ZoneInfo(NYC_TARGET.timezone)
    day = climate_day or decision_utc.astimezone(tz).date()
    day_obs = _obs_day(obs_nyc, decision_utc, day)
    latest = day_obs[-1]
    precip_1h = sum((o.p01i or 0.0) for o in day_obs if o.valid_utc >= decision_utc - timedelta(hours=1))
    precip_flag = 1.0 if precip_1h > 0 else 0.0
    sky = _sky_code(latest.skyc1)
    overcast = 1.0 if sky >= 3 else 0.0
    extra = [float(precip_1h), precip_flag, overcast]
    return DecisionFeatures(
        climate_day=day,
        decision_time_utc=decision_utc,
        values=list(neigh.values) + extra,
        max_so_far=neigh.max_so_far,
        provenance={
            **neigh.provenance,
            "feature_set": "cloud_precip_obs_only",
            "satellite_radar": "not_used",
            "satellite_radar_blocker": (
                "GOES-R ABI / NEXRAD full archives not integrated this run; "
                "using METAR sky cover + ASOS precip only. NYS Mesonet access not configured."
            ),
        },
    )


FEATURE_SETS = {
    "baseline": (BASELINE_FEATURES, "baseline"),
    "local_v2": (LOCAL_V2_FEATURES, "local_v2"),
    "neighbor": (NEIGHBOR_FEATURES, "neighbor"),
    "cloud_precip": (CLOUD_PRECIP_FEATURES, "cloud_precip"),
}
