"""Shared station feature schema for training, replay, and live inference.

Schema version ``station_v2.1``:
- Explicit missing indicators (never invent calm wind, clear sky, zero precip, or 30.00 inHg).
- LST climate day for day-window selection (settlement-aligned).
- Same builder for historical ASOS and live METAR→HourlyObs.

Availability disclosure: archive rows use observation valid time only (no independent
first-seen). Live FeedStore rows carry first_seen_utc; historical tests must disclose
that valid_utc ≈ availability is an assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs, solar_elevation_approx
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import (
    LATE_START_HOUR_LST,
    MAX_GAP_HOURS,
    MIN_TEMP_OBS_FOR_COVERAGE,
    civil_local,
    climate_day_start_utc,
    lst_climate_day,
    lst_datetime,
)

FEATURE_SCHEMA_VERSION = "station_v2.1"

# Ordered feature vector consumed by station_corrected operating models.
STATION_V2_FEATURES: list[str] = [
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
]


def _sky_code(skyc1: str | None) -> float:
    if not skyc1:
        return -1.0  # explicit missing — not clear
    m = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 5}
    return float(m.get(skyc1.upper()[:3], -1.0))


def hpa_to_inhg(hpa: float | None) -> float | None:
    if hpa is None:
        return None
    return float(hpa) * 0.029529983071


@dataclass
class CoverageReport:
    climate_day: date
    decision_time_utc: datetime
    n_temp_obs: int
    n_obs_total: int
    first_obs_utc: str | None
    last_obs_utc: str | None
    max_gap_hours: float | None
    late_start: bool
    gaps_exceeded: bool
    adequate: bool
    max_so_far: float | None
    max_so_far_status: str  # insufficient | partial_window | adequate_daytime_coverage
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "climate_day": self.climate_day.isoformat(),
            "decision_time_utc": self.decision_time_utc.isoformat(),
            "n_temp_obs": self.n_temp_obs,
            "n_obs_total": self.n_obs_total,
            "first_obs_utc": self.first_obs_utc,
            "last_obs_utc": self.last_obs_utc,
            "max_gap_hours": self.max_gap_hours,
            "late_start": self.late_start,
            "gaps_exceeded": self.gaps_exceeded,
            "adequate": self.adequate,
            "max_so_far": self.max_so_far,
            "max_so_far_status": self.max_so_far_status,
            "notes": self.notes,
            "requirements": {
                "min_temp_obs": MIN_TEMP_OBS_FOR_COVERAGE,
                "max_gap_hours": MAX_GAP_HOURS,
                "late_start_hour_lst": LATE_START_HOUR_LST,
                "climate_day_basis": "LST_UTC-5",
            },
        }


def day_window_obs(
    obs: list[HourlyObs],
    *,
    climate_day: date,
    decision_utc: datetime,
) -> list[HourlyObs]:
    """Observations on the LST climate day with valid_utc <= decision_utc."""
    start = climate_day_start_utc(climate_day)
    out = [
        o
        for o in obs
        if o.valid_utc <= decision_utc
        and o.valid_utc >= start
        and lst_climate_day(o.valid_utc) == climate_day
    ]
    out.sort(key=lambda o: o.valid_utc)
    return out


def assess_coverage(day_obs: list[HourlyObs], *, climate_day: date, decision_utc: datetime) -> CoverageReport:
    notes: list[str] = []
    temps = [o for o in day_obs if o.tmpf is not None]
    if not temps:
        return CoverageReport(
            climate_day=climate_day,
            decision_time_utc=decision_utc,
            n_temp_obs=0,
            n_obs_total=len(day_obs),
            first_obs_utc=None,
            last_obs_utc=None,
            max_gap_hours=None,
            late_start=True,
            gaps_exceeded=True,
            adequate=False,
            max_so_far=None,
            max_so_far_status="insufficient",
            notes=["No temperature observations in LST climate-day window before decision time"],
        )
    first, last = temps[0], temps[-1]
    gaps = []
    for a, b in zip(temps, temps[1:]):
        gaps.append((b.valid_utc - a.valid_utc).total_seconds() / 3600.0)
    max_gap = max(gaps) if gaps else 0.0
    first_lst = lst_datetime(first.valid_utc)
    late_start = first_lst.hour > LATE_START_HOUR_LST or (
        first_lst.hour == LATE_START_HOUR_LST and first_lst.minute > 0
    )
    gaps_exceeded = max_gap > MAX_GAP_HOURS
    adequate = (
        len(temps) >= MIN_TEMP_OBS_FOR_COVERAGE
        and not late_start
        and not gaps_exceeded
    )
    max_so_far = max(float(o.tmpf) for o in temps if o.tmpf is not None)
    if not adequate:
        status = "partial_window"
        if late_start:
            notes.append(
                f"First temp obs at {first_lst.isoformat()} LST — daytime maximum may have been missed"
            )
        if gaps_exceeded:
            notes.append(f"Max inter-obs gap {max_gap:.2f}h exceeds {MAX_GAP_HOURS}h")
        if len(temps) < MIN_TEMP_OBS_FOR_COVERAGE:
            notes.append(f"Only {len(temps)} temp obs (need ≥{MIN_TEMP_OBS_FOR_COVERAGE})")
        notes.append("max_so_far is NOT a verified full-day maximum")
    else:
        status = "adequate_daytime_coverage"
        notes.append("Coverage meets documented minimums; still not a final CLI maximum")
    return CoverageReport(
        climate_day=climate_day,
        decision_time_utc=decision_utc,
        n_temp_obs=len(temps),
        n_obs_total=len(day_obs),
        first_obs_utc=first.valid_utc.isoformat(),
        last_obs_utc=last.valid_utc.isoformat(),
        max_gap_hours=max_gap,
        late_start=late_start,
        gaps_exceeded=gaps_exceeded,
        adequate=adequate,
        max_so_far=max_so_far,
        max_so_far_status=status,
        notes=notes,
    )


@dataclass
class StationFeatureBundle:
    climate_day: date
    decision_time_utc: datetime
    feature_names: list[str]
    values: list[float | None]
    feature_map: dict[str, float | None]
    max_so_far: float
    coverage: CoverageReport
    provenance: dict[str, Any]


def build_station_v2_features(
    obs: list[HourlyObs],
    decision_utc: datetime,
    *,
    climate_day: date | None = None,
    availability_assumption: str = "valid_utc_as_availability",
) -> StationFeatureBundle | None:
    """Shared feature builder. Returns None only if zero temp obs in window."""
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    day = climate_day or lst_climate_day(decision_utc)
    day_obs = day_window_obs(obs, climate_day=day, decision_utc=decision_utc)
    coverage = assess_coverage(day_obs, climate_day=day, decision_utc=decision_utc)
    if coverage.max_so_far is None:
        return None

    temps = [o for o in day_obs if o.tmpf is not None]
    latest = temps[-1]
    max_so_far = coverage.max_so_far
    min_so_far = min(float(o.tmpf) for o in temps if o.tmpf is not None)
    local = civil_local(decision_utc)
    lst_now = lst_datetime(decision_utc)

    def _prev(hours: float, attr: str):
        target = decision_utc - timedelta(hours=hours)
        cands = [o for o in day_obs if o.valid_utc <= target]
        if not cands:
            return None
        return getattr(cands[-1], attr)

    def _at_temp(hours: float) -> float | None:
        target = decision_utc - timedelta(hours=hours)
        cands = [o for o in temps if o.valid_utc <= target]
        if not cands:
            return None
        return cands[-1].tmpf

    t1 = _at_temp(1.0)
    t3 = _at_temp(3.0)
    dT1 = (latest.tmpf - t1) if t1 is not None and latest.tmpf is not None else None
    dT3 = (latest.tmpf - t3) if t3 is not None and latest.tmpf is not None else None

    # precip_6h: only sum known values; if any obs in window lacks p01i → missing
    window_6 = [o for o in day_obs if o.valid_utc >= decision_utc - timedelta(hours=6)]
    precip_vals = []
    precip_missing = False
    if not window_6:
        precip_missing = True
    else:
        for o in window_6:
            if o.p01i is None:
                precip_missing = True
            else:
                precip_vals.append(float(o.p01i))
    precip_6h = float(sum(precip_vals)) if precip_vals and not precip_missing else (float(sum(precip_vals)) if precip_vals else None)
    if precip_missing:
        precip_6h = None  # never invent zero precip

    tmax_time = max(temps, key=lambda o: (o.tmpf, o.valid_utc)).valid_utc
    time_since_max = (decision_utc - tmax_time).total_seconds() / 3600.0

    dw0, dw3 = latest.dwpf, _prev(3.0, "dwpf")
    al0, al3 = latest.alti, _prev(3.0, "alti")
    sk0, sk3 = latest.sknt, _prev(3.0, "sknt")
    d_dew = (dw0 - dw3) if dw0 is not None and dw3 is not None else None
    d_alti = (al0 - al3) if al0 is not None and al3 is not None else None
    d_sknt = (sk0 - sk3) if sk0 is not None and sk3 is not None else None

    drct = latest.drct
    if drct is None:
        wsin = wcos = None
    else:
        rad = math.radians(float(drct))
        wsin, wcos = math.sin(rad), math.cos(rad)

    solar = solar_elevation_approx(NYC_TARGET.lat, NYC_TARGET.lon, decision_utc)
    hours_to_20 = max(0.0, 20.0 - (local.hour + local.minute / 60.0))

    fmap: dict[str, float | None] = {
        "hour_local": float(local.hour) + local.minute / 60.0,
        "doy": float(lst_now.timetuple().tm_yday),
        "tmpf": float(latest.tmpf) if latest.tmpf is not None else None,
        "dwpf": float(latest.dwpf) if latest.dwpf is not None else None,
        "sknt": float(latest.sknt) if latest.sknt is not None else None,
        "alti": float(latest.alti) if latest.alti is not None else None,
        "max_so_far": float(max_so_far),
        "rise_from_min": float(max_so_far - min_so_far),
        "dT_1h": float(dT1) if dT1 is not None else None,
        "dT_3h": float(dT3) if dT3 is not None else None,
        "solar_el": float(solar),
        "hours_to_20_local": float(hours_to_20),
        "precip_6h": precip_6h,
        "time_since_max_h": float(time_since_max),
        "d_dewpt_3h": float(d_dew) if d_dew is not None else None,
        "d_alti_3h": float(d_alti) if d_alti is not None else None,
        "d_sknt_3h": float(d_sknt) if d_sknt is not None else None,
        "wind_dir_sin": float(wsin) if wsin is not None else None,
        "wind_dir_cos": float(wcos) if wcos is not None else None,
        "missing_sknt": 1.0 if latest.sknt is None else 0.0,
        "missing_alti": 1.0 if latest.alti is None else 0.0,
        "missing_dwpf": 1.0 if latest.dwpf is None else 0.0,
        "missing_precip_6h": 1.0 if precip_missing else 0.0,
        "sky_code": _sky_code(latest.skyc1),
    }
    values = [fmap[n] for n in STATION_V2_FEATURES]
    return StationFeatureBundle(
        climate_day=day,
        decision_time_utc=decision_utc,
        feature_names=list(STATION_V2_FEATURES),
        values=values,
        feature_map=fmap,
        max_so_far=float(max_so_far),
        coverage=coverage,
        provenance={
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "climate_day_basis": "LST_UTC-5",
            "availability_assumption": availability_assumption,
            "station": NYC_TARGET.metar_id,
            "n_day_obs": len(day_obs),
        },
    )


def vector_for_model(fmap: dict[str, float | None], feature_names: list[str]) -> list[float]:
    """Dense vector with NaN for missing (caller may sentinel at predict time)."""
    import math as _m

    out: list[float] = []
    for n in feature_names:
        v = fmap.get(n)
        if v is None or (isinstance(v, float) and _m.isnan(v)):
            out.append(float("nan"))
        else:
            out.append(float(v))
    return out
