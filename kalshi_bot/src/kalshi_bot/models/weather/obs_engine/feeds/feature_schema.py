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
    EXPECTED_CADENCE_HOURS,
    LATE_START_HOUR_LST,
    MAX_GAP_HOURS,
    MAX_START_GAP_HOURS,
    MAX_STALE_HOURS,
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
    start_gap_hours: float | None
    stale_hours: float | None
    late_start: bool
    gaps_exceeded: bool
    start_gap_exceeded: bool
    stale_exceeded: bool
    adequate: bool
    max_so_far: float | None
    max_so_far_status: str  # insufficient | partial_window | adequate_window_coverage
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
            "start_gap_hours": self.start_gap_hours,
            "stale_hours": self.stale_hours,
            "late_start": self.late_start,
            "gaps_exceeded": self.gaps_exceeded,
            "start_gap_exceeded": self.start_gap_exceeded,
            "stale_exceeded": self.stale_exceeded,
            "adequate": self.adequate,
            "max_so_far": self.max_so_far,
            "max_so_far_status": self.max_so_far_status,
            "notes": self.notes,
            "requirements": {
                "min_temp_obs": MIN_TEMP_OBS_FOR_COVERAGE,
                "max_gap_hours": MAX_GAP_HOURS,
                "max_start_gap_hours": MAX_START_GAP_HOURS,
                "max_stale_hours": MAX_STALE_HOURS,
                "expected_cadence_hours": EXPECTED_CADENCE_HOURS,
                "climate_day_basis": "LST_fixed_offset_per_location",
                "note": "Partial max_so_far is never treated as a verified full-day maximum",
            },
        }


def day_window_obs(
    obs: list[HourlyObs],
    *,
    climate_day: date,
    decision_utc: datetime,
    tz_name: str = NYC_TARGET.timezone,
    enforce_first_seen: bool = True,
) -> list[HourlyObs]:
    """Observations on the LST climate day with valid_utc <= decision_utc.

    When ``enforce_first_seen`` is True and an observation carries ``first_seen_utc``,
    rows first seen after the decision are excluded (replay safety). Archive rows
    without first_seen keep the disclosed valid_utc≈availability assumption.
    """
    start = climate_day_start_utc(climate_day, tz_name=tz_name)
    out: list[HourlyObs] = []
    for o in obs:
        if o.valid_utc > decision_utc or o.valid_utc < start:
            continue
        if lst_climate_day(o.valid_utc, tz_name) != climate_day:
            continue
        if enforce_first_seen and getattr(o, "first_seen_utc", None) is not None:
            fs = o.first_seen_utc
            if fs.tzinfo is None:
                fs = fs.replace(tzinfo=timezone.utc)
            if fs > decision_utc:
                continue
        out.append(o)
    out.sort(key=lambda o: o.valid_utc)
    return out


def assess_coverage(
    day_obs: list[HourlyObs],
    *,
    climate_day: date,
    decision_utc: datetime,
    tz_name: str = NYC_TARGET.timezone,
) -> CoverageReport:
    """Coverage across the full required window — start gap, inter-obs gaps, and freshness."""
    notes: list[str] = []
    temps = [o for o in day_obs if o.tmpf is not None]
    empty = CoverageReport(
        climate_day=climate_day,
        decision_time_utc=decision_utc,
        n_temp_obs=0,
        n_obs_total=len(day_obs),
        first_obs_utc=None,
        last_obs_utc=None,
        max_gap_hours=None,
        start_gap_hours=None,
        stale_hours=None,
        late_start=True,
        gaps_exceeded=True,
        start_gap_exceeded=True,
        stale_exceeded=True,
        adequate=False,
        max_so_far=None,
        max_so_far_status="insufficient",
        notes=["No temperature observations in LST climate-day window before decision time"],
    )
    if not temps:
        return empty

    first, last = temps[0], temps[-1]
    gaps = [(b.valid_utc - a.valid_utc).total_seconds() / 3600.0 for a, b in zip(temps, temps[1:])]
    max_gap = max(gaps) if gaps else 0.0
    day_start = climate_day_start_utc(climate_day, tz_name=tz_name)
    start_gap = (first.valid_utc - day_start).total_seconds() / 3600.0
    stale = (decision_utc - last.valid_utc).total_seconds() / 3600.0
    first_lst = lst_datetime(first.valid_utc, tz_name)
    late_start = first_lst.hour > LATE_START_HOUR_LST or (
        first_lst.hour == LATE_START_HOUR_LST and first_lst.minute > 0
    )
    start_gap_exceeded = start_gap > MAX_START_GAP_HOURS
    gaps_exceeded = max_gap > MAX_GAP_HOURS
    stale_exceeded = stale > MAX_STALE_HOURS
    adequate = (
        len(temps) >= MIN_TEMP_OBS_FOR_COVERAGE
        and not start_gap_exceeded
        and not gaps_exceeded
        and not stale_exceeded
    )
    max_so_far = max(float(o.tmpf) for o in temps if o.tmpf is not None)
    if not adequate:
        status = "partial_window"
        if start_gap_exceeded:
            notes.append(
                f"Start gap {start_gap:.2f}h from 00:00 LST to first temp exceeds {MAX_START_GAP_HOURS}h "
                f"(first={first_lst.isoformat()} LST) — early-window maximum may have been missed"
            )
        if late_start and not start_gap_exceeded:
            notes.append(f"First temp obs at {first_lst.isoformat()} LST (legacy late_start flag)")
        if gaps_exceeded:
            notes.append(f"Max inter-obs gap {max_gap:.2f}h exceeds {MAX_GAP_HOURS}h")
        if stale_exceeded:
            notes.append(
                f"Latest temp obs is {stale:.2f}h before decision (limit {MAX_STALE_HOURS}h) — stale coverage"
            )
        if len(temps) < MIN_TEMP_OBS_FOR_COVERAGE:
            notes.append(f"Only {len(temps)} temp obs (need ≥{MIN_TEMP_OBS_FOR_COVERAGE})")
        notes.append("max_so_far is NOT a verified full-day maximum")
    else:
        status = "adequate_window_coverage"
        notes.append("Coverage meets documented window requirements; still not a final CLI maximum")
    return CoverageReport(
        climate_day=climate_day,
        decision_time_utc=decision_utc,
        n_temp_obs=len(temps),
        n_obs_total=len(day_obs),
        first_obs_utc=first.valid_utc.isoformat(),
        last_obs_utc=last.valid_utc.isoformat(),
        max_gap_hours=max_gap,
        start_gap_hours=start_gap,
        stale_hours=stale,
        late_start=late_start,
        gaps_exceeded=gaps_exceeded,
        start_gap_exceeded=start_gap_exceeded,
        stale_exceeded=stale_exceeded,
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
    tz_name: str = NYC_TARGET.timezone,
    lat: float | None = None,
    lon: float | None = None,
    station_id: str | None = None,
    availability_assumption: str = "valid_utc_as_availability",
) -> StationFeatureBundle | None:
    """Shared feature builder. Returns None only if zero temp obs in window."""
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    day = climate_day or lst_climate_day(decision_utc, tz_name)
    day_obs = day_window_obs(obs, climate_day=day, decision_utc=decision_utc, tz_name=tz_name)
    coverage = assess_coverage(day_obs, climate_day=day, decision_utc=decision_utc, tz_name=tz_name)
    if coverage.max_so_far is None:
        return None

    temps = [o for o in day_obs if o.tmpf is not None]
    latest = temps[-1]
    max_so_far = coverage.max_so_far
    min_so_far = min(float(o.tmpf) for o in temps if o.tmpf is not None)
    local = civil_local(decision_utc, tz_name)
    lst_now = lst_datetime(decision_utc, tz_name)
    use_lat = lat if lat is not None else NYC_TARGET.lat
    use_lon = lon if lon is not None else NYC_TARGET.lon

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

    solar = solar_elevation_approx(use_lat, use_lon, decision_utc)
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
            "climate_day_basis": f"LST_{tz_name}",
            "availability_assumption": availability_assumption,
            "station": station_id or NYC_TARGET.metar_id,
            "tz_name": tz_name,
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
