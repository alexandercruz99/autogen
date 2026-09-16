"""Feature construction from observations only (no forecast inputs)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs, local_date_for, solar_elevation_approx


FEATURE_NAMES = [
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
]


@dataclass
class DecisionFeatures:
    climate_day: date
    decision_time_utc: datetime
    values: list[float]
    max_so_far: float
    provenance: dict[str, Any]


def _obs_before(obs: list[HourlyObs], decision_utc: datetime) -> list[HourlyObs]:
    return [o for o in obs if o.valid_utc <= decision_utc and o.tmpf is not None]


def build_features_at(
    obs: list[HourlyObs],
    decision_utc: datetime,
    *,
    climate_day: date | None = None,
) -> DecisionFeatures | None:
    """Build features using only observations with valid_utc <= decision_utc.

    Limitation: IEM/ASOS rows use observation valid time; first-seen/publication time is not
    separately available in this archive — disclosed in provenance.
    """
    tz = ZoneInfo(NYC_TARGET.timezone)
    local = decision_utc.astimezone(tz)
    day = climate_day or local.date()
    prior = _obs_before(obs, decision_utc)
    # Restrict to climate day local (approx using UTC converted local date)
    day_obs = [o for o in prior if local_date_for(o.valid_utc) == day]
    if not day_obs:
        return None
    latest = day_obs[-1]
    temps = [o.tmpf for o in day_obs if o.tmpf is not None]
    max_so_far = max(temps)
    min_so_far = min(temps)

    def _at_hours_ago(h: float) -> float | None:
        target = decision_utc - timedelta(hours=h)
        # nearest prior
        cands = [o for o in day_obs if o.valid_utc <= target]
        if not cands:
            return None
        return cands[-1].tmpf

    t1 = _at_hours_ago(1.0)
    t3 = _at_hours_ago(3.0)
    dT1 = (latest.tmpf - t1) if t1 is not None and latest.tmpf is not None else 0.0
    dT3 = (latest.tmpf - t3) if t3 is not None and latest.tmpf is not None else 0.0
    precip = sum((o.p01i or 0.0) for o in day_obs if o.valid_utc >= decision_utc - timedelta(hours=6))
    solar = solar_elevation_approx(NYC_TARGET.lat, NYC_TARGET.lon, decision_utc)
    hours_to_20 = max(0.0, 20.0 - (local.hour + local.minute / 60.0))

    values = [
        float(local.hour) + local.minute / 60.0,
        float(local.timetuple().tm_yday),
        float(latest.tmpf if latest.tmpf is not None else max_so_far),
        float(latest.dwpf if latest.dwpf is not None else latest.tmpf or max_so_far),
        float(latest.sknt if latest.sknt is not None else 0.0),
        float(latest.alti if latest.alti is not None else 30.0),
        float(max_so_far),
        float(max_so_far - min_so_far),
        float(dT1),
        float(dT3),
        float(solar),
        float(hours_to_20),
        float(precip),
    ]
    return DecisionFeatures(
        climate_day=day,
        decision_time_utc=decision_utc,
        values=values,
        max_so_far=float(max_so_far),
        provenance={
            "n_obs_used": len(day_obs),
            "latest_obs_utc": latest.valid_utc.isoformat(),
            "availability_note": (
                "Features use observation valid timestamps ≤ decision time. "
                "Publication/first-seen times not in IEM ASOS CSV — forward collection should store receiptTime."
            ),
            "station": NYC_TARGET.iem_asos_id,
            "source": "iem_asos_hourly",
        },
    )


def enumerate_training_rows(
    obs: list[HourlyObs],
    labels: dict[date, int],
    decision_hours: tuple[int, ...] = NYC_TARGET.decision_hours_local,
) -> list[dict[str, Any]]:
    """One row per (climate_day, decision_hour) with label = GHCND TMAX and y_remain = label - max_so_far."""
    tz = ZoneInfo(NYC_TARGET.timezone)
    days = sorted({local_date_for(o.valid_utc) for o in obs})
    rows: list[dict[str, Any]] = []
    for day in days:
        if day not in labels:
            continue
        label = labels[day]
        for hour in decision_hours:
            # decision at hour:00 local
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            feats = build_features_at(obs, decision_utc, climate_day=day)
            if feats is None:
                continue
            remain = float(label) - feats.max_so_far
            rows.append(
                {
                    "climate_day": day.isoformat(),
                    "decision_hour": hour,
                    "decision_time_utc": decision_utc.isoformat(),
                    "features": feats.values,
                    "max_so_far": feats.max_so_far,
                    "label_tmax_f": label,
                    "remain_f": remain,
                    "label_source": "GHCND:USW00094728",
                    "provenance": feats.provenance,
                }
            )
    return rows
