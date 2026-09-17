"""Climate-day helpers using local *standard* time (no DST) per location timezone.

NWS CLI / Kalshi daily-max settlement use LST for the climate day. During daylight
saving, civil midnight–00:59 is still the previous LST climate day.

Frozen baseline training historically used America/New_York civil dates via
``local_date_for``. Operating station_v2 uses ``lst_climate_day`` and discloses
the difference. Pass ``tz_name`` for non-Eastern locations.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from kalshi_bot.models.weather.obs_engine import NYC_TARGET

# Standard-time UTC offsets (hours) for common IANA zones — no DST.
_STANDARD_OFFSET_HOURS: dict[str, int] = {
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Phoenix": -7,
    "America/Los_Angeles": -8,
    "America/Anchorage": -9,
    "Pacific/Honolulu": -10,
    "UTC": 0,
}

SUPPORTED_DECISION_HOURS_LOCAL: tuple[int, ...] = (8, 11, 14)

MIN_TEMP_OBS_FOR_COVERAGE = 4
MAX_GAP_HOURS = 3.5
MAX_START_GAP_HOURS = 4.0
MAX_STALE_HOURS = 2.5
EXPECTED_CADENCE_HOURS = 1.0
LATE_START_HOUR_LST = 10  # legacy flag only


def lst_tz(tz_name: str = NYC_TARGET.timezone) -> timezone:
    hours = _STANDARD_OFFSET_HOURS.get(tz_name)
    if hours is None:
        # Fall back to January offset (standard time) of the zone
        jan = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo(tz_name))
        hours = int(jan.utcoffset().total_seconds() // 3600)  # type: ignore[union-attr]
    return timezone(timedelta(hours=hours), name=f"LST_{tz_name}")


def lst_climate_day(when: datetime, tz_name: str = NYC_TARGET.timezone) -> date:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(lst_tz(tz_name)).date()


def lst_datetime(when: datetime, tz_name: str = NYC_TARGET.timezone) -> datetime:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(lst_tz(tz_name))


def civil_local(when: datetime, tz_name: str = NYC_TARGET.timezone) -> datetime:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(ZoneInfo(tz_name))


def climate_day_start_utc(day: date, tz_name: str = NYC_TARGET.timezone) -> datetime:
    return datetime(day.year, day.month, day.day, 0, 0, tzinfo=lst_tz(tz_name)).astimezone(timezone.utc)


def climate_day_end_utc(day: date, tz_name: str = NYC_TARGET.timezone) -> datetime:
    return climate_day_start_utc(day + timedelta(days=1), tz_name=tz_name)


def next_supported_decision_utc(now: datetime, tz_name: str = NYC_TARGET.timezone) -> datetime:
    local = civil_local(now, tz_name)
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        cand = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cand > local:
            return cand.astimezone(timezone.utc)
    nxt = (local + timedelta(days=1)).replace(
        hour=SUPPORTED_DECISION_HOURS_LOCAL[0], minute=0, second=0, microsecond=0
    )
    return nxt.astimezone(timezone.utc)


def is_supported_decision_time(
    now: datetime, *, tz_name: str = NYC_TARGET.timezone, tolerance_minutes: int = 20
) -> tuple[bool, int | None]:
    local = civil_local(now, tz_name)
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        delta = abs((local.hour * 60 + local.minute) - hour * 60)
        if delta <= tolerance_minutes:
            return True, hour
    return False, None
