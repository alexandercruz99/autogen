"""Climate-day helpers for NYC Central Park settlement alignment.

NWS CLI / Kalshi daily-max settlement use **local standard time (LST)** for the
climate day. During Eastern Daylight Time the civil clock is UTC−4 while LST is
UTC−5, so civil midnight–00:59 is still the previous LST climate day.

Frozen baseline training historically used America/New_York civil dates via
``local_date_for``. Operating feeds / station_v2 use ``lst_climate_day`` and
disclose the difference.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from kalshi_bot.models.weather.obs_engine import NYC_TARGET

# Eastern Local Standard Time fixed offset (no DST).
_LST = timezone(timedelta(hours=-5), name="LST_UTC-5")

# Validated actionable decision hours (local *civil* clock, as in backtests).
SUPPORTED_DECISION_HOURS_LOCAL: tuple[int, ...] = (8, 11, 14)

# Coverage: need temperature obs spanning morning→decision on the LST climate day.
MIN_TEMP_OBS_FOR_COVERAGE = 4
MAX_GAP_HOURS = 3.5
# If first temp obs is after this local LST hour, daytime max may have been missed.
LATE_START_HOUR_LST = 10


def lst_climate_day(when: datetime) -> date:
    """Return the NWS/Kalshi LST climate day for an aware UTC/local timestamp."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(_LST).date()


def lst_datetime(when: datetime) -> datetime:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(_LST)


def civil_local(when: datetime) -> datetime:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(ZoneInfo(NYC_TARGET.timezone))


def climate_day_start_utc(day: date) -> datetime:
    """00:00 LST on ``day`` as UTC."""
    return datetime(day.year, day.month, day.day, 0, 0, tzinfo=_LST).astimezone(timezone.utc)


def climate_day_end_utc(day: date) -> datetime:
    """Exclusive end: next day's 00:00 LST as UTC."""
    return climate_day_start_utc(day + timedelta(days=1))


def next_supported_decision_utc(now: datetime) -> datetime:
    """Next upcoming supported decision hour in NYC civil time (today or tomorrow)."""
    local = civil_local(now)
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        cand = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cand > local:
            return cand.astimezone(timezone.utc)
    # Next calendar day 08:00 civil
    nxt = (local + timedelta(days=1)).replace(
        hour=SUPPORTED_DECISION_HOURS_LOCAL[0], minute=0, second=0, microsecond=0
    )
    return nxt.astimezone(timezone.utc)


def is_supported_decision_time(now: datetime, *, tolerance_minutes: int = 20) -> tuple[bool, int | None]:
    """True if ``now`` is within tolerance of a validated decision hour (civil local)."""
    local = civil_local(now)
    for hour in SUPPORTED_DECISION_HOURS_LOCAL:
        delta = abs((local.hour * 60 + local.minute) - hour * 60)
        # also allow wrap near midnight not relevant for 8/11/14
        if delta <= tolerance_minutes:
            return True, hour
    return False, None
