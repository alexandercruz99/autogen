"""Load Central Park GHCND daily TMAX labels and IEM ASOS hourly observations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from zoneinfo import ZoneInfo

from kalshi_bot.models.weather.obs_engine import NYC_TARGET


@dataclass
class HourlyObs:
    valid_utc: datetime
    tmpf: float | None
    dwpf: float | None
    sknt: float | None
    drct: float | None
    alti: float | None
    p01i: float | None
    skyc1: str | None
    station: str
    source: str = "iem_asos"
    # When known (live FeedStore), first time this observation entered our store.
    # Archive/replay without this field must disclose valid_utc≈availability assumption.
    first_seen_utc: datetime | None = None
    receipt_utc: datetime | None = None


def default_data_dir() -> Path:
    return Path("data/obs_engine")


def load_ghcnd_tmax(path: Path | None = None) -> dict[date, int]:
    """Official Central Park daily TMAX (°F integers). Provenance: GHCND USW00094728."""
    path = path or default_data_dir() / "nyc_central_park_ghcnd_tmax_f.csv"
    out: dict[date, int] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            out[date.fromisoformat(row["date"])] = int(row["tmax_f"])
    return out


def load_asos_csv(path: Path, station: str | None = None) -> list[HourlyObs]:
    if not path.exists():
        return []
    rows: list[HourlyObs] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if station and row.get("station") and row["station"] != station:
                continue
            valid_raw = row.get("valid") or ""
            if not valid_raw:
                continue
            # IEM "YYYY-MM-DD HH:MM" in requested tz; we requested UTC
            try:
                valid = datetime.strptime(valid_raw.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            def _f(key: str) -> float | None:
                v = (row.get(key) or "").strip()
                if v in ("", "M", "null", "None"):
                    return None
                # IEM ASOS "T" = trace precip — known near-zero measurement, not missing
                if key == "p01i" and v.upper() == "T":
                    return 0.0
                try:
                    return float(v)
                except ValueError:
                    return None

            rows.append(
                HourlyObs(
                    valid_utc=valid,
                    tmpf=_f("tmpf"),
                    dwpf=_f("dwpf"),
                    sknt=_f("sknt"),
                    drct=_f("drct"),
                    alti=_f("alti"),
                    p01i=_f("p01i"),
                    skyc1=(row.get("skyc1") or None),
                    station=row.get("station") or station or NYC_TARGET.iem_asos_id,
                )
            )
    rows.sort(key=lambda r: r.valid_utc)
    return rows


def load_nyc_hourly_bundle(data_dir: Path | None = None) -> list[HourlyObs]:
    data_dir = data_dir or default_data_dir()
    rows: list[HourlyObs] = []
    for name in sorted(data_dir.glob("asos_NYC_*.csv")):
        rows.extend(load_asos_csv(name, station="NYC"))
    rows.sort(key=lambda r: r.valid_utc)
    return rows


def load_hourly_asos_bundle(
    data_dir: Path,
    *,
    glob_pattern: str,
    station: str,
) -> list[HourlyObs]:
    """Load IEM ASOS hourly CSVs for an arbitrary station (e.g. MDW)."""
    rows: list[HourlyObs] = []
    for name in sorted(Path(data_dir).glob(glob_pattern)):
        rows.extend(load_asos_csv(name, station=station))
    rows.sort(key=lambda r: r.valid_utc)
    return rows


def solar_elevation_approx(lat: float, lon: float, when: datetime) -> float:
    """Rough solar elevation degrees (no forecast model). Good enough as a feature."""
    import math

    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    # day of year
    doy = when.timetuple().tm_yday
    # declination
    dec = 23.44 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    # UTC hour angle
    utc_hours = when.hour + when.minute / 60.0
    lst = (utc_hours + lon / 15.0) % 24
    ha = (lst - 12.0) * 15.0
    lat_r = math.radians(lat)
    dec_r = math.radians(dec)
    ha_r = math.radians(ha)
    sin_el = math.sin(lat_r) * math.sin(dec_r) + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    sin_el = max(-1.0, min(1.0, sin_el))
    return math.degrees(math.asin(sin_el))


def local_date_for(when: datetime, tz_name: str = NYC_TARGET.timezone) -> date:
    return when.astimezone(ZoneInfo(tz_name)).date()
