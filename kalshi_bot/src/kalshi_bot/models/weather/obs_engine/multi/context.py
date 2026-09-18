"""Forecast context passed through collection → features → inference → paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class ForecastContext:
    location_id: str
    series_ticker: str
    measurement: str  # daily_max_temp_f | daily_min_temp_f | ...
    settlement_source_family: str  # nws_cli | weather_company | unknown
    climate_day: date
    decision_time_utc: datetime
    horizon: str  # same_day | day_ahead | unsupported
    model_version: str | None = None
    feature_schema_version: str | None = None
    decision_hour_local: int | None = None
    timezone: str = "America/New_York"
    uses_lst_climate_day: bool = True
    metar_id: str | None = None
    cli_location_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    elev_m: float | None = None
    mode: str = "RESEARCH"  # RESEARCH | HISTORICAL_REPLAY
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["climate_day"] = self.climate_day.isoformat()
        d["decision_time_utc"] = self.decision_time_utc.isoformat()
        return d

    @property
    def artifact_key(self) -> str:
        return f"{self.location_id}__{self.measurement}"
