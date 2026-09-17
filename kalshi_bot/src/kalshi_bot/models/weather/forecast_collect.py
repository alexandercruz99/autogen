"""Forecast collectors: NWS grid + optional Open-Meteo previous-runs layer."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import httpx

from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.nws_client import NWSClient
from kalshi_bot.models.weather.stations import StationSpec

logger = logging.getLogger(__name__)
OM_UA = "KalshiBotWeatherResearch/0.2"


class ForecastCollector:
    def __init__(self, archive: WeatherArchive) -> None:
        self.archive = archive
        self.nws = NWSClient()
        self._http = httpx.Client(
            timeout=45.0,
            headers={"User-Agent": OM_UA},
            follow_redirects=True,
        )

    def close(self) -> None:
        self.nws.close()
        self._http.close()

    def collect_nws_daytime_high(self, station: StationSpec, target_day: date) -> dict[str, Any] | None:
        try:
            fc = self.nws.daily_high_forecast(station.lat, station.lon, target_day)
        except Exception as exc:
            logger.warning("NWS forecast failed %s %s: %s", station.city_key, target_day, exc)
            return None
        if not fc or fc.get("temp_f") is None:
            return None
        retrieved = datetime.now(timezone.utc).isoformat()
        available = fc.get("fetched_at") or retrieved
        # Period start is when that forecast period begins — not when the grid was issued.
        init_time = fc.get("update_time") or fc.get("generated_at") or None
        self.archive.save_forecast(
            station_key=station.city_key,
            source="nws_grid",
            model_name="nws_daytime_period",
            init_time=init_time,
            available_at=available,
            valid_date=target_day,
            retrieved_at=retrieved,
            temp_max_f=float(fc["temp_f"]),
            payload=fc,
        )
        return fc

    def collect_open_meteo_daily(
        self,
        station: StationSpec,
        start: date,
        end: date,
        *,
        model: str = "gfs_seamless",
    ) -> dict[str, Any] | None:
        """Optional access layer. Previous-runs API identifies model family; still not CLI settlement.

        Open-Meteo terms: check https://open-meteo.com/en/terms — non-commercial free tier limits apply.
        """
        url = "https://previous-runs-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": station.lat,
            "longitude": station.lon,
            "daily": "temperature_2m_max",
            "timezone": station.timezone,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "models": model,
            "temperature_unit": "fahrenheit",
        }
        try:
            r = self._http.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("Open-Meteo failed %s: %s", station.city_key, exc)
            return None
        retrieved = datetime.now(timezone.utc).isoformat()
        times = (data.get("daily") or {}).get("time") or []
        vals = (data.get("daily") or {}).get("temperature_2m_max") or []
        for d_str, val in zip(times, vals):
            if val is None:
                continue
            self.archive.save_forecast(
                station_key=station.city_key,
                source="open_meteo",
                model_name=model,
                init_time=None,  # previous-runs endpoint does not always expose run hour here
                available_at=retrieved,  # honest: we only know retrieval time for this pull
                valid_date=date.fromisoformat(d_str),
                retrieved_at=retrieved,
                temp_max_f=float(val),
                payload={"note": "available_at=retrieval; init_time unknown for this call", "raw": data.get("daily")},
            )
        return data
