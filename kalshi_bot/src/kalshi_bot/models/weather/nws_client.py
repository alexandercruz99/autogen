from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "(kalshi-bot, research-use; contact: local-operator)"


class NWSClient:
    """National Weather Service API client (api.weather.gov). Free, no key required."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def get_points(self, lat: float, lon: float) -> dict[str, Any]:
        resp = self._client.get(f"https://api.weather.gov/points/{lat},{lon}")
        resp.raise_for_status()
        return resp.json()

    def get_gridpoint_forecast(self, forecast_url: str) -> dict[str, Any]:
        resp = self._client.get(forecast_url)
        resp.raise_for_status()
        return resp.json()

    def get_gridpoint_raw(self, grid_url: str) -> dict[str, Any]:
        resp = self._client.get(grid_url)
        resp.raise_for_status()
        return resp.json()

    def daily_high_forecast(
        self, lat: float, lon: float, target_day: date
    ) -> dict[str, Any] | None:
        """Return max forecast temperature (°F) for target local calendar day from NWS periods.

        Uses the textual forecast periods (daytime highs). Returns None if unavailable —
        never synthesizes a temperature.
        """
        points = self.get_points(lat, lon)
        props = points.get("properties") or {}
        forecast_url = props.get("forecast")
        if not forecast_url:
            return None
        forecast = self.get_gridpoint_forecast(forecast_url)
        fetched_at = datetime.now(timezone.utc)
        periods = (forecast.get("properties") or {}).get("periods") or []
        best: dict[str, Any] | None = None
        for period in periods:
            if not period.get("isDaytime", True):
                # Nighttime periods are lows; skip for daily high markets.
                continue
            start = period.get("startTime")
            if not start:
                continue
            # Parse date in the period's local offset.
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError:
                continue
            if start_dt.date() != target_day:
                continue
            temp = period.get("temperature")
            unit = period.get("temperatureUnit") or "F"
            if temp is None:
                continue
            temp_f = float(temp)
            if unit.upper().startswith("C"):
                temp_f = temp_f * 9.0 / 5.0 + 32.0
            best = {
                "temp_f": temp_f,
                "period_name": period.get("name"),
                "start_time": start,
                "end_time": period.get("endTime"),
                "detailed_forecast": period.get("detailedForecast"),
                "short_forecast": period.get("shortForecast"),
                "grid_id": props.get("gridId"),
                "cwa": props.get("cwa"),
                "forecast_url": forecast_url,
                "fetched_at": fetched_at.isoformat(),
                "source": "api.weather.gov/forecast",
                "station_note": f"NWS grid {props.get('gridId')} {props.get('gridX')},{props.get('gridY')}",
            }
            break
        return best


_THRESHOLD_RE = re.compile(
    r"-(?P<kind>[TB])(?P<a>\d+(?:\.\d+)?)(?:-(?P<b>\d+(?:\.\d+)?))?$"
)
# Kalshi often encodes buckets as B79.5 meaning 79-80
_BUCKET_RE = re.compile(r"-B(?P<mid>\d+(?:\.\d+)?)$")
_OVER_RE = re.compile(r"-T(?P< thr>\d+(?:\.\d+)?)$".replace(" ", ""))
_DATE_RE = re.compile(r"(?:HIGH|LOW|TEMP)[A-Z]*-(?P<y>\d{2})(?P<mon>[A-Z]{3})(?P<d>\d{2})")


MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_market_date(ticker: str) -> date | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker.upper())
    if not m:
        return None
    yy, mon, dd = int(m.group(1)), MONTHS.get(m.group(2)), int(m.group(3))
    if not mon:
        return None
    year = 2000 + yy
    try:
        return date(year, mon, dd)
    except ValueError:
        return None


def parse_temp_contract(ticker: str, title: str = "") -> dict[str, Any] | None:
    """Parse Kalshi high-temp contract into an interval on the temperature axis.

    Returns {'op': 'range'|'gt'|'lt', 'low': float|None, 'high': float|None} in °F
    where the YES outcome is temp in [low, high) or > / < thresholds per title cues.
    """
    t = ticker.upper()
    title_l = (title or "").lower()

    bm = re.search(r"-B(\d+(?:\.\d+)?)$", t)
    if bm:
        mid = float(bm.group(1))
        # Convention observed: B79.5 => 79-80° band (integer degrees centered near mid).
        low = int(mid - 0.5 + 1e-9)
        high = low + 2  # inclusive two-degree labels like "79-80"
        # YES if floor(temp) in {low, low+1} i.e. temp in [low, low+2)
        return {"op": "range", "low": float(low), "high": float(low + 2), "raw": bm.group(0)}

    tm = re.search(r"-T(\d+(?:\.\d+)?)$", t)
    if tm:
        thr = float(tm.group(1))
        if ">" in title_l or "above" in title_l or "higher than" in title_l or "more than" in title_l:
            return {"op": "gt", "low": thr, "high": None, "raw": tm.group(0)}
        if "<" in title_l or "below" in title_l or "lower than" in title_l or "less than" in title_l:
            return {"op": "lt", "low": None, "high": thr, "raw": tm.group(0)}
        # Ambiguous threshold without title direction — refuse rather than guess.
        return None

    # Title-based fallback for older HIGHNY tickers
    m = re.search(r"(\d+)\s*-\s*(\d+)", title_l)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return {"op": "range", "low": low, "high": high + 1.0, "raw": "title-range"}
    return None
