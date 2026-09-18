"""Fetch and parse NWS Daily Climate Report (CLI) products — settlement truth for daily markets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "KalshiBotWeatherResearch/0.2 (settlement-aligned research; local)"


@dataclass
class CliReport:
    station_cli_id: str
    climate_day: date
    max_temp_f: int | None
    min_temp_f: int | None
    issuance_time: datetime
    retrieved_at: datetime
    product_id: str
    is_preliminary: bool
    valid_as_of_local: str | None
    raw_text: str
    source_url: str

    @property
    def available_at(self) -> datetime:
        """Earliest time this report could be used as a decision feature."""
        return self.issuance_time


_MAX_RE = re.compile(
    r"MAXIMUM\s+(\d+|MM)\s+",
    re.IGNORECASE,
)
_MIN_RE = re.compile(
    r"MINIMUM\s+(\d+|MM)\s+",
    re.IGNORECASE,
)
_SUMMARY_DAY_RE = re.compile(
    r"CLIMATE SUMMARY FOR ([A-Z]+) (\d{1,2}) (\d{4})",
    re.IGNORECASE,
)
_VALID_TODAY_RE = re.compile(
    r"VALID TODAY AS OF\s+(.+)",
    re.IGNORECASE,
)
_MONTHS = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}


def parse_cli_product_text(text: str, *, station_cli_id: str, product_id: str, issuance_time: datetime) -> CliReport:
    climate_day = _parse_climate_day(text)
    max_t = _parse_int_field(_MAX_RE.search(text))
    min_t = _parse_int_field(_MIN_RE.search(text))
    valid = None
    m = _VALID_TODAY_RE.search(text)
    if m:
        valid = m.group(1).strip().rstrip(".")
    preliminary = "VALID TODAY" in text.upper() or "preliminary" in text.lower()
    return CliReport(
        station_cli_id=station_cli_id,
        climate_day=climate_day or date.today(),
        max_temp_f=max_t,
        min_temp_f=min_t,
        issuance_time=issuance_time,
        retrieved_at=datetime.now(timezone.utc),
        product_id=product_id,
        is_preliminary=preliminary,
        valid_as_of_local=valid,
        raw_text=text,
        source_url=f"https://api.weather.gov/products/{product_id}",
    )


def _parse_int_field(match: re.Match[str] | None) -> int | None:
    if not match:
        return None
    raw = match.group(1)
    if raw.upper() == "MM":
        return None
    return int(raw)


def _parse_climate_day(text: str) -> date | None:
    m = _SUMMARY_DAY_RE.search(text)
    if not m:
        return None
    mon = _MONTHS.get(m.group(1).upper())
    if not mon:
        return None
    return date(int(m.group(3)), mon, int(m.group(2)))


class CliReportClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._http = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._http.close()

    def list_recent(self, cli_location_id: str, limit: int = 20) -> list[dict[str, Any]]:
        url = f"https://api.weather.gov/products/types/CLI/locations/{cli_location_id}"
        r = self._http.get(url)
        r.raise_for_status()
        graph = r.json().get("@graph") or []
        return graph[:limit]

    def fetch_product(self, product_id: str) -> dict[str, Any]:
        r = self._http.get(f"https://api.weather.gov/products/{product_id}")
        r.raise_for_status()
        return r.json()

    def latest_for_day(self, cli_location_id: str, climate_day: date) -> CliReport | None:
        """Return the latest CLI product whose climate summary matches climate_day."""
        for item in self.list_recent(cli_location_id, limit=40):
            pid = item.get("id")
            if not pid:
                continue
            payload = self.fetch_product(pid)
            text = payload.get("productText") or ""
            issued = datetime.fromisoformat(
                str(payload.get("issuanceTime")).replace("Z", "+00:00")
            )
            report = parse_cli_product_text(
                text,
                station_cli_id=cli_location_id,
                product_id=pid,
                issuance_time=issued,
            )
            if report.climate_day == climate_day and report.max_temp_f is not None:
                return report
        return None

    def collect_recent_with_max(self, cli_location_id: str, limit: int = 30) -> list[CliReport]:
        out: list[CliReport] = []
        for item in self.list_recent(cli_location_id, limit=limit):
            pid = item.get("id")
            if not pid:
                continue
            try:
                payload = self.fetch_product(pid)
            except Exception as exc:
                logger.warning("CLI fetch failed %s: %s", pid, exc)
                continue
            text = payload.get("productText") or ""
            issued = datetime.fromisoformat(
                str(payload.get("issuanceTime")).replace("Z", "+00:00")
            )
            report = parse_cli_product_text(
                text,
                station_cli_id=cli_location_id,
                product_id=pid,
                issuance_time=issued,
            )
            if report.max_temp_f is None:
                continue
            out.append(report)
        return out
