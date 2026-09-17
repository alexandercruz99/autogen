"""Same-day posterior updates from preliminary observations (evidence, not settlement)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from kalshi_bot.models.weather.distribution import PredictiveDistribution, truncate_below
from kalshi_bot.models.weather.stations import StationSpec

UA = "KalshiBotWeatherResearch/0.2"


@dataclass
class SameDayEvidence:
    obs_max_f: float | None
    source: str
    retrieved_at: datetime
    note: str
    stale: bool = False


def apply_same_day(
    dist: PredictiveDistribution,
    evidence: SameDayEvidence,
) -> tuple[PredictiveDistribution, list[str]]:
    factors: list[str] = []
    if evidence.stale or evidence.obs_max_f is None:
        factors.append(evidence.note or "same-day obs unavailable/stale — no truncation")
        return dist, factors
    new = truncate_below(dist, evidence.obs_max_f, reason=evidence.note)
    factors.append(
        f"Same-day floor {evidence.obs_max_f:.0f}°F from {evidence.source} "
        f"(preliminary evidence — not CLI settlement). Material change to support ≥ floor."
    )
    return new, factors


class ObservationClient:
    """Pull recent METAR max as progressive evidence via aviationweather.gov (public)."""

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=30.0, headers={"User-Agent": UA}, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def metar_recent_max_f(self, station: StationSpec) -> SameDayEvidence:
        now = datetime.now(timezone.utc)
        if not station.metar_ids:
            return SameDayEvidence(None, "none", now, "no METAR id configured", stale=True)
        # aviationweather.gov API — public METAR JSON
        ids = ",".join(station.metar_ids)
        url = f"https://aviationweather.gov/api/data/metar?ids={ids}&format=json&hours=18"
        try:
            r = self._http.get(url)
            r.raise_for_status()
            rows = r.json()
        except Exception as exc:
            return SameDayEvidence(None, "metar", now, f"METAR fetch failed: {exc}", stale=True)
        temps = []
        for row in rows or []:
            t = row.get("temp")
            if t is not None:
                # API often returns °C
                temps.append(float(t) * 9 / 5 + 32)
        if not temps:
            return SameDayEvidence(None, "metar", now, "no METAR temps in window", stale=True)
        mx = max(temps)
        return SameDayEvidence(
            obs_max_f=mx,
            source=f"aviationweather METAR {ids}",
            retrieved_at=now,
            note=f"18h METAR max≈{mx:.1f}°F (C→F); not CLI settlement",
            stale=False,
        )
