from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.config import CombosConfig
from kalshi_bot.ev.combo import ComboLeg
from kalshi_bot.models.base import Prediction

logger = logging.getLogger(__name__)


@dataclass
class ComboCandidate:
    collection_ticker: str
    legs: list[ComboLeg]
    dependence_model: dict[str, Any] | None
    skip_reason: str | None = None
    eligible: bool = False


class ComboDiscoverer:
    def __init__(self, client: KalshiClient, config: CombosConfig) -> None:
        self.client = client
        self.config = config
        self._eligible_event_tickers: set[str] | None = None
        self._collections_cache: list[dict[str, Any]] | None = None

    def list_collections(self) -> list[dict[str, Any]]:
        if self._collections_cache is not None:
            return self._collections_cache
        out: list[dict[str, Any]] = []
        cursor = None
        while True:
            try:
                payload = self.client.get_multivariate_collections(status="open", limit=100, cursor=cursor)
            except Exception as exc:
                logger.warning("MVE collections fetch failed: %s", exc)
                break
            batch = payload.get("multivariate_contracts") or payload.get("collections") or []
            out.extend(batch)
            cursor = payload.get("cursor") or None
            if not cursor or not batch:
                break
        self._collections_cache = out
        return out

    def eligible_event_tickers(self) -> set[str]:
        if self._eligible_event_tickers is not None:
            return self._eligible_event_tickers
        tickers: set[str] = set()
        for c in self.list_collections():
            for e in c.get("associated_events") or []:
                if isinstance(e, dict) and e.get("ticker"):
                    tickers.add(e["ticker"])
            for t in c.get("associated_event_tickers") or []:
                tickers.add(t)
        self._eligible_event_tickers = tickers
        return tickers

    def collection_for_events(self, event_tickers: list[str]) -> dict[str, Any] | None:
        needed = set(event_tickers)
        for c in self.list_collections():
            have = set(c.get("associated_event_tickers") or [])
            for e in c.get("associated_events") or []:
                if isinstance(e, dict) and e.get("ticker"):
                    have.add(e["ticker"])
            if needed.issubset(have):
                size_min = c.get("size_min") or 2
                size_max = c.get("size_max") or 0
                n = len(needed)
                if n < size_min:
                    continue
                if size_max and n > size_max:
                    continue
                return c
        return None

    def build_candidates(
        self,
        predictions: list[Prediction],
        market_meta: dict[str, dict[str, Any]],
    ) -> list[ComboCandidate]:
        if not self.config.enabled:
            return []

        eligible_events = self.eligible_event_tickers()
        supported = [p for p in predictions if p.supported]
        candidates: list[ComboCandidate] = []

        # Eligibility report for modeled markets
        for p in supported:
            meta = market_meta.get(p.market_ticker) or {}
            et = meta.get("event_ticker") or ""
            if et and et not in eligible_events:
                candidates.append(
                    ComboCandidate(
                        collection_ticker="",
                        legs=[
                            ComboLeg(
                                p.market_ticker,
                                et,
                                "yes",
                                p.p_yes_conservative,
                            )
                        ],
                        dependence_model=None,
                        skip_reason=(
                            f"event {et} not in any open MVE collection associated_events "
                            "(verified via API — cannot form Kalshi combo)"
                        ),
                        eligible=False,
                    )
                )

        by_city: dict[str, list[Prediction]] = {}
        for p in supported:
            city = (p.details or {}).get("city") or "unknown"
            by_city.setdefault(str(city), []).append(p)

        cities = [c for c in by_city if c != "unknown"]
        for r in range(2, min(self.config.max_legs, len(cities)) + 1):
            for city_group in itertools.combinations(cities, r):
                picks: list[Prediction] = []
                for c in city_group:
                    ranked = sorted(by_city[c], key=lambda x: x.uncertainty)
                    picks.append(ranked[0])
                legs = []
                events = []
                for p in picks:
                    meta = market_meta.get(p.market_ticker) or {}
                    et = meta.get("event_ticker") or ""
                    events.append(et)
                    legs.append(
                        ComboLeg(
                            market_ticker=p.market_ticker,
                            event_ticker=et,
                            side="yes",
                            p_marginal=p.p_yes_conservative,
                            settlement_rules_note=(
                                "DNP/partial follows underlying; combo payout = product of leg values"
                            ),
                        )
                    )
                coll = self.collection_for_events(events) if all(events) else None
                if coll is None:
                    candidates.append(
                        ComboCandidate(
                            collection_ticker="",
                            legs=legs,
                            dependence_model={
                                "type": "independent_weather_cities",
                                "cities": list(city_group),
                            },
                            skip_reason=(
                                "weather city legs not jointly present in an open MVE collection; "
                                "infrastructure ready but combo not exchange-eligible now"
                            ),
                            eligible=False,
                        )
                    )
                else:
                    candidates.append(
                        ComboCandidate(
                            collection_ticker=coll.get("collection_ticker") or "",
                            legs=legs,
                            dependence_model={
                                "type": "independent_weather_cities",
                                "cities": list(city_group),
                                "note": "Distinct cities — independence optional via config",
                            },
                            eligible=True,
                        )
                    )
                if len(candidates) >= self.config.max_candidates_per_scan:
                    return candidates

        # Same-event dependent legs without joint model
        by_event: dict[str, list[Prediction]] = {}
        for p in supported:
            meta = market_meta.get(p.market_ticker) or {}
            et = meta.get("event_ticker") or ""
            if et:
                by_event.setdefault(et, []).append(p)
        for et, preds in by_event.items():
            if len(preds) >= 2:
                candidates.append(
                    ComboCandidate(
                        collection_ticker="",
                        legs=[
                            ComboLeg(preds[0].market_ticker, et, "yes", preds[0].p_yes_conservative),
                            ComboLeg(preds[1].market_ticker, et, "yes", preds[1].p_yes_conservative),
                        ],
                        dependence_model=None,
                        skip_reason=(
                            "same-event legs are dependent/mutually exclusive risk; "
                            "no validated joint model — skip combo"
                        ),
                        eligible=False,
                    )
                )
        return candidates[: self.config.max_candidates_per_scan]
