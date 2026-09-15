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


class ComboDiscoverer:
    def __init__(self, client: KalshiClient, config: CombosConfig) -> None:
        self.client = client
        self.config = config

    def list_collections(self) -> list[dict[str, Any]]:
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
        return out

    def build_candidates(
        self,
        predictions: list[Prediction],
        market_meta: dict[str, dict[str, Any]],
    ) -> list[ComboCandidate]:
        """Bound search of 2–N leg candidates from supported predictions.

        Only proposes independence across distinct weather cities when allowed.
        Same-city / same-game dependent legs are skipped without a joint model.
        """
        if not self.config.enabled:
            return []

        supported = [p for p in predictions if p.supported]
        candidates: list[ComboCandidate] = []

        # Group by city for weather independence heuristic
        by_city: dict[str, list[Prediction]] = {}
        for p in supported:
            city = (p.details or {}).get("city") or "unknown"
            by_city.setdefault(str(city), []).append(p)

        cities = list(by_city.keys())
        for r in range(2, min(self.config.max_legs, len(cities)) + 1):
            for city_group in itertools.combinations(cities, r):
                # Pick the single best-supported contract per city (highest |p-0.5| with lowest uncertainty)
                picks: list[Prediction] = []
                for c in city_group:
                    ranked = sorted(
                        by_city[c],
                        key=lambda x: (x.uncertainty, -abs(float(x.p_yes - 1) + float(x.p_yes))),
                    )
                    picks.append(ranked[0])
                legs = []
                for p in picks:
                    meta = market_meta.get(p.market_ticker) or {}
                    legs.append(
                        ComboLeg(
                            market_ticker=p.market_ticker,
                            event_ticker=meta.get("event_ticker") or "",
                            side="yes",
                            p_marginal=p.p_yes_conservative,
                            settlement_rules_note="DNP/partial settlement follows underlying market rules; combos are product of leg values",
                        )
                    )
                dependence = {
                    "type": "independent_weather_cities",
                    "cities": list(city_group),
                    "note": "Independence is an assumption — disabled unless config.allow_independence_assumption",
                }
                candidates.append(
                    ComboCandidate(
                        collection_ticker="",  # resolved when matching an open MVE collection
                        legs=legs,
                        dependence_model=dependence,
                    )
                )
                if len(candidates) >= self.config.max_candidates_per_scan:
                    return candidates

        # Explicitly record same-event multi-leg as unsupported without joint model
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
                            ComboLeg(
                                preds[0].market_ticker,
                                et,
                                "yes",
                                preds[0].p_yes_conservative,
                            ),
                            ComboLeg(
                                preds[1].market_ticker,
                                et,
                                "yes",
                                preds[1].p_yes_conservative,
                            ),
                        ],
                        dependence_model=None,
                        skip_reason=(
                            "same-event legs are dependent; no validated joint model — skip combo"
                        ),
                    )
                )
        return candidates[: self.config.max_candidates_per_scan]
