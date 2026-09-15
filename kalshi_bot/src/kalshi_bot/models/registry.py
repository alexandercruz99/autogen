from __future__ import annotations

from typing import Any

from kalshi_bot.models.base import ProbabilityModel


class ModelRegistry:
    def __init__(self) -> None:
        self._models: list[ProbabilityModel] = []

    def register(self, model: ProbabilityModel) -> None:
        self._models.append(model)

    def all(self) -> list[ProbabilityModel]:
        return list(self._models)

    def resolve(self, market: dict[str, Any], category: str) -> ProbabilityModel | None:
        for model in self._models:
            if model.supports(market, category):
                return model
        return None
