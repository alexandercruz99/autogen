from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass
class Prediction:
    """Evidence-based probability estimate. Never invent values outside a real model."""

    market_ticker: str
    p_yes: Decimal
    p_yes_conservative: Decimal
    uncertainty: Decimal
    model_version: str
    data_sources: list[dict[str, Any]]
    factors: list[str]
    validation_evidence: str
    as_of: datetime
    supported: bool = True
    skip_reason: str | None = None
    expected_settlement_yes: Decimal | None = None  # for non-binary / scalar
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def p_no(self) -> Decimal:
        return Decimal("1") - self.p_yes

    @property
    def p_no_conservative(self) -> Decimal:
        # Prefer stressed high YES ⇒ lower NO when available in details.
        high = self.details.get("p_yes_high") if self.details else None
        if high is not None:
            return Decimal("1") - Decimal(str(high))
        return Decimal("1") - self.p_yes_conservative


class ProbabilityModel(ABC):
    name: str
    version: str
    categories: list[str]

    @abstractmethod
    def supports(self, market: dict[str, Any], category: str) -> bool:
        ...

    @abstractmethod
    def predict(self, market: dict[str, Any], category: str) -> Prediction:
        ...
