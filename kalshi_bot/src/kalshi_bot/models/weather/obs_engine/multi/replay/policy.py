"""Versioned deterministic picker policy (frozen before final test)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from kalshi_bot.money import D


POLICY_VERSION = "picker.policy.v1"


@dataclass(frozen=True)
class PickerPolicy:
    """Frozen selection rules — do not retune against final test."""

    version: str = POLICY_VERSION
    evaluation_quantity: int = 1
    min_expected_net_profit_per_contract: str = "0.02"  # Decimal string
    max_strike_distance_f: float = 4.0
    apply_min_probability_preference: bool = True
    min_model_p_preference: str = "0.55"
    max_ask: str = "0.95"
    min_ask: str = "0.01"
    assume_taker_fees: bool = True
    fee_multiplier: str = "1"
    max_new_trades_per_event: int = 1
    score_quantize_places: int = 6  # quantize ENP before tie-break
    live_eligible: bool = False
    notes: tuple[str, ...] = (
        "Rank by expected net profit per contract at evaluation_quantity after fees.",
        "Ties: higher purchased-side probability, then ticker, then side (YES before NO).",
        "Most-likely bracket is reported separately from the purchased contract.",
        "No arbitrary probability haircut; preference min_p is a filter not a calibration claim.",
        "Research/paper only — live_eligible remains false.",
    )

    def min_enp(self) -> Decimal:
        return D(self.min_expected_net_profit_per_contract)

    def min_p(self) -> Decimal:
        return D(self.min_model_p_preference)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


DEFAULT_PICKER_POLICY = PickerPolicy()
