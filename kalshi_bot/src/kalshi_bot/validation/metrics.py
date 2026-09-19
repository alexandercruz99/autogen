from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from kalshi_bot.money import D, ZERO


def brier_score(probs: Iterable[Decimal], outcomes: Iterable[int]) -> Decimal:
    pairs = list(zip(probs, outcomes, strict=True))
    if not pairs:
        return ZERO
    s = sum((D(p) - D(o)) ** 2 for p, o in pairs)
    return s / D(len(pairs))


def log_loss(probs: Iterable[Decimal], outcomes: Iterable[int], eps: Decimal = D("1e-6")) -> Decimal:
    import math

    pairs = list(zip(probs, outcomes, strict=True))
    if not pairs:
        return ZERO
    total = 0.0
    for p, o in pairs:
        pp = min(max(float(p), float(eps)), 1.0 - float(eps))
        total += -(o * math.log(pp) + (1 - o) * math.log(1 - pp))
    return D(str(total / len(pairs)))


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    avg_pred: float
    avg_outcome: float


def reliability_bins(
    probs: list[float], outcomes: list[int], n_bins: int = 10
) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(probs) if (p >= lo and (p < hi or (i == n_bins - 1 and p <= hi)))]
        if not idx:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        avg_p = sum(probs[j] for j in idx) / len(idx)
        avg_o = sum(outcomes[j] for j in idx) / len(idx)
        bins.append(CalibrationBin(lo, hi, len(idx), avg_p, avg_o))
    return bins


def assert_no_future_leakage(decision_time_iso: str, feature_times: list[str]) -> None:
    """Raise if any feature timestamp is after the decision time."""
    from datetime import datetime

    decision = datetime.fromisoformat(decision_time_iso.replace("Z", "+00:00"))
    for ft in feature_times:
        t = datetime.fromisoformat(ft.replace("Z", "+00:00"))
        if t > decision:
            raise ValueError(f"future leakage: feature at {ft} after decision {decision_time_iso}")
