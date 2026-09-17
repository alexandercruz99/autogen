from __future__ import annotations

"""Chronological walk-forward evaluation scaffolding.

Does not fabricate Kalshi historical fills. Operates on provided decision records
that include timestamps and outcomes when known.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_bot.validation.metrics import assert_no_future_leakage, brier_score, log_loss, reliability_bins


@dataclass
class DecisionRecord:
    decision_time: str
    feature_times: list[str]
    p_model: Decimal
    p_market: Decimal | None
    outcome: int | None  # 1/0 if settled, else None
    net_pnl: Decimal | None = None
    model_version: str = ""
    config_id: str = ""


@dataclass
class WalkForwardReport:
    n_train: int
    n_holdout: int
    n_holdout_settled: int
    brier_holdout: Decimal | None
    log_loss_holdout: Decimal | None
    brier_market_holdout: Decimal | None
    calibration_bins: list[dict[str, Any]]
    configs_tried: list[str]
    meets_promotion_sample: bool
    notes: list[str] = field(default_factory=list)


def walk_forward_split(
    records: list[DecisionRecord],
    *,
    holdout_fraction: float = 0.2,
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    ordered = sorted(records, key=lambda r: r.decision_time)
    if not ordered:
        return [], []
    cut = max(1, int(len(ordered) * (1 - holdout_fraction)))
    if cut >= len(ordered):
        cut = len(ordered) - 1 if len(ordered) > 1 else 0
    return ordered[:cut], ordered[cut:]


def evaluate_holdout(
    holdout: list[DecisionRecord],
    *,
    configs_tried: list[str],
    min_sample: int = 100,
) -> WalkForwardReport:
    notes: list[str] = []
    for r in holdout:
        assert_no_future_leakage(r.decision_time, r.feature_times)

    settled = [r for r in holdout if r.outcome is not None]
    if not settled:
        return WalkForwardReport(
            n_train=0,
            n_holdout=len(holdout),
            n_holdout_settled=0,
            brier_holdout=None,
            log_loss_holdout=None,
            brier_market_holdout=None,
            calibration_bins=[],
            configs_tried=configs_tried,
            meets_promotion_sample=False,
            notes=["No settled holdout outcomes yet — forward paper collection required"],
        )

    probs = [r.p_model for r in settled]
    outcomes = [int(r.outcome) for r in settled]
    brier = brier_score(probs, outcomes)
    ll = log_loss(probs, outcomes)
    market_brier = None
    with_mkt = [r for r in settled if r.p_market is not None]
    if with_mkt:
        market_brier = brier_score([r.p_market for r in with_mkt], [int(r.outcome) for r in with_mkt])

    bins = [
        {"lo": b.lo, "hi": b.hi, "count": b.count, "avg_pred": b.avg_pred, "avg_outcome": b.avg_outcome}
        for b in reliability_bins([float(p) for p in probs], outcomes)
    ]
    meets = len(settled) >= min_sample
    if not meets:
        notes.append(f"Holdout settled n={len(settled)} < promotion minimum {min_sample}")
    notes.append(f"Configs/models logged as tried: {configs_tried}")
    return WalkForwardReport(
        n_train=0,
        n_holdout=len(holdout),
        n_holdout_settled=len(settled),
        brier_holdout=brier,
        log_loss_holdout=ll,
        brier_market_holdout=market_brier,
        calibration_bins=bins,
        configs_tried=configs_tried,
        meets_promotion_sample=meets,
        notes=notes,
    )
