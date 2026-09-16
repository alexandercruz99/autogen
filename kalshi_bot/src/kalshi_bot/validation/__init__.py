"""Validation utilities — walk-forward scaffolding.

Full historical Kalshi+NWS alignment requires collected forward observations.
This module provides metrics and leakage checks; it does not fabricate backtests.
"""

from kalshi_bot.validation.metrics import (
    assert_no_future_leakage,
    brier_score,
    log_loss,
    reliability_bins,
)

__all__ = [
    "assert_no_future_leakage",
    "brier_score",
    "log_loss",
    "reliability_bins",
]
