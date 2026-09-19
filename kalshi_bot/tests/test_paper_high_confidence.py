"""High-confidence paper policy unit tests."""

from __future__ import annotations

from kalshi_bot.models.weather.obs_engine.multi.paper_policy import (
    HighConfidencePaperPolicy,
    select_high_confidence_paper,
)
from kalshi_bot.money import D


def _eval(ticker, side, p, ask, ev, lo, hi):
    return {
        "ticker": ticker,
        "side": side,
        "p": str(p),
        "ask": str(ask),
        "ev": str(ev),
        "qty": "1",
        "fees": "0.01",
        "interval": {"op": "range_inclusive", "low": lo, "high": hi},
    }


def test_bans_no_even_if_high_p():
    evs = [
        _eval("T-B76.5", "no", "0.99", "0.10", "0.80", 76, 77),
        _eval("T-B78.5", "yes", "0.96", "0.50", "0.40", 78, 79),
    ]
    best, rejected = select_high_confidence_paper(
        evs,
        median_f=78.5,
        decision_hour_local=14,
        max_so_far=78.2,
        remain_q10_q50_q90=[0.0, 0.2, 1.0],
    )
    assert best is not None
    assert best["side"] == "yes"
    assert best["ticker"] == "T-B78.5"
    assert any(r.get("reject_class") == "yes_only_policy" for r in rejected)


def test_requires_p95_and_hour_14():
    evs = [_eval("T-B78.5", "yes", "0.96", "0.50", "0.40", 78, 79)]
    best, rejected = select_high_confidence_paper(
        evs,
        median_f=78.5,
        decision_hour_local=11,
        max_so_far=78.2,
        remain_q10_q50_q90=[0.0, 0.2, 1.0],
    )
    assert best is None
    assert any(r.get("reject_class") == "off_hour" for r in rejected)

    best2, _ = select_high_confidence_paper(
        [_eval("T-B78.5", "yes", "0.80", "0.50", "0.20", 78, 79)],
        median_f=78.5,
        decision_hour_local=14,
        max_so_far=78.2,
        remain_q10_q50_q90=[0.0, 0.2, 1.0],
    )
    assert best2 is None  # p < 0.95


def test_requires_max_so_far_in_bracket():
    evs = [_eval("T-B78.5", "yes", "0.97", "0.50", "0.40", 78, 79)]
    best, rejected = select_high_confidence_paper(
        evs,
        median_f=78.5,
        decision_hour_local=14,
        max_so_far=76.0,  # outside 78-79
        remain_q10_q50_q90=[0.0, 0.2, 1.0],
    )
    assert best is None
    assert any(r.get("reject_class") == "max_so_far_outside" for r in rejected)


def test_accepts_locked_modal_yes():
    pol = HighConfidencePaperPolicy(min_model_p=D("0.95"))
    evs = [_eval("T-B78.5", "yes", "0.97", "0.40", "0.50", 78, 79)]
    best, _ = select_high_confidence_paper(
        evs,
        median_f=78.4,
        decision_hour_local=14,
        max_so_far=78.1,
        remain_q10_q50_q90=[0.0, 0.1, 0.8],
        policy=pol,
    )
    assert best is not None
    assert best["side"] == "yes"
    assert best["selection_policy"] == pol.version
