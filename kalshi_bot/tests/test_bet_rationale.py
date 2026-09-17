"""Data-first bet selection with explicit rejection classes."""

from __future__ import annotations

from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
    DEFAULT_MIN_MODEL_P,
    format_bet_rationale,
    select_forecast_consistent,
    strike_distance_f,
)
from kalshi_bot.money import D


def test_min_model_p_preference_default():
    assert DEFAULT_MIN_MODEL_P >= D("0.55")


def test_strike_distance_threshold_yes_matches_forecast():
    iv = {"op": "gt", "low": 80.0, "high": None}
    assert strike_distance_f(iv, side="yes", median_f=81.0) == 0.0
    assert strike_distance_f(iv, side="yes", median_f=78.0) == 2.0


def test_strike_distance_band_no_when_forecast_outside():
    iv = {"op": "range_inclusive", "low": 75.0, "high": 76.0}
    assert strike_distance_f(iv, side="no", median_f=81.0) == 0.0
    assert strike_distance_f(iv, side="no", median_f=75.5) > 0


def test_select_prefers_high_confidence_over_cheap_lottery_ev():
    """63¢ with 80% model confidence beats 10¢ with 56% even if EV is smaller."""
    median = 81.0
    evals = [
        {
            "ticker": "KXHIGHLAX-T80",
            "side": "yes",
            "p": "0.56",
            "ask": "0.10",
            "ev": "0.40",
            "interval": {"op": "gt", "low": 80.0, "high": None},
            "decision": "ev_evaluated",
        },
        {
            "ticker": "KXHIGHLAX-B81",
            "side": "yes",
            "p": "0.80",
            "ask": "0.63",
            "ev": "0.12",
            "interval": {"op": "range_inclusive", "low": 81.0, "high": 81.0},
            "decision": "ev_evaluated",
        },
    ]
    best, _rejected = select_forecast_consistent(evals, median_f=median)
    assert best is not None
    assert best["ticker"] == "KXHIGHLAX-B81"
    assert best["ask"] == "0.63"


def test_select_rejects_far_band_even_if_ev_huge():
    median = 81.0
    evals = [
        {
            "ticker": "KXHIGHLAX-B73.5",
            "side": "yes",
            "p": "0.60",
            "ask": "0.02",
            "ev": "0.53",
            "interval": {"op": "range_inclusive", "low": 73.0, "high": 74.0},
            "decision": "ev_evaluated",
        },
        {
            "ticker": "KXHIGHLAX-T80",
            "side": "yes",
            "p": "0.65",
            "ask": "0.50",
            "ev": "0.10",
            "interval": {"op": "gt", "low": 80.0, "high": None},
            "decision": "ev_evaluated",
        },
    ]
    best, rejected = select_forecast_consistent(evals, median_f=median)
    assert best is not None
    assert best["ticker"] == "KXHIGHLAX-T80"
    far = next(r for r in rejected if r["ticker"] == "KXHIGHLAX-B73.5")
    assert far["reject_class"] == "weather_alignment"


def test_select_rejects_low_model_p_as_preference_not_no_edge():
    evals = [
        {
            "ticker": "KXHIGHCHI-B75.5",
            "side": "no",
            "p": "0.32",
            "ask": "0.03",
            "ev": "0.24",
            "interval": {"op": "range_inclusive", "low": 75.0, "high": 76.0},
            "decision": "ev_evaluated",
        }
    ]
    best, rejected = select_forecast_consistent(evals, median_f=76.5)
    assert best is None
    assert rejected[0]["reject_class"] == "preference_min_probability"
    assert "preference" in rejected[0]["reject_reason"]
    assert "no edge" not in rejected[0]["reject_reason"].lower() or "not labeled" in rejected[0][
        "reject_reason"
    ]


def test_inadequate_ev_distinct_from_preference():
    evals = [
        {
            "ticker": "KXHIGHNY-T70",
            "side": "yes",
            "p": "0.70",
            "ask": "0.69",
            "ev": "0.005",
            "interval": {"op": "gt", "low": 70.0, "high": None},
            "decision": "ev_evaluated",
        }
    ]
    best, rejected = select_forecast_consistent(evals, median_f=71.0)
    assert best is None
    assert rejected[0]["reject_class"] == "inadequate_ev"


def test_why_buy_states_uncertainty_not_growth_claims():
    text = format_bet_rationale(
        point_median_f=81.0,
        max_so_far=78.0,
        ticker="KXHIGHLAX-26SEP16-B81",
        side="yes",
        p="0.80",
        ask="0.63",
        interval={"op": "range_inclusive", "low": 81.0, "high": 81.0},
        conservative_ev="0.12",
    )
    assert "80.0%" in text or "80%" in text
    assert "63.0¢" in text or "63¢" in text
    assert "steady growth" not in text.lower()
    assert "high-confidence compounding" not in text.lower()
    assert "Point forecast ≠ calibrated probability" in text
