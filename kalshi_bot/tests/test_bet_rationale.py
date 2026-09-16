"""Data-first bet selection: high confidence compounds over lottery EV."""

from __future__ import annotations

from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
    MIN_MODEL_P,
    format_bet_rationale,
    select_forecast_consistent,
    strike_distance_f,
)
from kalshi_bot.money import D


def test_min_model_p_is_high_conviction():
    assert MIN_MODEL_P >= D("0.55")


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
            "ev": "0.40",  # bigger EV number
            "interval": {"op": "gt", "low": 80.0, "high": None},
            "decision": "ev_evaluated",
        },
        {
            "ticker": "KXHIGHLAX-B81",
            "side": "yes",
            "p": "0.80",
            "ask": "0.63",
            "ev": "0.12",  # smaller EV, higher win rate
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
    assert any(r["ticker"] == "KXHIGHLAX-B73.5" for r in rejected)


def test_select_rejects_low_model_p_longshot():
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
    assert rejected and "model_p" in rejected[0]["reject_reason"]


def test_why_buy_is_data_and_conviction_not_payout():
    text = format_bet_rationale(
        point_median_f=81.0,
        max_so_far=78.0,
        ticker="KXHIGHLAX-26SEP16-B81",
        side="yes",
        p="0.80",
        ask="0.63",
        interval={"op": "range_inclusive", "low": 81.0, "high": 81.0},
    )
    assert "data pick" in text
    assert "80%" in text
    assert "63¢" in text
    assert "37¢ profit if right" in text
    assert "not the cheapest ticket" in text
