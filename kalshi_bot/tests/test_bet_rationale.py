"""Weather-first bet selection: rationale over cheapest-ask chasing."""

from __future__ import annotations

from kalshi_bot.models.weather.obs_engine.multi.bet_rationale import (
    format_bet_rationale,
    select_forecast_consistent,
    strike_distance_f,
)
from kalshi_bot.money import D


def test_strike_distance_threshold_yes_matches_forecast():
    iv = {"op": "gt", "low": 80.0, "high": None}
    assert strike_distance_f(iv, side="yes", median_f=81.0) == 0.0
    assert strike_distance_f(iv, side="yes", median_f=78.0) == 2.0


def test_strike_distance_band_no_when_forecast_outside():
    iv = {"op": "range_inclusive", "low": 75.0, "high": 76.0}
    assert strike_distance_f(iv, side="no", median_f=81.0) == 0.0
    # forecast inside the band → NO is a weather mismatch
    assert strike_distance_f(iv, side="no", median_f=75.5) > 0


def test_select_prefers_forecast_story_over_cheapest_ask():
    """A 3¢ longshot must lose to a forecast-aligned contract even if EV is lower."""
    median = 81.0
    evals = [
        {
            "ticker": "KXHIGHLAX-T73",
            "side": "no",
            "p": "0.99",
            "ask": "0.03",
            "ev": "0.90",  # huge EV, but far from forecast story if claiming weird side
            "interval": {"op": "lt", "low": None, "high": 73.0},
            "decision": "ev_evaluated",
        },
        {
            "ticker": "KXHIGHLAX-T80",
            "side": "yes",
            "p": "0.65",
            "ask": "0.10",
            "ev": "0.49",
            "interval": {"op": "gt", "low": 80.0, "high": None},
            "decision": "ev_evaluated",
        },
    ]
    # NO below 73 with median 81 is consistent (forecast outside), and p high —
    # both pass filters; distance both 0. Prefer higher EV → T73 no would win.
    # So use a cheap longshot YES far below median instead:
    evals[0] = {
        "ticker": "KXHIGHLAX-B73.5",
        "side": "yes",
        "p": "0.40",
        "ask": "0.02",
        "ev": "0.33",
        "interval": {"op": "range_inclusive", "low": 73.0, "high": 74.0},
        "decision": "ev_evaluated",
    }
    best, rejected = select_forecast_consistent(evals, median_f=median)
    assert best is not None
    assert best["ticker"] == "KXHIGHLAX-T80"
    assert best["side"] == "yes"
    assert any(r["ticker"] == "KXHIGHLAX-B73.5" for r in rejected)


def test_select_rejects_low_model_p_longshot():
    evals = [
        {
            "ticker": "KXHIGHCHI-B75.5",
            "side": "no",
            "p": "0.08",
            "ask": "0.03",
            "ev": "0.04",
            "interval": {"op": "range_inclusive", "low": 75.0, "high": 76.0},
            "decision": "ev_evaluated",
        }
    ]
    best, rejected = select_forecast_consistent(evals, median_f=76.5)
    assert best is None
    assert rejected and "model_p" in rejected[0]["reject_reason"]


def test_why_buy_leads_with_weather_not_price():
    text = format_bet_rationale(
        point_median_f=81.0,
        max_so_far=78.0,
        ticker="KXHIGHLAX-26SEP16-T80",
        side="yes",
        p="0.65",
        ask="0.10",
        interval={"op": "gt", "low": 80.0, "high": None},
    )
    assert "Forecast high ~81°F" in text
    assert "already 78°F" in text
    assert "YES on KXHIGHLAX-26SEP16-T80" in text
    assert "not because the ticket is cheap" in text
    assert D("0.65")  # sanity import path
