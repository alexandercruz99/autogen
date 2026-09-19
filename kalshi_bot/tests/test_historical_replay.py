"""Tests for historical replay scoring and deterministic pick_contract."""

from __future__ import annotations

from decimal import Decimal

import pytest

from kalshi_bot.models.weather.distribution import PredictiveDistribution
from kalshi_bot.models.weather.obs_engine.multi.replay.execution_sim import execute_decision_simulated
from kalshi_bot.models.weather.obs_engine.multi.replay.picker import pick_contract
from kalshi_bot.models.weather.obs_engine.multi.replay.policy import DEFAULT_PICKER_POLICY
from kalshi_bot.models.weather.obs_engine.multi.replay.scoring import crps_pmf, modal_degree
from kalshi_bot.models.weather.settlement_rules import TempInterval


def test_crps_hand_calculated_two_point():
    # p(70)=0.5, p(72)=0.5, y=70
    # term1 = 0.5*0 + 0.5*2 = 1
    # term2 = 0.5 * (0.5*0.5*2 + 0.5*0.5*2) = 0.5 * 1 = 0.5
    # CRPS = 1 - 0.5 = 0.5
    assert crps_pmf([70, 72], [0.5, 0.5], 70) == pytest.approx(0.5)


def test_crps_point_mass_zero():
    assert crps_pmf([75], [1.0], 75) == pytest.approx(0.0)


def test_modal_tie_break_lower_temp():
    dist = PredictiveDistribution(temps_f=[70, 71], probs=[0.5, 0.5], method="t")
    assert modal_degree(dist) == 70


def test_pick_contract_no_book_is_no_trade():
    dist = PredictiveDistribution(temps_f=[75, 76, 77], probs=[0.2, 0.5, 0.3], method="t")
    contracts = [
        {
            "ticker": "T-B75.5",
            "status": "open",
            "interval": {"op": "range_inclusive", "low": 75, "high": 76},
            "climate_day": "2026-09-01",
        }
    ]
    d = pick_contract(dist, contracts, None, {"cash": "100"}, "2026-09-01T18:00:00+00:00")
    assert d.kind == "NO_TRADE"
    assert "missing_orderbook_snapshot" in d.reasons


def test_pick_contract_deterministic_under_shuffle():
    dist = PredictiveDistribution(
        temps_f=list(range(70, 81)),
        probs=[0.02] * 4 + [0.3, 0.3, 0.2] + [0.02] * 4,
        method="t",
    )
    contracts = [
        {
            "ticker": "T-B74.5",
            "status": "open",
            "interval": {"op": "range_inclusive", "low": 74, "high": 75},
            "climate_day": "2026-09-01",
        },
        {
            "ticker": "T-B76.5",
            "status": "open",
            "interval": {"op": "range_inclusive", "low": 76, "high": 77},
            "climate_day": "2026-09-01",
        },
        {
            "ticker": "T-B78.5",
            "status": "open",
            "interval": {"op": "range_inclusive", "low": 78, "high": 79},
            "climate_day": "2026-09-01",
        },
    ]
    books = {
        "T-B74.5": {"yes_ask": "0.40", "no_ask": "0.65"},
        "T-B76.5": {"yes_ask": "0.35", "no_ask": "0.70"},
        "T-B78.5": {"yes_ask": "0.20", "no_ask": "0.85"},
    }
    acct = {
        "cash": "100",
        "open_notional": "0",
        "reserved": "0",
        "max_total_exposure": "80",
        "max_per_market_exposure": "25",
    }
    d1 = pick_contract(dist, contracts, books, acct, "2026-09-01T18:00:00+00:00", point_median_f=76.0)
    d2 = pick_contract(
        dist,
        list(reversed(contracts)),
        books,
        acct,
        "2026-09-01T18:00:00+00:00",
        point_median_f=76.0,
    )
    assert d1.kind == d2.kind
    assert d1.ticker == d2.ticker
    assert d1.side == d2.side
    assert d1.limit_price == d2.limit_price


def test_pick_contract_depth_and_fees_can_kill_edge():
    dist = PredictiveDistribution(temps_f=[80], probs=[1.0], method="t")
    contracts = [
        {
            "ticker": "T-B80",
            "status": "open",
            "interval": {"op": "range_inclusive", "low": 80, "high": 80},
        }
    ]
    # Extremely expensive ask → inadequate ENP
    books = {"T-B80": {"yes_ask": "0.99", "no_ask": "0.99"}}
    d = pick_contract(
        dist,
        contracts,
        books,
        {"cash": "100", "max_total_exposure": "80", "max_per_market_exposure": "25"},
        "2026-09-01T18:00:00+00:00",
        point_median_f=80.0,
    )
    assert d.kind == "NO_TRADE"


def test_execution_sim_never_submits_live():
    dist = PredictiveDistribution(temps_f=[75], probs=[1.0], method="t")
    d = pick_contract(dist, [], {}, {"cash": "100"}, "t")
    result = execute_decision_simulated(d)
    assert result["live_order_submitted"] is False
    assert result["submitted"] is False


def test_interval_endpoints_open_ended():
    from kalshi_bot.models.weather.settlement_rules import yes_from_observation

    dist = PredictiveDistribution(temps_f=[70, 80, 90], probs=[0.2, 0.5, 0.3], method="t")
    gt = TempInterval("gt", 85.0, None, "t")
    lt = TempInterval("lt", None, 75.0, "t")
    assert yes_from_observation(90.0, gt)
    assert not yes_from_observation(80.0, gt)
    assert yes_from_observation(70.0, lt)
    assert not yes_from_observation(80.0, lt)
    # probability sums
    from kalshi_bot.models.weather.obs_engine.multi.replay.scoring import yes_probability

    assert yes_probability(dist, gt) == pytest.approx(0.3)
    assert yes_probability(dist, lt) == pytest.approx(0.2)


def test_policy_version_frozen():
    assert DEFAULT_PICKER_POLICY.version == "picker.policy.v1"
    assert DEFAULT_PICKER_POLICY.live_eligible is False
