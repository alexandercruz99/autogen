from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from kalshi_bot.models.weather.cli_reports import parse_cli_product_text
from kalshi_bot.models.weather.distribution import (
    PredictiveDistribution,
    assert_brackets_sum_near_one,
    coherent_bracket_probabilities,
    from_empirical_residuals,
    from_normal,
)
from kalshi_bot.models.weather.settlement_rules import TempInterval
from kalshi_bot.models.weather.stations import station_for_ticker
from kalshi_bot.validation.metrics import assert_no_future_leakage


SAMPLE_CLI = """
...THE CENTRAL PARK NY CLIMATE SUMMARY FOR SEPTEMBER 14 2026...

TEMPERATURE (F)
 TODAY
  MAXIMUM         75    300 PM  90    1927  77     -2       79
  MINIMUM         60    600 AM  44    1873  63     -3       66
"""


def test_parse_cli_maximum():
    issued = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    r = parse_cli_product_text(SAMPLE_CLI, station_cli_id="NYC", product_id="x", issuance_time=issued)
    assert r.climate_day.isoformat() == "2026-09-14"
    assert r.max_temp_f == 75
    assert r.min_temp_f == 60


def test_station_registry_nyc():
    s = station_for_ticker("KXHIGHNY-26SEP15-T70")
    assert s is not None
    assert s.cli_location_id == "NYC"
    assert "NWS" in s.settlement_source


def test_empirical_distribution_and_brackets_sum():
    dist = from_empirical_residuals(72.0, [-2, -1, 0, 0, 1, 2, 3])
    assert abs(sum(dist.probs) - 1.0) < 1e-6
    intervals = [
        TempInterval("lt", None, 70.0, "t"),
        TempInterval("range_inclusive", 70.0, 72.0, "t"),
        TempInterval("range_inclusive", 73.0, 75.0, "t"),
        TempInterval("gt", 75.0, None, "t"),
    ]
    # Not necessarily exhaustive ME for these intervals — build exhaustive integer partition:
    intervals = [
        TempInterval("lt", None, 70.0, "t"),
        TempInterval("range_inclusive", 70.0, 74.0, "t"),
        TempInterval("gt", 74.0, None, "t"),
    ]
    probs = coherent_bracket_probabilities(dist, intervals)
    assert_brackets_sum_near_one(probs, tol=Decimal("0.05"))


def test_continuity_normal_not_fifty_at_strike():
    dist = from_normal(82.0, 3.0, support_low=70, support_high=95)
    iv = TempInterval("gt", 82.0, None, "t")
    p = dist.p_interval(iv)
    assert p < Decimal("0.5")


def test_leakage_guard_still_enforced():
    assert_no_future_leakage("2026-09-15T12:00:00+00:00", ["2026-09-15T11:00:00+00:00"])
