from __future__ import annotations

import pytest

from kalshi_bot.models.weather.nws_client import parse_market_date, parse_temp_contract
from kalshi_bot.validation.metrics import assert_no_future_leakage, brier_score
from kalshi_bot.money import D


def test_parse_bucket_and_threshold():
    c = parse_temp_contract("KXHIGHNY-26SEP16-B79.5", "Will the maximum temperature be 79-80°")
    assert c is not None
    assert c["op"] == "range"
    assert c["low"] == 79.0
    assert c["high"] == 81.0

    c2 = parse_temp_contract("KXHIGHNY-26SEP16-T82", "Will the maximum temperature be >82°")
    assert c2["op"] == "gt"
    assert c2["low"] == 82.0


def test_parse_date():
    assert str(parse_market_date("KXHIGHNY-26SEP16-B79.5")) == "2026-09-16"


def test_ambiguous_threshold_refused():
    assert parse_temp_contract("KXHIGHNY-26SEP16-T82", "Temperature market T82") is None


def test_leakage_guard():
    assert_no_future_leakage("2026-09-15T12:00:00+00:00", ["2026-09-15T11:00:00+00:00"])
    with pytest.raises(ValueError):
        assert_no_future_leakage("2026-09-15T12:00:00+00:00", ["2026-09-15T13:00:00+00:00"])


def test_brier():
    score = brier_score([D("0.7"), D("0.2")], [1, 0])
    assert score == D("0.065")
