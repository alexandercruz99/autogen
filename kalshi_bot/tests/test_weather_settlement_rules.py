from __future__ import annotations

from kalshi_bot.models.weather.settlement_rules import interval_from_market, yes_from_observation


def test_interval_from_market_greater_less_between():
    g = interval_from_market({"strike_type": "greater", "floor_strike": 82, "rules_primary": "CLINYC"})
    assert g and g.op == "gt" and g.low == 82
    assert yes_from_observation(83, g) is True
    assert yes_from_observation(82, g) is False

    less = interval_from_market({"strike_type": "less", "cap_strike": 75})
    assert less and yes_from_observation(74, less) is True
    assert yes_from_observation(75, less) is False

    band = interval_from_market({"strike_type": "between", "floor_strike": 79, "cap_strike": 80})
    assert band and yes_from_observation(79, band) and yes_from_observation(80, band)
    assert yes_from_observation(81, band) is False
