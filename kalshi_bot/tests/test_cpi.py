from __future__ import annotations

from kalshi_bot.models.economics.cpi import mom_series, parse_cpi_market


def test_parse_cpi_ticker():
    c = parse_cpi_market("KXCPI-26SEP-T0.3", "Will CPI rise more than 0.3% in September 2026?")
    assert c is not None
    assert c.year == 2026 and c.month == 9 and c.threshold_pct == 0.3 and c.op == "gt"


def test_mom_series():
    levels = [
        {"year": 2024, "month": 1, "value": 100.0, "bls_period": "2024M01"},
        {"year": 2024, "month": 2, "value": 100.5, "bls_period": "2024M02"},
    ]
    moms = mom_series(levels)
    assert len(moms) == 1
    assert abs(moms[0]["mom_pct"] - 0.5) < 1e-9
