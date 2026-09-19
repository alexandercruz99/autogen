"""Session spend ledger: $5/bet cannot exceed $20 overall."""

from __future__ import annotations

from decimal import Decimal

from kalshi_bot.models.weather.obs_engine.multi.session_budget import (
    SessionBudgetPolicy,
    commit,
    release,
    remaining,
    reserve,
    snapshot,
)


def test_session_budget_caps_at_20(tmp_path):
    pol = SessionBudgetPolicy(session_id="test-20", max_spend=Decimal("20"), per_bet_max=Decimal("5"))
    root = tmp_path
    assert remaining("test-20", policy=pol, root=root) == Decimal("20")

    for i, series in enumerate(["KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHNY"]):
        r = reserve(5, series_ticker=series, session_id="test-20", policy=pol, root=root)
        assert r["ok"], r
        c = commit(r["reservation_id"], actual_spend=5, session_id="test-20", policy=pol, root=root)
        assert c["ok"], c

    assert remaining("test-20", policy=pol, root=root) == Decimal("0")
    blocked = reserve(5, series_ticker="KXHIGHNY", session_id="test-20", policy=pol, root=root)
    assert not blocked["ok"]
    assert "exhausted" in blocked["error"]


def test_per_bet_max_and_allowlist(tmp_path):
    pol = SessionBudgetPolicy(session_id="test-allow", max_spend=Decimal("20"), per_bet_max=Decimal("5"))
    bad_amt = reserve(6, series_ticker="KXHIGHNY", session_id="test-allow", policy=pol, root=tmp_path)
    assert not bad_amt["ok"]
    bad_series = reserve(5, series_ticker="KXHIGHMIA", session_id="test-allow", policy=pol, root=tmp_path)
    assert not bad_series["ok"]


def test_release_returns_budget(tmp_path):
    pol = SessionBudgetPolicy(session_id="test-rel", max_spend=Decimal("20"), per_bet_max=Decimal("5"))
    r = reserve(5, series_ticker="KXHIGHLAX", session_id="test-rel", policy=pol, root=tmp_path)
    assert r["ok"]
    assert remaining("test-rel", policy=pol, root=tmp_path) == Decimal("15")
    release(r["reservation_id"], session_id="test-rel", policy=pol, root=tmp_path, reason="no_fill")
    assert remaining("test-rel", policy=pol, root=tmp_path) == Decimal("20")
    snap = snapshot("test-rel", policy=pol, root=tmp_path)
    assert snap["spent"] == "0"
    assert snap["reserved"] == "0"
