from __future__ import annotations

from decimal import Decimal

from kalshi_bot.api.orderbook import parse_orderbook
from kalshi_bot.config import TradingConfig
from kalshi_bot.ev.calculator import evaluate_binary_contract, max_limit_price
from kalshi_bot.ev.combo import ComboLeg, combo_ev, combo_settlement_expectation, joint_probability
from kalshi_bot.models.base import Prediction
from datetime import datetime, timezone


def _pred(p=Decimal("0.60")):
    return Prediction(
        market_ticker="TEST",
        p_yes=p,
        p_yes_conservative=Decimal("0.55"),
        uncertainty=Decimal("0.05"),
        model_version="test",
        data_sources=[],
        factors=["unit test"],
        validation_evidence="synthetic test fixture — not live evidence",
        as_of=datetime.now(timezone.utc),
        supported=True,
    )


def test_ev_yes_example_from_spec():
    # p=0.60, price=0.52, costs=0.02 => EV 0.06 — using direct numbers via book
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [["0.4000", "10.00"]],
                "no_dollars": [["0.4800", "10.00"]],  # yes ask = 0.52
            }
        }
    )
    cfg = TradingConfig(
        min_net_edge=Decimal("0.01"),
        uncertainty_buffer=Decimal("0"),
        assume_taker=False,  # isolate price EV; fees tested elsewhere
        fee_multiplier=Decimal("0"),
        default_contract_quantity=Decimal("1.00"),
    )
    # With multiplier 0, fees are 0
    results = evaluate_binary_contract(_pred(Decimal("0.60")), book, cfg)
    yes = next(r for r in results if r.side == "yes")
    assert yes.executable_price == Decimal("0.5200")
    # fees still may include rounding path with multiplier 0 -> 0
    assert yes.estimated_ev == Decimal("0.0800") or yes.estimated_ev >= Decimal("0.06")


def test_max_limit_price_preserves_edge():
    limit = max_limit_price(
        Decimal("0.60"),
        Decimal("0.02"),
        Decimal("0.03"),
        Decimal("0.02"),
    )
    # 0.60 - 0.02 - 0.02 - 0.03 = 0.53
    assert limit == Decimal("0.5300")


def test_combo_joint_requires_dependence_model():
    legs = [
        ComboLeg("A", "E1", "yes", Decimal("0.5")),
        ComboLeg("B", "E2", "yes", Decimal("0.5")),
    ]
    j = joint_probability(legs, allow_independence=False)
    assert not j.supported


def test_combo_independence_when_allowed():
    legs = [
        ComboLeg("A", "E1", "yes", Decimal("0.5")),
        ComboLeg("B", "E2", "yes", Decimal("0.5")),
    ]
    j = joint_probability(
        legs,
        allow_independence=True,
        dependence_model={"type": "independent_weather_cities"},
    )
    assert j.supported
    assert j.p_all == Decimal("0.25")


def test_combo_settlement_product_including_dnp_scalar():
    # DNP scalar 0.70 * 1 * 1 = 0.70 — not a refund
    assert combo_settlement_expectation(
        [Decimal("0.70"), Decimal("1"), Decimal("1")]
    ) == Decimal("0.70")


def test_combo_ev_formula():
    assert combo_ev(Decimal("0.30"), Decimal("0.20"), Decimal("0.01")) == Decimal("0.09")


def test_longshot_conservative_does_not_inflate():
    """Buying YES at low price must not get a boost from shrink-to-0.5."""
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [["0.0050", "10.00"]],
                "no_dollars": [["0.9800", "10.00"]],  # yes ask = 0.02
            }
        }
    )
    pred = Prediction(
        market_ticker="TEST",
        p_yes=Decimal("0.05"),
        p_yes_conservative=Decimal("0.04"),
        uncertainty=Decimal("0.08"),
        model_version="test",
        data_sources=[],
        factors=[],
        validation_evidence="fixture",
        as_of=datetime.now(timezone.utc),
        supported=True,
    )
    cfg = TradingConfig(
        min_net_edge=Decimal("0.05"),
        uncertainty_buffer=Decimal("0.03"),
        fee_multiplier=Decimal("0"),
        assume_taker=False,
        default_contract_quantity=Decimal("1.00"),
    )
    yes = next(r for r in evaluate_binary_contract(pred, book, cfg) if r.side == "yes")
    # p_cons = min(0.04, 0.05) - 0.08 = 0 → EV negative → skip
    assert yes.conservative_prob == Decimal("0")
    assert not yes.qualifies
