from __future__ import annotations

from decimal import Decimal

from kalshi_bot.api.orderbook import parse_orderbook


def test_derive_yes_ask_from_no_bid():
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [["0.4200", "13.00"]],
                "no_dollars": [["0.5600", "17.00"]],
            }
        }
    )
    assert book.best_yes_bid == Decimal("0.4200")
    assert book.best_yes_ask == Decimal("0.4400")  # 1 - 0.56
    assert book.best_no_ask == Decimal("0.5800")  # 1 - 0.42


def test_fillable_yes_walks_depth():
    book = parse_orderbook(
        {
            "orderbook_fp": {
                "yes_dollars": [["0.4000", "5.00"]],
                "no_dollars": [
                    ["0.5000", "2.00"],  # yes ask 0.50
                    ["0.4500", "3.00"],  # yes ask 0.55
                ],
            }
        }
    )
    # yes asks from no bids: 0.50 (size 2), 0.55 (size 3)
    qty, vwap = book.fillable_yes(Decimal("4.00"), max_price=Decimal("0.55"))
    assert qty == Decimal("4.00")
    # 2@0.50 + 2@0.55 = 2.10 / 4 = 0.525
    assert vwap == Decimal("0.5250")
