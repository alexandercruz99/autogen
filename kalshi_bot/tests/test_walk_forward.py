from __future__ import annotations

from decimal import Decimal

from kalshi_bot.validation.walk_forward import DecisionRecord, evaluate_holdout, walk_forward_split


def test_walk_forward_split_and_metrics():
    records = [
        DecisionRecord(
            decision_time=f"2026-01-{i:02d}T12:00:00+00:00",
            feature_times=[f"2026-01-{i:02d}T11:00:00+00:00"],
            p_model=Decimal("0.6"),
            p_market=Decimal("0.55"),
            outcome=1 if i % 2 == 0 else 0,
            model_version="test",
            config_id="c1",
        )
        for i in range(1, 21)
    ]
    train, hold = walk_forward_split(records, holdout_fraction=0.25)
    assert len(train) + len(hold) == 20
    assert hold[0].decision_time > train[-1].decision_time
    report = evaluate_holdout(hold, configs_tried=["c1", "c2"], min_sample=100)
    assert report.n_holdout_settled == len(hold)
    assert report.meets_promotion_sample is False
    assert report.brier_holdout is not None
