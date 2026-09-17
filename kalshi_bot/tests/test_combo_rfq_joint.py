from __future__ import annotations

from decimal import Decimal

from kalshi_bot.execution.rfq_fsm import PaperRfqFsm, RfqState
from kalshi_bot.combo.joint_sim import simulate_independent_binary, simulate_shared_gaussian_driver
from kalshi_bot.ev.combo import combo_settlement_expectation
from kalshi_bot.money import D


def test_rfq_fsm_full_lifecycle_with_fixture_quote():
    fsm = PaperRfqFsm(wait_seconds=30, hvm=True)
    sess = fsm.create("COMBO-A+B", D("1.00"), "intent-1")
    assert sess.state == RfqState.CREATED
    q = fsm.ingest_fixture_quote(sess.rfq_id, D("0.20"), D("0.70"))
    assert "FIXTURE" in q["label"]
    sess = fsm.accept(sess.rfq_id, q["id"], "yes", max_price=D("0.25"))
    assert sess.state == RfqState.ACCEPTED
    sess = fsm.confirm_maker(sess.rfq_id, within_window=True)
    assert sess.state == RfqState.CONFIRMED
    sess = fsm.execute(sess.rfq_id, fill_fraction=D("1"))
    assert sess.state == RfqState.EXECUTED


def test_rfq_rejects_over_max_and_duplicate_open():
    fsm = PaperRfqFsm()
    sess = fsm.create("M", D("1"), "i1")
    q = fsm.ingest_fixture_quote(sess.rfq_id, D("0.40"), D("0.50"))
    sess = fsm.accept(sess.rfq_id, q["id"], "yes", max_price=D("0.30"))
    assert sess.state == RfqState.REJECTED
    # new open while prior rejected is ok; create another open then conflict
    s2 = fsm.create("M2", D("1"), "i2")
    try:
        fsm.create("M2", D("1"), "i3")
        assert False, "expected conflict"
    except RuntimeError as e:
        assert "409" in str(e)


def test_rfq_uncertain_keeps_nonterminal_state():
    fsm = PaperRfqFsm()
    sess = fsm.create("M", D("1"), "i")
    q = fsm.ingest_fixture_quote(sess.rfq_id, D("0.10"), D("0.80"))
    fsm.accept(sess.rfq_id, q["id"], "yes", D("0.20"))
    fsm.confirm_maker(sess.rfq_id)
    sess = fsm.execute(sess.rfq_id, uncertain=True)
    assert sess.state == RfqState.UNCERTAIN
    assert fsm.requires_fund_reservation(sess.state)


def test_joint_independence_mc_and_shared_driver():
    j = simulate_independent_binary([D("0.5"), D("0.5")], seed=1, n_samples=2000)
    assert j.supported
    assert D("0.20") < j.p_all < D("0.30")
    j2 = simulate_shared_gaussian_driver(
        [{"a_shared": 1.0, "b_idio": 0.1, "c": 0.0}, {"a_shared": 1.0, "b_idio": 0.1, "c": 0.0}],
        seed=2,
        n_samples=2000,
    )
    assert j2.supported
    # Shared driver induces dependence; joint >> independent 0.25 typically
    assert j2.p_all > D("0.25")


def test_dnp_product_expectation():
    assert combo_settlement_expectation([D("0.7"), D("1"), D("1")]) == D("0.7")
