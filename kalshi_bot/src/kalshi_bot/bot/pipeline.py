from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from kalshi_bot.accounting.settlement import SettlementReconciler
from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.orderbook import market_implied_yes_prob
from kalshi_bot.combo.discovery import ComboDiscoverer
from kalshi_bot.combo.joint_sim import simulate_independent_binary
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OpportunityRecord, Store, dumps, utcnow
from kalshi_bot.discovery.scanner import MarketScanner
from kalshi_bot.ev.calculator import evaluate_binary_contract
from kalshi_bot.ev.combo import joint_probability
from kalshi_bot.execution.engine import ExecutionEngine
from kalshi_bot.execution.rfq import ComboRFQExecutor
from kalshi_bot.execution.rfq_fsm import PaperRfqFsm, RfqState
from kalshi_bot.models.economics.cpi import CpiMomModel
from kalshi_bot.models.registry import ModelRegistry
from kalshi_bot.models.weather.high_temp import WeatherHighTempModel
from kalshi_bot.money import D

logger = logging.getLogger(__name__)


class TradingPipeline:
    def __init__(self, config: AppConfig, store: Store, client: KalshiClient) -> None:
        self.config = config
        self.store = store
        self.client = client
        self.scanner = MarketScanner(client, store, config.scan)
        self.registry = ModelRegistry()
        self.weather_model = WeatherHighTempModel(config.models.weather, store=store)
        self.cpi_model = CpiMomModel(store=store)
        self.registry.register(self.weather_model)
        if config.models.economics_cpi_enabled:
            self.registry.register(self.cpi_model)
        self.execution = ExecutionEngine(client, store, config)
        self.combos = ComboDiscoverer(client, config.combos)
        self.rfq = ComboRFQExecutor(client, store, config)
        self.paper_rfq = PaperRfqFsm(
            wait_seconds=config.combos.rfq_wait_seconds,
            hvm=True,
        )
        self.settlement = SettlementReconciler(client, store)
        self.model_configs_tried: list[str] = [
            self.weather_model.version,
            self.cpi_model.version,
        ]

    def close(self) -> None:
        self.weather_model.close()
        self.cpi_model.close()

    def health_check(self) -> dict[str, Any]:
        try:
            status = self.client.get_exchange_status()
            ok = bool(status.get("trading_active") or status.get("exchange_active"))
            self.store.update_state(connection_ok=ok, last_error="" if ok else "exchange not active")
            return {"ok": ok, "status": status}
        except Exception as exc:
            self.store.update_state(connection_ok=False, last_error=str(exc))
            return {"ok": False, "error": str(exc)}

    def validation_status(self) -> dict[str, Any]:
        return {
            "promotion_criteria": "docs/PROMOTION_CRITERIA.md",
            "live_eligible_strategies": [],
            "paper_only": [
                {
                    "model": self.weather_model.version,
                    "reason": "NWS proxy vs Weather Company CLINYC; σ unvalidated; holdout n insufficient",
                },
                {
                    "model": self.cpi_model.version,
                    "reason": "Climatology prior only; holdout n insufficient for promotion",
                },
            ],
            "blocked": [
                {
                    "strategy": "sports/props",
                    "missing": "Licensed sports data (lineups, injuries, odds) — not purchased/wired",
                },
                {
                    "strategy": "weather/CPI Kalshi combos",
                    "missing": "Events not in open MVE associated_events (API-verified)",
                },
            ],
            "configs_tried": self.model_configs_tried,
        }

    def run_scan_once(self) -> dict[str, Any]:
        health = self.health_check()
        if not health.get("ok"):
            self.store.update_state(last_scan_ok=False, last_scan_at=utcnow(), last_error=str(health))
            self.store.audit("scan_abort", "exchange/data health check failed", level="error", details=health)
            return {"ok": False, "health": health, "opportunities": 0, "orders": 0}

        # Reconcile any exchange-settled markets before new risk
        settle_result = self.settlement.reconcile_open_positions()

        scan = self.scanner.scan()
        opportunities: list[OpportunityRecord] = []
        orders_placed = 0
        predictions = []
        market_meta: dict[str, dict[str, Any]] = {}
        decision_records: list[dict[str, Any]] = []

        for snap in scan.snapshots:
            market = snap.market
            ticker = market["ticker"]
            market_meta[ticker] = market
            model = self.registry.resolve(market, snap.category)
            if model is None:
                opp = self._skip_opp(
                    market,
                    snap.category,
                    reason=f"no validated model for category/series ({snap.category})",
                )
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                continue

            age = (datetime.now(timezone.utc) - snap.captured_at).total_seconds()
            if age > self.config.scan.max_data_age_seconds:
                opp = self._skip_opp(market, snap.category, reason="orderbook/data stale")
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                continue

            pred = model.predict(market, snap.category)
            predictions.append(pred)
            if not pred.supported:
                opp = self._skip_opp(market, snap.category, reason=pred.skip_reason or "unsupported")
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                continue

            implied = market_implied_yes_prob(snap.book)
            evs = evaluate_binary_contract(pred, snap.book, self.config.trading)
            evs_sorted = sorted(evs, key=lambda e: e.conservative_ev, reverse=True)
            best = evs_sorted[0]
            for ev in evs_sorted:
                opp_id = str(uuid.uuid4())
                decision = "buy" if ev.qualifies else "skip"
                reason = ev.reason
                if implied is not None:
                    reason += f"; market mid≈{implied} (benchmark only)"
                feature_times = [
                    s.get("fetched_at") or s.get("available_at")
                    for s in pred.data_sources
                    if s.get("fetched_at") or s.get("available_at")
                ]
                opp = OpportunityRecord(
                    id=opp_id,
                    scanned_at=utcnow(),
                    kind="individual",
                    market_ticker=ticker,
                    event_ticker=market.get("event_ticker") or "",
                    category=snap.category,
                    side=ev.side,
                    quantity=str(ev.quantity),
                    executable_price=str(ev.executable_price),
                    estimated_prob=str(ev.estimated_prob),
                    conservative_prob=str(ev.conservative_prob),
                    uncertainty=str(ev.uncertainty),
                    breakeven_prob=str(ev.breakeven_prob),
                    estimated_ev=str(ev.estimated_ev),
                    conservative_ev=str(ev.conservative_ev),
                    max_loss=str(ev.max_loss),
                    fees=str(ev.fees_total),
                    data_freshness=snap.captured_at.isoformat(),
                    decision=decision,
                    reason=reason,
                    model_version=pred.model_version,
                    factors_json=dumps(pred.factors),
                    validation_note=pred.validation_evidence,
                    details_json=dumps(
                        {
                            "sources": pred.data_sources,
                            "details": pred.details,
                            "best_for_market": ev is best,
                            "feature_times": feature_times,
                            "decision_time": pred.as_of.isoformat(),
                            "live_eligible": False,
                        }
                    ),
                )
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                decision_records.append(
                    {
                        "decision_time": pred.as_of.isoformat(),
                        "feature_times": feature_times,
                        "p_model": str(pred.p_yes),
                        "p_market": str(implied) if implied is not None else None,
                        "model_version": pred.model_version,
                    }
                )

                if ev.qualifies and ev is best and self.store.get_state().mode in ("paper", "live"):
                    # Never live in this agent run — force paper path if live somehow set without ack
                    mode = self.store.get_state().mode
                    if mode == "live" and not self.store.get_state().live_enabled:
                        mode = "paper"
                    city = (pred.details or {}).get("city")
                    corr = [
                        f"city:{city}",
                        f"date:{(pred.details or {}).get('target_day')}",
                        f"event:{market.get('event_ticker')}",
                    ]
                    order = self.execution.place_individual(
                        market=market,
                        ev=ev,
                        opportunity_id=opp_id,
                        correlation_keys=[c for c in corr if c and not c.endswith(":None")],
                        mode="paper" if mode != "live" else "paper",  # hard paper for this verification run
                    )
                    if order and order.status in ("filled", "resting", "partial", "submitted"):
                        orders_placed += 1

        # Combos
        combo_notes = []
        for cand in self.combos.build_candidates(predictions, market_meta):
            if cand.skip_reason or not cand.eligible:
                opp = OpportunityRecord(
                    id=str(uuid.uuid4()),
                    scanned_at=utcnow(),
                    kind="combo",
                    market_ticker="+".join(l.market_ticker for l in cand.legs),
                    event_ticker=",".join(sorted({l.event_ticker for l in cand.legs})),
                    category="combo",
                    side="yes",
                    quantity=str(self.config.trading.default_contract_quantity),
                    executable_price="",
                    estimated_prob="",
                    conservative_prob="",
                    uncertainty="",
                    breakeven_prob="",
                    estimated_ev="",
                    conservative_ev="",
                    max_loss="",
                    fees="",
                    data_freshness=utcnow(),
                    decision="skip",
                    reason=cand.skip_reason or "not eligible",
                    model_version="combo.joint.v0",
                    factors_json=dumps([l.market_ticker for l in cand.legs]),
                    validation_note="Combo not live_eligible",
                    details_json=dumps({"eligible": cand.eligible, "collection": cand.collection_ticker}),
                )
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                combo_notes.append(cand.skip_reason or "ineligible")
                continue

            joint = joint_probability(
                cand.legs,
                allow_independence=self.config.combos.allow_independence_assumption,
                dependence_model=cand.dependence_model,
            )
            if joint.supported and cand.dependence_model and cand.dependence_model.get("type") == "independent_weather_cities":
                if self.config.combos.allow_independence_assumption:
                    joint = simulate_independent_binary(
                        [l.p_marginal for l in cand.legs],
                        seed=42,
                    )

            quote_price = None
            fixture = self.config.combos.paper_fixture_yes_price
            used_fixture = False
            if fixture is not None and self.store.get_state().mode == "paper":
                quote_price = D(fixture)
                used_fixture = True

            if not joint.supported or quote_price is None:
                opp = OpportunityRecord(
                    id=str(uuid.uuid4()),
                    scanned_at=utcnow(),
                    kind="combo",
                    market_ticker="+".join(l.market_ticker for l in cand.legs),
                    event_ticker=",".join(sorted({l.event_ticker for l in cand.legs})),
                    category="combo",
                    side="yes",
                    quantity=str(self.config.trading.default_contract_quantity),
                    executable_price="",
                    estimated_prob=str(joint.p_all) if joint.supported else "",
                    conservative_prob=str(joint.p_all_conservative) if joint.supported else "",
                    uncertainty="",
                    breakeven_prob="",
                    estimated_ev="",
                    conservative_ev="",
                    max_loss="",
                    fees="",
                    data_freshness=utcnow(),
                    decision="skip",
                    reason=(
                        joint.skip_reason
                        or "eligible collection but no RFQ quote (will not invent); "
                        "set combos.paper_fixture_yes_price only for labeled simulator tests"
                    ),
                    model_version="combo.joint.v0",
                    factors_json=dumps([l.market_ticker for l in cand.legs]),
                    validation_note="Awaiting quote",
                    details_json=dumps({"joint": joint.method, "collection": cand.collection_ticker}),
                )
                opportunities.append(opp)
                self.store.save_opportunity(opp)
                continue

            # Paper RFQ FSM with fixture quote (labeled)
            qty = D(self.config.trading.default_contract_quantity)
            evaluation = self.rfq.evaluate_candidate(
                cand.legs,
                quoted_yes_price=quote_price,
                quantity=qty,
                dependence_model=cand.dependence_model,
            )
            # Compare combo vs best individual vs no trade is recorded in reason
            compare = (
                f"fixture_quote={quote_price} (FIXTURE_NOT_PROFIT_EVIDENCE); "
                f"combo_cons_ev={evaluation.get('conservative_ev')}; "
                f"joint={joint.method}"
            )
            opp_id = str(uuid.uuid4())
            decision = "buy" if evaluation.get("qualifies") else "skip"
            opp = OpportunityRecord(
                id=opp_id,
                scanned_at=utcnow(),
                kind="combo",
                market_ticker="+".join(l.market_ticker for l in cand.legs),
                event_ticker=",".join(sorted({l.event_ticker for l in cand.legs})),
                category="combo",
                side="yes",
                quantity=str(qty),
                executable_price=str(quote_price),
                estimated_prob=str(joint.p_all),
                conservative_prob=str(joint.p_all_conservative),
                uncertainty="",
                breakeven_prob=str(quote_price),
                estimated_ev=str(evaluation.get("estimated_ev", "")),
                conservative_ev=str(evaluation.get("conservative_ev", "")),
                max_loss=str(evaluation.get("capital", "")),
                fees=str(evaluation.get("fees", "")),
                data_freshness=utcnow(),
                decision=decision,
                reason=(evaluation.get("reason") or "") + "; " + compare,
                model_version="combo.joint.v0",
                factors_json=dumps([l.market_ticker for l in cand.legs]),
                validation_note=(
                    "FIXTURE_QUOTE_NOT_LIVE_EVIDENCE" if used_fixture else "live quote"
                ),
                details_json=dumps({"collection": cand.collection_ticker, "fsm": True}),
            )
            opportunities.append(opp)
            self.store.save_opportunity(opp)

            if decision == "buy" and self.store.get_state().mode == "paper":
                # Full FSM: create → fixture quote → accept → confirm → execute
                intent = opp_id
                # Prevent duplicate intent
                existing = [
                    o
                    for o in self.store.list_orders(limit=500)
                    if o.get("opportunity_id") == opp_id
                ]
                if existing:
                    continue
                try:
                    sess = self.paper_rfq.create(opp.market_ticker, qty, intent)
                    q = self.paper_rfq.ingest_fixture_quote(
                        sess.rfq_id, quote_price, D("1") - quote_price
                    )
                    max_px = quote_price  # already constrained by evaluate
                    sess = self.paper_rfq.accept(sess.rfq_id, q["id"], "yes", max_px)
                    if sess.state == RfqState.REJECTED:
                        continue
                    sess = self.paper_rfq.confirm_maker(sess.rfq_id, within_window=True)
                    sess = self.paper_rfq.execute(sess.rfq_id)
                    if sess.state == RfqState.EXECUTED:
                        order = self.rfq.paper_execute(
                            market_ticker=opp.market_ticker,
                            event_ticker=opp.event_ticker,
                            evaluation=evaluation,
                            opportunity_id=opp_id,
                        )
                        if order:
                            orders_placed += 1
                except Exception as exc:
                    self.store.audit("combo_rfq_error", str(exc), level="error")

        self.store.update_state(
            last_scan_ok=True,
            last_scan_at=utcnow(),
            last_error="",
            extra_json=dumps(
                {
                    "validation": self.validation_status(),
                    "settlement": settle_result,
                    "decision_records_n": len(decision_records),
                }
            ),
        )
        self.store.audit(
            "scan_complete",
            f"snapshots={len(scan.snapshots)} opps={len(opportunities)} orders={orders_placed} settled={settle_result.get('settled')}",
        )
        return {
            "ok": True,
            "snapshots": len(scan.snapshots),
            "skipped": scan.skipped,
            "errors": scan.errors,
            "opportunities": len(opportunities),
            "orders": orders_placed,
            "combo_notes": combo_notes[:10],
            "settlement": settle_result,
            "validation": self.validation_status(),
        }

    def _skip_opp(self, market: dict[str, Any], category: str, reason: str) -> OpportunityRecord:
        return OpportunityRecord(
            id=str(uuid.uuid4()),
            scanned_at=utcnow(),
            kind="individual",
            market_ticker=market.get("ticker") or "",
            event_ticker=market.get("event_ticker") or "",
            category=category,
            side="",
            quantity="",
            executable_price="",
            estimated_prob="",
            conservative_prob="",
            uncertainty="",
            breakeven_prob="",
            estimated_ev="",
            conservative_ev="",
            max_loss="",
            fees="",
            data_freshness=utcnow(),
            decision="skip",
            reason=reason,
            model_version="",
            factors_json="[]",
            validation_note="",
        )
