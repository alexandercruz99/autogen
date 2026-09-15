from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from kalshi_bot.api.client import KalshiClient
from kalshi_bot.api.orderbook import market_implied_yes_prob
from kalshi_bot.combo.discovery import ComboDiscoverer
from kalshi_bot.config import AppConfig
from kalshi_bot.data.store import OpportunityRecord, Store, dumps, utcnow
from kalshi_bot.discovery.scanner import MarketScanner
from kalshi_bot.ev.calculator import evaluate_binary_contract
from kalshi_bot.ev.combo import joint_probability
from kalshi_bot.execution.engine import ExecutionEngine
from kalshi_bot.execution.rfq import ComboRFQExecutor
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
        self.registry.register(self.weather_model)
        self.execution = ExecutionEngine(client, store, config)
        self.combos = ComboDiscoverer(client, config.combos)
        self.rfq = ComboRFQExecutor(client, store, config)

    def close(self) -> None:
        self.weather_model.close()

    def health_check(self) -> dict[str, Any]:
        try:
            status = self.client.get_exchange_status()
            ok = bool(status.get("trading_active") or status.get("exchange_active"))
            self.store.update_state(connection_ok=ok, last_error="" if ok else "exchange not active")
            return {"ok": ok, "status": status}
        except Exception as exc:
            self.store.update_state(connection_ok=False, last_error=str(exc))
            return {"ok": False, "error": str(exc)}

    def run_scan_once(self) -> dict[str, Any]:
        health = self.health_check()
        if not health.get("ok"):
            self.store.update_state(last_scan_ok=False, last_scan_at=utcnow(), last_error=str(health))
            self.store.audit("scan_abort", "exchange/data health check failed", level="error", details=health)
            return {"ok": False, "health": health, "opportunities": 0, "orders": 0}

        scan = self.scanner.scan()
        opportunities: list[OpportunityRecord] = []
        orders_placed = 0
        predictions = []
        market_meta: dict[str, dict[str, Any]] = {}

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

            # Staleness
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
            # Rank by conservative EV
            evs_sorted = sorted(evs, key=lambda e: e.conservative_ev, reverse=True)
            best = evs_sorted[0]
            for ev in evs_sorted:
                opp_id = str(uuid.uuid4())
                decision = "buy" if ev.qualifies else "skip"
                reason = ev.reason
                if implied is not None:
                    reason += f"; market mid≈{implied} (benchmark only)"
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
                        }
                    ),
                )
                opportunities.append(opp)
                self.store.save_opportunity(opp)

                if ev.qualifies and ev is best and self.store.get_state().mode in ("paper", "live"):
                    city = (pred.details or {}).get("city")
                    corr = [f"city:{city}", f"date:{(pred.details or {}).get('target_day')}"]
                    order = self.execution.place_individual(
                        market=market,
                        ev=ev,
                        opportunity_id=opp_id,
                        correlation_keys=[c for c in corr if c and not c.endswith(":None")],
                    )
                    if order and order.status in ("filled", "resting", "partial", "submitted"):
                        orders_placed += 1

        # Combos — evaluate; only paper-execute when a real quote is supplied (none invented here)
        combo_notes = []
        for cand in self.combos.build_candidates(predictions, market_meta):
            if cand.skip_reason:
                combo_notes.append(cand.skip_reason)
                continue
            joint = joint_probability(
                cand.legs,
                allow_independence=self.config.combos.allow_independence_assumption,
                dependence_model=cand.dependence_model,
            )
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
                    or "combo candidate identified; awaiting RFQ quote — will not invent price"
                ),
                model_version="combo.joint.v0",
                factors_json=dumps([l.market_ticker for l in cand.legs]),
                validation_note="Combo RFQ fills require live quotes; paper needs explicit quote input",
                details_json=dumps({"dependence": cand.dependence_model, "joint_method": joint.method}),
            )
            opportunities.append(opp)
            self.store.save_opportunity(opp)

        self.store.update_state(last_scan_ok=True, last_scan_at=utcnow(), last_error="")
        self.store.audit(
            "scan_complete",
            f"snapshots={len(scan.snapshots)} opps={len(opportunities)} orders={orders_placed}",
        )
        return {
            "ok": True,
            "snapshots": len(scan.snapshots),
            "skipped": scan.skipped,
            "errors": scan.errors,
            "opportunities": len(opportunities),
            "orders": orders_placed,
            "combo_notes": combo_notes[:10],
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
