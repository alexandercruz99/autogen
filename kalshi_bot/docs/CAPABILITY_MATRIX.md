# Capability matrix and documentation verification

**Generated / last code audit:** 2026-09-16 (corrections addendum)  
**Environment for claims:** paper/research against production *public* market data + BLS/NWS. **No real-money orders in correction tests.**

## Official documentation re-checked (2026-09-16)

| Topic | URL | Notes verified |
| --- | --- | --- |
| API index | https://docs.kalshi.com/ | REST/WS, OpenAPI |
| Fills | https://docs.kalshi.com/api-reference/portfolio/get-fills | `outcome_side`, `book_side`, `yes_price_dollars`, `no_price_dollars`, `fee_cost` |
| Settlements | https://docs.kalshi.com/api-reference/portfolio/get-settlements | Paginated; `market_result`, counts, `fee_cost`, `revenue` |
| Orders V2 | https://docs.kalshi.com/api-reference/orders/create-order-v2.md | YES-book bid/ask |
| Orderbook | https://docs.kalshi.com/getting_started/orderbook_responses.md | Bid-only; YES ask = 1 − best NO bid |
| Fees | https://kalshi.com/regulatory/fee-schedule | Taker schedule + rounding docs |

## Status legend

- **implemented_tested** — code + automated tests
- **connected_real_data** — exercised against live APIs / free public data
- **forward_paper_obs** — paper entries recorded
- **statistically_validated** — walk-forward metrics meet promotion criteria
- **live_eligible** — meets promotion criteria + explicit live gate (**none today**)
- **blocked** — missing dependency listed
- **transfer_research** — same-ICAO / cross-source transfer; not settlement-calibrated for live

## Counts (do not collapse aliases)

| Metric | Approx | Notes |
| --- | ---: | --- |
| Discovered climate series | 392 | discovery scan |
| Unique daily-max locations (verified map) | NWS 8 + TWC 7 | aliases not double-counted as locations |
| Currently tradable markets | varies intraday | exchange status |
| Verified settlement mappings | see `multi/registry.py` | |
| Collecting locations | NYC, CHI/MDW, LAX + feed worker | |
| Trained station_v2 models | nyc, chi_midway, lax_airport | |
| Calibrated (location-scoped) | same three; no silent NYC fallback | |
| End-to-end demonstrated (collect→predict→paper) | NYC, CHI, LAX TWC paths | live-eligible: **0** |

## Matrix

| Capability | Status | Code | Tests | Evidence / blocker |
| --- | --- | --- | --- | --- |
| Market discovery | implemented_tested, connected_real_data | `discovery/scanner.py` | `test_scanner_smoke.py` | |
| Individual EV/fees/book | implemented_tested | `api/orderbook.py`, `fees.py`, `ev/` | fees, orderbook, EV | |
| Unified live execution | implemented_tested | `execution/engine.py` **only** `create_order_v2` | `test_audit_execution_gates.py`, `test_live_eligibility_gate.py` | TWC path routed through engine; ineligible → 0 submits |
| EV requalify at submit | implemented_tested | `multi/live_bet.py` | ask 0.40→0.90 fixture | |
| Fill YES/NO normalization | implemented_tested | `accounting/fills.py`, engine | `test_audit_accounting.py` | |
| Settlement reconcile | implemented_tested | `accounting/settlement.py`, `client.get_settlements` | settlement + portfolio API mocks | scan + CLI `reconcile`; checkpoint |
| Risk / reservations / intents | implemented_tested | `risk/limits.py`, `order_intents` | concurrency + ambiguous reconcile | deposits ≠ budget |
| Obs NYC / multi / TWC | research/paper, transfer_research | `obs_engine/*` | multi, feeds, calib scope | **not live_eligible** |
| CPI | implemented_tested | `economics/cpi.py` | unit | not live_eligible |
| Combo / RFQ live | blocked / paper only | `execution/rfq.py` | FSM | not wired to scan live |
| Walk-forward / promotion | framework only | `validation/` | | **no strategy meets criteria** |

## Live-readiness

Documented in `docs/PROMOTION_CRITERIA.md` and `docs/BOT_AUDIT_ADDENDUM.md`.  
**No model is live-eligible.** Transfer TWC models remain research until calibrated on TWC settlement outcomes.

See also: `docs/MULTI_LOCATION_STATUS.md`, `docs/TRADE_RECONCILIATION_CORRECTED.md`.
