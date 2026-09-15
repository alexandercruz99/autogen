# Capability matrix and documentation verification

**Generated / last code audit:** 2026-09-15  
**Environment for claims:** paper mode against production *public* market data + BLS/NWS (no live orders).

## Official documentation re-checked (2026-09-15)

| Topic | URL | Notes verified |
| --- | --- | --- |
| API index | https://docs.kalshi.com/ | REST/WS, OpenAPI |
| Environments | https://docs.kalshi.com/getting_started/api_environments.md | `external-api.kalshi.com`, demo hosts |
| Auth | https://docs.kalshi.com/getting_started/quick_start_authenticated_requests.md | RSA-PSS SHA256 headers |
| Orderbook | https://docs.kalshi.com/getting_started/orderbook_responses.md | Bid-only; YES ask = 1 − best NO bid |
| Fees | https://kalshi.com/regulatory/fee-schedule (+ PDF) | Taker ≈ M×0.07×C×P×(1−P); rounding docs |
| Fee rounding | https://docs.kalshi.com/getting_started/fee_rounding.md | 6dp trade fee + balance precision |
| Orders V2 | https://docs.kalshi.com/api-reference/orders/create-order-v2.md | YES-book bid/ask, fp prices |
| RFQ | https://docs.kalshi.com/getting_started/rfqs.md | Create → quote → accept → confirm → execute; HVM timers |
| Combos help | https://help.kalshi.com/en/articles/13823820-combos | Product settlement; DNP scalar ≠ refund |
| MVE collections | https://docs.kalshi.com/api-reference/multivariate/get-multivariate_event_collections.md | Eligibility discovery |

## Status legend

- **implemented_tested** — code + automated tests
- **connected_real_data** — exercised against live APIs / free public data
- **forward_paper_obs** — paper entries recorded (settlements only when exchange settles)
- **statistically_validated** — walk-forward metrics meet documented promotion criteria
- **live_eligible** — meets promotion criteria + explicit live gate (none today)
- **blocked** — missing dependency listed

## Matrix

| Capability | Status | Code | Tests | Evidence / blocker |
| --- | --- | --- | --- | --- |
| Market discovery | implemented_tested, connected_real_data | `discovery/scanner.py`, `api/client.py` | `tests/test_scanner_smoke.py` | DB markets/snapshots from open series |
| Weather model | implemented_tested, connected_real_data, forward_paper_obs | `models/weather/*` | parse + leakage + settlement + AI guards | **Daily max target = NWS CLI** (not Weather Co hourly). AI module `weather.ai_cli.v1.0-collecting` with empirical residuals; **not** statistically_validated / **not** live_eligible; decision-time forecast archive still thin |
| Obs-driven NYC engine | implemented_tested, research/paper | `models/weather/obs_engine/*` | `tests/test_obs_engine.py`, `tests/test_live_eligibility_gate.py` | Multi-year ASOS (2022–2026); backtest by hour vs clim/continuation/OM archive; **not** decision-time NWS superiority; **live blocked**; trading PnL **unvalidated**; see `docs/OBS_ENGINE.md` |
| Economics CPI model | implemented_tested, connected_real_data | `models/economics/cpi.py` | CPI parse + BLS mom tests | Prior from BLS history; **not statistically_validated**; **not live_eligible** |
| Sports / props | blocked | — | — | Need licensed lineup/odds feeds; no free reliable end-to-end source wired |
| Individual EV/fees/book | implemented_tested, connected_real_data | `api/orderbook.py`, `api/fees.py`, `ev/calculator.py` | fees, orderbook, EV tests | — |
| Individual paper execution | implemented_tested, forward_paper_obs | `execution/engine.py` | risk/dup/engine paper tests | Paper fills in SQLite; not live |
| Settlement reconciliation | implemented_tested | `accounting/settlement.py` | settlement unit tests | Applies **only** when Kalshi market `result`/`settlement_value` present; never invents |
| Combo discovery / eligibility | implemented_tested, connected_real_data | `combo/discovery.py` | combo eligibility tests | Open MVE collections queried; weather/CPI **not** in current associated events → combo of those markets **blocked** until eligible |
| Joint probability | implemented_tested | `ev/combo.py`, `combo/joint_sim.py` | joint tests | Independence only when justified; sim for shared-driver fixtures |
| RFQ lifecycle (paper) | implemented_tested | `execution/rfq.py`, `execution/rfq_fsm.py` | RFQ FSM + paper lifecycle tests | Paper simulator with **labeled fixture quotes**; live RFQ code present but **disabled** (paper/demo only this run) |
| Walk-forward validation | implemented_tested | `validation/walk_forward.py` | walk-forward / leakage tests | Framework + promotion criteria; weather/CPI **fail** live promotion until criteria met |
| Risk / reservations | implemented_tested | `risk/limits.py`, engine lock | budget, kill, concurrent reservation tests | Deposits do not raise budget |
| Recovery / kill / dup IDs | implemented_tested | engine, dashboard, store | dup + kill + restart tests | — |
| Dashboard | implemented_tested (manual + API) | `dashboard/app.py` | status API test | Mode, health, validation panel, pause/cancel/close/live-gate |

## Live-readiness (promotion) criteria — locked before final eval inspection

Documented in `docs/PROMOTION_CRITERIA.md`. **No strategy currently meets them.**
