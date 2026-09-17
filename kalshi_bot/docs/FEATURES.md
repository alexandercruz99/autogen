# Implemented features and remaining dependencies

## Implemented

- Kalshi REST client (public + RSA-PSS auth), rate limiting, retries
- Market discovery for configured series; SQLite persistence of markets/books/opps/orders/positions
- Orderbook YES/NO ask derivation from bid-only books (decimal-safe)
- Fee estimation (quadratic taker/maker + rounding approximation)
- Weather daily-high model (NWS) with stress-σ conservatism and skip-on-stale/missing data
- Individual EV, ranking, risk caps, paper fills, duplicate opportunity prevention
- Combo candidate search; joint probability only with justified dependence; RFQ accept path for live
- Dashboard: mode, health, cash/budget/exposure, opportunities table, pause/cancel/kill, live enable gate
- Autonomous scan loop; kill switch; audit log
- Unit tests for fees, books, EV/combo/DNP product, risk, leakage, parsing

## Unvalidated / limited

- Weather σ prior is **not** walk-forward calibrated against Kalshi settlements
- Sports, props, economics models: **not built** (markets skipped with reason)
- Same-game / dependent combo joints: skipped without a supplied simulation model
- Paper fills assume decision-time depth is available (no latency/partial-fill microstructure)
- Live order cancel path and full WS reconciliation are minimal
- Historical backtests: scaffolding only — **no fabricated profitability reports**

## Dependencies for you

1. Kalshi account (fund via Kalshi UI — bot does not deposit for you)
2. Optional API key files on disk for live/demo trading (`api_key_id` + private key path)
3. Host that stays running for unattended mode
4. Optional: paid sports data if/when those models are added later
5. Forward paper observations to validate weather calibration before live enablement
