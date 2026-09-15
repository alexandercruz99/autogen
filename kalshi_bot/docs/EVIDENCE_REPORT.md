# Evidence report — 2026-09-15 verification run (paper/demo only)

## Tests run

```text
pytest -q
```

(Run in this session after changes; see agent log for pass count.)

## Observed end-to-end evidence

### Weather individual path
- Public Kalshi markets for `KXHIGHNY` / `KXHIGHLAX` fetched; orderbooks parsed.
- NWS forecasts fetched with timestamps; settlement rules audited: **Weather Company CLINYC**.
- Opportunities persisted with model version, fees, conservative EV, skip/buy reasons.
- Paper orders/positions recorded when conservative gates pass; **no threshold relaxation**.
- Settlement reconciler runs each scan; applies PnL **only** when exchange `result`/`settlement_value` present (not invented).

### Economics CPI path
- `KXCPI` series scanned; BLS `CUSR0000SA0` MoM climatology model registered.
- Skips months already printed by BLS (leakage guard).

### Combos
- Open MVE collections queried; **weather and CPI events are not in associated_events** (API-verified) → cannot form exchange-eligible combos of those markets today.
- RFQ FSM paper lifecycle tested with **labeled fixture quotes**.
- Joint MC + shared-driver simulation tested; independence gated by config.

## Live-eligible strategies

**None.** See `docs/PROMOTION_CRITERIA.md`.

## Paper/research-only models
- `weather.high_temp.v0.2-unvalidated`
- `economics.cpi_mom.v0.1-unvalidated`

## Remaining blockers
| Item | Dependency |
| --- | --- |
| Weather live | Weather Company–aligned forecasts/obs + walk-forward meeting promotion criteria |
| CPI live | Better nowcast than climatology + holdout n + promotion criteria |
| Weather/CPI combos | Events must appear in open MVE collections |
| Sports/props | Licensed data feed (not purchased) |
| Historical depth backtests | Kalshi historical orderbook depth not assumed; collect forward paper fills |

## Startup

```bash
cd kalshi_bot
pip install -e ".[dev]"
cp config.example.yaml config.yaml
kalshi-bot --config config.yaml scan
kalshi-bot --config config.yaml reconcile
kalshi-bot --config config.yaml run
```

Process runs on the host you start; SQLite at `data/kalshi_bot.db` recovers state after restart. Kill switch / pause prevent new purchases; completed `client_order_id`s are not replayed.

## Latest paper scan (2026-09-15T19:12Z)

- snapshots=49 (weather + CPI)
- opportunities=124
- paper orders=2 (weather; thresholds not relaxed)
- CPI: model evaluated with BLS climatology; all skipped on conservative EV / depth (valid no-trade)
- Combos: MVE eligibility check — weather events not in open collections
- live_eligible_strategies=[]
- pytest: 33 passed
