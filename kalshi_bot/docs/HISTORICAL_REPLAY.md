# Historical weather replay & deterministic picker

**Mode:** research / paper only · `live_eligible=false` · **no orders submitted**

## Reproduce

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli weather-historical-replay
PYTHONPATH=src python3 -m pytest tests/test_historical_replay.py -q
```

Artifacts: `data/obs_engine/multi/replay/` · summary: `.../replay/_reports/latest.json`

## 1. System inventory

| Component | Status |
|-----------|--------|
| `predict_station_v2` shared inference | Exists — reused |
| Chronological split manifests | Exists — reused |
| `select_forecast_consistent` | Exists — **superseded for new path** by `pick_contract` (ENP ranking) |
| Paper ledger + live books | Exists — not used for historical P&L |
| Historical Kalshi order books | **Missing** (prospective.db has ~12 snapshots) |
| Injected replay clock | **Implemented** via decision_time on each row |
| Outcome-blind prediction save | **Implemented** under `predictions_blind/` |
| Exact energy-form CRPS | **Implemented** in `replay/scoring.py` |

## 2. Data contract

Each forecast row records location, decision UTC/local hour, max_so_far, model/calib versions, full PMF, then (after save) GHCND outcome labeled **`proxy_weather_evaluation`** — not verified exchange settlement.

Availability assumption: `archive_valid_utc_equals_availability_DISCLOSED`.

Unsupported here: MIA/AUS/DEN/HOU (no train profiles).

## 3–5. Weather replay results (untouched final test)

Labels = GHCND TMAX (proxy). Primary metrics: MAE of unrounded median; mean CRPS (exact PMF energy form).

| Location | Rows / days | MAE °F | Mean CRPS | Modal exact-degree hit | Median within ±1°F | 14:00 MAE |
|----------|------------:|-------:|----------:|-----------------------:|-------------------:|----------:|
| NYC | 747 / 249 | 2.37 | 1.69 | 20.5% | 34.1% | **1.10** |
| CHI | 770 / 257 | 2.47 | 1.80 | 20.0% | 34.9% | **1.11** |
| LAX | 771 / 257 | 1.51 | 1.14 | 32.4% | 58.5% | **0.54** |

±0.1°F rates exist in JSON but **do not** establish tenth-degree accuracy against whole-degree labels.

Morning (08:00) remains the weak hour (~2.7–3.9°F MAE).

## 6–9. Deterministic picker

- Interface: `pick_contract(dist, contracts, books, account, decision_time, policy) -> Decision`
- Policy: frozen `picker.policy.v1` — rank by expected net profit / contract after Kalshi quadratic taker fees; ties by p → ticker → side
- Most-likely bracket reported **separately** from purchased contract
- Execution adapter `execute_decision_simulated` never sets `live_order_submitted=true`

### Trading replay

**Blocked:** no dense historical executable books. Every test decision → `NO_TRADE` / `missing_orderbook_snapshot`. **No historical P&L claimed.**

Prospective collection should continue via existing research collector for books.

## 10. Exports

Per location stamp dir:

- `forecast_rows.json` — predictions + errors + CRPS
- `decision_rows.json` — picker outputs / rejection reasons
- `split_manifest.json`, `picker_policy.json`, `summary.json`

## 11. Tests

`tests/test_historical_replay.py`: hand-calculated CRPS, shuffle-stable picks, missing books, fees killing edge, open-ended brackets, sim adapter never submits.

## 12. Limitations

- GHCND ≠ Kalshi TWC/CLI settlement
- Archive availability approximated
- Trading skill **unvalidated** without books
- Passing tests ≠ forecast skill or profitability
