# Predict vs actual (TWC settlement)

**Goal:** production `station_v2` median matches official weather.com/kalshi climate `maxTemp` (the Kalshi TWC settlement source).

## Command

```bash
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-score-twc --days 5
```

Writes `data/obs_engine/multi/eval/predict_vs_twc_latest.json` (runtime; under gitignored trees as applicable).

## Latest scorecard (2026-09-12 … 2026-09-15 official; Sep 16 not yet official)

| Location | Overall MAE | 14:00 MAE | ±1°F | ±2°F | Quality |
|----------|------------:|----------:|------:|------:|---------|
| **LAX** | 0.79°F | **0.25°F** | 67% | 100% | **strong** |
| **CHI/MDW** | 2.66°F | 1.09°F | 42% | 50% | weak |
| **NYC** | 2.83°F | 2.50°F | 25% | 50% | weak |

At the **14:00** decision hour (most actionable), LAX is essentially matching settlement; CHI is usable; NYC still misses (e.g. Sep 13 −5°F).

## Sep 16 (provisional — TWC official report not published yet)

Research dry-run (forced 14:00, no live):

| City | Pred median | Seen so far | Note |
|------|------------:|------------:|------|
| NYC | ~77°F | 76°F | NYC multi calib restored; probabilities available again |
| CHI | ~75°F | 75°F | Already near floor |
| LAX | ~79°F | 78°F | Consistent with strong LAX track record |

Re-score after TWC posts official Sep 16 `maxTemp`.

## How we improve the match

1. **Keep scoring against TWC**, not only GHCND train labels (`weather-score-twc`).
2. **LAX:** maintain; same-ICAO GHCND→TWC transfer is already close at 14:00.
3. **CHI/NYC:** next work is TWC-label calibration / retrain (current models train on GHCND; residual transfer is exploratory). Morning (08h) errors dominate overall MAE.
4. Do **not** treat paper +EV at 1¢ tickets as proof the high matched — point forecast error is the primary target.

## Limitation

Matching the daily high well is necessary but not sufficient for contract PnL; bracket boundaries and fees still matter.
