# Predict vs actual (TWC settlement)

**Goal:** production `station_v2` median matches official weather.com/kalshi climate `maxTemp` (the Kalshi TWC settlement source).

## Commands

```bash
# Score predictions vs official TWC highs
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-score-twc --days 60

# Tune hour bias / optional TWC retrain; promotes only if chronological holdout improves
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-tune-twc --days 100
```

## Backtest (60 days, Jul 19 → Sep 15 official)

**Before tune (14:00):** NYC 0.96°F · CHI 0.93°F · LAX 0.51°F

**After TWC tune (promoted where holdout improved):**

| Location | Overall MAE | 14:00 MAE | ±1°F @14h | Notes |
|----------|------------:|----------:|----------:|-------|
| **LAX** | 1.06°F | **0.51°F** | **98%** | Baseline already best — left unchanged |
| **CHI** | 2.19°F | **0.76°F** | **89%** | Morning-only hour bias promoted |
| **NYC** | 1.67°F | 1.02°F | 81% | Hour bias promoted (holdout MAE 1.89→1.72) |

Retraining the GBM on short TWC history **hurt** holdout (overfit) — not promoted. Bias correction is safer with ~2–3 months of TWC labels.

## What the tweak does

1. Keep the GHCND-trained remain GBM.
2. Learn hour-specific point bias from TWC residuals on earlier days.
3. Recalibrate empirical residual distribution on TWC.
4. Promote only if the **latest chronological holdout** improves (and 14:00 does not get worse).

Artifacts: `station_corrected_v2_calibration.json` (active), `*_pre_twc_tune.json` backups, `*_twc_tuned.json` candidates under each location’s multi artifact dir.

## Limitation

TWC official reports only start mid-2026 for these stations — not enough history for a full TWC retrain yet. Matching the daily high well is necessary but not sufficient for contract PnL.
