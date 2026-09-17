# Chicago Midway — noon trading prep

**Goal:** Same research/paper capabilities as NYC for `HIGHCHI` (NWS CLI Midway), ready for **~11:00 CDT** tomorrow (closest validated hour to “around noon”). Live Kalshi orders remain **blocked**.

## What’s ready tonight

| Item | Status |
|------|--------|
| Settlement mapping | Midway CLI `MDW` / METAR `KMDW` (not O’Hare) |
| Hourly history | IEM ASOS MDW 2022-01-01 → 2026-09-16 (~41k hours) |
| Daily labels | GHCND `USW00014819` TMAX °F (proxy for CLI MAXIMUM; settlement still CLI) |
| Location model | `data/obs_engine/multi/artifacts/chi_midway__daily_max_temp_f/station_corrected_v2.joblib` |
| Calibration | hour-specific residuals; n≈257 per hour |
| Holdout (test) at **11 CDT** | production median MAE **~2.45°F**, 80% interval coverage **~0.79** |
| Forced 11 CDT replay | Probabilities available; paper blocked only because that day had no open markets |

## Tomorrow plan (Chicago time)

1. **Before 11:00 CDT** — ensure collector has Midway METAR for the morning (`weather-multi-once` or feeds cycle). Need adequate coverage (≥4 temps, start gap ≤4h, stale ≤2.5h).
2. **At 11:00 CDT (±20 min)** — run:
   ```bash
   cd kalshi_bot
   PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-once
   # or status:
   PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-status
   ```
3. System will: collect KMDW (+KORD neighbor), build Midway features, load **Chicago** model (not NYC), emit calibrated probabilities for same-day `HIGHCHI`, evaluate YES/NO EV, **simulate paper fill** if EV>0 after fees/buffer.
4. **14:00 CDT** is a second shot if 11:00 coverage was thin or markets still open.

## Accuracy notes (honest)

- Midway model is **newly trained**; do not treat first paper fills as proven edge.
- Labels are GHCND rounded °F; settlement is CLI whole °F — small label noise remains.
- External Weather Company Chicago market (`KXHIGHCHI`) is a **different** contract — do not trade it with this model.
- Live trading is still off (`live_eligible=false`).

## Commands

```bash
# Retrain if data updated
PYTHONPATH=src python3 -m kalshi_bot.cli weather-train-location --location chi_midway

# Discovery + multi cycle
PYTHONPATH=src python3 -m kalshi_bot.cli weather-discover
PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-once
```
