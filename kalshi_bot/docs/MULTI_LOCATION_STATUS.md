# Multi-location weather engine — status

Generated: 2026-09-16 (UTC). Live orders: **blocked** (`model_live_eligible` false for all weather paths after audit corrections).

## Separate counts

| Metric | Value |
|--------|------:|
| Climate/Weather series discovered | 392 |
| Daily max temp series | 57 |
| Daily min temp series | 54 |
| Verified NWS CLI daily-max mappings | 8 |
| Verified TWC daily-max mappings | 7 |
| Unique locations (aliases not double-counted) | see registry |
| Collecting (feeds / multi) | NYC, CHI Midway, LAX |
| Trained station_v2 | nyc_central_park, chi_midway, lax_airport |
| Location-scoped calibration | same three (no NYC fallback) |
| End-to-end paper demonstrated | NYC, CHI, LAX |
| Live-eligible | **0** |

Operating NWS CLI: `HIGHAUS`, `HIGHNY`, `HIGHCHI`, `HIGHMIA`, `KXDENHIGH`, `KXHIGHOU`, `KXHIGHHOU`, `KXHOUHIGH`.

Operating TWC (weather.com/kalshi adapter): `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHLAX`, `KXHIGHAUS`, `KXHIGHDEN`, `KXHIGHPHIL`, `KXHIGHMIA`.

Registry entry ≠ operating. Missing artifacts → documented prep (`weather-train-location`, calibration, eval), not a silent dead end.

## Prep workflow (repeatable)

```bash
# Capability matrix (separate counts; prep path for missing artifacts)
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-multi-status

# Backfill / collect observations for a location profile
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-feeds-run

# Train location model (nyc_central_park | chi_midway | lax_airport only today)
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-train-location --location lax_airport

# Research / paper inference (no --live)
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-twc-bet --series KXHIGHLAX
```

Locations without `LOCATION_TRAIN_PROFILES` (MIA/AUS/DEN/HOU/PHL, …) need IEM ASOS + GHCND backfill before train — status output lists `prep_workflow` per row. Do not mark them operating or live_eligible until trained, calibrated, and evaluated.

## LAX

- Settlement: TWC CLILAX / KLAX airport (`33.9425, -118.4081`), not downtown.
- Profile `lax_airport`; GHCND USW00023174 labels; transfer to `KXHIGHLAX` labeled `same_icao_transfer`.

## Corrections note

`weather-twc-bet --live` no longer bypasses `ExecutionEngine`. See `docs/BOT_AUDIT_ADDENDUM.md`.

## Remaining limitations

- TWC residual models not trained on TWC climate labels (exploratory transfer).
- MIA/AUS/DEN/HOU/PHL: mapped but not all trained+calibrated end-to-end in this environment.
- Accuracy / paper PnL claims require prospective evidence; no reliable-win claim.
