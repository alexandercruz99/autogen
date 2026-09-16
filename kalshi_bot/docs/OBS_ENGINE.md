# Observation-driven weather engine (NYC Central Park)

## Purpose

Predict the **official NWS CLI daily maximum** for Central Park using **measured conditions**,
not third-party forecasts. Published forecasts are **benchmarks only**.

## Target station (verified)

| Field | Value |
| --- | --- |
| Display | New York City Central Park |
| GHCND | `USW00094728` |
| CLI location | `NYC` (WFO OKX) |
| METAR | `KNYC` |
| IEM ASOS | `NYC` |
| Lat/Lon | 40.77898, −73.96925 |
| Elevation | ~42.7 m |
| Timezone | America/New_York (CLI climate day uses **LST**; DST caveat per Kalshi) |
| Unit / rounding | Whole °F as printed in CLI MAXIMUM / GHCND TMAX |
| Kalshi series | `KXHIGHNY`, `HIGHNY` |
| Settlement | Final NWS Daily Climate Report (Kalshi weather help, Jul 2026) |

Do **not** substitute LGA or city-center coordinates without documented reason.
Max of hourly METAR ≠ official daily max; METAR is same-day evidence / features only.

## Label reconciliation (CLI vs GHCND)

Settlement **controls**: NWS CLI MAXIMUM. Training labels use GHCND TMAX (same station).

Recent reconcile (`weather-obs-reconcile`): final CLI matched GHCND on sampled finals;
one prelim CLI was 1°F below later final/GHCND — treat prelim as revisable floor evidence,
not an immutable physical constraint beyond observed METAR.

## Historical coverage

| Dataset | Span | Independent climate days |
| --- | --- | --- |
| IEM ASOS NYC hourly | 2022-01-01 → 2026-09-15 | ~1,719 local days (0 missing in range) |
| GHCND TMAX | 2020-01-01 → 2026-09-13 | 2,448 days |
| Feature+label rows @ 08/11/14 | — | **1,661** independent days (4,981 rows) |

**Availability limitation:** IEM `valid` time ≠ proven publication time. Disclosed in backtests.

## Commands

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-train
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-backtest
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-reconcile
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-validate
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-predict
```

Artifacts: `data/obs_engine/models/obs_nyc_q50.joblib`, `backtest_report.json`,
`latest_research_prediction.json`, `label_reconciliation_cli_ghcnd.json`.

## Backtest summary (v1.1)

Chronological split by climate day; metrics **by local decision hour** (not pooled headline):

| Split | Days | First → last |
| --- | --- | --- |
| Train | 996 | 2022-01-01 → 2024-10-25 |
| Val | 332 | 2024-10-26 → 2025-10-04 |
| Test | 333 | 2025-10-05 → 2026-09-13 |

**Test MAE (°F)** — model vs baselines:

| Hour | Model | Climatology | Continuation | Persistence | Open-Meteo hist. fcst* |
| --- | --- | --- | --- | --- | --- |
| 08:00 | ~3.45 | ~7.48 | ~8.36 | ~5.56 | ~1.40 |
| 11:00 | ~2.45 | ~7.48 | ~5.06 | ~5.56 | ~1.40 |
| 14:00 | ~1.08 | ~7.48 | ~1.64 | ~5.56 | ~1.40 |

\*Open-Meteo GFS seamless **retrospective forecast archive** — not decision-time NWS; not a training feature.

- Beats climatology & continuation at all hours (day-block bootstrap CIs exclude 0).
- Does **not** beat Open-Meteo archive at 08:00/11:00; slight edge at 14:00 vs that archive.
- **Does not** claim outperformance of decision-time NWS (archive not matched).
- Trading profitability: **UNVALIDATED** (no historical executable books).

### Prior claim reconciliation

Earlier report (~1.36°F MAE / 58 holdout days) used a **summer-only ASOS window** and
hours 10/13/16. Multi-year 08/11/14 test MAE is higher in the morning — expected.
Both are reproducible from saved data; they are not the same evaluation.

## Live gate

All weather models set `model_live_eligible=False`. Execution boundary blocks live
submit unless that flag is **explicitly True**. Config flag alone + unit tests do **not**
authorize live. Paper/research remain available.

## Fresh prediction

`weather-obs-predict` writes `data/obs_engine/latest_research_prediction.json`.
Never places live orders.

## Continuous public feeds

See [FEEDS.md](FEEDS.md) for METAR / CLI / GOES-19 / NEXRAD adapters, LST climate-day
handling, `station_v2.1` shared features, calibrated residual probabilities, supported
decision hours (08/11/14), paper ledger, and worker health semantics.
