# Observation-driven weather engine (NYC Central Park)

## Purpose

Predict the **official NWS CLI daily maximum** for Central Park using **measured conditions**,
not third-party forecasts. NWS grid / Open-Meteo forecasts are **benchmarks only**.

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
| Timezone | America/New_York (CLI climate day uses **LST**; DST caveat) |
| Unit / rounding | Whole °F as printed in CLI MAXIMUM / GHCND TMAX |
| Kalshi series | `KXHIGHNY`, `HIGHNY` |

Do **not** substitute LGA or city-center coordinates without a documented reason.
Max of hourly METAR ≠ official daily max; METAR is same-day evidence / features only.

## What learns vs what is reused

| Component | Learns from observations? | Notes |
| --- | --- | --- |
| `obs_engine` remaining-rise quantile GBM | **Yes** | Features from ASOS/METAR; labels GHCND TMAX |
| Seasonal climatology / persistence / continuation baselines | **Yes** (simple) | Holdout MAE reported at train time |
| AI forecaster (`weather.ai_cli.*`) | Residual model on forecast−CLI | Uses NWS/OM **forecasts** as the point forecast |
| Legacy `high_temp` Gaussian | No | NWS grid + fixed σ |
| NWS grid in obs engine | **Benchmark only** | Never a training feature |

## Data provenance

| Dataset | Type | Role |
| --- | --- | --- |
| IEM ASOS hourly `NYC` | Direct station observations (METAR-derived archive) | Training features |
| Aviation Weather METAR `KNYC` | Direct observations | Live features |
| GHCND `USW00094728` TMAX | Official daily climate observation | Training labels (CLI proxy when historical CLI text unavailable) |
| NWS CLI products | Official settlement source | Live floor / settlement evidence |
| NWS grid daytime high | Forecast product | Benchmark comparison only |
| Satellite / radar | Deferred | Add only after incremental holdout gain |

**Availability limitation:** IEM CSV rows use observation `valid` time, not first-seen/publication
time. Forward collection should store `receiptTime` / retrieval time. Disclosed in feature provenance.

## Commands

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-train
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-validate
```

Artifacts: `data/obs_engine/models/obs_nyc_q50.joblib`, registry row in `data/weather_archive.db`,
report `data/obs_engine/last_train_report.json`.

## Live gate

- `models.weather.obs_engine_enabled: true` — register model for NYC markets
- `models.weather.obs_engine_live_eligible: false` — **required default**
- Predictions always set `details.model_live_eligible=False` until promotion criteria are met
- Pipeline demotes any live order attempt to **paper** when `model_live_eligible` is false
- Existing `live.enabled` / account connection does **not** authorize this engine

## Validation status

Trading profitability: **UNVALIDATED** (no historical executable book replay).
Promotion requires multi-season chronological holdout vs climatology/persistence **and**
same-decision-time NWS benchmarks, plus forward paper PnL — see `docs/PROMOTION_CRITERIA.md`.
