# Data sources and model notes

## Kalshi exchange

- Public market data does not require auth.
- Trading, balances, RFQs, and MVE market creation require API keys (RSA-PSS).
- Production and demo credentials are **not** interchangeable.
- Rate limits are tiered; the client uses a token bucket and retries on HTTP 429.

## What we predict (daily max)

**Target:** the official daily maximum temperature (°F) printed in the **NWS Daily Climate Report (CLI)** for the market’s settlement station — the value Kalshi uses for daily high/low markets (Kalshi Help Center; NHIGH/CHIHIGH terms).

**Not the target:** Weather Company / CLINYC *hourly* products, phone-app weather, or “max of METAR so far” (METAR is same-day *evidence* only).

**Stations (v1):** NYC Central Park (`CLI` location `NYC`, WFO OKX), LAX downtown CLI (`LAX`), Chicago Midway (`MDW`, WFO LOT). Each market’s `rules_primary` must still be checked.

**Time:** CLI climate days use **local standard time** (DST caveat documented by Kalshi).

## AI weather module (`weather.ai_cli.v1.0-collecting`)

| Piece | Status |
| --- | --- |
| Station registry + settlement audit fields | implemented |
| CLI fetch/parse via `api.weather.gov/products` | implemented_tested |
| NWS grid daytime forecast snapshots with timestamps | implemented |
| Open-Meteo previous-runs optional layer | implemented (retrieval-time honest; init often unknown) |
| Empirical residual distribution → contract P(YES) | implemented |
| Coherent bracket probs from one discrete °F distribution | implemented_tested |
| Same-day METAR floor truncation | implemented |
| Quantile GBM | scaffolding (needs ≥40 paired days) |
| Walk-forward CLI (`weather-validate`) | implemented; sample sizes still small |
| Historical forecast archives deep enough for promotion | **gap — collecting** |
| Executable-price trading profitability backtest | **unvalidated** |

### Commands

```bash
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-collect
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-train
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-validate
```

Archive DB: `data/weather_archive.db` (gitignored pattern under `data/*.db`).

### Legacy model

`weather.high_temp.v0.3-*` remains as fallback. Prefer `models.weather.use_ai_forecaster: true`.

## Combos

- Eligible sets come from `GET /multivariate_event_collections`.
- Dependent legs without a joint model are skipped.

## Categories not yet modeled

Sports, player props, politics, and other series are discoverable but evaluated as **skip: no validated model** unless a registered model supports them.
