# Weather feeds pipeline (research / paper)

Public observation adapters for the NYC Central Park daily-max research engine.
**Live Kalshi orders stay disabled** (`live_eligible=false`; paper simulation never
calls live order submission).

## Settlement day definition

NWS CLI / Kalshi daily-max markets use **local standard time (LST, UTC−5)** for the
climate day. During Eastern Daylight Time, civil midnight–00:59 is still the previous
LST climate day. Operating code uses `lst_climate_day()` in
`feeds/climate_day.py`. Historical ASOS archives expose observation `valid_utc` only;
live FeedStore also records `first_seen_utc`. Historical tests disclose the assumption
that archive `valid_utc` ≈ availability.

## Connections

| Feed | Source | Notes |
|------|--------|-------|
| METAR | aviationweather.gov | KNYC/KLGA/KJFK; `altim` hPa→inHg; precip often missing (explicit) |
| CLI | NWS API | Matched by station + target LST day + decision-time availability |
| GOES | `s3://noaa-goes19` | Collected; RTM model-assist disclosed; **not** consumed by station_v2 model |
| NEXRAD | `s3://unidata-nexrad-level3` | Collected; **not** consumed by station_v2 model |
| NYS Mesonet | — | Optional / blocked — not required for these fixes |

## Feature schema `station_v2.1`

Shared builder: `feeds/feature_schema.py` (`build_station_v2_features`).

- Same path for training, historical replay, and live inference.
- Missing wind / alti / dewpoint / precip are **never** invented as calm / 30.00 inHg / zero precip.
- Coverage requirements (configurable in `climate_day.py`): ≥4 temp obs,
  start gap from 00:00 LST ≤4.0h, max inter-obs gap ≤3.5h, stale (last obs → decision) ≤2.5h.
  Partial windows set `max_so_far_status=partial_window` and **block** paper entry (`insufficient_data`).
  Partial maxima are never treated as verified full-day maxima.
- Fractional METAR `max_so_far` (e.g. 69.08°F) is **not** treated as official floor 70°F.
  Only whole-°F CLI values may apply an integer soft floor.
- Live FeedStore rows carry `first_seen_utc`; replay excludes obs first seen after the decision.
  Archive rows without availability timestamps use the disclosed `valid_utc≈availability` assumption.

## Supported decision times

Actionable forecasts: local civil **08:00, 11:00, 14:00** only (validated hours).
Other times return `unsupported_decision_time` with `next_supported_run_*`.
Same-day distribution is **never** applied to another market day (`unsupported_horizon`).

## Probability method

**Calibrated empirical residual distribution by decision hour** (not three quantiles alone).

1. Fit quantile GBM remain models on chronological train days.
2. Fit residuals on a held-out **calibration** day set (excluded from train and test).
3. Production shifts calibration residuals onto `max_so_far + q50`.
4. If calibration `n < 40`, probabilities are **unavailable** and paper entry is blocked.
5. No fabricated residual fallbacks.

Production evaluation reports the **emitted distribution median MAE** and 80% interval
coverage — not raw regressor MAE as a substitute.

## Attribution

Reports separately:

- `feeds_collected`
- `feeds_quality_ok`
- `features_consumed_by_model` / `feeds_contributing_to_prediction` (station METAR for station_v2)
- `settlement_constraints_applied` (same-day CLI only)

GOES/NEXRAD must not appear as contributing to the station-only prediction.

## Paper simulation

- Evaluates YES and NO EV using executable asks, depth, Kalshi quadratic taker fees
  (`fees = M * 0.07 * C * P * (1-P)` + conservative rounding; see `api/fees.py`),
  and configured uncertainty buffer — **no 55% probability gate before EV**.
- Distinct reasons: unsupported model/time/horizon, insufficient data, calibration
  unavailable, unavailable prices, insufficient depth, negative EV, duplicate, cash/risk,
  total/per-market exposure.
- Positions keyed by `(ticker, side)` — YES then NO are **separate** positions (not netted into one side).
- `PaperLedger.settle_market` applies payouts once (idempotent `settlement_key`), updates cash,
  clears positions, records realized PnL including fees.
- Shared configurable paper budget / exposure across locations (`data/obs_engine/multi/paper_ledger.db`
  for multi-location cycles; NYC worker still uses `feeds/paper_ledger.db`).
- Simulated fills labeled `unvalidated`; `live_order_submitted=false` always.

## Multi-location architecture

Scope: every Kalshi Climate/Weather series discovered via the official API, with a
persistent registry (`data/obs_engine/multi/registry.db`) linking series → settlement
target. Settlement stations are **never** inferred from city names without evidence.

- Verified NWS CLI daily-max mappings (initial): HIGHNY (Central Park), HIGHCHI (Midway),
  HIGHMIA, HIGHAUS, KXDENHIGH, KXHIGHOU (+ Houston aliases).
- Weather Company (TWC) daily-max adapter (`feeds/twc_kalshi.py`) reads the public
  weather.com/kalshi portal JSON (`/kalshi/api/climate/primary`, `/kalshi/api/metar`).
  Verified series: KXHIGHNY (CLINYC/KNYC), KXHIGHCHI (CLIMDW/KMDW), KXHIGHLAX, KXHIGHAUS,
  KXHIGHDEN, KXHIGHPHIL, KXHIGHMIA. Settlement floors use TWC climate reports — never NWS CLI.
  Same-ICAO residual transfer (e.g. NYC station_v2 → KXHIGHNY) is labeled exploratory.
- Unmapped TWC series stay `blocked_incomplete_mapping`.
- Ambiguous CLI-only mappings (`issuedby` known, METAR unsettled) are recorded explicitly
  and skipped for operating inference.
- `ForecastContext` partitions location / measurement / climate day / decision time /
  model / feature schema through collection → features → unified predict → paper.
- Unified predict: `multi/predict.py::predict_station_v2` (shared by calib/eval/replay/operating).
- Location-specific models required; NYC `station_corrected_v2` is **not** transferred silently.
- One location failure does not stop others.

## Commands

```bash
cd kalshi_bot
# NYC feed worker (legacy path, still location-parameterized defaults)
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-train
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-once
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-feeds-run --interval 300
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-status
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-stop

# Multi-location
PYTHONPATH=src python3 -m kalshi_bot.cli weather-discover
PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-once
PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-status

# Tests
PYTHONPATH=src python3 -m pytest tests/test_multi_location.py tests/test_feeds_pipeline_fixes.py -q
```

Worker health distinguishes fetch success, usable METAR, and inference outcomes
(`ok` / `degraded_blocked_forecast` / `unhealthy_*`). Off-hours blocked runs are
**not** reported as end-to-end forecast success.

## Artifacts

- Operating model (NYC): `data/obs_engine/feeds/models/station_corrected_v2.joblib`
- Calibration (NYC): `data/obs_engine/feeds/models/station_corrected_v2_calibration.json`
- Per-location artifacts: `data/obs_engine/multi/artifacts/<location_id>__<measurement>/`
- Registry / discovery: `data/obs_engine/multi/registry.db`, `last_discovery.json`
- Multi cycle report: `data/obs_engine/multi/last_multi_cycle.json`
- Frozen baseline (untouched): `data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib`
- Paper ledger (NYC worker): `data/obs_engine/feeds/paper_ledger.db`
- Shared multi paper ledger: `data/obs_engine/multi/paper_ledger.db`
