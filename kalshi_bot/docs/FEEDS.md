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
- Coverage requirements: ≥4 temp obs, first obs by 10:00 LST, max gap ≤3.5h. Partial windows
  set `max_so_far_status=partial_window` and **block** paper entry (`insufficient_data`).
- Fractional METAR `max_so_far` (e.g. 69.08°F) is **not** treated as official floor 70°F.
  Only whole-°F CLI values may apply an integer soft floor.

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
  unavailable, unavailable prices, insufficient depth, negative EV, duplicate, cash/risk.
- `PaperLedger` persists cash/fills/positions across restarts; duplicate `client_order_id` blocked.
- Simulated fills labeled `unvalidated`; `live_order_submitted=false` always.

## Commands

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-train
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-once
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-feeds-run --interval 300
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-status
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-stop
```

Worker health distinguishes fetch success, usable METAR, and inference outcomes
(`ok` / `degraded_blocked_forecast` / `unhealthy_*`).

## Artifacts

- Operating model: `data/obs_engine/feeds/models/station_corrected_v2.joblib`
- Calibration: `data/obs_engine/feeds/models/station_corrected_v2_calibration.json`
- Frozen baseline (untouched): `data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib`
- Paper ledger: `data/obs_engine/feeds/paper_ledger.db`
