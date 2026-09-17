# Forecast candidate evaluation (A–D)

**Status:** research only · `live_eligible=false` · **no orders submitted**  
**Primary selection metric (declared before test inspection):** mean CRPS on the model-selection split; final untouched test CRPS reported separately.  
**Artifacts:** `data/obs_engine/multi/candidates/` (versioned; production artifacts frozen, not overwritten)

## Reproduce

```bash
cd kalshi_bot
pip install 'quantile-forest>=1.3'   # already in pyproject.toml
PYTHONPATH=src python3 -m kalshi_bot.cli weather-eval-candidates
PYTHONPATH=src python3 -m pytest tests/test_forecast_candidates.py -q
```

Report JSON: `data/obs_engine/multi/candidates/_reports/latest.json`

## What was verified

| Item | Result |
|------|--------|
| Shared predict path bias→floor ordering | Fixed in `multi/predict.py`; unit-tested |
| Settlement targets / label mismatch | Documented in `candidates/targets.py` (GHCND ≠ TWC/CLI) |
| Negative remain labels | Kept; audited (NYC 8, CHI 3, LAX 0) — mostly ASOS max above GHCND |
| Chronological 4-way split manifests | Persisted per location before tuning |
| Production freeze | Copied into each run’s `frozen_production_snapshot/` |
| Live eligibility | Still 0; multi-status `live_orders=false` |
| External NWS/HRRR same-vintage benchmark | **Unavailable / partial** — not claimed |

## Models trained

| ID | Method |
|----|--------|
| A | Corrected remain quantile-GBM + empirical residuals (bias before floor) |
| B | Same GBM + adaptive location-scale residuals \(Y^*=\mu + s\cdot z\), \(s=\max(q90-q10,s_{\min})\) |
| C | Quantile regression forest (`quantile-forest`, Meinshausen 2006) on remain |
| D | Direct-high quantile-GBM ablation (predict \(Y\) with \(M\) in features) |

Labels: GHCND TMAX. Decision hours 08/11/14 local. Availability assumption disclosed: archive `valid_utc` ≈ availability.

## Final test results (untouched chronological test)

Mean CRPS / MAE (°F) — lower is better. Searching four candidates increases selection uncertainty.

| Location | A CRPS / MAE | B CRPS / MAE | **C CRPS / MAE** | D CRPS / MAE |
|----------|-------------:|-------------:|-----------------:|-------------:|
| NYC | 1.691 / 2.355 | 1.675 / 2.363 | **1.649 / 2.349** | 2.123 / 2.914 |
| CHI | 1.802 / 2.438 | 1.775 / 2.453 | **1.620 / 2.247** | 2.140 / 2.921 |
| LAX | 1.136 / 1.512 | 1.118 / 1.505 | **0.999 / 1.355** | 1.344 / 1.787 |

**Winner (test CRPS):** C (QRF) at all three supported locations.  
**Loser:** D (direct-high) — remaining-rise formulation helps under the same features/budget.  
**B vs A:** small CRPS gain from adaptive residuals; not a large calibration leap.

Unsupported / insufficient in this checkout: MIA, AUS, DEN, HOU (no train profiles / ASOS+GHCND bundles).

## Probability calibration

- CRPS improved under C vs A on the final test at all locations.
- 80% interval coverage and contract Brier probes are in per-candidate `predictions_test.json`.
- Location-scale (B) and QRF PMFs are **not** guaranteed calibrated; tails outside training experience remain weakly supported.
- TWC settlement transfer still requires paired TWC evaluation; GHCND win ≠ live settlement eligibility.

## External benchmark

- NWS point forecast client exists for prospective pulls; **no archived same-vintage** series wired into this harness.
- HRRR: **not ingested** ([rapidrefresh.noaa.gov/hrrr](https://rapidrefresh.noaa.gov/hrrr/)). Do not substitute reanalysis.
- **No claim** that observation-only candidates beat external forecasts.

## Trading / profit

- Forecast evaluation only. **No profit evidence.** Paper/shadow paths unchanged; live blocks preserved.

## Rollback

1. Do not promote candidate joblibs into `multi/artifacts/*/station_corrected_v2.joblib` without a separate promotion review.
2. Production path remains the frozen snapshot + existing tuned calib.
3. Revert this branch / restore prior `predict.py` if bias-then-floor ordering must be undone (not recommended).

## Remaining blockers

- Matched historical NWS/HRRR issue-time archives.
- Longer official TWC history for settlement-source calibration without leakage.
- Date-block resampling CIs (not yet computed; temporal dependence acknowledged).
- Prospective shadow logging of full candidate distributions at decision time (infra ready via versioned artifacts; continuous collector not newly scheduled here).
