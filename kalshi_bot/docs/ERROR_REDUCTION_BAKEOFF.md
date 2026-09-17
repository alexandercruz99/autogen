# Midway error-reduction bake-off

Chronological test on **Chicago Midway** (train earlier days → hold out later). Goal: move toward **±0.5°F**.

## What the books / ops practice say

- **Diurnal cycle:** Daily max usually lags solar noon; heating continues until energy surplus flips to deficit (Penn State Meteo; WKU temperature lab).
- **Forecasters’ Reference Book:** Daytime surface-temperature rise is driven by insolation, clouds, and the existing sounding — remaining rise shrinks as the afternoon wears on.
- **MOS / cloud corrections:** Sky cover modulates Tmax error; cloudy days warm less.
- **Nowcasting:** Final max ≥ max-so-far; once cooling onset is clear, remaining rise ≈ 0.

## Methods tested

| # | Method | Idea |
|---|--------|------|
| 1 | `baseline_station_v2` | Current observation GBM remaining-rise model |
| 2 | `diurnal_climatology` | Typical remain by season + hour |
| 3 | `sky_conditioned_clim` | Same, split clear vs cloudy (MOS-style) |
| 4 | `peak_lock_hybrid` | Model, but lock to max-so-far on cooling onset |
| 5 | `blend_model_clim` | 60% model + 40% climatology |
| 6 | `persistence_max_so_far` | Assume high already in |
| + | `hour_specialist` (extra) | Separate GBM trained only on that decision hour |

Neighbor ORD−MDW temp delta vs remaining rise at 11 CDT: **correlation ≈ 0.09** (weak — not a free lunch).

## Core operating hours (8 / 11 / 14) — test set

| Method | MAE °F | Within ±0.5°F | Within ±1°F | Bias |
|--------|-------:|--------------:|------------:|-----:|
| **baseline_station_v2** | **2.43** | 20.9% | 36.9% | −0.61 |
| peak_lock_hybrid | 2.46 | 19.4% | 36.6% | −0.67 |
| blend_model_clim | 2.55 | 21.0% | 36.0% | −0.54 |
| diurnal_climatology | 2.98 | 21.2% | 41.8% | −0.45 |
| sky_conditioned_clim | 2.99 | 22.1% | 41.4% | −0.02 |
| persistence_max_so_far | 5.98 | 5.6% | 23.4% | −5.98 |

**Winner at noon-relevant hours: still the baseline model.** Climatology/sky/blend did not beat it on MAE for 8/11/14.

## By local hour — when ±0.5°F becomes plausible

| Hour CDT | Baseline MAE | ±0.5°F | Hour-specialist MAE | Specialist ±0.5°F | Persistence ±0.5°F |
|---------:|-------------:|-------:|--------------------:|------------------:|-------------------:|
| 8 | 3.79 | 10% | — | — | 2% |
| **11** | **2.42** | **14%** | — | — | 4% |
| 12 | 2.01 | 17% | — | — | 5% |
| 13 | 1.59 | 25% | — | — | 7% |
| 14 | 1.08 | 38% | 1.07 | 38% | 11% |
| **15** | 0.92 | 37% | **0.79** | **48%** | 19% |
| **16** | 0.88 | 35% | **0.66** | **50%** | 28% |

Only **blend at hour 16** and **hour-specialist near 15–16** reach ~**50%** of days within ±0.5°F. Morning/noon never do on this test set.

## Honest conclusion

1. **You cannot get reliable ±0.5°F at ~11–12 CDT** with these observation-only methods on Midway. Best noon window is still ~2.0–2.4°F MAE (~14–17% within half a degree).
2. **Path toward ±0.5°F:** wait until **15–16 CDT**, use an **hour-specific** remain model (or persistence once the high is locked). Specialist MAE ≈ **0.66–0.79°F**, ~**48–50%** within ±0.5°F — closest we got.
3. **Climatology / sky / peak-lock / ORD neighbor** did not unlock half-degree skill at noon.
4. **Trading:** for ~noon bets, optimize **bracket probabilities / EV**, not half-degree points. For half-degree-ish points, trade **late afternoon** when markets are tighter and edge may be gone.

## Artifacts

- `data/obs_engine/multi/artifacts/chi_midway__daily_max_temp_f/error_reduction_bakeoff.json`
- `data/obs_engine/multi/artifacts/chi_midway__daily_max_temp_f/hour_specialist_extra.json`
- Script: `scripts/error_reduction_bakeoff.py`
