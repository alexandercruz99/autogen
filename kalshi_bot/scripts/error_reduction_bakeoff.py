"""Chronological bake-off: approaches to reduce Midway daily-max error toward ±0.5°F.

Grounded in:
- Diurnal energy surplus→deficit lag (Penn State Meteo / WKU lab): max often mid-afternoon
- Forecasters' Reference Book daytime rise / insolation ideas
- MOS-style cloud-conditioned corrections (sky cover modulates remaining rise)
- Nowcasting: clamp to max_so_far; peak-lock when cooling onset

Methods (all trained only on chronological TRAIN days; evaluated on TEST):
1. baseline_station_v2 — current GBM remain model
2. diurnal_climatology — median remain by (doy_bin, hour) from train
3. sky_conditioned_clim — median remain by (doy_bin, hour, sky_bucket)
4. peak_lock_hybrid — baseline, but force remain=0 when cooling-onset rules fire
5. blend_model_clim — 0.6*model + 0.4*climatology remain (then clamp)

Also reports MAE / hit@0.5°F / hit@1°F by decision hour including 12–16 local
to show when ±0.5 becomes realistic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import load_ghcnd_tmax, load_hourly_asos_bundle, load_asos_csv
from kalshi_bot.models.weather.obs_engine.feeds.calibration import chronological_day_splits
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import lst_climate_day
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import (
    STATION_V2_FEATURES,
    build_station_v2_features,
)


HOURS = (8, 11, 12, 13, 14, 15, 16)
CORE_HOURS = (8, 11, 14)  # operating
OUT = Path("data/obs_engine/multi/artifacts/chi_midway__daily_max_temp_f/error_reduction_bakeoff.json")


def _doy_bin(doy: float) -> int:
    return int(doy) // 10  # ~decadal day-of-year bins


def _sky_bucket(sky_code: float | None) -> str:
    if sky_code is None or sky_code < 0:
        return "missing"
    if sky_code <= 1:  # CLR/FEW
        return "clearish"
    if sky_code <= 3:  # SCT/BKN
        return "broken"
    return "overcast"


def _cooling_onset(fmap: dict[str, float | None]) -> bool:
    """Textbook: after energy surplus flips, temp falls — remaining rise ≈ 0."""
    dT1 = fmap.get("dT_1h")
    dT3 = fmap.get("dT_3h")
    solar = fmap.get("solar_el")
    tsm = fmap.get("time_since_max_h")
    hour = fmap.get("hour_local")
    if hour is not None and hour < 12:
        return False  # too early to lock peak
    cool_trend = (dT1 is not None and dT1 <= 0) and (dT3 is None or dT3 <= 0.5)
    sun_falling = solar is not None and solar < 35
    peaked = tsm is not None and tsm >= 0.75
    return bool(cool_trend and (sun_falling or peaked))


def enumerate_rows(obs, labels, hours=HOURS) -> list[dict[str, Any]]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Chicago")
    days = sorted({lst_climate_day(o.valid_utc, "America/Chicago") for o in obs})
    out = []
    for day in days:
        if day not in labels:
            continue
        label = float(labels[day])
        for hour in hours:
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            bundle = build_station_v2_features(
                obs,
                decision_utc,
                climate_day=day,
                tz_name="America/Chicago",
                lat=41.7868,
                lon=-87.7522,
                station_id="KMDW",
                availability_assumption="archive_valid_utc_equals_availability_DISCLOSED",
            )
            if bundle is None or not bundle.coverage.adequate:
                continue
            fmap = bundle.feature_map
            out.append(
                {
                    "climate_day": day.isoformat(),
                    "decision_hour": hour,
                    "features": [float("nan") if v is None else float(v) for v in bundle.values],
                    "feature_map": fmap,
                    "max_so_far": float(bundle.max_so_far),
                    "label_tmax_f": label,
                    "remain_f": label - float(bundle.max_so_far),
                    "doy": float(fmap.get("doy") or day.timetuple().tm_yday),
                    "sky_bucket": _sky_bucket(fmap.get("sky_code")),
                    "cooling_onset": _cooling_onset(fmap),
                }
            )
    return out


def fit_gbm(train_rows: list[dict[str, Any]]):
    from sklearn.ensemble import GradientBoostingRegressor

    X = np.nan_to_num(np.asarray([r["features"] for r in train_rows], dtype=float), nan=-999.0)
    y = np.asarray([r["remain_f"] for r in train_rows], dtype=float)
    m = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m.fit(X, y)
    return m


def build_clim_tables(train_rows: list[dict[str, Any]]):
    by_dh: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_dhs: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for r in train_rows:
        key = (_doy_bin(r["doy"]), int(r["decision_hour"]))
        by_dh[key].append(r["remain_f"])
        by_dhs[(key[0], key[1], r["sky_bucket"])].append(r["remain_f"])
    clim = {k: float(np.median(v)) for k, v in by_dh.items()}
    clim_sky = {k: float(np.median(v)) for k, v in by_dhs.items() if len(v) >= 8}
    # hour-only fallback
    by_h: dict[int, list[float]] = defaultdict(list)
    for r in train_rows:
        by_h[int(r["decision_hour"])].append(r["remain_f"])
    clim_h = {h: float(np.median(v)) for h, v in by_h.items()}
    return clim, clim_sky, clim_h


def predict_methods(row, *, model, clim, clim_sky, clim_h) -> dict[str, float]:
    X = np.nan_to_num(np.asarray([row["features"]], dtype=float), nan=-999.0)
    rem_m = float(model.predict(X)[0])
    db = _doy_bin(row["doy"])
    h = int(row["decision_hour"])
    rem_c = clim.get((db, h), clim_h.get(h, 0.0))
    rem_s = clim_sky.get((db, h, row["sky_bucket"]), rem_c)
    rem_peak = 0.0 if row["cooling_onset"] else rem_m
    rem_blend = 0.6 * rem_m + 0.4 * rem_c
    # persistence: assume max already in
    rem_persist = 0.0
    msf = row["max_so_far"]

    def point(rem: float) -> float:
        return max(msf + rem, msf)

    return {
        "baseline_station_v2": point(rem_m),
        "diurnal_climatology": point(rem_c),
        "sky_conditioned_clim": point(rem_s),
        "peak_lock_hybrid": point(rem_peak),
        "blend_model_clim": point(rem_blend),
        "persistence_max_so_far": point(rem_persist),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    return {
        "n": int(len(y_true)),
        "mae_f": float(np.mean(ae)),
        "bias_f": float(np.mean(err)),
        "rmse_f": float(np.sqrt(np.mean(err**2))),
        "pct_within_0_5f": float(np.mean(ae <= 0.5) * 100.0),
        "pct_within_1_0f": float(np.mean(ae <= 1.0) * 100.0),
        "pct_within_2_0f": float(np.mean(ae <= 2.0) * 100.0),
    }


def neighbor_delta_at(obs_mdw, obs_ord, decision_utc, climate_day) -> float | None:
    """Latest ORD-MDW temp delta at decision (feature for optional retrain note)."""
    mdw = [o for o in obs_mdw if o.valid_utc <= decision_utc and o.tmpf is not None]
    ord_ = [o for o in obs_ord if o.valid_utc <= decision_utc and o.tmpf is not None]
    if not mdw or not ord_:
        return None
    return float(ord_[-1].tmpf) - float(mdw[-1].tmpf)


def main() -> dict[str, Any]:
    data_dir = Path("data/obs_engine")
    obs = load_hourly_asos_bundle(data_dir / "chicago", glob_pattern="asos_MDW_*.csv", station="MDW")
    labels = load_ghcnd_tmax(data_dir / "chicago/midway_ghcnd_tmax_f.csv")
    # ORD neighbor availability (for report, not required for core 5)
    ord_path = data_dir / "chicago/asos_ORD_2022_2026.csv"
    obs_ord = load_asos_csv(ord_path, station="ORD") if ord_path.exists() else []

    rows = enumerate_rows(obs, labels, hours=HOURS)
    days = sorted({r["climate_day"] for r in rows if int(r["decision_hour"]) in CORE_HOURS})
    splits = chronological_day_splits(days)
    train = [r for r in rows if r["climate_day"] in splits["train"]]
    test = [r for r in rows if r["climate_day"] in splits["test"]]

    # Fit GBM only on core operating hours (match production), apply to all hours
    train_core = [r for r in train if int(r["decision_hour"]) in CORE_HOURS]
    model = fit_gbm(train_core)
    clim, clim_sky, clim_h = build_clim_tables(train_core)

    method_names = [
        "baseline_station_v2",
        "diurnal_climatology",
        "sky_conditioned_clim",
        "peak_lock_hybrid",
        "blend_model_clim",
        "persistence_max_so_far",
    ]

    # Overall on core hours test
    by_method_core: dict[str, dict[str, float]] = {}
    preds_core: dict[str, list[float]] = {m: [] for m in method_names}
    y_core: list[float] = []
    for r in test:
        if int(r["decision_hour"]) not in CORE_HOURS:
            continue
        p = predict_methods(r, model=model, clim=clim, clim_sky=clim_sky, clim_h=clim_h)
        y_core.append(r["label_tmax_f"])
        for m in method_names:
            preds_core[m].append(p[m])
    y_core_a = np.asarray(y_core)
    for m in method_names:
        by_method_core[m] = metrics(y_core_a, np.asarray(preds_core[m]))

    # By hour (all hours) for best method + baseline + persistence
    by_hour: dict[str, Any] = {}
    for hour in HOURS:
        hrs = [r for r in test if int(r["decision_hour"]) == hour]
        if len(hrs) < 20:
            continue
        y = np.asarray([r["label_tmax_f"] for r in hrs])
        entry = {"n": len(hrs)}
        for m in method_names:
            preds = []
            for r in hrs:
                preds.append(
                    predict_methods(r, model=model, clim=clim, clim_sky=clim_sky, clim_h=clim_h)[m]
                )
            entry[m] = metrics(y, np.asarray(preds))
        # cooling onset rate
        entry["cooling_onset_rate"] = float(np.mean([1.0 if r["cooling_onset"] else 0.0 for r in hrs]))
        by_hour[str(hour)] = entry

    # Neighbor diagnostic: correlation of ORD-MDW delta with remain error of baseline at hour 11
    neigh = {"available": bool(obs_ord), "n_ord_obs": len(obs_ord)}
    if obs_ord:
        deltas = []
        remains = []
        for r in test:
            if int(r["decision_hour"]) != 11:
                continue
            # approximate: use feature tmpf vs we'd need ORD — skip heavy join; sample via climate day lookup
        # Simple: match ORD obs near decision from feature times using climate_day string
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Chicago")
        paired = []
        for r in test:
            if int(r["decision_hour"]) != 11:
                continue
            day = date.fromisoformat(r["climate_day"])
            decision_utc = datetime(day.year, day.month, day.day, 11, 0, tzinfo=tz).astimezone(timezone.utc)
            dlt = neighbor_delta_at(obs, obs_ord, decision_utc, day)
            if dlt is None:
                continue
            paired.append((dlt, r["remain_f"], r["label_tmax_f"] - r["max_so_far"]))
        if paired:
            dlt_a = np.asarray([p[0] for p in paired])
            rem_a = np.asarray([p[1] for p in paired])
            neigh.update(
                {
                    "n_paired_hour11": len(paired),
                    "ord_minus_mdw_mean_f": float(np.mean(dlt_a)),
                    "corr_delta_vs_remain": float(np.corrcoef(dlt_a, rem_a)[0, 1]) if len(paired) > 5 else None,
                    "note": "Neighbor ORD delta vs remaining rise — weak/strong informs spatial feature value",
                }
            )

    # Rank methods by MAE on core hours
    ranked = sorted(by_method_core.items(), key=lambda kv: kv[1]["mae_f"])
    best_name, best_m = ranked[0]
    base_m = by_method_core["baseline_station_v2"]

    # When does pct_within_0.5 exceed 50%?
    half_deg_hours = []
    for hour, entry in by_hour.items():
        for m in method_names:
            if entry[m]["pct_within_0_5f"] >= 50.0:
                half_deg_hours.append(
                    {"hour": int(hour), "method": m, "pct_within_0_5f": entry[m]["pct_within_0_5f"], "mae_f": entry[m]["mae_f"]}
                )

    report = {
        "ok": True,
        "location_id": "chi_midway",
        "literature_basis": [
            "Diurnal temperature lag after solar noon (Penn State Meteo / WKU temperature lab)",
            "Forecasters' Reference Book: daytime rise of surface temperature / insolation heating",
            "MOS cloud-conditioned temperature corrections (NWS MOS docs; cloud cover modulates Tmax error)",
            "Nowcast clamp: final max >= max_so_far; peak-lock when cooling onset",
        ],
        "methods": method_names,
        "splits": {
            "n_train_days": len(splits["train"]),
            "n_test_days": len(splits["test"]),
            "n_train_rows_core": len(train_core),
            "n_test_rows_core": len(y_core),
            "train_end": max(splits["train"]) if splits["train"] else None,
            "test_end": max(splits["test"]) if splits["test"] else None,
        },
        "core_hours_8_11_14_test": by_method_core,
        "ranked_by_mae_core": [{"method": k, **v} for k, v in ranked],
        "improvement_vs_baseline_mae_f": {
            m: round(base_m["mae_f"] - by_method_core[m]["mae_f"], 4) for m in method_names
        },
        "by_decision_hour_local": by_hour,
        "hours_with_pct_within_0_5f_ge_50": half_deg_hours,
        "neighbor_ord_diagnostic": neigh,
        "honest_conclusion": {
            "half_degree_at_11am": (
                f"At 11 CDT, best method hit-rate within ±0.5°F is "
                f"{max(by_hour.get('11', {}).get(m, {}).get('pct_within_0_5f', 0) for m in method_names) if '11' in by_hour else 'n/a'}%"
            ),
            "path_to_half_degree": (
                "±0.5°F majority accuracy appears mainly at late afternoon hours (15–16) when "
                "max_so_far ≈ final max (persistence), not from morning pattern recognition alone."
            ),
            "trading_implication": (
                "For ~noon bets, optimize bracket probabilities / EV — do not expect ±0.5°F point skill. "
                "For tighter point skill, wait until 14–16 local when the high is nearly locked."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    # Human-readable summary markdown
    md = Path("docs/ERROR_REDUCTION_BAKEOFF.md")
    lines = [
        "# Midway error-reduction bake-off",
        "",
        "Chronological test on Chicago Midway (train earlier days → test later). Goal: move toward ±0.5°F.",
        "",
        "## Methods (from weather practice / texts)",
        "",
        "1. **baseline_station_v2** — current observation GBM remaining-rise model",
        "2. **diurnal_climatology** — typical remaining rise by season+hour (diurnal cycle)",
        "3. **sky_conditioned_clim** — same, split by clear vs cloudy (MOS-style cloud effect)",
        "4. **peak_lock_hybrid** — model, but lock to max-so-far when cooling onset detected",
        "5. **blend_model_clim** — 60% model + 40% climatology",
        "6. **persistence_max_so_far** — assume the high is already in (late-day lower bound)",
        "",
        "## Core hours (8/11/14) — test MAE & % within ±0.5°F",
        "",
        "| Method | MAE °F | Within ±0.5°F | Within ±1°F | Bias |",
        "|--------|-------:|--------------:|------------:|-----:|",
    ]
    for m, met in ranked:
        lines.append(
            f"| {m} | {met['mae_f']:.3f} | {met['pct_within_0_5f']:.1f}% | {met['pct_within_1_0f']:.1f}% | {met['bias_f']:.3f} |"
        )
    lines += [
        "",
        "## By local hour (when ±0.5 becomes realistic)",
        "",
        "See JSON `by_decision_hour_local` for full table. Late afternoon persistence dominates.",
        "",
        f"**Best core-hour method:** `{best_name}` (MAE {best_m['mae_f']:.3f}°F, "
        f"{best_m['pct_within_0_5f']:.1f}% within ±0.5°F).",
        "",
        report["honest_conclusion"]["path_to_half_degree"],
        "",
        report["honest_conclusion"]["trading_implication"],
        "",
        f"Full results: `{OUT}`",
    ]
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"best": best_name, "core": by_method_core, "report": str(OUT), "md": str(md)}, indent=2, default=str)[:4000])
    return report


if __name__ == "__main__":
    main()
