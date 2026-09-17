"""Controlled chronological feature experiments + diagnostics export."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.backtest import (
    BACKTEST_HOURS,
    _block_bootstrap_mae_diff,
    _chrono_day_splits,
    load_open_meteo_historical_highs,
)
from kalshi_bot.models.weather.obs_engine.data import (
    default_data_dir,
    load_asos_csv,
    load_ghcnd_tmax,
    load_nyc_hourly_bundle,
    local_date_for,
)
from kalshi_bot.models.weather.obs_engine.research.features_v2 import (
    BASELINE_FEATURES,
    CLOUD_PRECIP_FEATURES,
    LOCAL_V2_FEATURES,
    NEIGHBOR_FEATURES,
    build_baseline,
    build_cloud_precip,
    build_local_v2,
    build_neighbor,
)

logger = logging.getLogger(__name__)


def _load_lga(data_dir: Path):
    rows = []
    for p in sorted(data_dir.glob("asos_LGA_*.csv")):
        rows.extend(load_asos_csv(p, station="LGA"))
    rows.sort(key=lambda r: r.valid_utc)
    return rows


def _enumerate(
    builder: Callable,
    obs_nyc,
    obs_lga,
    labels: dict,
    hours: tuple[int, ...] = BACKTEST_HOURS,
) -> list[dict[str, Any]]:
    tz = ZoneInfo = __import__("zoneinfo").ZoneInfo
    tz = ZoneInfo(NYC_TARGET.timezone)
    days = sorted({local_date_for(o.valid_utc) for o in obs_nyc})
    out = []
    for day in days:
        if day not in labels:
            continue
        label = labels[day]
        for hour in hours:
            local_dt = datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)
            decision_utc = local_dt.astimezone(timezone.utc)
            if builder in (build_neighbor, build_cloud_precip):
                feats = builder(obs_nyc, obs_lga, decision_utc, climate_day=day)
            else:
                feats = builder(obs_nyc, decision_utc, climate_day=day)
            if feats is None:
                continue
            remain = float(label) - feats.max_so_far
            out.append(
                {
                    "climate_day": day.isoformat(),
                    "decision_hour": hour,
                    "decision_time_utc": decision_utc.isoformat(),
                    "features": feats.values,
                    "max_so_far": feats.max_so_far,
                    "label_tmax_f": label,
                    "remain_f": remain,
                    "provenance": feats.provenance,
                }
            )
    return out


def _fit_predict_mae(train_rows, eval_rows) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor

    if len(train_rows) < 50 or len(eval_rows) < 10:
        return {"ok": False, "reason": "insufficient rows"}
    Xtr = np.asarray([r["features"] for r in train_rows], dtype=float)
    ytr = np.asarray([r["remain_f"] for r in train_rows], dtype=float)
    # Replace nan
    Xtr = np.nan_to_num(Xtr, nan=-999.0)
    m = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m.fit(Xtr, ytr)
    by_h: dict[str, Any] = {}
    all_err_m, all_err_c, all_err_cont, days = [], [], [], []
    for hour in BACKTEST_HOURS:
        hrs = [r for r in eval_rows if int(r["decision_hour"]) == hour]
        if not hrs:
            continue
        X = np.nan_to_num(np.asarray([r["features"] for r in hrs], dtype=float), nan=-999.0)
        rem = m.predict(X)
        max_so = np.asarray([r["max_so_far"] for r in hrs], dtype=float)
        y = np.asarray([r["label_tmax_f"] for r in hrs], dtype=float)
        pred = np.maximum(max_so + rem, max_so)
        err = list(y - pred)
        err_cont = list(y - max_so)
        # month clim from train
        month_vals = defaultdict(list)
        seen = set()
        for r in train_rows:
            if r["climate_day"] in seen:
                continue
            seen.add(r["climate_day"])
            month_vals[date.fromisoformat(r["climate_day"]).month].append(int(r["label_tmax_f"]))
        clim = {k: float(np.mean(v)) for k, v in month_vals.items()}
        gclim = float(np.mean([x for xs in month_vals.values() for x in xs])) if month_vals else 70.0
        err_clim = [float(r["label_tmax_f"]) - clim.get(date.fromisoformat(r["climate_day"]).month, gclim) for r in hrs]
        day_ids = [r["climate_day"] for r in hrs]
        by_h[str(hour)] = {
            "n_rows": len(hrs),
            "n_days": len(set(day_ids)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "mae_continuation": float(np.mean(np.abs(err_cont))),
            "mae_climatology": float(np.mean(np.abs(err_clim))),
            "bootstrap_vs_continuation": _block_bootstrap_mae_diff(err, err_cont, day_ids=day_ids),
        }
        all_err_m.extend(err)
        all_err_c.extend(err_clim)
        all_err_cont.extend(err_cont)
        days.extend(day_ids)
    return {"ok": True, "by_hour": by_h, "n_eval_rows": len(eval_rows)}


def run_feature_experiments(*, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    research_dir = data_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    obs = load_nyc_hourly_bundle(data_dir)
    lga = _load_lga(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")

    builders = {
        "baseline": (build_baseline, BASELINE_FEATURES),
        "local_v2": (build_local_v2, LOCAL_V2_FEATURES),
        "neighbor": (build_neighbor, NEIGHBOR_FEATURES),
        "cloud_precip": (build_cloud_precip, CLOUD_PRECIP_FEATURES),
    }

    # Build once per set
    row_sets = {}
    for name, (builder, feats) in builders.items():
        rows = _enumerate(builder, obs, lga, labels)
        row_sets[name] = rows

    # Align on common (day, hour) keys present in ALL candidates (coverage fairness)
    def key(r):
        return (r["climate_day"], r["decision_hour"])

    common = set(key(r) for r in row_sets["baseline"])
    for name, rows in row_sets.items():
        common &= set(key(r) for r in rows)
    # Also require neighbor not using sentinel-only? Keep all common keys; report LGA missing rate separately

    aligned = {}
    for name, rows in row_sets.items():
        m = {key(r): r for r in rows}
        aligned[name] = [m[k] for k in sorted(common)]

    days = sorted({r["climate_day"] for r in aligned["baseline"]})
    # Chronological: train 60%, val 20%, final test 20%
    # After using val to pick winner, test is one-shot. Prior inspected period (2025-10+)
    # guided earlier work — disclose that final test still overlaps previously reviewed span.
    n = len(days)
    i_tr = int(n * 0.60)
    i_va = int(n * 0.80)
    split = {
        "train": set(days[:i_tr]),
        "val": set(days[i_tr:i_va]),
        "test": set(days[i_va:]),
    }

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_independent_days_aligned": len(days),
        "n_rows_aligned": len(aligned["baseline"]),
        "coverage_loss_vs_baseline_only": {
            name: len(row_sets["baseline"]) - len(aligned[name]) for name in aligned
        },
        "splits": {k: {"n_days": len(v), "first": min(v) if v else None, "last": max(v) if v else None} for k, v in split.items()},
        "previously_reviewed_disclosure": (
            "Final test days may overlap the earlier v1.1 backtest inspection window. "
            "Treat as reviewed for promotion purposes; prospective collector needed for untouched confirmation."
        ),
        "candidates": {},
        "satellite_radar": {
            "status": "not_integrated",
            "blocker": "No GOES-R/NEXRAD/NYS Mesonet credentials or scoped download this run; cloud_precip uses METAR/ASOS only.",
        },
        "live_eligible": False,
    }

    val_scores = {}
    for name, rows in aligned.items():
        train = [r for r in rows if r["climate_day"] in split["train"]]
        val = [r for r in rows if r["climate_day"] in split["val"]]
        test = [r for r in rows if r["climate_day"] in split["test"]]
        val_m = _fit_predict_mae(train, val)
        test_m = _fit_predict_mae(train + val, test)  # refit including val after selection? 
        # For selection use val only; for final report refit on train+val then score test once
        results["candidates"][name] = {
            "n_features": len(builders[name][1]),
            "feature_names": builders[name][1],
            "validation": val_m,
            "test_refit_train_plus_val": test_m,
        }
        # Selection score: mean MAE across hours on val
        if val_m.get("ok"):
            maes = [v["mae"] for v in val_m["by_hour"].values()]
            val_scores[name] = float(np.mean(maes))

    if val_scores:
        winner = min(val_scores, key=val_scores.get)
        baseline_val = val_scores.get("baseline")
        results["selection"] = {
            "metric": "macro_mean_mae_across_hours_on_validation",
            "val_scores": val_scores,
            "selected": winner,
            "improved_vs_baseline_on_val": (
                baseline_val is not None and val_scores[winner] < baseline_val - 0.05
            ),
            "note": "Require >=0.05°F macro MAE gain on val to prefer over baseline for inference.",
        }
    else:
        results["selection"] = {"selected": "baseline", "reason": "no val scores"}

    # OM retrospective comparison on test for selected vs baseline (disclosed)
    om = load_open_meteo_historical_highs(data_dir / "open_meteo_nyc_tmax_historical.csv")
    results["external_benchmark"] = {
        "open_meteo_historical_forecast_api": {
            "provenance": "RETROSPECTIVE — Historical Forecast API stitches early hours of successive runs",
            "docs": "https://open-meteo.com/en/docs/historical-forecast-api",
            "single_runs_api": "https://open-meteo.com/en/docs/single-runs-api — not wired for decision-time cutoffs this run",
            "n_days_cached": len(om),
            "fair_decision_time_nws": False,
        }
    }

    out_path = research_dir / "experiment_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    results["saved_path"] = str(out_path)
    return results


def export_diagnostics(*, data_dir: Path | None = None, split: str = "test") -> dict[str, Any]:
    """One record per station-day × decision hour for baseline model errors."""
    data_dir = data_dir or default_data_dir()
    research_dir = data_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    obs = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    rows = _enumerate(build_baseline, obs, [], labels)
    days = sorted({r["climate_day"] for r in rows})
    n = len(days)
    i_tr = int(n * 0.60)
    i_va = int(n * 0.80)
    sets = {"train": set(days[:i_tr]), "val": set(days[i_tr:i_va]), "test": set(days[i_va:])}
    train = [r for r in rows if r["climate_day"] in sets["train"]]
    hold = [r for r in rows if r["climate_day"] in sets[split]]

    from sklearn.ensemble import GradientBoostingRegressor

    Xtr = np.asarray([r["features"] for r in train], dtype=float)
    ytr = np.asarray([r["remain_f"] for r in train], dtype=float)
    m50 = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.1, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.9, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m50.fit(Xtr, ytr)
    m10.fit(Xtr, ytr)
    m90.fit(Xtr, ytr)

    om = load_open_meteo_historical_highs(data_dir / "open_meteo_nyc_tmax_historical.csv")
    path = research_dir / f"diagnostics_{split}.csv"
    fields = [
        "climate_day",
        "decision_hour",
        "model_version",
        "train_cutoff",
        "max_so_far",
        "pred_q10",
        "pred_q50",
        "pred_q90",
        "label_tmax_f",
        "signed_error",
        "abs_error",
        "remain_label",
        "precip_6h",
        "sky_proxy",
        "om_hist_tmax",
        "om_error",
    ]
    records = []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in hold:
            x = np.asarray([r["features"]], dtype=float)
            q10 = float(r["max_so_far"] + m10.predict(x)[0])
            q50 = float(r["max_so_far"] + m50.predict(x)[0])
            q90 = float(r["max_so_far"] + m90.predict(x)[0])
            q10, q50, q90 = sorted([q10, q50, q90])
            q50 = max(q50, float(r["max_so_far"]))
            y = float(r["label_tmax_f"])
            d = date.fromisoformat(r["climate_day"])
            om_t = om.get(d)
            rec = {
                "climate_day": r["climate_day"],
                "decision_hour": r["decision_hour"],
                "model_version": "baseline_diag",
                "train_cutoff": max(sets["train"]) if sets["train"] else "",
                "max_so_far": r["max_so_far"],
                "pred_q10": q10,
                "pred_q50": q50,
                "pred_q90": q90,
                "label_tmax_f": y,
                "signed_error": y - q50,
                "abs_error": abs(y - q50),
                "remain_label": r["remain_f"],
                "precip_6h": r["features"][12] if len(r["features"]) > 12 else "",
                "sky_proxy": "",
                "om_hist_tmax": om_t if om_t is not None else "",
                "om_error": (y - om_t) if om_t is not None else "",
            }
            w.writerow(rec)
            records.append(rec)

    # Summaries
    abs_err = sorted(records, key=lambda r: -float(r["abs_error"]))
    by_hour = defaultdict(list)
    for r in records:
        by_hour[str(r["decision_hour"])].append(float(r["abs_error"]))
    summary = {
        "n_records": len(records),
        "n_days": len({r["climate_day"] for r in records}),
        "mae_by_hour": {h: float(np.mean(v)) for h, v in by_hour.items()},
        "largest_errors": abs_err[:15],
        "negative_remain_frac": float(np.mean([float(r["remain_label"]) < -0.05 for r in records])),
        "mean_signed_error_by_hour": {
            h: float(np.mean([float(r["signed_error"]) for r in records if str(r["decision_hour"]) == h]))
            for h in by_hour
        },
        "hypotheses_not_claims": [
            "Large morning errors may coincide with post-decision clearing/warming — needs cloud feature tests.",
            "Negative remain labels indicate ASOS max_so_far > GHCND — measurement/rounding audit.",
            "Do not attribute individual days to wind/clouds without controlled ablation.",
        ],
        "csv_path": str(path),
    }
    (research_dir / f"diagnostics_{split}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
