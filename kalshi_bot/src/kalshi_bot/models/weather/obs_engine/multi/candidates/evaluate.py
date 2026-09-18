"""Train and chronologically evaluate forecast candidates A–D (research only).

Primary selection metric (declared BEFORE inspecting final test): mean CRPS on the
final untouched test split, with MAE and 80% interval coverage as supporting evidence.
live_eligible remains false; no orders are submitted.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir
from kalshi_bot.models.weather.obs_engine.feeds.climate_day import SUPPORTED_DECISION_HOURS_LOCAL
from kalshi_bot.models.weather.obs_engine.feeds.feature_schema import FEATURE_SCHEMA_VERSION, STATION_V2_FEATURES
from kalshi_bot.models.weather.obs_engine.multi.candidates.data import load_location_rows, write_json
from kalshi_bot.models.weather.obs_engine.multi.candidates.metrics_prob import (
    contract_brier,
    crps_discrete,
    interval_coverage,
    p_inclusive_band,
    summarize_predictions,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.pipeline import (
    FEATURE_NAMES,
    build_residual_dist,
    fit_quantile_gbms,
    from_quantile_grid,
    from_standardized_residuals,
    hour_bias_map,
    matrix_from_rows,
    predict_remain_quantiles,
    remain_point_from_quantiles,
    synthetic_between_interval,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.splits import (
    build_split_manifest,
    filter_rows_by_days,
    save_split_manifest,
)
from kalshi_bot.models.weather.obs_engine.multi.candidates.targets import (
    SETTLEMENT_TARGETS,
    UNSUPPORTED_OR_INSUFFICIENT,
    targets_manifest,
)
from kalshi_bot.models.weather.settlement_rules import TempInterval, yes_from_observation

CANDIDATE_VERSIONS = {
    "A_frozen_baseline": "candidate.A.frozen_remain_gbm_empirical.v1",
    "A_corrected_baseline": "candidate.A.corrected_shared_pipeline.v1",
    "B_adaptive_residuals": "candidate.B.adaptive_location_scale.v1",
    "C_quantile_regression_forest": "candidate.C.qrf_meinshausen.v1",
    "D_direct_high_gbm": "candidate.D.direct_high_gbm.v1",
}

# Declared before test inspection.
PRIMARY_SELECTION_METRIC = "mean_crps_final_test"


def _artifact_root(data_dir: Path) -> Path:
    return data_dir / "multi" / "candidates"


def _score_row(
    *,
    y: float,
    point: float,
    dist,
    decision_hour: int,
    climate_day: str,
    method: str,
) -> dict[str, Any]:
    q10 = float(dist.quantile(0.1))
    q50 = float(dist.quantile(0.5))
    q90 = float(dist.quantile(0.9))
    # Prefer distribution median for error (consistent with constrained dist)
    med = q50
    err = med - float(y)
    band = synthetic_between_interval(int(round(med)), width=0)
    return {
        "climate_day": climate_day,
        "decision_hour_local": decision_hour,
        "actual_f": float(y),
        "point_f": float(point),
        "median_f": med,
        "error_f": err,
        "crps": crps_discrete(dist, y),
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "cov80": interval_coverage(y, q10, q90),
        "width80": q90 - q10,
        "brier_between_1f": contract_brier(dist, y, band),
        "p_inclusive_median_band": p_inclusive_band(dist, int(round(med)), int(round(med))),
        "method": method,
        "model_data_conflict": bool((dist.details or {}).get("model_data_conflict")),
    }


def _group_summaries(scored: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summarize_predictions(scored)
    by_hour = {}
    for h in SUPPORTED_DECISION_HOURS_LOCAL:
        sub = [r for r in scored if int(r["decision_hour_local"]) == h]
        by_hour[str(h)] = summarize_predictions(sub, group_key=f"hour_{h}")
    return {"overall": overall, "by_hour": by_hour}


def _freeze_existing_baseline(data_dir: Path, out_dir: Path, location_id: str) -> dict[str, Any]:
    """Copy current production artifacts into a versioned frozen snapshot (no overwrite of prod)."""
    src = data_dir / "multi" / "artifacts" / f"{location_id}__daily_max_temp_f"
    dest = out_dir / "frozen_production_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in (
        "station_corrected_v2.joblib",
        "station_corrected_v2_calibration.json",
        "station_corrected_v2_report.json",
    ):
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dest / name)
            copied[name] = "copied"
        else:
            # NYC legacy fallback
            alt = data_dir / "feeds" / "models" / name
            if location_id == "nyc_central_park" and alt.exists():
                shutil.copy2(alt, dest / name)
                copied[name] = "copied_from_feeds_models"
            else:
                copied[name] = "missing"
    meta = {
        "location_id": location_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "copied": copied,
        "live_eligible": False,
        "note": "Frozen snapshot of production artifacts at evaluation start; not modified.",
    }
    write_json(dest / "freeze_manifest.json", meta)
    return meta


def _fit_bias_from_rows(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    target: str = "remain",
) -> dict[str, float]:
    by: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
    if not rows:
        return {}
    X = matrix_from_rows(rows)
    if target == "remain":
        _, q50, _, _ = predict_remain_quantiles(models, X)
        for r, rem in zip(rows, q50):
            point = remain_point_from_quantiles(float(r["max_so_far"]), float(rem), bias=0.0)
            resid = float(r["label_tmax_f"]) - point
            h = str(int(r["decision_hour"]))
            by[h].append(resid)
            by["all"].append(resid)
    else:
        _, q50, _, _ = predict_remain_quantiles(models, X)
        for r, pred in zip(rows, q50):
            point = max(float(pred), float(r["max_so_far"]))
            resid = float(r["label_tmax_f"]) - point
            h = str(int(r["decision_hour"]))
            by[h].append(resid)
            by["all"].append(resid)
    return hour_bias_map(by)


def _empirical_residuals(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    bias_by_hour: dict[str, float],
    *,
    target: str = "remain",
) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
    if not rows:
        return by
    X = matrix_from_rows(rows)
    _, q50, _, _ = predict_remain_quantiles(models, X)
    for r, pred in zip(rows, q50):
        h = str(int(r["decision_hour"]))
        b = float(bias_by_hour.get(h, 0.0))
        if target == "remain":
            point = remain_point_from_quantiles(float(r["max_so_far"]), float(pred), bias=b)
        else:
            point = max(float(pred) + b, float(r["max_so_far"]))
        resid = float(r["label_tmax_f"]) - point
        by[h].append(resid)
        by["all"].append(resid)
    return by


def _predict_corrected_baseline_row(
    r: dict[str, Any],
    models: dict[str, Any],
    residuals_by_h: dict[str, list[float]],
    bias_by_hour: dict[str, float],
) -> dict[str, Any]:
    X = matrix_from_rows([r])
    q10, q50, q90, crossed = predict_remain_quantiles(models, X)
    h = str(int(r["decision_hour"]))
    bias = float(bias_by_hour.get(h, 0.0))
    point = remain_point_from_quantiles(float(r["max_so_far"]), float(q50[0]), bias=bias)
    res = residuals_by_h.get(h) or residuals_by_h.get("all") or []
    dist, point2 = build_residual_dist(
        point,
        res,
        method=f"corrected_baseline_hour_{h}",
        floor_f=None,  # no CLI floor in historical GHCND replay without settlement floor evidence
    )
    scored = _score_row(
        y=float(r["label_tmax_f"]),
        point=point2,
        dist=dist,
        decision_hour=int(r["decision_hour"]),
        climate_day=str(r["climate_day"]),
        method="A_corrected_baseline",
    )
    scored["quantiles_crossed"] = bool(crossed[0])
    scored["remain_q10_q50_q90"] = [float(q10[0]), float(q50[0]), float(q90[0])]
    scored["bias_f_applied"] = bias
    return scored


def _train_eval_A(
    rows_train: list[dict[str, Any]],
    rows_sel: list[dict[str, Any]],
    rows_calib: list[dict[str, Any]],
    rows_test: list[dict[str, Any]],
    *,
    out_dir: Path,
    location_id: str,
) -> dict[str, Any]:
    Xtr = matrix_from_rows(rows_train)
    ytr = np.asarray([r["remain_f"] for r in rows_train], dtype=float)
    models = fit_quantile_gbms(Xtr, ytr)
    # Bias from selection only (not test); residuals from calib after applying that bias
    bias = _fit_bias_from_rows(rows_sel, models, target="remain")
    resid = _empirical_residuals(rows_calib, models, bias, target="remain")
    # Require adequate pooled residuals
    if len(resid.get("all") or []) < 40:
        resid = _empirical_residuals(rows_sel + rows_calib, models, bias, target="remain")

    blob = {
        "candidate": "A_corrected_baseline",
        "model_version": CANDIDATE_VERSIONS["A_corrected_baseline"],
        "models": models,
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "bias_by_hour": bias,
        "residuals_by_hour": resid,
        "location_id": location_id,
        "live_eligible": False,
        "known_defects_in_frozen_baseline": [
            "Historical production applied hour bias after max_so_far floor then rebuilt dist; "
            "point_median_f was not always synced to CLI-truncated distribution.",
            "hours_to_20_local was sometimes read as if it defined the settlement horizon.",
        ],
    }
    joblib.dump(blob, out_dir / "model.joblib")
    write_json(
        out_dir / "calibration.json",
        {
            "method": "empirical_residual_by_hour_bias_first",
            "residuals_by_hour": resid,
            "meta": {
                "location_id": location_id,
                "point_bias_by_hour": bias,
                "live_eligible": False,
                "model_version": blob["model_version"],
            },
        },
    )

    def eval_split(name: str, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [_predict_corrected_baseline_row(r, models, resid, bias) for r in split_rows]
        return {"split": name, "n": len(scored), **_group_summaries(scored), "rows": scored}

    return {
        "candidate": "A_corrected_baseline",
        "model_version": blob["model_version"],
        "selection": eval_split("selection", rows_sel),
        "calib_sanity": eval_split("calib", rows_calib),
        "test": eval_split("test", rows_test),
        "artifact_dir": str(out_dir),
        "live_eligible": False,
    }


def _train_eval_B(
    rows_train: list[dict[str, Any]],
    rows_sel: list[dict[str, Any]],
    rows_calib: list[dict[str, Any]],
    rows_test: list[dict[str, Any]],
    *,
    out_dir: Path,
    location_id: str,
) -> dict[str, Any]:
    Xtr = matrix_from_rows(rows_train)
    ytr = np.asarray([r["remain_f"] for r in rows_train], dtype=float)
    models = fit_quantile_gbms(Xtr, ytr)
    bias = _fit_bias_from_rows(rows_sel, models, target="remain")

    # Choose s_min on selection only
    Xsel = matrix_from_rows(rows_sel)
    q10s, q50s, q90s, _ = predict_remain_quantiles(models, Xsel)
    spreads = np.maximum(q90s - q10s, 0.0)
    # grid search s_min to minimize selection CRPS
    best_smin = 1.0
    best_crps = 1e9
    for s_min in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        z_list: list[float] = []
        # build z on calib using this s_min for scoring selection via calib z — use sel for z when choosing s_min
        for r, q10, q50, q90 in zip(rows_sel, q10s, q50s, q90s):
            h = str(int(r["decision_hour"]))
            b = float(bias.get(h, 0.0))
            mu = remain_point_from_quantiles(float(r["max_so_far"]), float(q50), bias=b)
            s = max(float(q90 - q10), s_min)
            z_list.append((float(r["label_tmax_f"]) - mu) / s)
        # leave-one-style approx: use all z including self (selection only — not final test)
        crps_vals = []
        for r, q10, q50, q90, z_self in zip(rows_sel, q10s, q50s, q90s, z_list):
            h = str(int(r["decision_hour"]))
            b = float(bias.get(h, 0.0))
            mu = remain_point_from_quantiles(float(r["max_so_far"]), float(q50), bias=b)
            s = max(float(q90 - q10), s_min)
            # exclude own z
            z_pool = [z for j, z in enumerate(z_list) if rows_sel[j]["climate_day"] != r["climate_day"]]
            if len(z_pool) < 20:
                z_pool = z_list
            dist = from_standardized_residuals(mu, s, z_pool)
            crps_vals.append(crps_discrete(dist, float(r["label_tmax_f"])))
        mean_crps = float(np.mean(crps_vals)) if crps_vals else 1e9
        if mean_crps < best_crps:
            best_crps = mean_crps
            best_smin = s_min

    # Standardized residuals on calib (and optionally sel+calib if thin) using selected s_min
    def collect_z(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not split_rows:
            return []
        X = matrix_from_rows(split_rows)
        q10, q50, q90, _ = predict_remain_quantiles(models, X)
        out = []
        for r, a, b50, c in zip(split_rows, q10, q50, q90):
            h = str(int(r["decision_hour"]))
            bias_h = float(bias.get(h, 0.0))
            mu = remain_point_from_quantiles(float(r["max_so_far"]), float(b50), bias=bias_h)
            s = max(float(c - a), best_smin)
            z = (float(r["label_tmax_f"]) - mu) / s
            out.append({"climate_day": r["climate_day"], "hour": h, "z": z, "s": s, "mu": mu})
        return out

    z_calib = collect_z(rows_calib)
    if len(z_calib) < 40:
        z_calib = collect_z(rows_sel + rows_calib)
    z_values = [e["z"] for e in z_calib]

    # Diagnose residual structure
    z_by_hour: dict[str, list[float]] = {"8": [], "11": [], "14": []}
    for e in z_calib:
        z_by_hour.setdefault(e["hour"], []).append(e["z"])
    z_diag = {
        hour: {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }
        for hour, vals in z_by_hour.items()
    }

    blob = {
        "candidate": "B_adaptive_residuals",
        "model_version": CANDIDATE_VERSIONS["B_adaptive_residuals"],
        "models": models,
        "feature_names": FEATURE_NAMES,
        "s_min": best_smin,
        "s_min_selection_crps": best_crps,
        "bias_by_hour": bias,
        "z_residuals": z_values,
        "z_diag_by_hour": z_diag,
        "pooling": "pooled_z_across_hours",
        "location_id": location_id,
        "live_eligible": False,
        "limitation": "location-scale approximation; not guaranteed calibrated",
    }
    joblib.dump(blob, out_dir / "model.joblib")
    write_json(
        out_dir / "calibration.json",
        {
            "method": "adaptive_location_scale",
            "s_min": best_smin,
            "z_residuals": z_values,
            "z_diag_by_hour": z_diag,
            "meta": {
                "location_id": location_id,
                "point_bias_by_hour": bias,
                "n_z": len(z_values),
                "live_eligible": False,
            },
        },
    )

    def eval_split(name: str, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        if not split_rows:
            return {"split": name, "n": 0, "overall": {"n": 0}, "by_hour": {}, "rows": []}
        X = matrix_from_rows(split_rows)
        q10, q50, q90, crossed = predict_remain_quantiles(models, X)
        for r, a, b50, c, cr in zip(split_rows, q10, q50, q90, crossed):
            h = str(int(r["decision_hour"]))
            bias_h = float(bias.get(h, 0.0))
            mu = remain_point_from_quantiles(float(r["max_so_far"]), float(b50), bias=bias_h)
            s = max(float(c - a), best_smin)
            # exclude same climate day from z pool
            z_pool = [
                e["z"] for e in z_calib if e["climate_day"] != r["climate_day"]
            ] or z_values
            dist = from_standardized_residuals(mu, s, z_pool)
            scored.append(
                {
                    **_score_row(
                        y=float(r["label_tmax_f"]),
                        point=mu,
                        dist=dist,
                        decision_hour=int(r["decision_hour"]),
                        climate_day=str(r["climate_day"]),
                        method="B_adaptive_residuals",
                    ),
                    "s": s,
                    "quantiles_crossed": bool(cr),
                    "bias_f_applied": bias_h,
                }
            )
        return {"split": name, "n": len(scored), **_group_summaries(scored), "rows": scored}

    return {
        "candidate": "B_adaptive_residuals",
        "model_version": blob["model_version"],
        "s_min": best_smin,
        "z_diag_by_hour": z_diag,
        "selection": eval_split("selection", rows_sel),
        "calib_sanity": eval_split("calib", rows_calib),
        "test": eval_split("test", rows_test),
        "artifact_dir": str(out_dir),
        "live_eligible": False,
    }


def _train_eval_C(
    rows_train: list[dict[str, Any]],
    rows_sel: list[dict[str, Any]],
    rows_calib: list[dict[str, Any]],
    rows_test: list[dict[str, Any]],
    *,
    out_dir: Path,
    location_id: str,
) -> dict[str, Any]:
    from quantile_forest import RandomForestQuantileRegressor

    Xtr = matrix_from_rows(rows_train)
    ytr = np.asarray([r["remain_f"] for r in rows_train], dtype=float)
    qrf = RandomForestQuantileRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        max_samples_leaf=None,
        random_state=0,
        n_jobs=1,
    )
    qrf.fit(Xtr, ytr)
    # Optional bias from selection using QRF median remain
    q_grid = np.linspace(0.01, 0.99, 99)

    def qrf_point_and_dist(r: dict[str, Any]):
        X = matrix_from_rows([r])
        qs_remain = qrf.predict(X, quantiles=list(q_grid))[0]
        # bias: use 0 in core; selection median residual optional
        med_remain = float(np.median(qs_remain))
        point = remain_point_from_quantiles(float(r["max_so_far"]), med_remain, bias=0.0)
        y_grid = float(r["max_so_far"]) + qs_remain
        # clamp samples to max_so_far (physical: high cannot be below observed max)
        y_grid = np.maximum(y_grid, float(r["max_so_far"]))
        dist = from_quantile_grid(y_grid, method="qrf_quantile_grid_to_pmf")
        return point, dist, med_remain

    # Selection bias (median residual) — applied once at point, not again in dist
    sel_resid = []
    for r in rows_sel:
        point, dist, _ = qrf_point_and_dist(r)
        sel_resid.append(float(r["label_tmax_f"]) - float(dist.quantile(0.5)))
    bias = float(np.median(sel_resid)) if len(sel_resid) >= 8 else 0.0

    joblib.dump(
        {
            "candidate": "C_quantile_regression_forest",
            "model_version": CANDIDATE_VERSIONS["C_quantile_regression_forest"],
            "qrf": qrf,
            "quantile_grid": list(q_grid),
            "selection_point_bias": bias,
            "feature_names": FEATURE_NAMES,
            "location_id": location_id,
            "live_eligible": False,
            "reference": "https://jmlr.org/papers/v7/meinshausen06a.html",
            "implementation": "quantile-forest.RandomForestQuantileRegressor",
            "limitation": (
                "Extremes outside training leaf experience are poorly supported; "
                "PMF is built from a dense quantile grid (weighted leaf empirical via QRF)."
            ),
        },
        out_dir / "model.joblib",
    )

    def eval_split(name: str, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        for r in split_rows:
            point, dist, med_remain = qrf_point_and_dist(r)
            # apply selection bias to point then rebuild by shifting grid equivalently
            if bias:
                point = remain_point_from_quantiles(float(r["max_so_far"]), med_remain, bias=bias)
                X = matrix_from_rows([r])
                qs_remain = qrf.predict(X, quantiles=list(q_grid))[0]
                y_grid = np.maximum(float(r["max_so_far"]) + qs_remain + bias, float(r["max_so_far"]))
                dist = from_quantile_grid(y_grid, method="qrf_quantile_grid_to_pmf+sel_bias")
            scored.append(
                _score_row(
                    y=float(r["label_tmax_f"]),
                    point=point,
                    dist=dist,
                    decision_hour=int(r["decision_hour"]),
                    climate_day=str(r["climate_day"]),
                    method="C_qrf",
                )
            )
        return {"split": name, "n": len(scored), **_group_summaries(scored), "rows": scored}

    return {
        "candidate": "C_quantile_regression_forest",
        "model_version": CANDIDATE_VERSIONS["C_quantile_regression_forest"],
        "selection_point_bias": bias,
        "selection": eval_split("selection", rows_sel),
        "calib_sanity": eval_split("calib", rows_calib),
        "test": eval_split("test", rows_test),
        "artifact_dir": str(out_dir),
        "live_eligible": False,
    }


def _train_eval_D(
    rows_train: list[dict[str, Any]],
    rows_sel: list[dict[str, Any]],
    rows_calib: list[dict[str, Any]],
    rows_test: list[dict[str, Any]],
    *,
    out_dir: Path,
    location_id: str,
) -> dict[str, Any]:
    """Direct-high ablation: predict Y with M as a feature (same feature vector includes max_so_far)."""
    Xtr = matrix_from_rows(rows_train)
    ytr = np.asarray([r["label_tmax_f"] for r in rows_train], dtype=float)
    models = fit_quantile_gbms(Xtr, ytr)
    bias = _fit_bias_from_rows(rows_sel, models, target="direct")
    resid = _empirical_residuals(rows_calib, models, bias, target="direct")
    if len(resid.get("all") or []) < 40:
        resid = _empirical_residuals(rows_sel + rows_calib, models, bias, target="direct")

    joblib.dump(
        {
            "candidate": "D_direct_high_gbm",
            "model_version": CANDIDATE_VERSIONS["D_direct_high_gbm"],
            "models": models,
            "target": "label_tmax_f",
            "feature_names": FEATURE_NAMES,
            "bias_by_hour": bias,
            "residuals_by_hour": resid,
            "location_id": location_id,
            "live_eligible": False,
        },
        out_dir / "model.joblib",
    )

    def eval_split(name: str, split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        if not split_rows:
            return {"split": name, "n": 0, "overall": {"n": 0}, "by_hour": {}, "rows": []}
        X = matrix_from_rows(split_rows)
        q10, q50, q90, crossed = predict_remain_quantiles(models, X)
        for r, a, b50, c, cr in zip(split_rows, q10, q50, q90, crossed):
            h = str(int(r["decision_hour"]))
            bias_h = float(bias.get(h, 0.0))
            # bias first, then floor to max_so_far
            raw = float(b50) + bias_h
            point = max(raw, float(r["max_so_far"]))
            res = resid.get(h) or resid.get("all") or []
            dist, point2 = build_residual_dist(point, res, method=f"direct_high_hour_{h}")
            scored.append(
                {
                    **_score_row(
                        y=float(r["label_tmax_f"]),
                        point=point2,
                        dist=dist,
                        decision_hour=int(r["decision_hour"]),
                        climate_day=str(r["climate_day"]),
                        method="D_direct_high",
                    ),
                    "quantiles_crossed": bool(cr),
                    "bias_f_applied": bias_h,
                }
            )
        return {"split": name, "n": len(scored), **_group_summaries(scored), "rows": scored}

    return {
        "candidate": "D_direct_high_gbm",
        "model_version": CANDIDATE_VERSIONS["D_direct_high_gbm"],
        "selection": eval_split("selection", rows_sel),
        "calib_sanity": eval_split("calib", rows_calib),
        "test": eval_split("test", rows_test),
        "artifact_dir": str(out_dir),
        "live_eligible": False,
    }


def _strip_rows_for_report(result: dict[str, Any]) -> dict[str, Any]:
    """Keep metrics; drop per-row arrays from summary (full rows saved separately)."""
    out = dict(result)
    for key in ("selection", "calib_sanity", "test"):
        block = out.get(key)
        if isinstance(block, dict) and "rows" in block:
            rows = block["rows"]
            write_json(Path(result["artifact_dir"]) / f"predictions_{key}.json", {"rows": rows})
            block = {k: v for k, v in block.items() if k != "rows"}
            out[key] = block
    return out


def evaluate_location(location_id: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    loaded = load_location_rows(location_id, data_dir=data_dir)
    if not loaded.get("ok"):
        return {
            "ok": False,
            "location_id": location_id,
            "error": loaded.get("error"),
            "status": loaded.get("status", "insufficient_data"),
            "live_eligible": False,
        }

    rows = loaded["rows"]
    days = sorted({r["climate_day"] for r in rows})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    loc_root = _artifact_root(data_dir) / location_id / stamp
    loc_root.mkdir(parents=True, exist_ok=True)

    freeze = _freeze_existing_baseline(data_dir, loc_root, location_id)
    manifest = build_split_manifest(
        location_id=location_id,
        metric="daily_max_temp_f",
        label_source="ghcnd_tmax",
        climate_days=days,
        decision_hours_local=list(SUPPORTED_DECISION_HOURS_LOCAL),
        notes=[
            "Final test is untouched for s_min / bias / candidate selection.",
            "Prior TWC 60d scorecards are development benchmarks if they influenced earlier work.",
            loaded["availability_assumption"],
        ],
    )
    save_split_manifest(loc_root / "split_manifest.json", manifest)
    write_json(loc_root / "negative_remain_audit.json", loaded["negative_remain"])
    write_json(loc_root / "target.json", SETTLEMENT_TARGETS[location_id].as_dict())

    rows_train = filter_rows_by_days(rows, manifest.train_days)
    rows_sel = filter_rows_by_days(rows, manifest.selection_days)
    rows_calib = filter_rows_by_days(rows, manifest.calib_days)
    rows_test = filter_rows_by_days(rows, manifest.test_days)

    results = []
    # A corrected
    a_dir = loc_root / "A_corrected_baseline"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_res = _train_eval_A(rows_train, rows_sel, rows_calib, rows_test, out_dir=a_dir, location_id=location_id)
    results.append(_strip_rows_for_report(a_res))

    # B adaptive
    b_dir = loc_root / "B_adaptive_residuals"
    b_dir.mkdir(parents=True, exist_ok=True)
    b_res = _train_eval_B(rows_train, rows_sel, rows_calib, rows_test, out_dir=b_dir, location_id=location_id)
    results.append(_strip_rows_for_report(b_res))

    # C QRF
    c_dir = loc_root / "C_quantile_regression_forest"
    c_dir.mkdir(parents=True, exist_ok=True)
    c_res = _train_eval_C(rows_train, rows_sel, rows_calib, rows_test, out_dir=c_dir, location_id=location_id)
    results.append(_strip_rows_for_report(c_res))

    # D direct high
    d_dir = loc_root / "D_direct_high_gbm"
    d_dir.mkdir(parents=True, exist_ok=True)
    d_res = _train_eval_D(rows_train, rows_sel, rows_calib, rows_test, out_dir=d_dir, location_id=location_id)
    results.append(_strip_rows_for_report(d_res))

    # Selection on selection-split CRPS (not test) to pick a provisional winner;
    # report test metrics for honesty.
    def sel_crps(res: dict[str, Any]) -> float:
        return float(((res.get("selection") or {}).get("overall") or {}).get("crps") or 9e9)

    ranked_sel = sorted(results, key=sel_crps)
    provisional = ranked_sel[0]["candidate"]
    # Final test ranking (reported separately; searching candidates increases selection uncertainty)
    def test_crps(res: dict[str, Any]) -> float:
        return float(((res.get("test") or {}).get("overall") or {}).get("crps") or 9e9)

    ranked_test = sorted(results, key=test_crps)

    summary = {
        "ok": True,
        "location_id": location_id,
        "stamp": stamp,
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "selection_rule": "Minimize mean CRPS on model-selection split; final test reported separately.",
        "provisional_winner_by_selection_crps": provisional,
        "test_ranking_by_crps": [r["candidate"] for r in ranked_test],
        "test_crps": {r["candidate"]: test_crps(r) for r in results},
        "test_mae": {
            r["candidate"]: ((r.get("test") or {}).get("overall") or {}).get("mae_f") for r in results
        },
        "split_manifest_hash": manifest.day_hash(),
        "n_train_days": len(manifest.train_days),
        "n_selection_days": len(manifest.selection_days),
        "n_calib_days": len(manifest.calib_days),
        "n_test_days": len(manifest.test_days),
        "negative_remain_count": loaded["negative_remain"]["count"],
        "frozen_baseline": freeze,
        "candidates": results,
        "live_eligible": False,
        "no_orders_submitted": True,
        "artifact_root": str(loc_root),
    }
    write_json(loc_root / "comparison.json", summary)
    # latest pointer
    latest = _artifact_root(data_dir) / location_id / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        latest.symlink_to(loc_root.resolve())
    except OSError:
        shutil.copytree(loc_root, latest, dirs_exist_ok=True)
    return summary


def run_candidate_evaluation(*, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = _artifact_root(data_dir) / "_reports" / stamp
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "targets_manifest.json", targets_manifest())

    locations = []
    for loc_id in SETTLEMENT_TARGETS:
        locations.append(evaluate_location(loc_id, data_dir=data_dir))

    # External benchmark status
    external = {
        "nws_point_forecast": {
            "status": "partial",
            "detail": (
                "nws_client.daily_high_forecast exists for live/prospective pulls; "
                "no archived same-vintage NWS forecast time series matched to decision hours "
                "is wired into this candidate harness yet."
            ),
        },
        "hrrr": {
            "status": "unavailable",
            "detail": (
                "HRRR (https://rapidrefresh.noaa.gov/hrrr/) is not ingested. "
                "Requires AWS/NOMADS archival access, station extraction, and issue-time alignment. "
                "Do not substitute reanalysis. Prospective collection recommended."
            ),
            "reference": "https://rapidrefresh.noaa.gov/hrrr/",
        },
        "claim": "No claim is made that observation-only candidates beat external forecasts.",
    }
    write_json(root / "external_benchmark_status.json", external)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "candidate_versions": CANDIDATE_VERSIONS,
        "locations": locations,
        "unsupported_or_insufficient": UNSUPPORTED_OR_INSUFFICIENT,
        "external_benchmark": external,
        "live_eligible": False,
        "no_orders_submitted": True,
        "profit_evidence": None,
        "profit_evidence_note": "No trading P&L evaluated; forecast metrics only.",
        "report_dir": str(root),
    }
    write_json(root / "executive_summary.json", report)
    write_json(_artifact_root(data_dir) / "_reports" / "latest.json", report)
    return report
