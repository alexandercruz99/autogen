"""Train observation-driven baselines and quantile remaining-rise model for NYC."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.obs_engine import ARTIFACT_NAME, MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.features import FEATURE_NAMES, enumerate_training_rows

logger = logging.getLogger(__name__)


def _chronological_split(rows: list[dict[str, Any]], holdout_frac: float = 0.25):
    days = sorted({r["climate_day"] for r in rows})
    cut = max(1, int(len(days) * (1 - holdout_frac)))
    train_days = set(days[:cut])
    hold_days = set(days[cut:])
    train = [r for r in rows if r["climate_day"] in train_days]
    hold = [r for r in rows if r["climate_day"] in hold_days]
    return train, hold, days[:cut], days[cut:]


def train_obs_nyc(
    *,
    data_dir: Path | None = None,
    archive_path: str | Path = "data/weather_archive.db",
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    obs = load_nyc_hourly_bundle(data_dir)
    rows = enumerate_training_rows(obs, labels)
    report: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "n_hourly_obs": len(obs),
        "n_label_days_available": len(labels),
        "n_training_rows": len(rows),
        "n_unique_days_in_rows": len({r["climate_day"] for r in rows}),
        "feature_names": FEATURE_NAMES,
        "label_provenance": "GHCND USW00094728 TMAX (°F). Settlement is CLI; GHCND is same station climate record.",
        "forecast_features_used": False,
    }
    if len({r["climate_day"] for r in rows}) < 40:
        report["ok"] = False
        report["reason"] = (
            f"Only {len({r['climate_day'] for r in rows})} independent climate days with features+labels; "
            "need more ASOS history (target ≥90 summer days)."
        )
        return report

    train, hold, train_days, hold_days = _chronological_split(rows)
    report["n_train_rows"] = len(train)
    report["n_hold_rows"] = len(hold)
    report["n_train_days"] = len(train_days)
    report["n_hold_days"] = len(hold_days)

    # --- Baselines on holdout (point error on final max) ---
    # Climatology by month from train labels
    month_vals: dict[int, list[int]] = defaultdict(list)
    for r in train:
        month_vals[date.fromisoformat(r["climate_day"]).month].append(int(r["label_tmax_f"]))
    clim = {m: float(np.mean(v)) for m, v in month_vals.items()}

    def clim_pred(day_s: str) -> float:
        m = date.fromisoformat(day_s).month
        return clim.get(m, float(np.mean([r["label_tmax_f"] for r in train])))

    # Persistence: previous day's label
    label_by_day = {r["climate_day"]: int(r["label_tmax_f"]) for r in rows}
    sorted_days = sorted(label_by_day)

    def persist_pred(day_s: str) -> float | None:
        if day_s not in sorted_days:
            return None
        i = sorted_days.index(day_s)
        if i == 0:
            return None
        return float(label_by_day[sorted_days[i - 1]])

    # Continuation: predict final = max_so_far (no more warming)
    hold_clim_err, hold_pers_err, hold_cont_err = [], [], []
    for r in hold:
        y = float(r["label_tmax_f"])
        hold_clim_err.append(y - clim_pred(r["climate_day"]))
        hold_cont_err.append(y - float(r["max_so_far"]))
        pp = persist_pred(r["climate_day"])
        if pp is not None:
            hold_pers_err.append(y - pp)

    report["baselines_holdout"] = {
        "climatology_mae": float(np.mean(np.abs(hold_clim_err))) if hold_clim_err else None,
        "persistence_mae": float(np.mean(np.abs(hold_pers_err))) if hold_pers_err else None,
        "continuation_max_so_far_mae": float(np.mean(np.abs(hold_cont_err))) if hold_cont_err else None,
        "n": len(hold),
    }

    # --- Quantile GBM on remaining rise ---
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        report["ok"] = False
        report["reason"] = "scikit-learn missing"
        return report

    X_train = np.asarray([r["features"] for r in train], dtype=float)
    y_train = np.asarray([r["remain_f"] for r in train], dtype=float)
    X_hold = np.asarray([r["features"] for r in hold], dtype=float)
    y_hold = np.asarray([r["remain_f"] for r in hold], dtype=float)

    quantiles = {}
    mae_point = None
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            max_depth=2,
            n_estimators=120,
            learning_rate=0.05,
            random_state=0,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_hold)
        quantiles[name] = {
            "alpha": q,
            "hold_pinball": float(np.mean(np.maximum(q * (y_hold - pred), (q - 1) * (y_hold - pred)))),
        }
        if name == "q50":
            # final max pred = max_so_far + remain_q50
            final_err = []
            for r, rem in zip(hold, pred):
                final_err.append(float(r["label_tmax_f"]) - (float(r["max_so_far"]) + float(rem)))
            mae_point = float(np.mean(np.abs(final_err)))
            quantiles[name]["hold_final_mae"] = mae_point
        # Store tree params by predicting on a grid is heavy; store training residuals of remain for mixture
    # Empirical remain residuals around q50 for distribution construction without pickling huge trees:
    m50 = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m50.fit(X_train, y_train)
    remain_resid = list((y_train - m50.predict(X_train)).astype(float))
    # Also store leaf predictions via compact approach: save sklearn with joblib if available
    artifact_dir = data_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "obs_nyc_q50.joblib"
    try:
        import joblib

        models = {}
        for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
            m = GradientBoostingRegressor(
                loss="quantile", alpha=q, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
            )
            m.fit(X_train, y_train)
            models[name] = m
        joblib.dump({"models": models, "feature_names": FEATURE_NAMES}, model_path)
        report["artifact_path"] = str(model_path)
    except Exception as exc:
        report["joblib_warning"] = str(exc)
        model_path = None

    artifact = {
        "type": ARTIFACT_NAME,
        "feature_names": FEATURE_NAMES,
        "remain_residuals": remain_resid[:500],
        "climatology_by_month": clim,
        "train_end_day": train_days[-1] if train_days else None,
        "model_path": str(model_path) if model_path else None,
        "n_train_days": len(train_days),
        "n_hold_days": len(hold_days),
    }
    metrics = {
        "baselines_holdout": report["baselines_holdout"],
        "quantiles": quantiles,
        "hold_final_mae_q50": mae_point,
        "beats_climatology": (
            mae_point is not None
            and report["baselines_holdout"]["climatology_mae"] is not None
            and mae_point < report["baselines_holdout"]["climatology_mae"]
        ),
        "beats_continuation": (
            mae_point is not None
            and report["baselines_holdout"]["continuation_max_so_far_mae"] is not None
            and mae_point < report["baselines_holdout"]["continuation_max_so_far_mae"]
        ),
    }
    archive = WeatherArchive(archive_path)
    archive.save_artifact(
        NYC_TARGET.city_key,
        ARTIFACT_NAME,
        artifact,
        metrics=metrics,
        train_end_day=artifact["train_end_day"],
    )
    report["ok"] = True
    report["metrics"] = metrics
    report["live_eligible"] = False
    report["notes"] = [
        "NWS forecasts were NOT used as training features (benchmark only).",
        "Holdout is chronological by climate day.",
        "Trading profitability not evaluated here.",
        "model_live_eligible remains false.",
    ]
    # Persist report
    (data_dir / "last_train_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
