"""Chronological calibration residuals for probabilistic forecasts (no fabricated fallbacks)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


MIN_CALIB_RESIDUALS = 40


def chronological_day_splits(
    days: list[date] | list[str],
    *,
    train_frac: float = 0.70,
    calib_frac: float = 0.15,
) -> dict[str, set[str]]:
    """Split sorted climate days into train / calib / test (no shuffle)."""
    norm = sorted({d.isoformat() if hasattr(d, "isoformat") else str(d) for d in days})
    n = len(norm)
    i_tr = int(n * train_frac)
    i_ca = int(n * (train_frac + calib_frac))
    return {
        "train": set(norm[:i_tr]),
        "calib": set(norm[i_tr:i_ca]),
        "test": set(norm[i_ca:]),
    }


def residuals_by_hour(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    feature_key: str = "features",
) -> dict[str, list[float]]:
    """outcome_tmax − (max_so_far + q50_remain) on calibration rows, keyed by decision hour."""
    by: dict[str, list[float]] = {"8": [], "11": [], "14": [], "all": []}
    for r in rows:
        hour = str(int(r["decision_hour"]))
        X = np.nan_to_num(np.asarray([r[feature_key]], dtype=float), nan=-999.0)
        rem = float(models["q50"].predict(X)[0])
        # Same clamp as multi.predict.predict_station_v2 (operating + eval parity)
        pred = max(float(r["max_so_far"]) + rem, float(r["max_so_far"]))
        resid = float(r["label_tmax_f"]) - pred
        by.setdefault(hour, []).append(resid)
        by["all"].append(resid)
    return by


def save_calibration_artifact(
    path: Path,
    *,
    residuals_by_hour: dict[str, list[float]],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "calibrated_empirical_residual_by_decision_hour",
        "min_residuals_required": MIN_CALIB_RESIDUALS,
        "residuals_by_hour": residuals_by_hour,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pick_residuals(calib: dict[str, Any] | None, decision_hour: int | None) -> tuple[list[float] | None, str]:
    """Return residuals for hour or None if inadequate (no fabricated fallback)."""
    if not calib:
        return None, "calibration_missing"
    by = calib.get("residuals_by_hour") or {}
    key = str(decision_hour) if decision_hour is not None else "all"
    res = list(by.get(key) or [])
    if len(res) < MIN_CALIB_RESIDUALS:
        # try pooled
        pooled = list(by.get("all") or [])
        if len(pooled) >= MIN_CALIB_RESIDUALS:
            return pooled, "calibration_pooled_all_hours"
        return None, f"calibration_inadequate_n={len(res)}_need>={MIN_CALIB_RESIDUALS}"
    return res, f"calibration_hour_{key}"
