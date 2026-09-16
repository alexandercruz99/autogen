"""Train optional research candidate (neighbor) without promoting over baseline by default."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.research.experiments import _enumerate, _fit_predict_mae, _load_lga
from kalshi_bot.models.weather.obs_engine.research.features_v2 import NEIGHBOR_FEATURES, build_neighbor


def train_neighbor_candidate(*, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    research = data_dir / "research"
    research.mkdir(parents=True, exist_ok=True)
    exp_path = research / "experiment_comparison.json"
    selection = {}
    if exp_path.exists():
        selection = json.loads(exp_path.read_text()).get("selection") or {}

    obs = load_nyc_hourly_bundle(data_dir)
    lga = _load_lga(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    rows = _enumerate(build_neighbor, obs, lga, labels)
    days = sorted({r["climate_day"] for r in rows})
    i_tr = int(len(days) * 0.80)  # train through val for candidate artifact
    train_days = set(days[:i_tr])
    train = [r for r in rows if r["climate_day"] in train_days]
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib

    X = np.nan_to_num(np.asarray([r["features"] for r in train], dtype=float), nan=-999.0)
    y = np.asarray([r["remain_f"] for r in train], dtype=float)
    models = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
        )
        m.fit(X, y)
        models[name] = m
    path = research / "obs_nyc_neighbor_candidate.joblib"
    joblib.dump(
        {
            "models": models,
            "feature_names": NEIGHBOR_FEATURES,
            "feature_set": "neighbor",
            "train_end_day": max(train_days) if train_days else None,
            "promoted_for_inference": False,
            "reason": (
                "Val macro MAE improved ~0.1°F vs baseline but no decision-time NWS advantage; "
                "baseline remains default for weather-obs-predict."
            ),
        },
        path,
    )
    report = {
        "ok": True,
        "artifact": str(path),
        "feature_set": "neighbor",
        "train_end_day": max(train_days) if train_days else None,
        "n_train_days": len(train_days),
        "selection_from_experiments": selection,
        "promoted_for_default_predict": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_eligible": False,
    }
    (research / "neighbor_candidate_report.json").write_text(json.dumps(report, indent=2))
    return report
