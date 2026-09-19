"""Two-part remaining-rise probe (research): P(no further rise) + rise|rise>0.

Not promoted — diagnostic alternative after measurement audit.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle
from kalshi_bot.models.weather.obs_engine.research.experiments import _enumerate
from kalshi_bot.models.weather.obs_engine.research.features_v2 import build_baseline


def run_two_part_probe(*, data_dir: Path | None = None) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    data_dir = data_dir or default_data_dir()
    obs = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    rows = _enumerate(build_baseline, obs, [], labels)
    days = sorted({r["climate_day"] for r in rows})
    i_tr = int(len(days) * 0.8)
    train_days = set(days[:i_tr])
    test_days = set(days[i_tr:])
    train = [r for r in rows if r["climate_day"] in train_days]
    test = [r for r in rows if r["climate_day"] in test_days]

    # Label: further rise if remain > 0.5°F (accounts for rounding noise)
    y_rise = np.asarray([1 if r["remain_f"] > 0.5 else 0 for r in train], dtype=int)
    Xtr = np.asarray([r["features"] for r in train], dtype=float)
    clf = GradientBoostingClassifier(max_depth=2, n_estimators=80, learning_rate=0.05, random_state=0)
    clf.fit(Xtr, y_rise)
    # Conditional rise magnitude among rise days
    rise_rows = [r for r in train if r["remain_f"] > 0.5]
    reg = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    if rise_rows:
        reg.fit(
            np.asarray([r["features"] for r in rise_rows], dtype=float),
            np.asarray([r["remain_f"] for r in rise_rows], dtype=float),
        )

    # Evaluate point MAE using E[remain] ≈ p_rise * q50_rise
    errs = []
    for hour in (8, 11, 14):
        hrs = [r for r in test if int(r["decision_hour"]) == hour]
        e = []
        for r in hrs:
            x = np.asarray([r["features"]], dtype=float)
            p = float(clf.predict_proba(x)[0, 1])
            rem = float(reg.predict(x)[0]) if rise_rows else 0.0
            rem = max(0.0, rem)
            pred = float(r["max_so_far"]) + p * rem
            pred = max(pred, float(r["max_so_far"]))
            e.append(abs(float(r["label_tmax_f"]) - pred))
        errs.append((hour, float(np.mean(e)) if e else None, len(hrs)))

    # Compare to single quantile on same split
    m50 = GradientBoostingRegressor(
        loss="quantile", alpha=0.5, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
    )
    m50.fit(Xtr, np.asarray([r["remain_f"] for r in train], dtype=float))
    base_errs = []
    for hour in (8, 11, 14):
        hrs = [r for r in test if int(r["decision_hour"]) == hour]
        e = []
        for r in hrs:
            rem = float(m50.predict(np.asarray([r["features"]], dtype=float))[0])
            pred = max(float(r["max_so_far"]) + rem, float(r["max_so_far"]))
            e.append(abs(float(r["label_tmax_f"]) - pred))
        base_errs.append((hour, float(np.mean(e)) if e else None))

    report = {
        "ok": True,
        "method": "two_part_p_rise_times_conditional_q50",
        "rise_threshold_f": 0.5,
        "n_train_days": len(train_days),
        "n_test_days": len(test_days),
        "two_part_mae_by_hour": {str(h): mae for h, mae, _ in errs},
        "baseline_quantile_mae_by_hour": {str(h): mae for h, mae in base_errs},
        "improved": all(
            (t[1] is not None and b[1] is not None and t[1] < b[1] - 0.05) for t, b in zip(errs, base_errs)
        ),
        "promoted": False,
        "note": "Probe only; negative remains retained in training labels for classifier (rise=0).",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = data_dir / "research" / "two_part_probe.json"
    out.write_text(json.dumps(report, indent=2))
    report["saved_path"] = str(out)
    return report
