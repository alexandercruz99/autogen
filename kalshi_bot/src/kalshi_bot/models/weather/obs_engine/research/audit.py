"""Measurement / settlement audit for NYC Central Park obs engine."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine import NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import (
    HourlyObs,
    default_data_dir,
    load_asos_csv,
    load_ghcnd_tmax,
    load_nyc_hourly_bundle,
    local_date_for,
)
from kalshi_bot.models.weather.obs_engine.features import build_features_at, enumerate_training_rows


def _c_to_f_whole(c: float) -> float:
    return c * 9 / 5 + 32


def audit_measurement_settlement(*, data_dir: Path | None = None) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    obs = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    report: dict[str, Any] = {
        "station": {
            "ghcnd": NYC_TARGET.ghcnd_id,
            "cli": NYC_TARGET.cli_location_id,
            "metar": NYC_TARGET.metar_id,
            "coords": [NYC_TARGET.lat, NYC_TARGET.lon],
            "settlement": NYC_TARGET.settlement_source,
            "lst_climate_day": NYC_TARGET.uses_lst_climate_day,
            "unit": NYC_TARGET.unit,
        },
        "documentation_checked": [
            "https://help.kalshi.com/en/articles/13823837-weather-markets",
            "https://mesonet.agron.iastate.edu/request/asos/1min.phtml",
            "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
        ],
    }

    # Hourly vs daily max gap: max hourly tmpf vs GHCND TMAX
    by_day: dict[date, list[float]] = defaultdict(list)
    for o in obs:
        if o.tmpf is None:
            continue
        by_day[local_date_for(o.valid_utc)].append(float(o.tmpf))

    gaps = []
    whole_c_suspect = 0
    frac_temps = 0
    for d, temps in by_day.items():
        if d not in labels:
            continue
        hmax = max(temps)
        g = float(labels[d])
        gaps.append(g - hmax)
        # Fractional °F presence (METAR often 0.0 from whole °C)
        for t in temps:
            if abs(t - round(t)) > 1e-6:
                frac_temps += 1
            # whole °C converted: tmpf = 32 + 1.8*n for integer n
            c = (t - 32) * 5 / 9
            if abs(c - round(c)) < 1e-6:
                whole_c_suspect += 1

    gaps_a = np.asarray(gaps) if gaps else np.asarray([0.0])
    report["hourly_vs_ghcnd"] = {
        "n_days": len(gaps),
        "mean_ghcnd_minus_hourly_max_f": float(np.mean(gaps_a)),
        "median_ghcnd_minus_hourly_max_f": float(np.median(gaps_a)),
        "p90_gap_f": float(np.percentile(gaps_a, 90)),
        "frac_days_ghcnd_gt_hourly_max": float(np.mean(gaps_a > 0.05)),
        "frac_days_ghcnd_lt_hourly_max": float(np.mean(gaps_a < -0.05)),
        "interpretation": (
            "Positive gap ⇒ official daily max exceeded published hourly ASOS max "
            "(peak between hours, different sensor window, or rounding)."
        ),
    }

    # Negative remaining-rise labels at decision hours
    rows = enumerate_training_rows(obs, labels, decision_hours=(8, 11, 14))
    neg = [r for r in rows if r["remain_f"] < -0.05]
    report["negative_remaining_rise"] = {
        "n_rows": len(rows),
        "n_negative": len(neg),
        "frac_negative": float(len(neg) / len(rows)) if rows else None,
        "mean_negative_remain_f": float(np.mean([r["remain_f"] for r in neg])) if neg else None,
        "by_hour": {
            str(h): sum(1 for r in neg if r["decision_hour"] == h) for h in (8, 11, 14)
        },
        "hypothesis": (
            "Negative remain = GHCND TMAX < max_so_far from ASOS at decision time. "
            "Causes may include: whole-°C METAR rounding above true max, station/sensor "
            "differences vs CLI/GHCND, or climate-day LST boundary mismatch. Not clipped blindly."
        ),
        "sample": neg[:8],
    }

    # Missing sky/wind encoding risk in features
    missing_sky = sum(1 for o in obs if not o.skyc1)
    missing_wind = sum(1 for o in obs if o.sknt is None)
    report["missingness"] = {
        "n_obs": len(obs),
        "frac_missing_skyc1": float(missing_sky / len(obs)) if obs else None,
        "frac_missing_sknt": float(missing_wind / len(obs)) if obs else None,
        "baseline_feature_risk": (
            "features.py substitutes sknt=0 and alti=30 when missing — can look like calm/high pressure. "
            "Candidate local_v2 uses explicit missing indicators."
        ),
    }

    # Whole °C conversion prevalence
    with_tmp = [o for o in obs if o.tmpf is not None]
    report["precision"] = {
        "n_with_tmpf": len(with_tmp),
        "n_exact_whole_celsius_grid": whole_c_suspect,
        "frac_on_whole_c_grid": float(whole_c_suspect / len(with_tmp)) if with_tmp else None,
        "n_fractional_f_readings": frac_temps,
        "note": (
            "IEM ASOS tmpf often lands on whole-°C×1.8+32 grid. Aviation Weather may expose "
            "finer fields (tempFloat) — prospective collector stores precision_note."
        ),
    }

    # CLI reconcile file if present
    recon_path = data_dir / "label_reconciliation_cli_ghcnd.json"
    if recon_path.exists():
        report["cli_vs_ghcnd"] = json.loads(recon_path.read_text())
    else:
        report["cli_vs_ghcnd"] = {"note": "Run weather-obs-reconcile"}

    report["preliminary_vs_final"] = {
        "policy": (
            "CLI prelim used as soft floor evidence only; Kalshi settles on FINAL CLI. "
            "Prelim must not be treated as irrevocable physical maximum."
        )
    }
    report["leakage_policy"] = {
        "features": "valid_utc <= decision_utc only",
        "limitation": "IEM publication/first-seen not available historically — disclosed",
    }

    # 1-minute ASOS: attempt small sample diagnose (optional)
    report["one_minute_asos"] = {
        "status": "not_downloaded_in_this_run",
        "blocker_or_plan": (
            "IEM 1-minute ASOS can diagnose peaks between hourly reports "
            "(https://mesonet.agron.iastate.edu/request/asos/1min.phtml). "
            "Scoped download deferred unless hourly-gap audit warrants; "
            "operational use requires matching live cadence."
        ),
        "hourly_gap_already_quantified": True,
    }

    out = data_dir / "research" / "data_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    report["saved_path"] = str(out)
    return report
