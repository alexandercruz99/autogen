"""Chronological backtests for observation-driven NYC daily-max engine.

Decision times evaluated separately (local): 08:00, 11:00, 14:00.
NWS/Open-Meteo comparisons are labeled by provenance — never silent forecast features.
Trading profitability is UNVALIDATED without executable historical books.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.models.weather.obs_engine import ARTIFACT_NAME, MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import (
    default_data_dir,
    load_ghcnd_tmax,
    load_nyc_hourly_bundle,
    local_date_for,
)
from kalshi_bot.models.weather.obs_engine.features import FEATURE_NAMES, build_features_at, enumerate_training_rows
from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.settlement_rules import TempInterval

logger = logging.getLogger(__name__)

# Fixed local decision hours for reporting (not pooled into one headline).
BACKTEST_HOURS = (8, 11, 14)


def _block_bootstrap_mae_diff(
    err_a: list[float],
    err_b: list[float],
    *,
    day_ids: list[str],
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Day-blocked bootstrap CI for MAE(A)−MAE(B). Negative ⇒ A better."""
    by_day: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for d, a, b in zip(day_ids, err_a, err_b):
        by_day[d].append((a, b))
    days = sorted(by_day)
    if len(days) < 5:
        return {"n_days": float(len(days)), "diff_mae": float("nan"), "ci80_lo": float("nan"), "ci80_hi": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(days, size=len(days), replace=True)
        ea, eb = [], []
        for d in sample:
            for a, b in by_day[d]:
                ea.append(abs(a))
                eb.append(abs(b))
        diffs.append(float(np.mean(ea) - np.mean(eb)))
    return {
        "n_days": float(len(days)),
        "diff_mae": float(np.mean(np.abs(err_a)) - np.mean(np.abs(err_b))),
        "ci80_lo": float(np.percentile(diffs, 10)),
        "ci80_hi": float(np.percentile(diffs, 90)),
    }


def _fit_quantile_models(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingRegressor

    models = {}
    for q, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, max_depth=2, n_estimators=120, learning_rate=0.05, random_state=0
        )
        m.fit(X, y)
        models[name] = m
    return models


def _chrono_day_splits(
    days: list[str],
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> dict[str, set[str]]:
    n = len(days)
    i_train = max(1, int(n * train_frac))
    i_val = max(i_train + 1, int(n * (train_frac + val_frac)))
    return {
        "train": set(days[:i_train]),
        "val": set(days[i_train:i_val]),
        "test": set(days[i_val:]),
    }


def load_open_meteo_historical_highs(path: Path | None = None) -> dict[date, float]:
    """Optional external point benchmark (GFS seamless archive). Not decision-time-safe NWS."""
    path = path or default_data_dir() / "open_meteo_nyc_tmax_historical.csv"
    out: dict[date, float] = {}
    if not path.exists():
        return out
    import csv

    with path.open() as f:
        for row in csv.DictReader(f):
            out[date.fromisoformat(row["date"])] = float(row["tmax_f"])
    return out


def fetch_open_meteo_historical_highs(
    start: date,
    end: date,
    *,
    data_dir: Path | None = None,
) -> dict[date, float]:
    """Download Open-Meteo historical forecast daily max (RETROSPECTIVE archive).

    Provenance: model forecast archive, not NWS CLI and not guaranteed available at
    a specific local decision hour. Used only as an external point-error benchmark.
    """
    import httpx

    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "open_meteo_nyc_tmax_historical.csv"
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_TARGET.lat,
        "longitude": NYC_TARGET.lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": NYC_TARGET.timezone,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": "gfs_seamless",
    }
    try:
        r = httpx.get(url, params=params, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        js = r.json()
    except Exception as exc:
        logger.warning("Open-Meteo historical fetch failed: %s", exc)
        return load_open_meteo_historical_highs(out_path)

    days = js.get("daily", {}).get("time") or []
    temps = js.get("daily", {}).get("temperature_2m_max") or []
    rows = []
    mapping: dict[date, float] = {}
    for d_s, t in zip(days, temps):
        if t is None:
            continue
        d = date.fromisoformat(d_s)
        mapping[d] = float(t)
        rows.append({"date": d_s, "tmax_f": f"{float(t):.2f}", "source": "open_meteo_gfs_seamless_historical_forecast"})
    import csv

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "tmax_f", "source"])
        w.writeheader()
        w.writerows(rows)
    return mapping


def reconcile_cli_vs_ghcnd(
    *,
    archive_path: str | Path = "data/weather_archive.db",
    max_reports: int = 40,
) -> dict[str, Any]:
    """Compare archived NWS CLI MAXIMUM vs GHCND TMAX for Central Park."""
    from kalshi_bot.models.weather.cli_reports import CliReportClient

    labels = load_ghcnd_tmax()
    archive = WeatherArchive(archive_path)
    client = CliReportClient()
    pairs = []
    try:
        reports = client.collect_recent_with_max(NYC_TARGET.cli_location_id, limit=max_reports)
        for rep in reports:
            archive.save_cli(station_key="NYC", report=rep)
            g = labels.get(rep.climate_day)
            if g is None or rep.max_temp_f is None:
                continue
            pairs.append(
                {
                    "climate_day": rep.climate_day.isoformat(),
                    "cli_max_f": int(rep.max_temp_f),
                    "ghcnd_tmax_f": int(g),
                    "diff_cli_minus_ghcnd": int(rep.max_temp_f) - int(g),
                    "cli_preliminary": bool(rep.is_preliminary),
                    "issuance": rep.issuance_time.isoformat() if rep.issuance_time else None,
                }
            )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n_pairs": 0}
    finally:
        client.close()

    diffs = [p["diff_cli_minus_ghcnd"] for p in pairs]
    report = {
        "ok": True,
        "n_pairs": len(pairs),
        "n_exact_match": sum(1 for d in diffs if d == 0),
        "n_abs_diff_ge_1": sum(1 for d in diffs if abs(d) >= 1),
        "n_abs_diff_ge_2": sum(1 for d in diffs if abs(d) >= 2),
        "mean_diff": float(np.mean(diffs)) if diffs else None,
        "mae_diff": float(np.mean(np.abs(diffs))) if diffs else None,
        "settlement_source_controls": "NWS CLI MAXIMUM (Kalshi daily high settlement)",
        "training_label_source": "GHCND USW00094728 TMAX (same station climate record; proxy when historical CLI text unavailable)",
        "pairs_sample": pairs[:15],
        "note": (
            "Mismatches can arise from preliminary vs final CLI, rounding, or rare revisions. "
            "Kalshi settles on final CLI; GHCND is used for multi-year labels."
        ),
    }
    out = default_data_dir() / "label_reconciliation_cli_ghcnd.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def run_obs_backtest(
    *,
    data_dir: Path | None = None,
    archive_path: str | Path = "data/weather_archive.db",
    decision_hours: tuple[int, ...] = BACKTEST_HOURS,
    fetch_external_benchmark: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    obs = load_nyc_hourly_bundle(data_dir)
    # Build rows for backtest hours (may differ from train default hours)
    rows = enumerate_training_rows(obs, labels, decision_hours=decision_hours)
    days = sorted({r["climate_day"] for r in rows})
    splits = _chrono_day_splits(days)
    report: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_hourly_obs": len(obs),
        "n_independent_climate_days": len(days),
        "n_rows": len(rows),
        "decision_hours_local": list(decision_hours),
        "splits": {k: {"n_days": len(v), "first": min(v) if v else None, "last": max(v) if v else None} for k, v in splits.items()},
        "availability_limitation": (
            "ASOS features use observation valid timestamps. Publication/first-seen times are not in IEM CSV; "
            "historical availability at decision time is assumed ≈ valid time (DISCLOSED LIMITATION)."
        ),
        "trading_profitability": "UNVALIDATED — no historical executable Kalshi order-book replay",
        "by_hour": {},
        "overall_test": {},
    }

    om_bench: dict[date, float] = {}
    if fetch_external_benchmark and days:
        om_bench = fetch_open_meteo_historical_highs(
            date.fromisoformat(days[0]),
            date.fromisoformat(days[-1]),
            data_dir=data_dir,
        )
    report["external_benchmark"] = {
        "name": "open_meteo_gfs_seamless_historical_forecast",
        "role": "point_error_benchmark_only",
        "provenance": "RETROSPECTIVE_FORECAST_ARCHIVE — not NWS CLI; not proven available at local decision hour",
        "n_days": len(om_bench),
        "nws_decision_time_matched": False,
        "note": "True same-decision-time NWS grid snapshots require forward collection into weather_archive.db",
    }

    train_rows = [r for r in rows if r["climate_day"] in splits["train"]]
    if len(splits["train"]) < 40:
        report["ok"] = False
        report["reason"] = f"Insufficient train days ({len(splits['train'])})"
        return report

    X_train = np.asarray([r["features"] for r in train_rows], dtype=float)
    y_train = np.asarray([r["remain_f"] for r in train_rows], dtype=float)
    models = _fit_quantile_models(X_train, y_train)

    # Month climatology from train labels (one value per day)
    month_vals: dict[int, list[int]] = defaultdict(list)
    seen_day = set()
    for r in train_rows:
        if r["climate_day"] in seen_day:
            continue
        seen_day.add(r["climate_day"])
        month_vals[date.fromisoformat(r["climate_day"]).month].append(int(r["label_tmax_f"]))
    clim = {m: float(np.mean(v)) for m, v in month_vals.items()}
    global_clim = float(np.mean([x for xs in month_vals.values() for x in xs]))

    label_by_day = {r["climate_day"]: int(r["label_tmax_f"]) for r in rows}
    sorted_days = sorted(label_by_day)

    def persist(day_s: str) -> float | None:
        if day_s not in sorted_days:
            return None
        i = sorted_days.index(day_s)
        return float(label_by_day[sorted_days[i - 1]]) if i > 0 else None

    # Persist fitted artifact for inference consistency with backtest train cutoff
    try:
        import joblib

        model_path = data_dir / "models" / "obs_nyc_q50.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "feature_names": FEATURE_NAMES}, model_path)
        remain_resid = list((y_train - models["q50"].predict(X_train)).astype(float))
        artifact = {
            "type": ARTIFACT_NAME,
            "feature_names": FEATURE_NAMES,
            "remain_residuals": remain_resid[:500],
            "climatology_by_month": clim,
            "train_end_day": max(splits["train"]) if splits["train"] else None,
            "model_path": str(model_path),
            "n_train_days": len(splits["train"]),
            "backtest_hours": list(decision_hours),
        }
        archive = WeatherArchive(archive_path)
        archive.save_artifact("NYC", ARTIFACT_NAME, artifact, metrics={"source": "obs_backtest_train"}, train_end_day=artifact["train_end_day"])
        report["artifact_path"] = str(model_path)
        report["train_end_day"] = artifact["train_end_day"]
    except Exception as exc:
        report["artifact_warning"] = str(exc)

    def eval_split(split_name: str) -> dict[str, Any]:
        subset = [r for r in rows if r["climate_day"] in splits[split_name]]
        by_h: dict[str, Any] = {}
        for hour in decision_hours:
            hrs = [r for r in subset if int(r["decision_hour"]) == hour]
            if not hrs:
                continue
            y = np.asarray([r["label_tmax_f"] for r in hrs], dtype=float)
            max_so = np.asarray([r["max_so_far"] for r in hrs], dtype=float)
            X = np.asarray([r["features"] for r in hrs], dtype=float)
            rem = models["q50"].predict(X)
            pred = max_so + rem
            # Enforce physical floor at observed max
            pred = np.maximum(pred, max_so)
            err_model = list(y - pred)
            err_clim = []
            err_cont = list(y - max_so)
            err_pers = []
            err_om = []
            day_ids = []
            for r, e in zip(hrs, err_model):
                day_ids.append(r["climate_day"])
                m = date.fromisoformat(r["climate_day"]).month
                err_clim.append(float(r["label_tmax_f"]) - clim.get(m, global_clim))
                pp = persist(r["climate_day"])
                if pp is not None:
                    err_pers.append(float(r["label_tmax_f"]) - pp)
                d = date.fromisoformat(r["climate_day"])
                if d in om_bench:
                    err_om.append(float(r["label_tmax_f"]) - om_bench[d])

            # Interval coverage from q10/q90 remain
            q10 = models["q10"].predict(X)
            q90 = models["q90"].predict(X)
            lo = max_so + np.minimum(q10, q90)
            hi = max_so + np.maximum(q10, q90)
            lo = np.maximum(lo, max_so)
            covered = ((y >= lo) & (y <= hi)).mean()

            # Contract probe: P(final > max_so_far) via remain>0 — weak calibration probe
            # Better: bracket around integer final using empirical residual mixture
            from kalshi_bot.models.weather.distribution import from_empirical_residuals, truncate_below

            resid = list((y_train - models["q50"].predict(X_train)).astype(float))[:200]
            brier_gt_clim = []
            for r, p50, rem_hat in zip(hrs, pred, rem):
                dist = from_empirical_residuals(float(p50), resid, method="backtest_probe")
                dist = truncate_below(dist, float(r["max_so_far"]), reason="max_so_far")
                # Probe: YES if final >= round(climatology)
                thr = clim.get(date.fromisoformat(r["climate_day"]).month, global_clim)
                iv = TempInterval("gt", float(thr) - 0.5, None, "probe")  # rough
                # Use greater-than floor=thr-1 so ~ >= thr for integers... keep simple range
                iv = TempInterval("range_inclusive", int(round(thr)), int(round(thr)), "probe_eq_clim")
                p = float(dist.p_interval(iv))
                y_bin = 1.0 if int(r["label_tmax_f"]) == int(round(thr)) else 0.0
                brier_gt_clim.append((p - y_bin) ** 2)

            by_h[str(hour)] = {
                "n_rows": len(hrs),
                "n_days": len(set(day_ids)),
                "mae_model": float(np.mean(np.abs(err_model))),
                "rmse_model": float(np.sqrt(np.mean(np.square(err_model)))),
                "mae_climatology": float(np.mean(np.abs(err_clim))),
                "mae_continuation_max_so_far": float(np.mean(np.abs(err_cont))),
                "mae_persistence": float(np.mean(np.abs(err_pers))) if err_pers else None,
                "mae_open_meteo_hist_forecast": float(np.mean(np.abs(err_om))) if err_om else None,
                "n_open_meteo": len(err_om),
                "interval_q10_q90_coverage": float(covered),
                "mean_interval_width_f": float(np.mean(hi - lo)),
                "brier_probe_eq_month_clim": float(np.mean(brier_gt_clim)) if brier_gt_clim else None,
                "bootstrap_model_minus_clim_mae": _block_bootstrap_mae_diff(err_model, err_clim, day_ids=day_ids),
                "bootstrap_model_minus_continuation_mae": _block_bootstrap_mae_diff(err_model, err_cont, day_ids=day_ids),
                "bootstrap_model_minus_om_mae": (
                    _block_bootstrap_mae_diff(
                        [e for e, r in zip(err_model, hrs) if date.fromisoformat(r["climate_day"]) in om_bench],
                        err_om,
                        day_ids=[r["climate_day"] for r in hrs if date.fromisoformat(r["climate_day"]) in om_bench],
                    )
                    if err_om
                    else None
                ),
            }
        return by_h

    for split in ("val", "test"):
        report["by_hour"][split] = eval_split(split)

    # Headline: test-set metrics MUST stay broken out by hour — also store macro-average with disclosure
    test_hours = report["by_hour"].get("test") or {}
    if test_hours:
        maes = [v["mae_model"] for v in test_hours.values()]
        report["overall_test"] = {
            "macro_mean_mae_across_hours": float(np.mean(maes)),
            "note": "Macro-average across decision hours only; do not treat as pooled accuracy claim",
            "hours": {h: test_hours[h]["mae_model"] for h in test_hours},
            "beats_climatology_all_hours": all(
                test_hours[h]["mae_model"] < test_hours[h]["mae_climatology"] for h in test_hours
            ),
            "beats_open_meteo_all_hours": all(
                test_hours[h].get("mae_open_meteo_hist_forecast") is not None
                and test_hours[h]["mae_model"] < test_hours[h]["mae_open_meteo_hist_forecast"]
                for h in test_hours
            )
            if all(test_hours[h].get("mae_open_meteo_hist_forecast") is not None for h in test_hours)
            else False,
            "outperforms_established_nws_decision_time": False,
            "outperforms_established_nws_reason": (
                "No decision-time-matched NWS grid archive in this backtest. "
                "Open-Meteo historical forecast is a retrospective external point benchmark only."
            ),
        }

    report["ok"] = True
    report["live_eligible"] = False
    out_path = data_dir / "backtest_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(out_path)
    return report
