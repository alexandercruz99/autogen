"""Freeze baseline artifacts, reproduce metrics, write manifests + hashes."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.models.weather.obs_engine import MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import default_data_dir, load_ghcnd_tmax, load_nyc_hourly_bundle, local_date_for
from kalshi_bot.models.weather.obs_engine.features import FEATURE_NAMES
from kalshi_bot.models.weather.obs_engine.research import BASELINE_MODEL_VERSION


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_baseline(
    *,
    data_dir: Path | None = None,
    reproduce_backtest: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    freeze_dir = data_dir / "research" / "baseline_freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    # Copy frozen model artifact
    src_model = data_dir / "models" / "obs_nyc_q50.joblib"
    dst_model = freeze_dir / "obs_nyc_q50_baseline.joblib"
    if src_model.exists():
        shutil.copy2(src_model, dst_model)

    # Dataset manifest
    asos = load_nyc_hourly_bundle(data_dir)
    labels = load_ghcnd_tmax(data_dir / "nyc_central_park_ghcnd_tmax_f.csv")
    days = sorted({local_date_for(o.valid_utc) for o in asos})
    overlap = [d for d in days if d in labels]
    asos_files = sorted(data_dir.glob("asos_NYC_*.csv"))
    manifest = {
        "frozen_at_utc": now.isoformat(),
        "station": {
            "display": NYC_TARGET.display_name,
            "ghcnd": NYC_TARGET.ghcnd_id,
            "cli": NYC_TARGET.cli_location_id,
            "metar": NYC_TARGET.metar_id,
            "lat": NYC_TARGET.lat,
            "lon": NYC_TARGET.lon,
            "timezone": NYC_TARGET.timezone,
            "settlement_source": NYC_TARGET.settlement_source,
        },
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "current_code_model_version": MODEL_VERSION,
        "feature_names_baseline": FEATURE_NAMES,
        "decision_hours_local": list(NYC_TARGET.decision_hours_local),
        "asos_files": [{"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in asos_files],
        "ghcnd_path": str(data_dir / "nyc_central_park_ghcnd_tmax_f.csv"),
        "ghcnd_sha256": _sha256(data_dir / "nyc_central_park_ghcnd_tmax_f.csv"),
        "n_hourly_obs": len(asos),
        "n_asos_local_days": len(days),
        "asos_span": [days[0].isoformat(), days[-1].isoformat()] if days else None,
        "n_ghcnd_days": len(labels),
        "n_overlap_climate_days": len(overlap),
        "independent_station_days_note": (
            "Independent outcomes = unique climate days with labels, not hourly row counts."
        ),
        "backtest_claim": {
            "test_n_days": 333,
            "test_span": ["2025-10-05", "2026-09-13"],
            "mae_hours_f": {"8": 3.45, "11": 2.45, "14": 1.08},
            "beats_climatology": True,
            "beats_decision_time_nws": False,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    try:
        import numpy
        import sklearn

        manifest["software"]["numpy"] = numpy.__version__
        manifest["software"]["sklearn"] = sklearn.__version__
    except Exception:
        pass

    hashes = {
        "obs_nyc_q50_baseline.joblib": _sha256(dst_model),
        "backtest_report.json": _sha256(data_dir / "backtest_report.json"),
        "features.py": _sha256(Path("src/kalshi_bot/models/weather/obs_engine/features.py")),
        "backtest.py": _sha256(Path("src/kalshi_bot/models/weather/obs_engine/backtest.py")),
    }
    manifest["artifact_sha256"] = hashes

    reproduce: dict[str, Any] | None = None
    if reproduce_backtest:
        from kalshi_bot.models.weather.obs_engine.backtest import run_obs_backtest

        # Use cached OM CSV; do not refetch
        r = run_obs_backtest(
            data_dir=data_dir,
            archive_path="data/weather_archive.db",
            fetch_external_benchmark=False,
        )
        # Load OM from disk for fair comparison numbers if present
        from kalshi_bot.models.weather.obs_engine.backtest import load_open_meteo_historical_highs

        # Re-run with OM loaded via existing CSV: patch by setting fetch True only if file exists
        if (data_dir / "open_meteo_nyc_tmax_historical.csv").exists():
            r = run_obs_backtest(
                data_dir=data_dir,
                archive_path="data/weather_archive.db",
                fetch_external_benchmark=True,
            )
        hours = (r.get("overall_test") or {}).get("hours") or {}
        claimed = manifest["backtest_claim"]["mae_hours_f"]
        reproduce = {
            "ok": r.get("ok"),
            "n_independent_climate_days": r.get("n_independent_climate_days"),
            "test_split": r.get("splits", {}).get("test"),
            "reproduced_mae_hours": hours,
            "claimed_mae_hours_approx": claimed,
            "match_within_0_05f": {
                h: (abs(float(hours.get(h, 99)) - float(claimed[h])) < 0.05) for h in claimed if str(h) in hours or h in hours
            },
            "beats_climatology_all_hours": (r.get("overall_test") or {}).get("beats_climatology_all_hours"),
            "outperforms_established_nws_decision_time": (r.get("overall_test") or {}).get(
                "outperforms_established_nws_decision_time"
            ),
            "note": "Reproduction uses same chronological split code; OM archive still labeled retrospective.",
        }
        # Fix match keys to string hours
        reproduce["match_within_0_05f"] = {
            str(h): abs(float(hours.get(str(h), hours.get(h, 99))) - float(v)) < 0.05 for h, v in claimed.items()
        }
        (freeze_dir / "reproduction_report.json").write_text(json.dumps(reproduce, indent=2, default=str))

    (freeze_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # Copy feature definition snapshot
    (freeze_dir / "feature_names_baseline.json").write_text(json.dumps(FEATURE_NAMES, indent=2))

    out = {
        "ok": True,
        "freeze_dir": str(freeze_dir),
        "manifest_path": str(freeze_dir / "dataset_manifest.json"),
        "baseline_model_sha256": hashes.get("obs_nyc_q50_baseline.joblib"),
        "reproduction": reproduce,
        "live_eligible": False,
    }
    (freeze_dir / "freeze_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out
