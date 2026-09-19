"""Shared data loading and negative-remain audit for candidate evaluation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from kalshi_bot.models.weather.obs_engine.data import (
    default_data_dir,
    load_ghcnd_tmax,
    load_hourly_asos_bundle,
    load_nyc_hourly_bundle,
)
from kalshi_bot.models.weather.obs_engine.feeds.train_operating import (
    LOCATION_TRAIN_PROFILES,
    _enumerate_station_v2,
)


def load_location_rows(
    location_id: str,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Load station_v2 feature rows with GHCND labels; audit negative remain."""
    if location_id not in LOCATION_TRAIN_PROFILES:
        return {
            "ok": False,
            "location_id": location_id,
            "error": "no_train_profile",
            "status": "insufficient_data",
        }
    profile = LOCATION_TRAIN_PROFILES[location_id]
    data_dir = data_dir or default_data_dir()
    asos_dir = data_dir / profile["asos_subdir"] if profile.get("asos_subdir") else data_dir
    if location_id == "nyc_central_park" and not profile.get("asos_subdir"):
        obs = load_nyc_hourly_bundle(data_dir)
    else:
        obs = load_hourly_asos_bundle(
            asos_dir,
            glob_pattern=profile["asos_glob"],
            station=profile["iem_station"],
        )
    labels = load_ghcnd_tmax(data_dir / profile["label_csv"])
    if not obs or not labels:
        return {
            "ok": False,
            "location_id": location_id,
            "error": "missing_obs_or_labels",
            "n_obs": len(obs or []),
            "n_labels": len(labels or {}),
            "status": "insufficient_data",
        }
    rows, n_neg = _enumerate_station_v2(
        obs,
        labels,
        tz_name=profile["timezone"],
        lat=float(profile["lat"]),
        lon=float(profile["lon"]),
        station_id=profile["location_id"],
    )
    neg_examples = [r for r in rows if r.get("negative_remain")]
    # Cause taxonomy (identifiable from available fields)
    causes = {
        "asos_max_above_ghcnd_tmax": 0,
        "rounding_or_source_mismatch_suspect": 0,
        "near_zero_neg": 0,
    }
    for r in neg_examples:
        rem = float(r["remain_f"])
        if rem >= -0.5:
            causes["near_zero_neg"] += 1
        elif float(r["max_so_far"]) > float(r["label_tmax_f"]) + 0.5:
            causes["asos_max_above_ghcnd_tmax"] += 1
        else:
            causes["rounding_or_source_mismatch_suspect"] += 1

    return {
        "ok": True,
        "location_id": location_id,
        "profile": {
            "timezone": profile["timezone"],
            "metar_id": profile["metar_id"],
            "ghcnd_id": profile.get("ghcnd_id"),
            "label_csv": profile["label_csv"],
            "label_note": profile.get("label_note"),
            "series_ticker": profile["series_ticker"],
            "measurement": profile["measurement"],
        },
        "rows": rows,
        "n_rows": len(rows),
        "n_days": len({r["climate_day"] for r in rows}),
        "negative_remain": {
            "count": n_neg,
            "rate": (n_neg / len(rows)) if rows else None,
            "causes": causes,
            "policy": (
                "Negative remain labels are KEPT for training. They can reflect future cooling "
                "after an early peak missed by the label source, ASOS vs GHCND mismatch, "
                "rounding, or reporting-source differences. They are not clipped without audit."
            ),
            "examples": [
                {
                    "climate_day": r["climate_day"],
                    "decision_hour": r["decision_hour"],
                    "max_so_far": r["max_so_far"],
                    "label_tmax_f": r["label_tmax_f"],
                    "remain_f": r["remain_f"],
                }
                for r in neg_examples[:8]
            ],
        },
        "label_source": "ghcnd_tmax",
        "availability_assumption": "archive_valid_utc_equals_availability_DISCLOSED",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
