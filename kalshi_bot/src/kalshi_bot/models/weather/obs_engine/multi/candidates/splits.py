"""Chronological split manifests — persisted before tuning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


@dataclass
class SplitManifest:
    location_id: str
    metric: str
    label_source: str
    train_days: list[str]
    selection_days: list[str]
    calib_days: list[str]
    test_days: list[str]
    decision_hours_local: list[int]
    availability_assumption: str
    created_at_utc: str
    frac: dict[str, float]
    notes: list[str]

    def day_hash(self) -> str:
        payload = "|".join(
            [
                *self.train_days,
                "#",
                *self.selection_days,
                "#",
                *self.calib_days,
                "#",
                *self.test_days,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["manifest_hash"] = self.day_hash()
        return d


def build_split_manifest(
    *,
    location_id: str,
    metric: str,
    label_source: str,
    climate_days: Sequence[str],
    decision_hours_local: Sequence[int],
    train_frac: float = 0.55,
    selection_frac: float = 0.15,
    calib_frac: float = 0.15,
    # remainder = test
    availability_assumption: str = "archive_valid_utc_equals_availability_DISCLOSED",
    notes: list[str] | None = None,
) -> SplitManifest:
    """Create train / model-selection / calib / final-test day sets (chronological)."""
    days = sorted({str(d) for d in climate_days})
    n = len(days)
    if n < 40:
        raise ValueError(f"too few climate days ({n}) for four-way split")
    i_tr = int(n * train_frac)
    i_sel = int(n * (train_frac + selection_frac))
    i_ca = int(n * (train_frac + selection_frac + calib_frac))
    # Ensure non-empty tails
    i_tr = max(10, min(i_tr, n - 15))
    i_sel = max(i_tr + 5, min(i_sel, n - 10))
    i_ca = max(i_sel + 5, min(i_ca, n - 5))
    return SplitManifest(
        location_id=location_id,
        metric=metric,
        label_source=label_source,
        train_days=days[:i_tr],
        selection_days=days[i_tr:i_sel],
        calib_days=days[i_sel:i_ca],
        test_days=days[i_ca:],
        decision_hours_local=list(decision_hours_local),
        availability_assumption=availability_assumption,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        frac={
            "train": train_frac,
            "selection": selection_frac,
            "calib": calib_frac,
            "test": round(1.0 - train_frac - selection_frac - calib_frac, 4),
        },
        notes=notes
        or [
            "All decision hours for a climate day stay in the same split.",
            "Final test days must not be used for s_min, bias, or candidate selection.",
            "If prior reported windows influenced development, treat them as development benchmarks.",
        ],
    )


def save_split_manifest(path: Path, manifest: SplitManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.as_dict(), indent=2))
    return path


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def filter_rows_by_days(rows: list[dict[str, Any]], days: set[str] | list[str]) -> list[dict[str, Any]]:
    dayset = set(days)
    return [r for r in rows if str(r["climate_day"]) in dayset]
