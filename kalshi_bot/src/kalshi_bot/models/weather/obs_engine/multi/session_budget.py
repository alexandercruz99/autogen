"""Hard session spend ledger for capped live weather bets.

Tracks capital reserved/committed across scheduled runs so ``$5 × N`` cannot
exceed a session max (default $20). File-backed for cross-process timer runs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_bot.money import D, ZERO

DEFAULT_SESSION_MAX = D("20.00")
DEFAULT_PER_BET_MAX = D("5.00")
LEDGER_DIR = Path("data/obs_engine/multi/session_budgets")

_LOCK = threading.Lock()

LIVE_SERIES_ALLOWLIST = frozenset({"KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX"})


@dataclass(frozen=True)
class SessionBudgetPolicy:
    session_id: str
    max_spend: Decimal = DEFAULT_SESSION_MAX
    per_bet_max: Decimal = DEFAULT_PER_BET_MAX
    allowlist: frozenset[str] = LIVE_SERIES_ALLOWLIST


def default_session_id(day: date | None = None) -> str:
    d = day or datetime.now(timezone.utc).date()
    return f"weather-live-{d.isoformat()}"


def ledger_path(session_id: str, *, root: Path | None = None) -> Path:
    base = root or LEDGER_DIR
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return base / f"{safe}.json"


def _empty(session_id: str, policy: SessionBudgetPolicy) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "max_spend": str(policy.max_spend),
        "per_bet_max": str(policy.per_bet_max),
        "spent": "0",
        "reserved": "0",
        "entries": [],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load(path: Path, session_id: str, policy: SessionBudgetPolicy) -> dict[str, Any]:
    if not path.exists():
        return _empty(session_id, policy)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return _empty(session_id, policy)
    raw.setdefault("session_id", session_id)
    raw.setdefault("max_spend", str(policy.max_spend))
    raw.setdefault("per_bet_max", str(policy.per_bet_max))
    raw.setdefault("spent", "0")
    raw.setdefault("reserved", "0")
    raw.setdefault("entries", [])
    return raw


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def remaining(session_id: str | None = None, *, policy: SessionBudgetPolicy | None = None, root: Path | None = None) -> Decimal:
    pol = policy or SessionBudgetPolicy(session_id=session_id or default_session_id())
    sid = session_id or pol.session_id
    with _LOCK:
        data = _load(ledger_path(sid, root=root), sid, pol)
        used = D(data["spent"]) + D(data["reserved"])
        max_spend = D(data.get("max_spend") or pol.max_spend)
        return max(ZERO, max_spend - used)


def _snapshot_unlocked(data: dict[str, Any], pol: SessionBudgetPolicy) -> dict[str, Any]:
    used = D(data["spent"]) + D(data["reserved"])
    max_spend = D(data.get("max_spend") or pol.max_spend)
    return {
        **data,
        "used": str(used),
        "remaining": str(max(ZERO, max_spend - used)),
    }


def snapshot(session_id: str | None = None, *, policy: SessionBudgetPolicy | None = None, root: Path | None = None) -> dict[str, Any]:
    pol = policy or SessionBudgetPolicy(session_id=session_id or default_session_id())
    sid = session_id or pol.session_id
    with _LOCK:
        data = _load(ledger_path(sid, root=root), sid, pol)
        return _snapshot_unlocked(data, pol)


def reserve(
    amount: Decimal | str | float,
    *,
    series_ticker: str,
    session_id: str | None = None,
    policy: SessionBudgetPolicy | None = None,
    root: Path | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Reserve capital before submit. Returns reservation id or error."""
    pol = policy or SessionBudgetPolicy(session_id=session_id or default_session_id())
    sid = session_id or pol.session_id
    tick = series_ticker.upper()
    if tick not in pol.allowlist:
        return {
            "ok": False,
            "error": f"series {tick} not in live session allowlist {sorted(pol.allowlist)}",
        }
    want = D(str(amount))
    if want <= ZERO:
        return {"ok": False, "error": "reserve amount must be > 0"}
    if want > pol.per_bet_max:
        return {
            "ok": False,
            "error": f"per-bet amount {want} exceeds per_bet_max {pol.per_bet_max}",
        }
    with _LOCK:
        path = ledger_path(sid, root=root)
        data = _load(path, sid, pol)
        max_spend = D(data.get("max_spend") or pol.max_spend)
        used = D(data["spent"]) + D(data["reserved"])
        left = max_spend - used
        if want > left:
            return {
                "ok": False,
                "error": f"session budget exhausted: need {want}, remaining {left}",
                "remaining": str(max(ZERO, left)),
                "max_spend": str(max_spend),
                "spent": data["spent"],
            }
        res_id = f"res-{len(data['entries'])+1}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        data["reserved"] = str(D(data["reserved"]) + want)
        data["entries"].append(
            {
                "id": res_id,
                "type": "reserve",
                "series": tick,
                "amount": str(want),
                "note": note,
                "at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save(path, data)
        return {
            "ok": True,
            "reservation_id": res_id,
            "amount": str(want),
            "remaining_after": str(max_spend - D(data["spent"]) - D(data["reserved"])),
            "session_id": sid,
        }


def commit(
    reservation_id: str,
    *,
    actual_spend: Decimal | str | float,
    session_id: str | None = None,
    policy: SessionBudgetPolicy | None = None,
    root: Path | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    """Convert a reservation into spent capital (may be less than reserved)."""
    pol = policy or SessionBudgetPolicy(session_id=session_id or default_session_id())
    sid = session_id or pol.session_id
    actual = D(str(actual_spend))
    with _LOCK:
        path = ledger_path(sid, root=root)
        data = _load(path, sid, pol)
        reserved_amt = ZERO
        for e in data["entries"]:
            if e.get("id") == reservation_id and e.get("type") == "reserve":
                reserved_amt = D(e["amount"])
                break
        if reserved_amt <= ZERO:
            return {"ok": False, "error": f"unknown reservation_id {reservation_id}"}
        if actual < ZERO:
            return {"ok": False, "error": "actual_spend must be >= 0"}
        if actual > reserved_amt:
            actual = reserved_amt
        data["reserved"] = str(max(ZERO, D(data["reserved"]) - reserved_amt))
        data["spent"] = str(D(data["spent"]) + actual)
        data["entries"].append(
            {
                "id": f"commit-{reservation_id}",
                "type": "commit",
                "reservation_id": reservation_id,
                "amount": str(actual),
                "order_id": order_id,
                "at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save(path, data)
        return {"ok": True, "spent_added": str(actual), "snapshot": _snapshot_unlocked(data, pol)}


def release(
    reservation_id: str,
    *,
    session_id: str | None = None,
    policy: SessionBudgetPolicy | None = None,
    root: Path | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Release a reservation without spending (failed submit / no candidate)."""
    pol = policy or SessionBudgetPolicy(session_id=session_id or default_session_id())
    sid = session_id or pol.session_id
    with _LOCK:
        path = ledger_path(sid, root=root)
        data = _load(path, sid, pol)
        reserved_amt = ZERO
        for e in data["entries"]:
            if e.get("id") == reservation_id and e.get("type") == "reserve":
                reserved_amt = D(e["amount"])
                break
        if reserved_amt <= ZERO:
            return {"ok": False, "error": f"unknown reservation_id {reservation_id}"}
        data["reserved"] = str(max(ZERO, D(data["reserved"]) - reserved_amt))
        data["entries"].append(
            {
                "id": f"release-{reservation_id}",
                "type": "release",
                "reservation_id": reservation_id,
                "amount": str(reserved_amt),
                "reason": reason,
                "at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save(path, data)
        return {"ok": True, "released": str(reserved_amt), "snapshot": _snapshot_unlocked(data, pol)}
