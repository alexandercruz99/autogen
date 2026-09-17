from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from dataclasses import dataclass

from kalshi_bot.models.base import Prediction, ProbabilityModel
from kalshi_bot.money import D

logger = logging.getLogger(__name__)

MODEL_VERSION = "economics.cpi_mom.v0.1-unvalidated"
USER_AGENT = "(kalshi-bot, research-use; local-operator)"


@dataclass
class CpiContract:
    year: int
    month: int  # 1-12
    threshold_pct: float
    op: str  # gt


_TICKER_RE = re.compile(
    r"KXCPI-(?P<yy>\d{2})(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-T(?P<thr>-?\d+(?:\.\d+)?)$",
    re.I,
)
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_cpi_market(ticker: str, title: str = "", rules: str = "") -> CpiContract | None:
    m = _TICKER_RE.search(ticker.upper())
    if not m:
        return None
    thr = float(m.group("thr"))
    text = f"{title} {rules}".lower()
    if "more than" in text or "greater" in text or ">" in text or "rise more" in text:
        op = "gt"
    elif "less than" in text or "<" in text:
        op = "lt"
    else:
        # Kalshi KXCPI T thresholds are "rise more than X%" per series conventions.
        op = "gt"
    return CpiContract(
        year=2000 + int(m.group("yy")),
        month=_MONTHS[m.group("mon").upper()],
        threshold_pct=thr,
        op=op,
    )


class BLSClient:
    """BLS Public Data API (no key required for low volume)."""

    def __init__(self, timeout: float = 60.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})

    def close(self) -> None:
        self._client.close()

    def cpi_sa_index(self, start_year: int, end_year: int) -> list[dict[str, Any]]:
        payload = {
            "seriesid": ["CUSR0000SA0"],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        resp = self._client.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS error: {data.get('message')}")
        rows = data["Results"]["series"][0]["data"]
        out = []
        for r in rows:
            if not str(r.get("period", "")).startswith("M"):
                continue
            if r.get("value") in (None, ".", "-"):
                continue
            try:
                value = float(r["value"])
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "year": int(r["year"]),
                    "month": int(r["period"][1:]),
                    "value": value,
                    "period_name": r.get("periodName"),
                    # BLS does not always provide exact release timestamp here;
                    # we store fetch time separately and refuse to use future months.
                    "bls_period": f"{r['year']}{r['period']}",
                }
            )
        out.sort(key=lambda x: (x["year"], x["month"]))
        return out


def mom_series(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    moms = []
    for i in range(1, len(levels)):
        prev, cur = levels[i - 1], levels[i]
        mom = (cur["value"] / prev["value"] - 1.0) * 100.0
        moms.append(
            {
                "year": cur["year"],
                "month": cur["month"],
                "mom_pct": mom,
                "level": cur["value"],
                "bls_period": cur["bls_period"],
            }
        )
    return moms


class CpiMomModel(ProbabilityModel):
    """CPI MoM binary markets settled by BLS.

    Forecast: same-calendar-month historical mean MoM (climatology) with residual σ
    from all historical MoM innovations. This is a baseline, not a nowcast consensus.
    VALIDATION STATUS: unvalidated for live edge.
    """

    name = "economics_cpi_mom"
    version = MODEL_VERSION
    categories = ["Economics", "Financials", "economy"]

    def __init__(self, store: Any = None) -> None:
        self.store = store
        self.bls = BLSClient()
        self._cache: list[dict[str, Any]] | None = None
        self._fetched_at: datetime | None = None

    def close(self) -> None:
        self.bls.close()

    def supports(self, market: dict[str, Any], category: str) -> bool:
        ticker = (market.get("ticker") or "").upper()
        return ticker.startswith("KXCPI-") or ticker.startswith("CPI-")

    def _history(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        now = datetime.now(timezone.utc)
        if self.store:
            cached = self.store.get_forecast_cache("bls:CUSR0000SA0:moms")
            if cached:
                import json
                from datetime import datetime as dt

                fetched = dt.fromisoformat(cached["fetched_at"])
                if (now - fetched).total_seconds() < 3600 * 6:
                    payload = json.loads(cached["payload_json"])
                    self._cache = payload.get("moms") or []
                    self._fetched_at = fetched
                    if self._cache:
                        return self._cache
        # BLS may not accept a future end year cleanly; request through current calendar year.
        end_year = now.year
        levels = self.bls.cpi_sa_index(start_year=now.year - 8, end_year=end_year)
        moms = mom_series(levels)
        self._cache = moms
        self._fetched_at = now
        if self.store is not None:
            self.store.cache_forecast(
                "bls:CUSR0000SA0:moms",
                "api.bls.gov",
                {"moms": moms[-120:], "fetched_at": now.isoformat()},
            )
        return moms

    def predict(self, market: dict[str, Any], category: str) -> Prediction:
        from scipy.stats import norm

        now = datetime.now(timezone.utc)
        ticker = market.get("ticker") or ""
        contract = parse_cpi_market(
            ticker,
            market.get("title") or "",
            market.get("rules_primary") or "",
        )
        if contract is None:
            return Prediction(
                market_ticker=ticker,
                p_yes=D("0.5"),
                p_yes_conservative=D("0.5"),
                uncertainty=D("0.5"),
                model_version=self.version,
                data_sources=[],
                factors=[],
                validation_evidence="skipped",
                as_of=now,
                supported=False,
                skip_reason="could not parse CPI threshold from ticker/title",
            )

        # Refuse if target month already fully in the past relative to latest BLS print
        # without using that month's print when predicting that month before release —
        # for past months, skip trading (settlement likely known/near).
        try:
            hist = self._history()
        except Exception as exc:
            return Prediction(
                market_ticker=ticker,
                p_yes=D("0.5"),
                p_yes_conservative=D("0.5"),
                uncertainty=D("0.5"),
                model_version=self.version,
                data_sources=[],
                factors=[],
                validation_evidence="skipped",
                as_of=now,
                supported=False,
                skip_reason=f"BLS history unavailable: {exc}",
            )

        latest = hist[-1]
        # If BLS already published the target month, do not trade — avoids using final print as "forecast".
        if (latest["year"], latest["month"]) >= (contract.year, contract.month):
            return Prediction(
                market_ticker=ticker,
                p_yes=D("0.5"),
                p_yes_conservative=D("0.5"),
                uncertainty=D("0.5"),
                model_version=self.version,
                data_sources=[{"name": "BLS CUSR0000SA0", "fetched_at": self._fetched_at.isoformat() if self._fetched_at else None}],
                factors=[],
                validation_evidence="skipped",
                as_of=now,
                supported=False,
                skip_reason=(
                    f"BLS already has print through {latest['year']}-{latest['month']:02d}; "
                    "refusing to trade with potentially leaked settlement information"
                ),
            )

        same_month = [h["mom_pct"] for h in hist if h["month"] == contract.month]
        all_moms = [h["mom_pct"] for h in hist]
        if len(same_month) < 3 or len(all_moms) < 12:
            return Prediction(
                market_ticker=ticker,
                p_yes=D("0.5"),
                p_yes_conservative=D("0.5"),
                uncertainty=D("0.5"),
                model_version=self.version,
                data_sources=[],
                factors=[],
                validation_evidence="skipped",
                as_of=now,
                supported=False,
                skip_reason="insufficient BLS history for climatology prior",
            )

        import statistics

        mu = statistics.mean(same_month)
        # Residual around same-month mean across years + overall dispersion floor
        resid = [h["mom_pct"] - mu for h in hist if h["month"] == contract.month]
        sigma = max(statistics.pstdev(resid) if len(resid) > 1 else 0.3, 0.20)

        thr = contract.threshold_pct
        if contract.op == "gt":
            p = float(1.0 - norm.cdf(thr, loc=mu, scale=sigma))
            p_wide = float(1.0 - norm.cdf(thr, loc=mu, scale=sigma * 1.5))
        else:
            p = float(norm.cdf(thr, loc=mu, scale=sigma))
            p_wide = float(norm.cdf(thr, loc=mu, scale=sigma * 1.5))

        p_yes_low = min(p, p_wide)
        p_yes_high = max(p, p_wide)
        uncertainty = max(abs(p - p_wide), 0.08)

        return Prediction(
            market_ticker=ticker,
            p_yes=D(f"{p:.6f}"),
            p_yes_conservative=D(f"{p_yes_low:.6f}"),
            uncertainty=D(f"{uncertainty:.6f}"),
            model_version=self.version,
            data_sources=[
                {
                    "name": "BLS Public API CUSR0000SA0",
                    "url": "https://www.bls.gov/cpi/",
                    "fetched_at": self._fetched_at.isoformat() if self._fetched_at else None,
                    "latest_available_period": latest["bls_period"],
                    "note": "Settlement source matches Kalshi KXCPI (BLS). Forecast is climatology, not SPF consensus.",
                }
            ],
            factors=[
                f"Same-month climatology μ={mu:.3f}% σ={sigma:.3f}% for month={contract.month}",
                f"Contract: MoM {contract.op} {thr}% for {contract.year}-{contract.month:02d}",
                "UNVALIDATED baseline — not a professional nowcast",
            ],
            validation_evidence=(
                "UNVALIDATED: climatology prior only; no walk-forward edge vs market yet. "
                "Paper/research only."
            ),
            as_of=now,
            supported=True,
            details={
                "mu_mom": mu,
                "sigma_mom": sigma,
                "threshold": thr,
                "op": contract.op,
                "p_yes_high": p_yes_high,
                "category": "Economics",
            },
        )
