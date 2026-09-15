"""Obs-driven NYC ProbabilityModel — research/paper only; blocked from live execution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.config import WeatherModelConfig
from kalshi_bot.data.store import Store
from kalshi_bot.models.base import Prediction, ProbabilityModel
from kalshi_bot.models.weather.archive import WeatherArchive
from kalshi_bot.models.weather.cli_reports import CliReportClient
from kalshi_bot.models.weather.distribution import PredictiveDistribution, from_empirical_residuals
from kalshi_bot.models.weather.nws_client import NWSClient, parse_market_date
from kalshi_bot.models.weather.obs_engine import ARTIFACT_NAME, MODEL_VERSION, NYC_TARGET
from kalshi_bot.models.weather.obs_engine.data import HourlyObs, default_data_dir
from kalshi_bot.models.weather.obs_engine.features import FEATURE_NAMES, build_features_at
from kalshi_bot.models.weather.same_day import ObservationClient
from kalshi_bot.models.weather.settlement_rules import interval_from_market
from kalshi_bot.models.weather.stations import station_for_ticker
from kalshi_bot.money import D, clamp01

logger = logging.getLogger(__name__)


class ObsDrivenNycForecaster(ProbabilityModel):
    """Independent observation-driven engine for NYC Central Park daily max.

    Always emits model_live_eligible=False and RESEARCH validation evidence until
    config.models.weather.obs_engine_live_eligible is explicitly true AND promotion metrics pass.
    """

    name = "weather_obs_nyc"
    version = MODEL_VERSION
    categories = ["Climate and Weather"]

    def __init__(
        self,
        config: WeatherModelConfig,
        store: Store | None = None,
        archive_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.archive = WeatherArchive(archive_path or getattr(config, "archive_path", None) or "data/weather_archive.db")
        self.obs_client = ObservationClient()
        self.cli = CliReportClient()
        self.nws = NWSClient()  # benchmark only
        self._models = None
        self._artifact = None
        self._load_artifact()

    def close(self) -> None:
        self.obs_client.close()
        self.cli.close()
        self.nws.close()

    def supports(self, market: dict[str, Any], category: str) -> bool:
        if not getattr(self.config, "obs_engine_enabled", True):
            return False
        ticker = (market.get("ticker") or "").upper()
        st = station_for_ticker(ticker)
        if st is None or st.city_key != "NYC":
            return False
        if "HOUR" in ticker:
            return False
        return any(ticker.startswith(p) for p in NYC_TARGET.series_prefixes)

    def _load_artifact(self) -> None:
        row = self.archive.load_artifact(NYC_TARGET.city_key, ARTIFACT_NAME)
        if not row:
            return
        import json

        self._artifact = json.loads(row["artifact_json"])
        path = self._artifact.get("model_path")
        if path and Path(path).exists():
            try:
                import joblib

                blob = joblib.load(path)
                self._models = blob.get("models")
            except Exception as exc:
                logger.warning("Could not load obs joblib: %s", exc)

    def _live_metar_series(self) -> list[HourlyObs]:
        """Convert recent aviationweather METAR into HourlyObs list (Central Park KNYC)."""
        import httpx

        url = f"https://aviationweather.gov/api/data/metar?ids={NYC_TARGET.metar_id}&format=json&hours=18"
        try:
            r = httpx.get(url, timeout=30.0, headers={"User-Agent": "KalshiBotObsEngine/0.1"}, follow_redirects=True)
            r.raise_for_status()
            rows = r.json()
        except Exception as exc:
            logger.warning("METAR fetch failed: %s", exc)
            return []
        out: list[HourlyObs] = []
        for row in rows or []:
            # obsTime epoch seconds; receiptTime is publication-ish
            ot = row.get("obsTime")
            if ot is None:
                continue
            valid = datetime.fromtimestamp(int(ot), tz=timezone.utc)
            temp_c = row.get("temp")
            dewp_c = row.get("dewp")
            tmpf = float(temp_c) * 9 / 5 + 32 if temp_c is not None else None
            dwpf = float(dewp_c) * 9 / 5 + 32 if dewp_c is not None else None

            def _num(key: str) -> float | None:
                v = row.get(key)
                if v is None or v == "":
                    return None
                if isinstance(v, str) and v.upper() in ("VRB", "M", "NULL"):
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            out.append(
                HourlyObs(
                    valid_utc=valid,
                    tmpf=tmpf,
                    dwpf=dwpf,
                    sknt=_num("wspd"),
                    drct=_num("wdir"),
                    alti=None,
                    p01i=None,
                    skyc1=(row.get("cover") or None),
                    station=NYC_TARGET.metar_id,
                    source="aviationweather_metar",
                )
            )
        out.sort(key=lambda o: o.valid_utc)
        return out

    def _cli_prelim_max(self, climate_day) -> tuple[int | None, str]:
        try:
            rep = self.cli.latest_for_day(NYC_TARGET.cli_location_id, climate_day)
        except Exception as exc:
            return None, f"CLI fetch failed: {exc}"
        if not rep or rep.max_temp_f is None:
            return None, "no CLI max yet"
        note = f"CLI {'prelim' if rep.is_preliminary else 'final'} max={rep.max_temp_f} issued={rep.issuance_time.isoformat()}"
        return int(rep.max_temp_f), note

    def _nws_benchmark(self, target_day) -> dict[str, Any] | None:
        try:
            fc = self.nws.daily_high_forecast(NYC_TARGET.lat, NYC_TARGET.lon, target_day)
            if fc and fc.get("temp_f") is not None:
                return {
                    "source": "nws_grid_daytime_benchmark_only",
                    "temp_f": float(fc["temp_f"]),
                    "fetched_at": fc.get("fetched_at"),
                    "start_time": fc.get("start_time"),
                }
        except Exception as exc:
            return {"error": str(exc)}
        return None

    def _remain_quantiles(self, features: list[float]) -> dict[str, float]:
        if self._models:
            return {k: float(self._models[k].predict([features])[0]) for k in ("q10", "q50", "q90")}
        # Fallback: climatological remain residual around 0 with wide spread
        return {"q10": -1.0, "q50": 1.0, "q90": 5.0}

    def _distribution_from_remain(
        self, max_so_far: float, quantiles: dict[str, float], residuals: list[float]
    ) -> PredictiveDistribution:
        # Enforce quantile order
        q10, q50, q90 = quantiles["q10"], quantiles["q50"], quantiles["q90"]
        q10, q50, q90 = sorted([q10, q50, q90])
        # Build mixture of point masses at max_so_far+q plus residual noise
        centers = [max_so_far + q10, max_so_far + q50, max_so_far + q90]
        weights = [0.25, 0.5, 0.25]
        samples = []
        resid = residuals or [0.0, -1.0, 1.0, 2.0, -2.0]
        for c, w in zip(centers, weights):
            n = max(5, int(40 * w))
            for i in range(n):
                samples.append(c + resid[i % len(resid)])
        # Also use empirical helper for variance floor
        # Represent as residuals relative to max_so_far+q50
        point = max_so_far + q50
        res = [s - point for s in samples]
        dist = from_empirical_residuals(point, res, method="obs_nyc_remain_quantiles")
        # Hard floor at observed max (physical constraint on eventual max, ignoring reporting noise for now)
        from kalshi_bot.models.weather.distribution import truncate_below

        return truncate_below(dist, max_so_far, reason="observed max_so_far physical floor")

    def predict(self, market: dict[str, Any], category: str) -> Prediction:
        now = datetime.now(timezone.utc)
        ticker = market.get("ticker") or ""
        target_day = parse_market_date(ticker)
        if target_day is None:
            return self._skip(ticker, now, "unparsed market date")
        interval = interval_from_market(market)
        if interval is None:
            return self._skip(ticker, now, "unmapped settlement interval")

        live_obs = self._live_metar_series()
        feats = build_features_at(live_obs, now, climate_day=target_day)
        factors = [
            f"Target: {NYC_TARGET.settlement_source} ({NYC_TARGET.display_name})",
            f"GHCND={NYC_TARGET.ghcnd_id}; METAR={NYC_TARGET.metar_id}; coords=({NYC_TARGET.lat},{NYC_TARGET.lon})",
            "Features: observations only (no third-party forecast inputs)",
        ]
        if feats is None:
            return self._skip(ticker, now, "insufficient same-day observations for features")

        quantiles = self._remain_quantiles(feats.values)
        residuals = (self._artifact or {}).get("remain_residuals") or [0.0, 1.0, -1.0, 2.0]
        dist = self._distribution_from_remain(feats.max_so_far, quantiles, residuals)

        # CLI prelim floor (reporting evidence)
        cli_max, cli_note = self._cli_prelim_max(target_day)
        if cli_max is not None:
            from kalshi_bot.models.weather.distribution import truncate_below

            dist = truncate_below(dist, float(cli_max), reason=cli_note)
            factors.append(cli_note)

        factors.append(
            f"max_so_far={feats.max_so_far:.1f}°F remain q10/50/90={quantiles['q10']:.1f}/{quantiles['q50']:.1f}/{quantiles['q90']:.1f}"
        )
        factors.append(
            f"dist mean={dist.mean():.1f} q10/50/90={dist.quantile(0.1)}/{dist.quantile(0.5)}/{dist.quantile(0.9)} method={dist.method}"
        )

        bench = self._nws_benchmark(target_day)
        if bench and "temp_f" in bench:
            factors.append(f"NWS grid daytime benchmark (not a feature): {bench['temp_f']:.1f}°F")

        p_yes = dist.p_interval(interval)
        # Conservative: use lower of p at q10-centered stress
        q10 = quantiles["q10"]
        dist_lo = self._distribution_from_remain(feats.max_so_far, {"q10": q10 - 1, "q50": q10, "q90": quantiles["q50"]}, residuals)
        if cli_max is not None:
            from kalshi_bot.models.weather.distribution import truncate_below

            dist_lo = truncate_below(dist_lo, float(cli_max), reason="cli floor")
        p_cons = min(p_yes, dist_lo.p_interval(interval))
        unc = max(abs(p_yes - p_cons), D("0.05"))

        cfg_live = bool(getattr(self.config, "obs_engine_live_eligible", False))
        live_eligible = False  # hard default; promotion gate must flip config AND evidence
        validation = (
            "RESEARCH/PAPER ONLY — observation-driven NYC engine. "
            "Not live_eligible. NWS forecasts used only as benchmarks. "
            f"Artifact loaded={self._artifact is not None}; joblib={self._models is not None}."
        )

        return Prediction(
            market_ticker=ticker,
            p_yes=clamp01(p_yes),
            p_yes_conservative=clamp01(p_cons),
            uncertainty=clamp01(unc),
            model_version=self.version,
            data_sources=[
                {"name": "aviationweather_metar", "station": NYC_TARGET.metar_id, "role": "features"},
                {"name": "ghcnd_label_train", "id": NYC_TARGET.ghcnd_id, "role": "training_labels"},
                {"name": "nws_cli", "role": "settlement_evidence_floor"},
                {"name": "nws_grid", "role": "benchmark_only", "payload": bench},
            ],
            factors=factors,
            validation_evidence=validation,
            as_of=now,
            supported=True,
            details={
                "city": "NYC",
                "target_day": target_day.isoformat(),
                "settlement_aligned": True,
                "distribution": dist.as_dict(),
                "features": dict(zip(FEATURE_NAMES, feats.values)),
                "feature_provenance": feats.provenance,
                "remain_quantiles": quantiles,
                "nws_benchmark": bench,
                "model_live_eligible": live_eligible,
                "config_obs_engine_live_eligible": cfg_live,
                "engine": "obs_driven",
                "p_yes_high": str(p_yes),
            },
        )

    def _skip(self, ticker: str, now: datetime, reason: str) -> Prediction:
        return Prediction(
            market_ticker=ticker,
            p_yes=D("0.5"),
            p_yes_conservative=D("0.5"),
            uncertainty=D("0.5"),
            model_version=self.version,
            data_sources=[],
            factors=[],
            validation_evidence="RESEARCH skipped",
            as_of=now,
            supported=False,
            skip_reason=reason,
            details={"model_live_eligible": False, "engine": "obs_driven"},
        )
