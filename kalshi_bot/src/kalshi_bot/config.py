from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


Mode = Literal["research", "paper", "live"]
Environment = Literal["production", "demo"]


class ApiConfig(BaseModel):
    base_url: str = "https://external-api.kalshi.com/trade-api/v2"
    private_key_path: str = ""
    api_key_id: str = ""
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_tokens_per_second: float = 8.0


class StorageConfig(BaseModel):
    sqlite_path: str = "data/kalshi_bot.db"


class ScanConfig(BaseModel):
    categories: list[str] = Field(default_factory=lambda: ["Climate and Weather"])
    series_tickers: list[str] = Field(default_factory=list)
    max_markets_per_scan: int = 200
    scan_interval_seconds: int = 120
    orderbook_depth_levels: int = 10
    max_data_age_seconds: int = 900


class TradingConfig(BaseModel):
    budget_dollars: Decimal = Decimal("100.00")
    min_cash_reserve_dollars: Decimal = Decimal("25.00")
    max_loss_per_trade_dollars: Decimal = Decimal("10.00")
    max_event_exposure_dollars: Decimal = Decimal("25.00")
    max_portfolio_exposure_dollars: Decimal = Decimal("75.00")
    max_combo_exposure_dollars: Decimal = Decimal("20.00")
    max_daily_loss_dollars: Decimal = Decimal("20.00")
    max_drawdown_dollars: Decimal = Decimal("40.00")
    min_net_edge: Decimal = Decimal("0.05")
    uncertainty_buffer: Decimal = Decimal("0.03")
    max_model_market_divergence: Decimal = Decimal("0.25")
    default_contract_quantity: Decimal = Decimal("1.00")
    max_contracts_per_order: Decimal = Decimal("5.00")
    # Target capital per individual trade (price×qty + fees), capped by max_loss_per_trade.
    target_trade_dollars: Decimal = Decimal("0")
    hold_to_settlement: bool = True
    fee_multiplier: Decimal = Decimal("1.0")
    assume_taker: bool = True
    balance_precision: Decimal = Decimal("0.0001")

    @field_validator(
        "budget_dollars",
        "min_cash_reserve_dollars",
        "max_loss_per_trade_dollars",
        "max_event_exposure_dollars",
        "max_portfolio_exposure_dollars",
        "max_combo_exposure_dollars",
        "max_daily_loss_dollars",
        "max_drawdown_dollars",
        "min_net_edge",
        "uncertainty_buffer",
        "max_model_market_divergence",
        "default_contract_quantity",
        "max_contracts_per_order",
        "target_trade_dollars",
        "fee_multiplier",
        "balance_precision",
        mode="before",
    )
    @classmethod
    def _to_decimal(cls, v: Any) -> Decimal:
        return Decimal(str(v))


class WeatherCityConfig(BaseModel):
    series_prefixes: list[str]
    lat: float
    lon: float
    station_note: str = ""


class WeatherModelConfig(BaseModel):
    enabled: bool = True
    forecast_error_sigma_f: float = 3.0
    min_sigma_f: float = 2.0
    max_forecast_age_hours: float = 12.0
    cities: dict[str, WeatherCityConfig] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    weather: WeatherModelConfig = Field(default_factory=WeatherModelConfig)
    economics_cpi_enabled: bool = True


class CombosConfig(BaseModel):
    enabled: bool = True
    max_legs: int = 3
    max_candidates_per_scan: int = 20
    require_joint_model: bool = True
    allow_independence_assumption: bool = False
    rfq_poll_seconds: float = 2.0
    rfq_wait_seconds: float = 15.0
    # Labeled paper simulator only — never evidence of live profitability.
    paper_fixture_yes_price: Decimal | None = None

    @field_validator("paper_fixture_yes_price", mode="before")
    @classmethod
    def _opt_dec(cls, v: Any) -> Decimal | None:
        if v is None or v == "":
            return None
        return Decimal(str(v))


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8787
    # HTTP basic auth for remote/phone access. Empty password = auth disabled (local only).
    username: str = "kalshi"
    password: str = ""


class AlertsConfig(BaseModel):
    webhook_url: str = ""


class LiveConfig(BaseModel):
    enabled: bool = False
    require_budget_and_limits: bool = True


class AppConfig(BaseModel):
    mode: Mode = "paper"
    environment: Environment = "production"
    api: ApiConfig = Field(default_factory=ApiConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    combos: CombosConfig = Field(default_factory=CombosConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)


class EnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_base_url: str = ""
    kalshi_environment: str = ""
    kalshi_mode: str = ""
    kalshi_config: str = "config.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    env = EnvOverrides()
    cfg_path = Path(path or env.kalshi_config or "config.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open() as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be a mapping: {cfg_path}")
            raw = loaded
    elif Path("config.example.yaml").exists() and not cfg_path.exists():
        with Path("config.example.yaml").open() as f:
            raw = yaml.safe_load(f) or {}

    cfg = AppConfig.model_validate(raw)

    if env.kalshi_api_key_id:
        cfg.api.api_key_id = env.kalshi_api_key_id
    if env.kalshi_private_key_path:
        cfg.api.private_key_path = env.kalshi_private_key_path
    if env.kalshi_base_url:
        cfg.api.base_url = env.kalshi_base_url
    if env.kalshi_environment in ("production", "demo"):
        cfg.environment = env.kalshi_environment  # type: ignore[assignment]
        if env.kalshi_environment == "demo" and not env.kalshi_base_url:
            cfg.api.base_url = "https://external-api.demo.kalshi.co/trade-api/v2"
    if env.kalshi_mode in ("research", "paper", "live"):
        cfg.mode = env.kalshi_mode  # type: ignore[assignment]

    # Safety: never start in live from config alone without explicit live.enabled.
    if cfg.mode == "live" and not cfg.live.enabled:
        cfg.mode = "paper"

    return cfg


def project_root() -> Path:
    return Path(os.getcwd())
