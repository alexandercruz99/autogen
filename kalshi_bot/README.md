# Kalshi Research & Trading Bot

Evidence-based Kalshi market scanner, probability models, EV/risk checks, paper trading, combo RFQ support, and a gated live mode.

**Default mode: paper trading.** Live trading never auto-enables. Choosing **no trade** is a first-class outcome. This software does not promise profits.

## What works now (milestone status)

| Milestone | Status |
| --- | --- |
| Real market discovery + persistent storage | Implemented (Kalshi public Trade API) |
| One category model → conservative EV | **Weather daily highs** via NWS (`api.weather.gov`) — labeled **unvalidated** |
| Paper trading + risk + accounting | Implemented |
| Combo discovery + joint/RFQ handling | Implemented (quotes required; no invented combo prices) |
| Dashboard + autonomous loop + kill switch | Implemented |
| Live trading (explicit enable) | Implemented behind one-time dashboard gate + API keys |

### Honest limitations

- The weather forecast-error σ is a **configurable prior**, not a walk-forward calibrated edge. Do not treat paper fills as proof of live profitability.
- Sports / player props / economics models are **not implemented** — markets in those categories are listed as skipped with a reason.
- Combo joint probabilities for dependent legs (same game, related props) are **skipped** unless a validated dependence model is supplied. Independence across distinct weather cities is optional and **off by default**.
- Demo environment credentials are separate from production. Demo fills ≠ live edge.

## Docs verified for this build

- API: https://docs.kalshi.com/
- Environments: production `https://external-api.kalshi.com/trade-api/v2`, demo `https://external-api.demo.kalshi.co/trade-api/v2`
- Auth: RSA-PSS SHA256 headers (`KALSHI-ACCESS-*`)
- Orderbook: bid-only; YES ask = `1 − best NO bid`
- Fees: `fees ≈ M × 0.07 × C × P × (1−P)` taker model + rounding (see fee schedule)
- Orders V2: `POST /portfolio/events/orders` (YES-book `bid`/`ask`)
- Combos: MVE collections + RFQ lifecycle; settlement = product of leg values (DNP can be scalar, not a refund)

## Quick start (local)

```bash
cd kalshi_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
# Optional for live/demo auth — set paths locally, never paste private keys into chat:
#   api.api_key_id / api.private_key_path  or  KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH

# One scan: real Kalshi markets → weather model → EV → risk → paper orders
kalshi-bot --config config.yaml scan

# Dashboard (http://127.0.0.1:8787) + optional loop
kalshi-bot --config config.yaml run
```

### Where it runs

The bot runs on **the machine you start it on** (your laptop or a VPS you choose). That host must stay powered on and connected for unattended operation. This project does not purchase hosting for you.

Optional hosting path (your approval required before paid spend): any Linux VPS with Python 3.11+, systemd unit running `kalshi-bot run`, firewall allowing only your IP to the dashboard port, secrets via file permissions (`chmod 600` on the `.key`).

## Operating modes

1. **Research** — scan and evaluate only; no orders.
2. **Paper** — simulate fills, track positions/PnL (default).
3. **Live** — place real orders only after dashboard **Enable live trading** (budget + risk limits + typed confirmation + successful validation scan + API credentials). After enablement, qualifying buys do **not** ask for per-trade approval.

Deposits never auto-increase the configured **trading budget**.

## Architecture

```
kalshi_bot/
  src/kalshi_bot/
    api/           # Kalshi client, orderbook, fees
    discovery/     # market scan
    models/weather # NWS high-temp model (pluggable registry)
    ev/            # individual + combo EV
    risk/          # hard limits (no martingale; Kelly deferred)
    execution/     # paper/live orders + RFQ
    combo/         # MVE candidate search
    bot/           # pipeline + loop
    dashboard/     # FastAPI UI
    validation/    # Brier/log-loss/leakage helpers
  config.example.yaml
  tests/
```

## Data sources

| Source | Use | Cost |
| --- | --- | --- |
| Kalshi Trade API | Markets, books, orders, RFQs | Kalshi account; API tier limits apply |
| NWS `api.weather.gov` | Daily high forecasts for weather model | Free (requires User-Agent) |
| Kalshi weather index endpoints | Optional cross-check | Free with API |

Paid sports/odds feeds are **not** wired in. Do not invent substitutes for live decisions.

## Tests

```bash
pytest -q
```

Coverage includes fees, orderbook ask derivation, EV/combo settlement (incl. DNP product), risk limits, duplicate opportunity orders, leakage guards, and contract parsing.

## Configuration essentials

See `config.example.yaml`. Important knobs:

- `trading.budget_dollars`, per-trade / event / portfolio / combo / daily / drawdown limits
- `trading.min_net_edge`, `uncertainty_buffer`
- `models.weather.forecast_error_sigma_f` (prior — calibrate before trusting live)
- `combos.allow_independence_assumption` (default `false`)

## Security

- Store RSA private keys on disk; reference by path only.
- Logs redact key material (path logged, not PEM contents).
- `.gitignore` excludes `config.yaml`, `*.key`, `.env`, SQLite data.
