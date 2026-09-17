# Capped live session (NYC / CHI / LAX)

**User-authorized 2026-09-18:** `$5` per bet, **`$20` session max** across all scheduled slots.

## Caps

| Knob | Value |
|------|------:|
| Per bet (`--dollars`) | 5 |
| Session ledger max | 20 |
| Config `trading.budget_dollars` | 20 |
| Allowlist | `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHLAX` |

Ledger file: `data/obs_engine/multi/session_budgets/weather-live-YYYY-MM-DD.json`

After ~4 fills the ledger refuses further reserves even if more timers fire.

## Enable

1. `models.weather.obs_engine_live_eligible: true` in local `config.yaml` (gitignored)
2. `mode: live` + `live.enabled: true`
3. CLI: `weather-twc-bet --series KXHIGHNY --live --dollars 5`

`weather-twc-bet` arms the sqlite store when config is live. Off-hour `force_decision` remains research-only and cannot submit.

## Caution

PROMOTION_CRITERIA are **not** fully met (TWC transfer / settlement mismatch risk). This path is an explicit capped risk session, not a claim of edge or promotion.
