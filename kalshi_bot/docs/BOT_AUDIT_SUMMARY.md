# Bot Audit Summary

**Run ID:** `audit-20260916T204915Z`  
**Workspace commit:** `2a4ede86` (PR #1 branch `cursor/kalshi-trading-bot-5068`)  
**`origin/main`:** `c1646f21` (unrelated Autogen monorepo tip — Kalshi bot lives only on the PR branch)  
**Safety:** read-only exchange GETs + local DB reads; no orders, cancels, config changes, or retrains.

## What the bot actually does today

1. **Dashboard + scan loop (armed live in local `config.yaml`)** discovers Climate/Economics markets, scores them with registered models, and can paper-trade. **Live scan submits are blocked** unless `details.model_live_eligible is True` (no weather model currently sets this).
2. **Feeds collector** (tmux) ingests METAR/CLI/related feeds into `feeds.db` for research/multi paper.
3. **`weather-twc-bet --live`** is a **separate live path**: forecasts TWC `KXHIGH*` via station_v2 transfer, then calls `create_order_v2` with a dollar cap. It **does not** check `obs_engine_live_eligible` and **does not** use `RiskManager`.

## Live vs “blocked” discrepancy (resolved)

| Claim | Reality |
|--------|---------|
| Docs/`obs_engine_live_eligible: false` → weather live blocked | True for **scan/`ExecutionEngine`** |
| Purchases occurred | True via (a) **Sep 15 scan live** before gates existed, (b) **Sep 16 `weather-twc-bet --live`** bypass |

Account also has **NFL/MVE/UCL** fills **not** in `kalshi_bot.db` → other entry point or manual activity on the same Kalshi account.

## End-to-end locations (observation → probability → optional trade)

| Location | Trained station_v2 | TWC series | Scan live | TWC CLI live |
|----------|-------------------|------------|-----------|--------------|
| NYC Central Park | yes (feeds model) | KXHIGHNY | blocked now | used 2026-09-16 |
| CHI Midway | yes | KXHIGHCHI | blocked | used 2026-09-16 |
| LAX airport | yes | KXHIGHLAX | blocked | used 2026-09-16 |
| MIA/AUS/DEN/HOU | mapping only | some TWC verified | no model | no |

“Every available betting location” is **not** implemented end-to-end — only NYC/CHI/LAX have location models.

## Why losing / low-priced purchases happened

### A. 2026-09-15 — 1¢ YES longshots (scan path) — **confirmed losses / near-loss**

| Contract | Side | Model | Logged p / cons EV | Fill | Settlement | Outcome |
|----------|------|-------|--------------------|------|------------|---------|
| `KXHIGHNY-26SEP15-T70` (“high **&lt;70**”) | YES @ 1¢ ×458 | `weather.high_temp.v0.2-unvalidated` | p≈0.25, cons EV≈0.13 | live | **result=no** | **YES lost** (~premium+fees) |
| `KXHIGHLAX-26SEP15-T82` (“high **&gt;82**”) | YES @ 1¢ ×458 | same | p≈0.50, cons EV≈0.38 | live; later sold @1¢ | **result=no** | closed ≈flat minus fees |

**Timeline:** live enabled `19:30Z` → orders `19:32Z` → longshot guard `19:39Z` → model live gate `23:15Z`.  
**Class:** unvalidated NWS-proxy probabilities on TWC-settled markets + **risk controls not yet deployed** + EV formula making 1¢ look huge even with modest p.

### B. 2026-09-16 — TWC capped CLI (user-requested)

| Contract | Side | Logged median / p | Ask | Status at audit |
|----------|------|-------------------|-----|-----------------|
| `KXHIGHNY-26SEP16-B77.5` | NO | ~77°F / p≈57% | 29¢ | **closed; realized_pnl ≈ +$3.28** |
| `KXHIGHCHI-26SEP16-B75.5` | NO | ~76°F / p≈32% | 3¢ | **open** (~156 NO); weak weather story |
| `KXHIGHLAX-26SEP16-T80` | YES | ~81°F / p≈65% | 10¢ | **open** (~47 YES) |

CHI is the clearest “bought because cheap / max EV” case under the **then** selector. Later code raises min p to 55% and ranks confidence first — that would **reject CHI**; it does **not** rewrite history.

## Were purchases consistent with defensible probabilities / rules?

| Trade | Defensible at time? | Consistent with *intended* rules? |
|-------|---------------------|-----------------------------------|
| Sep15 NY/LAX 1¢ | **No** — model marked UNVALIDATED, `settlement_aligned=false`, “Paper only” | **No** — violated spirit; gates not yet in code |
| Sep16 NYC NO | Weakly — soft band vs median; p≈57% | Yes for then max-EV TWC path; RiskManager skipped |
| Sep16 CHI NO | **No** — p≈32% vs ~76°F forecast | Yes for then max-EV code; **fails** current filters |
| Sep16 LAX YES | Plausible weather story | Yes for then TWC path; RiskManager skipped |

## Accuracy / +EV evidence

- **Temperature holdout MAE** (station_v2) exists for NYC/CHI/LAX — this is **not** trading EV.
- **No** archived historical order books → cannot prove historical +EV after fees.
- Sep15 live longshots **lost** on settlement (NY) / closed near flat (LAX).
- Sep16 mixed: one closed profit (NYC cashout), two open — **not** a performance claim.

**Claim of demonstrated positive expected value after costs: not supported.**

## What must be corrected before results can be trusted

1. **Remove or hard-gate** `weather-twc-bet --live` behind the same eligibility + `RiskManager` as scan.  
2. Keep scan live blocked until promotion criteria met; never live-submit `settlement_aligned=false` models.  
3. Fix LAX downtown vs airport coordinates for any NWS-grid path.  
4. Fail closed if location calibration missing (no silent NYC calib).  
5. Reconcile Store positions to exchange settlements automatically.  
6. Train TWC labels before treating TWC transfer as validated.  
7. Separate this bot’s account activity from other strategies sharing the key.

## What remains unknowable

- Exact feature vectors at each TWC decision (not fully frozen in order details).  
- Official TWC daily max prints for open Sep16 markets (not settled at audit time).  
- Who placed NFL/MVE fills (not in bot Store).  
- True long-run calibration of TWC-transfer probabilities vs TWC settlement.

See `docs/BOT_AUDIT_REPORT.md` and `data/audit/audit-20260916T204915Z/`.
