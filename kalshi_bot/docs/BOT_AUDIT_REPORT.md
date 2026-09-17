# Bot Audit Report

> **Corrections addendum (2026-09-16):** Implementation fixes and corrected trade reconciliation are in  
> [`docs/BOT_AUDIT_ADDENDUM.md`](BOT_AUDIT_ADDENDUM.md) and [`docs/TRADE_RECONCILIATION_CORRECTED.md`](TRADE_RECONCILIATION_CORRECTED.md).  
> This report is preserved as the original audit narrative; several F-findings below are **superseded** by the addendum (notably F2 live bypass, F5 NO prices, F6 settlements, F7 LAX PnL).

**Run ID:** `audit-20260916T204915Z`  
**Generated (UTC):** 2026-09-16 (audit session)  
**Repository:** `alexandercruz99/autogen`  
**PR under review:** [#1](https://github.com/alexandercruz99/autogen/pull/1) branch `cursor/kalshi-trading-bot-5068` @ `2a4ede86`  
**Base `main`:** `c1646f21` (does not contain `kalshi_bot/`; bot exists on the PR branch)  

**Safety statement:** This audit used read-only filesystem inspection, read-only SQLite (`mode=ro`), and Kalshi **GET** portfolio/market endpoints only. No orders were created or cancelled, live settings were not changed, positions were not modified, and models were not retrained.

Companion artifacts: `data/audit/audit-20260916T204915Z/{manifest,findings,model_inventory,trade_audit.csv,orders_local,positions_local,exchange_readonly,pytest_focused}.json|txt|csv`.

---

## 1. What is actually running

### 1.1 Commits

| Ref | SHA | Role |
|-----|-----|------|
| Workspace / PR #1 HEAD | `2a4ede86` | Code + config under audit |
| `origin/main` | `c1646f21` | Upstream Autogen; **no kalshi_bot** |
| Deployed commit | **Unknown outside this VM** | No separate deploy manifest found |

### 1.2 Processes observed on this VM

| Process | Evidence |
|---------|----------|
| Dashboard | `tmux` session `kalshi-dashboard`: `python3 -m kalshi_bot.cli --config config.yaml dashboard` (port 8787) |
| Feeds collector | `tmux` session `weather-feeds-collector` |
| Continuous `run` scan loop | Not confirmed running at audit time (historical terminal starts failed/stopped) |

### 1.3 Configuration (redacted)

Local `config.yaml` (gitignored; values observed, secrets not copied):

- `mode: live`
- `live.enabled: true`
- `models.weather.obs_engine_live_eligible: false`
- Budget / per-trade caps: `$25` / `$5`
- `uncertainty_buffer: 0.05`, `unvalidated_longshot_max_price: 0.05`, `unvalidated_market_shrink: 0.65`

Example config defaults to `mode: paper` — **runtime config differs**.

### 1.4 Databases / artifacts

| Store | Path | Role |
|-------|------|------|
| Trading Store | `data/kalshi_bot.db` | orders, positions, opportunities, audit_log |
| FeedStore | `data/obs_engine/feeds/feeds.db` | samples, features, paper_decisions |
| Weather archive | `data/weather_archive.db` | CLI/forecast snapshots, AI artifacts |
| station_v2 models | `feeds/models/`, `multi/artifacts/{nyc,chi,lax}_*/` | joblib + calibration JSON |

### 1.5 Code present vs tested vs operated vs live-ordering

| Layer | Present | Tested | Operated here | Can place live orders |
|-------|---------|--------|---------------|------------------------|
| `TradingPipeline` scan | yes | yes | historically Sep15; dashboard available | only if `model_live_eligible is True` (**none today**) |
| `ExecutionEngine.create_order_v2` | yes | gate tests | Sep15 live fills | gated now |
| `weather-twc-bet` / `place_capped_live_bet` | yes | rationale tests; submit untested | Sep16 | **yes if mode+live.enabled** |
| Multi/feeds paper | yes | yes | yes | no |
| Combo live RFQ | yes | FSM only | not wired to scan | code exists, unused |
| CPI model | yes | unit | paper path | not live-eligible |

**Only two call sites invoke `create_order_v2`:** `execution/engine.py` and `multi/live_bet.py`.

---

## 2. System map

```text
                    ┌─────────────────────────────────────────┐
                    │ Entry points                            │
                    │  CLI scan/run | dashboard | weather-*    │
                    └───────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   MarketScanner              weather-discover          weather-feeds-run
   (bot/pipeline)             (multi/discovery)         (feeds/worker)
          │                         │                         │
          ▼                         ▼                         ▼
   ModelRegistry.resolve      LocationRegistry          FeedStore samples
          │                   VERIFIED_NWS / TWC              │
          ▼                         │                         ▼
   high_temp / ai_cli /       process_location          features_live /
   obs_nyc / cpi              (NWS or TWC branch)       features_twc
          │                         │                         │
          ▼                         ▼                         ▼
   PredictiveDistribution     predict_station_v2        infer_paper /
   + settlement_rules         + hour residual calib     PaperLedger
          │                         │
          ▼                         ▼
   ev.calculator              _paper_evaluate /
   (fees, shrink,             bet_rationale.select_*
    longshot guard)                 │
          │                         ├── paper only (multi)
          ▼                         └── weather-twc-bet --live
   RiskManager ──► ExecutionEngine        │
          │              │                 ▼
          │              └──── gated ──► create_order_v2
          │                                │
          └──── TWC path SKIPS RiskManager ┘
                                    │
                                    ▼
                          Store orders/positions
                                    │
                          SettlementReconciler (scan)
```

### Step → file:function

| Step | Primary implementation |
|------|------------------------|
| Market discovery (scan) | `markets/scanner.py` / `TradingPipeline.run_scan_once` |
| Weather discovery (multi) | `multi/discovery.py::discover_weather_markets` |
| Settlement interval | `settlement_rules.py::interval_from_market` |
| Data collection | `feeds/metar.py`, `feeds/cli_feed.py`, `feeds/twc_kalshi.py`, `nws_client.py` |
| Features | `feeds/features_live.py`, `feeds/features_twc.py`, `obs_engine/features.py` |
| Prediction (scan) | `high_temp.py`, `ai_forecaster.py`, `obs_engine/forecaster.py`, `economics/cpi.py` |
| Prediction (multi) | `multi/predict.py::predict_station_v2` |
| Calibration | `feeds/calibration.py` + per-location `*_calibration.json` |
| Probabilities | `distribution.py::{from_empirical_residuals,from_normal,p_interval,truncate_below}` |
| Pricing/fees | `api/orderbook.py`, `api/fees.py::estimate_net_fee`, `ev/calculator.py` |
| Selection (TWC live) | `multi/bet_rationale.py::select_forecast_consistent` |
| Risk | `risk/manager.py` via `execution/engine.py` (**not** TWC live) |
| Order submit | `KalshiClient.create_order_v2` |
| Accounting | `data/store.py` OrderRecord/PositionRecord |
| Settlement | `settlement/reconciler.py` (invoked from scan; **Sep15 local positions still open**) |

### Gaps (implemented ≠ connected ≠ demonstrated ≠ validated)

- Multi-location “all cities” **mapped** in registry; **trained** only NYC/CHI/LAX.
- Combo live RFQ **implemented**, **not connected** to scan.
- Paper settlement methods **tested**, continuous reconcile of live Store vs exchange **not demonstrated** for finalized Sep15 markets.
- TWC transfer **demonstrated live**, **not validated** on TWC labels.

---

## 3. Prediction model inventory

Full machine-readable inventory: `data/audit/.../model_inventory.json`.

### 3.1 Scan registry (order matters)

1. **`ObsDrivenNycForecaster`** (`obs_engine/forecaster.py`) — NYC HIGHNY; GBM remaining-rise quantiles; `model_live_eligible` forced false.  
2. **`AIWeatherForecaster`** (`ai_forecaster.py`) — multi-city CLI-aligned; empirical residuals or unfitted Normal(μ, 4.5).  
3. **`WeatherHighTempModel`** (`high_temp.py`) — Gaussian around **NWS grid** daytime high; σ from config (default 3°F); continuity correction; **explicitly not settlement-aligned** to TWC.  
4. **`CpiMomModel`** (`economics/cpi.py`) — Gaussian month-of-year climatology from BLS.

### 3.2 Multi / feeds station_v2 (observation path)

**Mathematics (plain language):**

1. Build features from METAR (temp, dewpoint, wind, pressure, sky, solar elevation, max-so-far, recent changes, …).  
2. Gradient boosting models predict **remaining rise** quantiles q10/q50/q90:  
   \(\widehat{\text{high}} = \max(\text{max\_so\_far} + q_{50},\ \text{max\_so\_far})\).  
3. On a calibration set, store residuals \(r = \text{true high} - \widehat{\text{high}}\) by decision hour (08/11/14 local).  
4. Build a discrete °F distribution by shifting those residuals onto today’s point (`from_empirical_residuals`).  
5. Optionally truncate below observed/CLI floor (`truncate_below`).  
6. Contract probability = sum of PMF mass on temperatures that make YES true (`p_interval`).

**Holdout production median MAE (°F)** from train reports (temperature skill, **not** trading EV):

| Location | 08h | 11h | 14h |
|----------|-----|-----|-----|
| NYC | 3.53 | 2.44 | 1.11 |
| CHI Midway | 3.75 | 2.45 | 1.04 |
| LAX | 2.70 | 1.24 | 0.54 |

### 3.3 Arbitrary constants / overconfidence risks

| Constant | Where | Risk |
|----------|-------|------|
| σ=3°F / stress×1.5 | `high_temp` | Unvalidated; drove Sep15 1¢ EVs |
| σ=4.5 unfitted | `ai_forecaster` | Wide prior, not TWC-calibrated |
| uncertainty_buffer 0.05 | config | Haircut, not calibration |
| unvalidated_market_shrink 0.65 | EV | Added **after** Sep15 longshots |
| Hardcoded remain quantiles if joblib missing | `obs` forecaster | Fabricated fallback |
| NYC calib default if path None | `predict_station_v2` | Silent wrong-location calib |

A **point forecast is not a validated probability**. Sep15 opportunities logged Gaussian p values from an UNVALIDATED proxy.

---

## 4. Settlement and data alignment

### 4.1 KXHIGH* vs HIGH*

Kalshi `rules_primary` on audited markets names **The Weather Company** stations (CLINYC, CLILAX, …).  
NWS CLI and TWC are **different products**. Code now splits:

- `HIGHNY` → NWS CLI / station_v2 NYC  
- `KXHIGHNY` → TWC path (`VERIFIED_TWC_DAILY_MAX`)

### 4.2 Location checks

| Issue | Severity | Evidence |
|-------|----------|----------|
| LAX downtown lat/lon in `stations.py` / `config.yaml` vs KLAX/CLILAX | high | 34.05,-118.24 vs 33.94,-118.41 |
| CHI Midway vs O'Hare | OK | MDW/KMDW consistent; ORD neighbor only |
| TWC uses GHCND labels for training | medium | exploratory transfer |
| Civil vs LST climate day | addressed in feeds fixes | see pipeline LST helpers |
| first_seen vs valid_utc | previously buggy; TWC METAR fix committed | still require vigilance on backfills |

### 4.3 Coverage claim vs reality

Discovery may list dozens of series; **unique trained operating locations** for daily max observation models: **3** (NYC, CHI, LAX). Aliases must not be counted as extra locations.

---

## 5. Reconstructed purchases

See `trade_audit.csv`. Period: bot Store live from **2026-09-15**; exchange fills also show earlier non-bot activity.

### 5.1 Bot-attributed live weather trades

#### Trade L1 — `KXHIGHNY-26SEP15-T70` YES @ 1¢

| Field | Value |
|-------|-------|
| Entry | `TradingPipeline` → `ExecutionEngine` |
| Exchange order | `01a0a68e-9408-7175-9ba3-3b82fa8a88a0` |
| Model | `weather.high_temp.v0.2-unvalidated` |
| Inputs logged | NWS grid μ=72°F, σ=3; contract lt 70; TWC settlement in rules |
| estimated_prob / conservative_prob | 0.2525 / 0.1725 |
| conservative_ev | ~0.132 |
| Qty filled | 458.24 @ 0.01 |
| Official result | **no** (YES loses) |
| Classification | **Confirmed loss** — poor/unvalidated probability + pre-guard live |
| Reproducible | Partial (opp logged; grid snapshot not frozen byte-for-byte) |

#### Trade L2 — `KXHIGHLAX-26SEP15-T82` YES @ 1¢

| Field | Value |
|-------|-------|
| Model / μ | high_temp v0.2; μ=82°F → p_yes≈0.50 at boundary |
| Result | **no** |
| Exit | Exchange sell fill same day @ 0.01 |
| Classification | Closed near flat minus fees (best-effort from fills) |
| Extra defect | Downtown NWS grid vs CLILAX settlement |

#### Trade T1 — `KXHIGHNY-26SEP16-B77.5` NO @ 29¢ (TWC CLI)

| Field | Value |
|-------|-------|
| Entry | `weather-twc-bet --live` (`user_requested: true`) |
| Median / p_no | ~77.1°F / ~0.57 |
| Model origin | `same_icao_transfer:nyc_feeds_default` |
| Exchange realized_pnl | **+$3.284** (position flat at audit) |
| Classification | Closed profit — **does not prove** process quality |
| RiskManager | **bypassed** |

#### Trade T2 — `KXHIGHCHI-26SEP16-B75.5` NO @ 3¢

| Field | Value |
|-------|-------|
| Median / p_no | ~76.0°F / ~0.323 |
| Ask | 0.03 → max-EV lottery shape |
| Position at audit | ≈ −156 (NO), exposure ~$4.68, unsettled |
| Classification | Open; **not defensible** under data-first / high-confidence policy |
| Current code | would reject (min p 0.55) |

#### Trade T3 — `KXHIGHLAX-26SEP16-T80` YES @ 10¢

| Field | Value |
|-------|-------|
| Median / p_yes | ~81.2°F / ~0.646 |
| Origin | `same_icao_transfer:lax_airport` |
| Position | ≈ +47 YES, open |
| Classification | Open; weather-aligned relative to CHI; TWC label transfer still exploratory |

### 5.2 Non-bot account activity

`non_bot_exchange_orders.json` lists NFL, multivariate combo, and UCL orders **absent** from bot Store live orders. **Attribution to this bot is not supported.**

---

## 6. Why low-priced contracts were selected

### EV definition in code (`ev/calculator.py`)

For a $1 binary (hold to settlement):

\[
\mathrm{EV}_{side} = P(side) - P_{\text{ask}} - \frac{\mathrm{fees}}{qty}
\]

Conservative qualification subtracts `uncertainty_buffer` (and uses conservative/shrunk p).

At **price = 0.01**, even \(P=0.17\) yields large positive conservative EV after fees — the **favorite–longshot numerical trap**.

### Sep15

- Selector maximized qualifying conservative EV.  
- `unvalidated_longshot_max_price` guard **did not exist yet** (landed 7 minutes later).  
- `model_live_eligible` execution gate **did not exist yet** (landed ~3.7h later).  
- Opportunity itself said `live_eligible: false` / “Paper only” — **not enforced**.

### Sep16 TWC

- Selector was **max EV among +EV sides** (before confidence ranking).  
- CHI 3¢ with p≈32% won on EV arithmetic.  
- Dollar cap sized qty ≈ `$5 / ask` → **more contracts at lower prices**, amplifying lottery shape.  
- No `RiskManager` correlation/budget checks on this path.

Win rate, payout size, calibration, and net returns are **separate**. Cheap tickets can show high EV while destroying wealth when probabilities are wrong.

---

## 7. Execution, risk, accounting

### Live routes

1. Scan → `place_individual(..., model_live_eligible=)` → block unless True → `create_order_v2`.  
2. `place_capped_live_bet` → balance + mode checks → `create_order_v2` (**no RiskManager**, **no obs flag**).  
3. Combo `live_rfq_buy` — not called from scan.

### Risk controls present on scan path

Budget, per-trade max loss, event/portfolio exposure, daily loss, drawdown — via `RiskManager.check_purchase`.

### Failures observed

- Sep15: live without eligibility.  
- Sep16 TWC: RiskManager bypass; correlated city-day exposure not enforced beyond informal $5 cap.  
- Local Sep15 positions remain `open` after `finalized` markets → reconciler gap in this environment.  
- Duplicate paper client_order_id skips seen in TWC reports (paper ledger), separate from live.

### Combinations

`require_joint_model: true`, `allow_independence_assumption: false` — independence multiply not used when disallowed. Live combo path unused.

---

## 8. Validation claims / regressions

### Tests run this audit

```text
pytest tests/test_live_eligibility_gate.py \
       tests/test_bet_rationale.py \
       tests/test_multi_location.py \
       tests/test_fees.py
→ 32 passed
```

### Earlier risks — status now

| Risk | Status |
|------|--------|
| Non-NYC model silent NYC **joblib** transfer | Blocked in NWS multi pipeline |
| NYC **calibration** if calib_path None | **Still a concern** |
| Stale coverage / first_seen | Partially fixed; not fully re-proven here |
| YES/NO overwrite | Not re-hit in this audit |
| Paper settlement unscheduled | Methods exist; live Store reconcile incomplete |
| Discovery counting aliases | Still a documentation risk |

### Evaluator vs production

station_v2 training reports call the same `predict_station_v2` family used in multi paper. Scan path uses **different** models (`high_temp`/`ai`/`obs_nyc`). **Agreeing helpers inside one stack ≠ cross-stack parity.**

---

## 9. Predictive and trading performance

### Temperature skill (holdout)

See §3.2 MAE table — limited independent days (~257 test days per CHI/LAX report).

### Trading

| Slice | Evidence |
|-------|----------|
| Sep15 live longshots | NY loss; LAX near-flat exit |
| Sep16 TWC | 1 closed +PnL; 2 open |
| By price ≤5¢ | Dominated by unvalidated/max-EV behavior |
| Historical +EV after costs | **Not demonstrated** (no executable book archive) |
| Baselines | NWS grid / climatology used as model guts, not as honest benchmark ledger |

**No claim of predictive improvement that implies tradable edge after costs is supported by this audit.**

---

## 10. Findings (severity)

| ID | Severity | Status |
|----|----------|--------|
| F1 Sep15 live longshots pre-guards | critical | confirmed |
| F2 TWC live bypasses obs flag + RiskManager | critical | confirmed |
| F3 CHI cheap max-EV selection | high | confirmed at trade time |
| F4 Docs “live blocked” vs reality | high | confirmed |
| F5 Store not reconciled after Sep15 finalize | medium | confirmed here |
| F6 LAX downtown vs airport | high | supported |
| F7 NYC calib fallback | medium | supported |
| F8 Non-bot account fills | info | confirmed |
| F9 No demonstrated +EV after costs | high | supported |
| F10 TWC transfer exploratory | medium | confirmed limitation |

Each finding’s acceptance test is listed in `findings.json`.

### Correction priority (by financial / trust impact)

1. Gate or disable uncapped-bypass live CLI; unify on RiskManager + eligibility.  
2. Never live-trade `settlement_aligned=false` / UNVALIDATED models.  
3. Keep longshot + confidence filters; do not revert to max-EV-only.  
4. Fix LAX coordinates / series mapping.  
5. Fail closed on missing location calibration.  
6. Automate settlement reconcile; fix Store drift.  
7. Isolate Kalshi API key from other strategies or tag all orders.

---

## Final answers

1. **What does the bot do today?** Research/paper weather+CPI scanning, feeds collection, multi TWC/NWS forecasting; local config arms live; scan live blocked by model flag; **TWC CLI can still live-trade**.  
2. **Which locations work end to end?** NYC/CHI/LAX observation models + TWC forecast path; not all discovered cities.  
3. **Why losing purchases?** Sep15: unvalidated Gaussian proxy + 1¢ EV trap before guards. Sep16 CHI: max-EV cheap NO with low p.  
4. **Defensible / rules-consistent?** Sep15 **no**. Sep16 CHI **no** (data). Sep16 NY/LA **arguable** under then TWC rules; RiskManager skipped.  
5. **Accuracy evidence?** Temperature MAE only; **not** trading calibration proof.  
6. **Demonstrated +EV after costs?** **No.**  
7. **Must correct before trust?** Live path unification, settlement alignment, reconcile, TWC label training, account isolation.  
8. **Unknowable?** Frozen feature vectors, unsettled Sep16 outcomes, non-bot order provenance, deploy outside this VM.

---

*End of report. Do not interpret any closed profitable trade as validation of the overall system.*
