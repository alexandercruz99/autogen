# Audit corrections addendum

**Original audit package (preserved):** `data/audit/audit-20260916T204915Z/`  
**This addendum:** corrections implemented on branch `cursor/audit-corrections-5068`  
**Generated (UTC):** 2026-09-16  

No original snapshot files were overwritten. Trade PnL numbers below are **historical fixture expectations**, not claims about the current account.

---

## Finding status after corrections

| ID | Original finding | Correction status |
|----|------------------|-------------------|
| F2 | `weather-twc-bet --live` called `create_order_v2` bypassing RiskManager / eligibility | **Fixed.** `place_capped_live_bet` routes only through `ExecutionEngine.place_individual`. Requires `model_live_eligible is True`, armed live mode, RiskManager, EV requalify at refreshed ask. Dollar cap / `user_requested` cannot bypass. `force_decision` is research-only. |
| F3 | Ask refresh resized qty but did not requalify EV | **Fixed.** Sequence: immutable prediction → refresh book → fees → conservative EV → risk → reserve → submit. Fixture: p=0.60, ask 0.40→0.90 → reject, **0** `create_order_v2` calls. |
| F4 | Order intents / ambiguous submits | **Improved.** Persistent `order_intents` + pending `orders` before submit; ambiguous status on network failure; `reconcile_live_order` before duplicate create. |
| F5 | NO fills stored with YES-book average (0.71 / 0.97) | **Fixed.** `accounting/fills.py` (`fill_price_norm.v1`) + engine uses `outcome_side` / `no_price_dollars`. NYC NO cost 0.29; CHI NO cost 0.03. |
| F6 | Settlement retrieval `'KalshiClient' object has no attribute '_request'` | **Fixed.** `KalshiClient.get_settlements` / `iter_settlements` via public `get()`. Reconciler uses portfolio settlements + market result; checkpoint persisted; settlement applied once. Hooked from scan + `reconcile` CLI. |
| F7 | Audit LAX loss −$4.90 incorrect; NYC profit fee treatment unclear; evaluator parity overstated | **Corrected in addendum** (below). Original CSV preserved. |
| F8 | Preference min-p labeled as “no edge” / marketing growth claims | **Fixed.** Reject classes: `preference_min_probability`, `inadequate_ev`, `unsupported_probability`, `weather_alignment`, `liquidity_or_price`. Rationale states uncertainty; no “steady growth”. |
| F9 | NYC calibration silent fallback | **Fixed (prior commit).** Missing location calib → `probabilities_unavailable`. |
| F10 | Shared predict path | **Already shared** via `predict_from_rows` / `predict_station_v2` in train eval. |
| F11 | LAX downtown coords in runtime config | **Fixed.** `config.yaml` LAX → KLAX 33.9425, −118.4081 (stations/registry already correct). |

---

## Corrected trade reconciliation (fixtures)

Preserve original `trade_audit.csv`. Corrected interpretation:

| Contract | Audit CSV `realized_pnl_best_effort` | Corrected | Notes |
|----------|--------------------------------------|-----------|-------|
| Sep15 NYC T70 YES | −4.900000 | **≈ −$4.9000** | Held to loss; premium + fees. Confirmed. |
| Sep15 LAX T82 YES | −4.900000 | **≈ −$0.6352** | Equal entry/exit @ 0.01 with **$0.3176 fee each side** → fee drag only. CSV reused NYC loss figure — **audit error**. |
| Sep16 NYC B77.5 NO | 3.284000 | **≈ +$2.9822 net** if 3.2840 is gross and fees 0.3018 are separate; **unresolved** whether exchange `realized_pnl_dollars` already nets fees. Local avg_fill 0.71 was YES-book; acquisition cost **0.29**. |
| Sep16 CHI B75.5 NO | 0.000000 (open) | Open / unresolved | Local avg 0.97 was YES-book; acquisition **0.03**. Weak weather alignment; preference would now reject low model_p. |
| Sep16 LAX T80 YES | 0.000000 (open) | Open / unresolved | Partial fill path; still transfer model, not live-eligible under new gates. |

**Unsupported claims removed:** evaluator parity between production and audit-time re-score was not demonstrated with frozen inputs — mark **unresolved**, do not treat as proven.

---

## Migration / rollback

**Forward (automatic on Store open):**

- Tables `reconcile_checkpoints`, `order_intents` created via `CREATE TABLE IF NOT EXISTS`.
- Fill normalization version stored in order/position `details_json` (`fill_price_norm.v1`).
- No destructive rewrite of historical orders/positions.

**Rollback:**

1. Deploy prior commit; new tables may remain empty/unused (safe).
2. Do not delete `data/audit/audit-20260916T204915Z/`.
3. To ignore new fill semantics, leave existing `avg_price` rows as-is (historical); only new fills are normalized.

---

## Remaining limitations (honest)

**Implementation gaps**

- Combo/RFQ live path still separate (`accept_quote`); not used by scan; not promoted.
- Not every discovered weather series has trained+calibrated artifacts (MIA/AUS/DEN/HOU prep workflow documented; not all end-to-end).
- Deposit vs budget attribution for *manual* exchange activity remains best-effort via local Store only.

**Missing data / access**

- Official portfolio settlements were not re-fetched in this correction run against production (mocked in tests).
- Prospective prediction+quote archives remain thin for profitability claims.

**Unproven performance**

- No strategy is live-eligible under `model_live_eligible is True` after gates.
- Transfer TWC models remain research/paper until settlement-calibrated on TWC outcomes.
- Do not claim reliable winning returns.
