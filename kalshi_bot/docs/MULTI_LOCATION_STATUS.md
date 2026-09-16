# Multi-location weather engine — run evidence

Generated: 2026-09-16 (UTC). Live orders: **blocked**.

## Discovery (`weather-discover`)

| Metric | Value |
|--------|------:|
| Climate/Weather series | 392 |
| Daily max temp series | 57 |
| Daily min temp series | 54 |
| Operating NWS CLI daily-max (verified mapping) | 8 |
| Ambiguous (CLI issuedby, METAR unsettled) | 3 |
| Unsupported / TWC / other | 383 |

Operating NWS CLI daily-max series: `HIGHAUS`, `HIGHNY`, `HIGHCHI`, `HIGHMIA`, `KXDENHIGH`, `KXHIGHOU`, `KXHIGHHOU`, `KXHOUHIGH`.

Settlement stations are evidence-backed (contract/API CLI URLs). TWC series are recorded as `blocked_unsupported_settlement_source` — the NWS station_v2 pipeline is not applied.

## Multi cycle (`weather-multi-once`)

Run at ~06:06 UTC (local night for all US locations).

| Location | Series | Status | Paper |
|----------|--------|--------|-------|
| nyc_central_park | HIGHNY | `unsupported_decision_time` | blocked (off-hours) |
| chi_midway | HIGHCHI | `unsupported_decision_time` | blocked |
| mia_cli | HIGHMIA | `unsupported_decision_time` | blocked |
| aus_cli | HIGHAUS | `unsupported_decision_time` | blocked |
| den_cli | KXDENHIGH | `unsupported_decision_time` | blocked |
| hou_cli | KXHIGHOU (+aliases) | `unsupported_decision_time` | blocked |

- METAR 30h backfill + CLI collected for mapped locations.
- `n_probabilities=0` — legitimate; not reported as end-to-end forecast success.
- `live_order_submitted_any=false`.

Early-morning coverage check (NYC, climate day just started): 1 temp obs → `partial_window` / inadequate — correct.

## NYC historical train/eval (preserved)

From `station_corrected_v2_report.json` (prior chronological fit; not re-tuned this change):

| Split | Days | Rows |
|-------|-----:|-----:|
| Train | 1162 | 3486 |
| Calib | 249 | 747 |
| Test | 249 | 747 |

Production distribution median MAE (°F) / 80% coverage on test: 08h 3.53 / 0.80; 11h 2.44 / 0.78; 14h 1.11 / 0.85.

Frozen baseline path unchanged. Other locations: `needs_historical_backfill` — no location model transfer from NYC.

## Tests run

```text
PYTHONPATH=src python3 -m pytest tests/test_multi_location.py tests/test_feeds_pipeline_fixes.py tests/test_feeds.py -q
→ 31 passed
```

Includes: stale/missing-window coverage, first_seen-after-decision, wrong-station CLI, timezone boundaries, cross-location isolation, unified predict parity, YES→NO separate positions, settlement idempotency, concurrent duplicates, restart persistence, exposure/cash, unsupported location independence, fill→settlement fixture, live-order source scan.

## Fixes verified

1. Coverage rejects 10h-stale and missing early-window obs.
2. Paper YES then NO → two positions, not two NO contracts.
3. Settlement payout idempotent; shared exposure limits.
4. Unified `predict_station_v2` used by operating infer path.
5. Off-hours multi cycle does not claim forecast success.
6. METAR backfill prefers receipt/report time for `first_seen_utc` on new inserts.

## Remaining limitations

- Non-NYC locations need historical ASOS/CLI backfill + location-specific train/calib before probabilities.
- Daily-min / rain / snow require metric-specific adapters (discovered, blocked).
- TWC settlement adapters not implemented.
- Ambiguous NWS CLI mappings need METAR/station evidence before operating.
- Accuracy / paper PnL claims require prospective evidence; fixture settlement is labeled simulated.
