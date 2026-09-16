# Multi-location weather engine — run evidence

Generated: 2026-09-16 (UTC). Live orders: **blocked**.

## Discovery (`weather-discover`)

| Metric | Value |
|--------|------:|
| Climate/Weather series | 392 |
| Daily max temp series | 57 |
| Daily min temp series | 54 |
| Operating NWS CLI daily-max (verified mapping) | 8 |
| Operating TWC daily-max (verified) | 7 |
| Ambiguous (CLI issuedby, METAR unsettled) | 3 |

Operating NWS CLI: `HIGHAUS`, `HIGHNY`, `HIGHCHI`, `HIGHMIA`, `KXDENHIGH`, `KXHIGHOU`, `KXHIGHHOU`, `KXHOUHIGH`.

Operating TWC (weather.com/kalshi adapter): `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHLAX`, `KXHIGHAUS`, `KXHIGHDEN`, `KXHIGHPHIL`, `KXHIGHMIA`.

## TWC adapter

- Public portal JSON: `/kalshi/api/climate/primary`, `/kalshi/api/metar`.
- Settlement floors from TWC climate reports (not NWS CLI).
- Progressive max from TWC portal METAR; rich features from AviationWeather same ICAO.
- Same-ICAO residual transfer (NYC/CHI) labeled exploratory; live blocked.
- See `docs/TWC_ADAPTER.md`.

## Live TWC preview (2026-09-16 ~18:30 UTC)

Both NYC and CHI were off supported decision hours (14:30 EDT / 13:30 CDT). Collection succeeded (`aviationweather_metar`, `twc_climate`, `twc_metar`). AviationWeather max-so-far ~75°F at Central Park / Midway. Paper correctly blocked as `unsupported_decision_time`. Next actionable windows: local 8 / 11 / 14.

## Tests run

```text
PYTHONPATH=src python3 -m pytest tests/test_multi_location.py tests/test_feeds_pipeline_fixes.py tests/test_feeds.py -q
→ 33 passed
```

## Remaining limitations

- TWC residual models not yet trained on TWC climate labels (exploratory same-ICAO transfer only for NYC/CHI).
- Daily-min / rain / snow / international TWC still blocked or unmapped.
- Accuracy / paper PnL claims require prospective evidence.
