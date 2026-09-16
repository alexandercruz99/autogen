# TWC (Weather Company) adapter

Settlement authority for Kalshi daily high/low temperature series such as
`KXHIGHNY` / `KXHIGHCHI` is **The Weather Company**, not NWS CLI. The live
`settlement_sources` URL is https://weather.com/kalshi.

## Public portal API (no key)

| Endpoint | Use |
|----------|-----|
| `GET /kalshi/api/climate/primary?date=YYYY-MM-DD` | Domestic daily max/min climate reports (`official` / `preliminary` / `no_report`) |
| `GET /kalshi/api/metar?primary=true&weekStart=YYYY-MM-DD` | Progressive hourly temps at settlement ICAOs |
| `GET /kalshi/api/climate/international?date=YYYY-MM-DD` | International cities (°C) — not operating yet |

Evidence: Kalshi `rules_primary` for `KXHIGHNY` names CLINYC / The Weather Company;
portal rows expose `cliId=NYC`, `icao=KNYC`.

## Code

- `feeds/twc_kalshi.py` — client + collect + climate floor + progressive max
- `feeds/features_twc.py` — station_v2 features from AviationWeather METAR, TWC max/floor
- `multi/registry.py` — `VERIFIED_TWC_DAILY_MAX`
- `multi/pipeline.py` — `_process_twc_location` (never applies NWS CLI)

## Operating series (verified)

KXHIGHNY, KXHIGHCHI, KXHIGHLAX, KXHIGHAUS, KXHIGHDEN, KXHIGHPHIL, KXHIGHMIA.

Residual models may reuse same-ICAO `station_v2` artifacts (NYC/CHI) and are labeled
**exploratory** until trained on TWC climate labels. Live orders remain blocked.

## Run

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli weather-discover
PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-once
PYTHONPATH=src python3 -m kalshi_bot.cli weather-multi-status
```
