# Weather feeds pipeline (research / paper)

Public observation adapters for the NYC Central Park daily-max research engine.
**Live Kalshi orders stay disabled** on this path (`live_eligible=false`; paper decisions only).

## Connections (verified endpoints)

| Feed | Source | Product / notes |
|------|--------|-----------------|
| METAR | `aviationweather.gov/api/data/metar` | KNYC, KLGA, KJFK — public, no key |
| CLI | NWS API climate reports | Prelim + final; settlement uses **final** CLI |
| GOES | `s3://noaa-goes19` (GOES-East) | `ABI-L2-ACMC` CONUS Clear Sky Mask; NYC window extract |
| NEXRAD | `s3://unidata-nexrad-level3` | OKX `N0B` base reflectivity; **not** deprecated `noaa-nexrad-level2` |
| NYS Mesonet | — | **Optional / blocked** until permitted access |

Historical ACM fallback: `s3://noaa-goes16` when GOES-19 object missing.

### Provenance disclosures

- GOES ABI L2 ACM includes RTM brightness-temperature comparison fields (`includes_rtm_model_assist=true`). It is a satellite cloud product, **not** an NWS temperature forecast.
- NEXRAD Level-III features use product level indices (relative intensity), not calibrated dBZ.
- Missing wind / cloud / radar is **never** filled as calm / clear / dry.

## Continuous collector

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-feeds-run --interval 300
# status / stop
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-status
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-stop
```

Or module entrypoint:

```bash
PYTHONPATH=src python3 -m kalshi_bot.models.weather.obs_engine.feeds.worker run --interval 300
```

State lives under `data/obs_engine/feeds/`:

- `feeds.db` — samples, features, predictions, paper decisions, checkpoints, heartbeat
- `collector.pid` — worker pidfile
- `last_cycle.json` / `latest_prediction.json`

Restart recovery: UNIQUE `(feed, source_key)` prevents duplicate inserts; checkpoints restore last success keys.

Systemd (optional; only if you install the unit yourself — this file is **not** auto-deployed):

```ini
# /etc/systemd/system/kalshi-weather-feeds.service
[Unit]
Description=Kalshi weather feeds collector (research)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/kalshi_bot
Environment=PYTHONPATH=src
ExecStart=/usr/bin/python3 -m kalshi_bot.cli --config config.yaml weather-feeds-run --interval 300
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
```

## Train / infer

```bash
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-train   # station_corrected.v1 + satrad candidate eval
PYTHONPATH=src python3 -m kalshi_bot.cli weather-feeds-once    # one collect + research infer + paper record
```

- Frozen baseline: `data/obs_engine/research/baseline_freeze/obs_nyc_q50_baseline.joblib`
- Operating fallback: `data/obs_engine/feeds/models/station_corrected_v1.joblib` (local_v2 / missing-wind indicators)
- Sat/radar candidate is **not** attached to inference until evaluation supports promotion

## CLI commands

| Command | Purpose |
|---------|---------|
| `weather-feeds-once` | Full cycle METAR/CLI/GOES/NEXRAD + features + research infer |
| `weather-feeds-run` | Persistent worker |
| `weather-feeds-status` | Heartbeat + checkpoints |
| `weather-feeds-stop` | SIGTERM worker |
| `weather-feeds-train` | Train / compare models |
