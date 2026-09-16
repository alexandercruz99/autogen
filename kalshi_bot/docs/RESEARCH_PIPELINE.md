# Observation-engine research pipeline

Research/paper only. **No live orders.** Default inference remains the frozen **baseline**.

## Commands

```bash
cd kalshi_bot
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-freeze
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-audit
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-diagnose
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-experiment
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-collect-once
PYTHONPATH=src python3 -m kalshi_bot.cli --config config.yaml weather-obs-predict
```

## Prospective collection

`weather-obs-collect-once` appends METAR (with first-seen), CLI, NWS benchmark, Kalshi books, and a research prediction into `data/obs_engine/research/prospective.db`.

**Scheduling:** run that command on a cron/timer, or keep `kalshi-bot run` alive and invoke collect periodically. This repo does **not** claim a collector continues after the cloud session exits unless you leave a process running.

## Outputs

| Path | Contents |
| --- | --- |
| `research/baseline_freeze/` | Manifest, hashes, frozen joblib, reproduction |
| `research/data_quality_report.json` | Measurement audit |
| `research/diagnostics_test.csv` | Per day×hour errors |
| `research/experiment_comparison.json` | Candidate MAE by hour |
| `research/two_part_probe.json` | Two-part rise model probe |
| `research/prospective.db` | Append-only forward evidence |
| `latest_research_prediction.json` | Current research forecast |

## Experiment outcome (this run)

Validation macro MAE: baseline 2.40 → local_v2 2.31 → neighbor **2.30** (°F).  
Test: neighbor slightly better at 08/11/14 (~0.05–0.16°F).  
**cloud_precip** (METAR/ASOS only) ≈ neighbor — no incremental gain; GOES/NEXRAD/NYS Mesonet **not integrated**.  
**No** decision-time NWS advantage. Neighbor candidate saved but **not** default for predict.  
Two-part probe: see `two_part_probe.json` (promote only if clearly better).
