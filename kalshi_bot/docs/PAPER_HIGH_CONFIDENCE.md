# High-confidence paper mode

**Live is off.** Paper only. `live_eligible=false`.

## Goal

Stop the losing live pattern (cheap NO / tails). Only simulate fills when the ticket is a **YES on the modal bracket** with **model p ≥ 0.95**, at **local hour 14**, with **max_so_far already inside that bracket** and **remain q90 ≤ 2°F**.

## Honest limits

| Claim | Reality |
|-------|---------|
| “Model wins 95% of the time” on *all* days | **False** — replay YES on modal 2° bracket @14h is ~56–77% by city |
| “When we *do* paper-fill under this filter” | Trades are rare; PMFs almost never put ≥95% on one bracket |
| Locked days (max already in bucket, tiny remain) | Hit rate approaches certainty on **proxy** labels — still ≠ TWC CLI proof |

We are **raising the bar until almost nothing trades**, not inventing edge.

## Policy id

`paper.high_confidence.v1` — see `paper_policy.py`.

## Run

```bash
cd kalshi_bot
# ensure models.weather.obs_engine_live_eligible: false
PYTHONPATH=src python3 -m kalshi_bot.cli weather-twc-bet --series KXHIGHNY
# do NOT pass --live
```

Expect `paper_skip_no_high_confidence` most of the time. That is success of the filter.
