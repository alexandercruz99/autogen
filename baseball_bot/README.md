# Baseball Picks Bot

Analyzes **today's MLB slate** using the public [MLB Stats API](https://statsapi.mlb.com/) and ranks:

1. **Who wins** each game (model win probability)
2. **Who is most likely to hit a HR**
3. **Best picks** across the slate (moneylines, HR spots, simple stacks)

> Model estimates for analysis/entertainment only. Not gambling advice.

## Quick start

```bash
cd baseball_bot
pip install -e .
baseball-bot
```

Options:

```bash
baseball-bot --date 2026-09-17
baseball-bot --json-out /tmp/mlb_picks.json --top-hrs 20
baseball-bot -v
```

## How it works

| Signal | Source |
|--------|--------|
| Schedule, venues, weather, probable pitchers | `statsapi.mlb.com` schedule hydrate |
| Team records / runs scored-allowed | Standings |
| Team OPS / ERA | Team season stats |
| Starter ERA, WHIP, HR/9 | Pitcher season stats |
| Lineups (if posted / in progress) | Live game feed boxscore |
| Otherwise HR leaders | Team hitting leaders |
| Park HR factor | Built-in venue table (Coors, GABP, etc.) |

**Win model:** blend of win%, Pythagorean run differential, team OPS/ERA, opposing starter quality, and a small home-field bump → logistic win probability.

**HR model:** batter HR/PA × park factor × pitcher HR/9 multiplier × weather × lineup/order → `P(≥1 HR) ≈ 1 - e^(-λ)`.

## Output

- Ranked **best picks**
- Per-game **winner** table
- Top **HR candidates**
- Slate snapshot (venue / weather / starters)

Optional `--json-out` writes the full structured report.

## Disclaimer

Baseball outcomes are noisy. This bot does **not** scrape sportsbooks or place bets. Use it as a research checklist, not a tip sheet.
