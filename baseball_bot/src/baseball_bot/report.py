from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from baseball_bot.models import SlateReport

console = Console()


def print_report(report: SlateReport, top_hrs: int = 15) -> None:
    console.print()
    console.print(
        Panel.fit(
            Text(f"MLB Picks Bot — {report.date}", style="bold white"),
            subtitle="Model estimates only · not betting advice",
            border_style="bright_blue",
        )
    )

    if not report.games:
        console.print("[yellow]No MLB games found for this date.[/yellow]")
        return

    _print_best_picks(report)
    _print_winners(report)
    _print_hrs(report, top_hrs=top_hrs)
    _print_slate(report)


def _print_best_picks(report: SlateReport) -> None:
    table = Table(title="Best Picks (ranked)", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Pick", style="bold")
    table.add_column("Conf", width=8)
    table.add_column("Why")

    if not report.best_picks:
        console.print("[yellow]No high-confidence picks on this slate.[/yellow]")
        return

    for i, pick in enumerate(report.best_picks, start=1):
        table.add_row(
            str(i),
            pick.category,
            pick.title,
            pick.confidence,
            pick.detail,
        )
    console.print(table)
    console.print()


def _print_winners(report: SlateReport) -> None:
    table = Table(title="Who Wins Today", show_lines=False)
    table.add_column("Matchup", style="bold")
    table.add_column("Pick", style="green")
    table.add_column("Win%", justify="right")
    table.add_column("Away%", justify="right", style="dim")
    table.add_column("Home%", justify="right", style="dim")
    table.add_column("Conf")
    table.add_column("Status", style="dim")

    for w in report.win_predictions:
        table.add_row(
            w.matchup,
            w.favorite_abbr,
            f"{w.win_prob:.0%}",
            f"{w.away_prob:.0%}",
            f"{w.home_prob:.0%}",
            w.confidence,
            w.status,
        )
    console.print(table)
    console.print()


def _print_hrs(report: SlateReport, top_hrs: int) -> None:
    table = Table(title=f"Top HR Candidates (top {top_hrs})", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Batter", style="bold")
    table.add_column("Tm", width=4)
    table.add_column("Game")
    table.add_column("P(HR)", justify="right", style="magenta")
    table.add_column("Season", justify="right")
    table.add_column("Park", justify="right")
    table.add_column("vs Pitcher")
    table.add_column("LU", width=3)

    for i, h in enumerate(report.hr_predictions[:top_hrs], start=1):
        table.add_row(
            str(i),
            h.name,
            h.team_abbr,
            h.matchup,
            f"{h.hr_prob:.0%}",
            str(h.season_hr),
            f"{h.park_factor:.2f}",
            h.pitcher_name,
            "Y" if h.in_lineup else "—",
        )
    console.print(table)
    console.print()


def _print_slate(report: SlateReport) -> None:
    table = Table(title="Slate Snapshot", show_lines=False)
    table.add_column("Game")
    table.add_column("Venue")
    table.add_column("Weather")
    table.add_column("Away SP")
    table.add_column("Home SP")
    table.add_column("Park HR")

    for g in report.games:
        weather_bits = []
        if g.weather_temp is not None:
            weather_bits.append(f"{g.weather_temp:.0f}°F")
        if g.weather_condition:
            weather_bits.append(g.weather_condition)
        if g.weather_wind:
            weather_bits.append(g.weather_wind)
        table.add_row(
            f"{g.away.abbreviation} @ {g.home.abbreviation}",
            g.venue_name,
            ", ".join(weather_bits) or "—",
            g.away.probable_pitcher.name if g.away.probable_pitcher else "TBD",
            g.home.probable_pitcher.name if g.home.probable_pitcher else "TBD",
            f"{g.park_hr_factor:.2f}",
        )
    console.print(table)


def report_to_dict(report: SlateReport) -> dict[str, Any]:
    return {
        "date": report.date,
        "disclaimer": "Model estimates for entertainment/analysis only. Not gambling advice.",
        "best_picks": [
            {
                "category": p.category,
                "title": p.title,
                "detail": p.detail,
                "score": round(p.score, 4),
                "confidence": p.confidence,
                "game_pk": p.game_pk,
            }
            for p in report.best_picks
        ],
        "winners": [
            {
                "matchup": w.matchup,
                "favorite": w.favorite,
                "favorite_abbr": w.favorite_abbr,
                "win_prob": round(w.win_prob, 4),
                "away_prob": round(w.away_prob, 4),
                "home_prob": round(w.home_prob, 4),
                "confidence": w.confidence,
                "status": w.status,
                "rationale": w.rationale,
                "game_pk": w.game_pk,
            }
            for w in report.win_predictions
        ],
        "hr_candidates": [
            {
                "name": h.name,
                "team": h.team_abbr,
                "matchup": h.matchup,
                "hr_prob": round(h.hr_prob, 4),
                "expected_hrs": round(h.expected_hrs, 4),
                "season_hr": h.season_hr,
                "park_factor": h.park_factor,
                "pitcher": h.pitcher_name,
                "in_lineup": h.in_lineup,
                "rationale": h.rationale,
                "game_pk": h.game_pk,
            }
            for h in report.hr_predictions
        ],
        "games": [
            {
                "game_pk": g.game_pk,
                "matchup": f"{g.away.name} @ {g.home.name}",
                "status": g.status,
                "venue": g.venue_name,
                "park_hr_factor": g.park_hr_factor,
                "weather": {
                    "temp": g.weather_temp,
                    "condition": g.weather_condition,
                    "wind": g.weather_wind,
                },
            }
            for g in report.games
        ],
    }


def write_json(report: SlateReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report_to_dict(report), fh, indent=2)
