from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pitcher:
    id: int | None
    name: str
    era: float | None = None
    whip: float | None = None
    hr_per_9: float | None = None
    innings: float | None = None
    wins: int | None = None
    losses: int | None = None
    strikeouts: int | None = None
    games_started: int | None = None


@dataclass
class Batter:
    id: int
    name: str
    team_abbr: str
    home_runs: int = 0
    plate_appearances: int = 0
    at_bats: int = 0
    avg: float | None = None
    ops: float | None = None
    games: int = 0
    in_lineup: bool = False
    batting_order: int | None = None


@dataclass
class TeamSide:
    id: int
    name: str
    abbreviation: str
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.5
    runs_scored: float | None = None
    runs_allowed: float | None = None
    team_era: float | None = None
    team_ops: float | None = None
    team_hr: int | None = None
    probable_pitcher: Pitcher | None = None
    batters: list[Batter] = field(default_factory=list)


@dataclass
class GameContext:
    game_pk: int
    status: str
    venue_name: str
    venue_id: int | None
    game_time: str
    weather_temp: float | None = None
    weather_condition: str | None = None
    weather_wind: str | None = None
    park_hr_factor: float = 1.0
    away: TeamSide = field(default_factory=lambda: TeamSide(0, "Away", "AWY"))
    home: TeamSide = field(default_factory=lambda: TeamSide(0, "Home", "HOM"))
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class WinPrediction:
    game_pk: int
    matchup: str
    favorite: str
    favorite_abbr: str
    win_prob: float
    away_prob: float
    home_prob: float
    edge: float
    confidence: str
    rationale: list[str] = field(default_factory=list)
    status: str = ""


@dataclass
class HRPrediction:
    batter_id: int
    name: str
    team_abbr: str
    opponent_abbr: str
    game_pk: int
    matchup: str
    hr_prob: float
    expected_hrs: float
    season_hr: int
    hr_rate: float
    park_factor: float
    pitcher_name: str
    rationale: list[str] = field(default_factory=list)
    in_lineup: bool = False


@dataclass
class Pick:
    category: str  # moneyline | hr | slate
    title: str
    detail: str
    score: float
    confidence: str
    game_pk: int | None = None


@dataclass
class SlateReport:
    date: str
    games: list[GameContext]
    win_predictions: list[WinPrediction]
    hr_predictions: list[HRPrediction]
    best_picks: list[Pick]
