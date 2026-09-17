from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from baseball_bot.models import Batter, GameContext, Pitcher, TeamSide
from baseball_bot.parks import park_hr_factor

logger = logging.getLogger(__name__)

STATS_API = "https://statsapi.mlb.com/api/v1"
STATS_API_11 = "https://statsapi.mlb.com/api/v1.1"
DEFAULT_TIMEOUT = 30


class MLBClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.session = session or requests.Session()
        self.timeout = timeout
        self._pitcher_cache: dict[int, Pitcher] = {}
        self._team_stats_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._standings_cache: dict[int, dict[int, dict[str, Any]]] = {}

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def today_et() -> date:
        return datetime.now(ZoneInfo("America/New_York")).date()

    def fetch_schedule(self, game_date: date | None = None) -> list[GameContext]:
        game_date = game_date or self.today_et()
        season = game_date.year
        payload = self.get(
            f"{STATS_API}/schedule",
            {
                "sportId": 1,
                "date": game_date.isoformat(),
                "hydrate": "probablePitcher,team,venue,weather,linescore",
            },
        )
        games: list[GameContext] = []
        for day in payload.get("dates", []):
            for raw in day.get("games", []):
                games.append(self._build_game(raw, season))
        return games

    def _build_game(self, raw: dict[str, Any], season: int) -> GameContext:
        away_raw = raw["teams"]["away"]
        home_raw = raw["teams"]["home"]
        venue = raw.get("venue", {}) or {}
        weather = raw.get("weather", {}) or {}
        home_abbr = home_raw["team"].get("abbreviation", "")
        venue_name = venue.get("name", "")
        away = self._build_side(away_raw, season)
        home = self._build_side(home_raw, season)

        # Prefer live/confirmed lineup; otherwise season HR leaders.
        self._attach_batters(raw["gamePk"], away, home, season)

        temp = None
        if weather.get("temp") not in (None, ""):
            try:
                temp = float(weather["temp"])
            except (TypeError, ValueError):
                temp = None

        return GameContext(
            game_pk=int(raw["gamePk"]),
            status=raw.get("status", {}).get("detailedState", "Unknown"),
            venue_name=venue_name or "Unknown",
            venue_id=venue.get("id"),
            game_time=raw.get("gameDate", ""),
            weather_temp=temp,
            weather_condition=weather.get("condition"),
            weather_wind=weather.get("wind"),
            park_hr_factor=park_hr_factor(venue_name, home_abbr),
            away=away,
            home=home,
            raw=raw,
        )

    def _build_side(self, side_raw: dict[str, Any], season: int) -> TeamSide:
        team = side_raw["team"]
        record = side_raw.get("leagueRecord", {}) or {}
        wins = int(record.get("wins") or 0)
        losses = int(record.get("losses") or 0)
        games = max(wins + losses, 1)
        win_pct = float(record.get("pct") or (wins / games))

        standings = self._team_standings_row(int(team["id"]), season)
        team_stats = self._team_season_stats(int(team["id"]), season)

        pitcher = None
        pp = side_raw.get("probablePitcher")
        if pp and pp.get("id"):
            pitcher = self.fetch_pitcher(int(pp["id"]), season, fallback_name=pp.get("fullName", "TBD"))

        return TeamSide(
            id=int(team["id"]),
            name=team.get("name", "Unknown"),
            abbreviation=team.get("abbreviation", "???"),
            wins=wins,
            losses=losses,
            win_pct=win_pct,
            runs_scored=_as_float(standings.get("runsScored")),
            runs_allowed=_as_float(standings.get("runsAllowed")),
            team_era=_as_float((team_stats.get("pitching") or {}).get("era")),
            team_ops=_as_float((team_stats.get("hitting") or {}).get("ops")),
            team_hr=_as_int((team_stats.get("hitting") or {}).get("homeRuns")),
            probable_pitcher=pitcher,
        )

    def _team_standings_row(self, team_id: int, season: int) -> dict[str, Any]:
        if season not in self._standings_cache:
            mapping: dict[int, dict[str, Any]] = {}
            try:
                payload = self.get(
                    f"{STATS_API}/standings",
                    {"leagueId": "103,104", "season": season},
                )
                for block in payload.get("records", []):
                    for row in block.get("teamRecords", []):
                        mapping[int(row["team"]["id"])] = row
            except requests.RequestException as exc:
                logger.warning("standings fetch failed: %s", exc)
            self._standings_cache[season] = mapping
        return self._standings_cache[season].get(team_id, {})

    def _team_season_stats(self, team_id: int, season: int) -> dict[str, Any]:
        key = (team_id, season)
        if key in self._team_stats_cache:
            return self._team_stats_cache[key]
        out: dict[str, Any] = {}
        for group in ("hitting", "pitching"):
            try:
                payload = self.get(
                    f"{STATS_API}/teams/{team_id}/stats",
                    {"season": season, "group": group, "stats": "season"},
                )
                for sg in payload.get("stats", []):
                    splits = sg.get("splits") or []
                    if splits:
                        out[group] = splits[0].get("stat", {})
                        break
            except requests.RequestException as exc:
                logger.warning("team stats %s/%s failed: %s", team_id, group, exc)
        self._team_stats_cache[key] = out
        return out

    def fetch_pitcher(self, pitcher_id: int, season: int, fallback_name: str = "TBD") -> Pitcher:
        if pitcher_id in self._pitcher_cache:
            return self._pitcher_cache[pitcher_id]
        pitcher = Pitcher(id=pitcher_id, name=fallback_name)
        try:
            payload = self.get(
                f"{STATS_API}/people/{pitcher_id}",
                {"hydrate": f"stats(group=[pitching],type=[season],season={season})"},
            )
            people = payload.get("people") or []
            if people:
                person = people[0]
                pitcher.name = person.get("fullName", fallback_name)
                for sg in person.get("stats") or []:
                    splits = sg.get("splits") or []
                    if not splits:
                        continue
                    st = splits[0].get("stat", {})
                    ip = _parse_innings(st.get("inningsPitched"))
                    hrs = _as_int(st.get("homeRuns")) or 0
                    pitcher.era = _as_float(st.get("era"))
                    pitcher.whip = _as_float(st.get("whip"))
                    pitcher.innings = ip
                    pitcher.wins = _as_int(st.get("wins"))
                    pitcher.losses = _as_int(st.get("losses"))
                    pitcher.strikeouts = _as_int(st.get("strikeOuts"))
                    pitcher.games_started = _as_int(st.get("gamesStarted"))
                    if ip and ip > 0:
                        pitcher.hr_per_9 = round(hrs * 9.0 / ip, 2)
                    break
        except requests.RequestException as exc:
            logger.warning("pitcher %s fetch failed: %s", pitcher_id, exc)
        self._pitcher_cache[pitcher_id] = pitcher
        return pitcher

    def _attach_batters(self, game_pk: int, away: TeamSide, home: TeamSide, season: int) -> None:
        lineup_away, lineup_home = self._lineup_from_live(game_pk, away.abbreviation, home.abbreviation)
        if lineup_away:
            away.batters = lineup_away
        else:
            away.batters = self._hr_leaders(away.id, away.abbreviation, season)
        if lineup_home:
            home.batters = lineup_home
        else:
            home.batters = self._hr_leaders(home.id, home.abbreviation, season)

    def _lineup_from_live(
        self, game_pk: int, away_abbr: str, home_abbr: str
    ) -> tuple[list[Batter], list[Batter]]:
        try:
            payload = self.get(f"{STATS_API_11}/game/{game_pk}/feed/live")
        except requests.RequestException:
            return [], []
        box = ((payload.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        return (
            self._batters_from_box(box.get("away") or {}, away_abbr),
            self._batters_from_box(box.get("home") or {}, home_abbr),
        )

    def _batters_from_box(self, side: dict[str, Any], team_abbr: str) -> list[Batter]:
        batters_ids = side.get("batters") or []
        players = side.get("players") or {}
        resolved_abbr = (
            team_abbr
            or ((side.get("team") or {}).get("abbreviation"))
            or ""
        )
        out: list[Batter] = []
        for idx, bid in enumerate(batters_ids[:9], start=1):
            pdata = players.get(f"ID{bid}") or {}
            person = pdata.get("person") or {}
            batting = (pdata.get("seasonStats") or {}).get("batting") or {}
            out.append(
                Batter(
                    id=int(bid),
                    name=person.get("fullName", f"Player {bid}"),
                    team_abbr=resolved_abbr,
                    home_runs=_as_int(batting.get("homeRuns")) or 0,
                    plate_appearances=_as_int(batting.get("plateAppearances")) or 0,
                    at_bats=_as_int(batting.get("atBats")) or 0,
                    avg=_as_float(batting.get("avg")),
                    ops=_as_float(batting.get("ops")),
                    games=_as_int(batting.get("gamesPlayed")) or 0,
                    in_lineup=True,
                    batting_order=idx,
                )
            )
        return out

    def _hr_leaders(self, team_id: int, team_abbr: str, season: int, limit: int = 9) -> list[Batter]:
        try:
            payload = self.get(
                f"{STATS_API}/stats",
                {
                    "stats": "season",
                    "group": "hitting",
                    "season": season,
                    "teamId": team_id,
                    "playerPool": "all",
                    "limit": limit,
                    "sortStat": "homeRuns",
                    "order": "desc",
                },
            )
        except requests.RequestException as exc:
            logger.warning("HR leaders for team %s failed: %s", team_id, exc)
            return []
        out: list[Batter] = []
        for sg in payload.get("stats") or []:
            for split in sg.get("splits") or []:
                player = split.get("player") or {}
                st = split.get("stat") or {}
                out.append(
                    Batter(
                        id=int(player["id"]),
                        name=player.get("fullName", "Unknown"),
                        team_abbr=team_abbr,
                        home_runs=_as_int(st.get("homeRuns")) or 0,
                        plate_appearances=_as_int(st.get("plateAppearances")) or 0,
                        at_bats=_as_int(st.get("atBats")) or 0,
                        avg=_as_float(st.get("avg")),
                        ops=_as_float(st.get("ops")),
                        games=_as_int(st.get("gamesPlayed")) or 0,
                        in_lineup=False,
                    )
                )
        return out


def _as_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_innings(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value)
    if "." in text:
        whole, frac = text.split(".", 1)
        try:
            # MLB encodes outs as .1 / .2
            return float(whole) + (int(frac) / 3.0)
        except ValueError:
            return _as_float(text)
    return _as_float(text)
