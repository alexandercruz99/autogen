from __future__ import annotations

from baseball_bot.models import Batter, GameContext, Pitcher, TeamSide
from baseball_bot.parks import park_hr_factor
from baseball_bot.predict import analyze_slate, predict_hrs, predict_winner


def _side(abbr: str, name: str, wins: int, losses: int, ops: float, era: float, pitcher: Pitcher) -> TeamSide:
    games = wins + losses
    return TeamSide(
        id=1,
        name=name,
        abbreviation=abbr,
        wins=wins,
        losses=losses,
        win_pct=wins / games,
        runs_scored=float(wins * 4.5),
        runs_allowed=float(losses * 4.5),
        team_era=era,
        team_ops=ops,
        team_hr=150,
        probable_pitcher=pitcher,
        batters=[
            Batter(
                id=10,
                name=f"{abbr} Slugger",
                team_abbr=abbr,
                home_runs=35,
                plate_appearances=550,
                at_bats=500,
                avg=0.280,
                ops=0.900,
                games=140,
                in_lineup=True,
                batting_order=3,
            ),
            Batter(
                id=11,
                name=f"{abbr} Contact",
                team_abbr=abbr,
                home_runs=8,
                plate_appearances=500,
                at_bats=450,
                avg=0.270,
                ops=0.720,
                games=140,
                in_lineup=True,
                batting_order=1,
            ),
        ],
    )


def test_park_factor_coors():
    assert park_hr_factor("Coors Field") > 1.2
    assert park_hr_factor(None, "COL") > 1.2


def test_favorite_is_stronger_team():
    ace = Pitcher(id=1, name="Ace", era=2.50, whip=0.95, hr_per_9=0.7, innings=160)
    soft = Pitcher(id=2, name="Soft", era=5.20, whip=1.45, hr_per_9=1.6, innings=140)
    game = GameContext(
        game_pk=1,
        status="Scheduled",
        venue_name="Yankee Stadium",
        venue_id=1,
        game_time="2026-09-17T23:05:00Z",
        weather_temp=80,
        weather_condition="Clear",
        weather_wind="10 mph, Out To RF",
        park_hr_factor=1.18,
        away=_side("AWY", "Away Weak", 60, 90, 0.680, 4.80, soft),
        home=_side("HOM", "Home Strong", 95, 55, 0.780, 3.40, ace),
    )
    pred = predict_winner(game)
    assert pred.favorite_abbr == "HOM"
    assert pred.win_prob > 0.60


def test_hr_ranks_slugger_above_contact():
    pitcher = Pitcher(id=3, name="Target", era=4.50, whip=1.35, hr_per_9=1.5, innings=120)
    game = GameContext(
        game_pk=2,
        status="Scheduled",
        venue_name="Coors Field",
        venue_id=2,
        game_time="2026-09-17T20:00:00Z",
        weather_temp=88,
        weather_condition="Sunny",
        weather_wind="12 mph, Out To LF",
        park_hr_factor=1.35,
        away=_side("SD", "Padres", 80, 70, 0.740, 3.90, pitcher),
        home=_side("COL", "Rockies", 55, 95, 0.700, 5.10, pitcher),
    )
    hrs = predict_hrs(game, top_n=5)
    assert hrs
    assert hrs[0].hr_prob > hrs[-1].hr_prob
    assert "Slugger" in hrs[0].name


def test_analyze_slate_builds_picks():
    ace = Pitcher(id=1, name="Ace", era=2.80, whip=1.00, hr_per_9=0.8, innings=150)
    soft = Pitcher(id=2, name="Soft", era=4.90, whip=1.40, hr_per_9=1.4, innings=140)
    game = GameContext(
        game_pk=9,
        status="Scheduled",
        venue_name="Great American Ball Park",
        venue_id=3,
        game_time="2026-09-17T23:00:00Z",
        park_hr_factor=1.22,
        away=_side("LAD", "Dodgers", 92, 60, 0.770, 3.50, ace),
        home=_side("CIN", "Reds", 70, 82, 0.710, 4.40, soft),
    )
    report = analyze_slate([game], "2026-09-17")
    assert report.win_predictions
    assert report.hr_predictions
    assert report.best_picks
