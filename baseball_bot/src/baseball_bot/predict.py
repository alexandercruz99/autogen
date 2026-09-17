from __future__ import annotations

import math
import re
from typing import Iterable

from baseball_bot.models import (
    Batter,
    GameContext,
    HRPrediction,
    Pick,
    Pitcher,
    SlateReport,
    TeamSide,
    WinPrediction,
)

LEAGUE_ERA = 4.10
LEAGUE_OPS = 0.720
LEAGUE_HR_PER_PA = 0.032
HOME_FIELD_BOOST = 0.035


def analyze_slate(games: list[GameContext], game_date: str) -> SlateReport:
    wins = [predict_winner(g) for g in games]
    hrs: list[HRPrediction] = []
    for g in games:
        hrs.extend(predict_hrs(g))
    hrs.sort(key=lambda h: h.hr_prob, reverse=True)
    picks = rank_best_picks(wins, hrs)
    return SlateReport(
        date=game_date,
        games=games,
        win_predictions=sorted(wins, key=lambda w: w.win_prob, reverse=True),
        hr_predictions=hrs,
        best_picks=picks,
    )


def predict_winner(game: GameContext) -> WinPrediction:
    away_strength = _team_strength(game.away, opposing_pitcher=game.home.probable_pitcher)
    home_strength = _team_strength(game.home, opposing_pitcher=game.away.probable_pitcher)
    home_strength += HOME_FIELD_BOOST

    # Softmax-style conversion of rating gap → win probability
    gap = home_strength - away_strength
    home_prob = 1.0 / (1.0 + math.exp(-gap * 5.5))
    home_prob = _clamp(home_prob, 0.22, 0.78)
    away_prob = 1.0 - home_prob

    if home_prob >= away_prob:
        favorite = game.home.name
        favorite_abbr = game.home.abbreviation
        win_prob = home_prob
    else:
        favorite = game.away.name
        favorite_abbr = game.away.abbreviation
        win_prob = away_prob

    edge = abs(home_prob - 0.5)
    confidence = _confidence_label(win_prob)
    rationale = _win_rationale(game, home_prob, away_strength, home_strength)

    return WinPrediction(
        game_pk=game.game_pk,
        matchup=f"{game.away.abbreviation} @ {game.home.abbreviation}",
        favorite=favorite,
        favorite_abbr=favorite_abbr,
        win_prob=win_prob,
        away_prob=away_prob,
        home_prob=home_prob,
        edge=edge,
        confidence=confidence,
        rationale=rationale,
        status=game.status,
    )


def predict_hrs(game: GameContext, top_n: int = 8) -> list[HRPrediction]:
    weather_mult = _weather_hr_multiplier(game)
    preds: list[HRPrediction] = []

    for batter, pitcher, opponent_abbr in _matchup_pairs(game):
        hr_rate = _batter_hr_rate(batter)
        pitcher_mult = _pitcher_hr_multiplier(pitcher)
        order_mult = _order_multiplier(batter.batting_order)
        lineup_mult = 1.08 if batter.in_lineup else 0.92

        # Expected PAs ~ 4.1 for typical starter; scale by order
        expected_pa = 4.15 * order_mult
        per_pa = hr_rate * game.park_hr_factor * pitcher_mult * weather_mult * lineup_mult
        # Cap extreme rates
        per_pa = _clamp(per_pa, 0.005, 0.12)
        expected_hrs = expected_pa * per_pa
        # P(at least 1 HR) ≈ 1 - exp(-λ)
        hr_prob = 1.0 - math.exp(-expected_hrs)
        hr_prob = _clamp(hr_prob, 0.01, 0.55)

        rationale = [
            f"Season HR rate {hr_rate:.3f}/PA ({batter.home_runs} HR in {batter.plate_appearances or batter.at_bats} PA/AB)",
            f"Park factor {game.park_hr_factor:.2f} at {game.venue_name}",
            f"vs {pitcher.name if pitcher else 'TBD'} (HR/9={pitcher.hr_per_9 if pitcher and pitcher.hr_per_9 is not None else 'n/a'})",
        ]
        if weather_mult != 1.0:
            rationale.append(f"Weather multiplier {weather_mult:.2f} ({game.weather_condition}, {game.weather_wind})")
        if batter.in_lineup and batter.batting_order:
            rationale.append(f"Confirmed batting order #{batter.batting_order}")

        preds.append(
            HRPrediction(
                batter_id=batter.id,
                name=batter.name,
                team_abbr=batter.team_abbr,
                opponent_abbr=opponent_abbr,
                game_pk=game.game_pk,
                matchup=f"{game.away.abbreviation} @ {game.home.abbreviation}",
                hr_prob=hr_prob,
                expected_hrs=expected_hrs,
                season_hr=batter.home_runs,
                hr_rate=hr_rate,
                park_factor=game.park_hr_factor,
                pitcher_name=pitcher.name if pitcher else "TBD",
                rationale=rationale,
                in_lineup=batter.in_lineup,
            )
        )

    preds.sort(key=lambda p: p.hr_prob, reverse=True)
    return preds[:top_n]


def rank_best_picks(
    wins: Iterable[WinPrediction],
    hrs: Iterable[HRPrediction],
    max_picks: int = 10,
) -> list[Pick]:
    picks: list[Pick] = []

    for w in sorted(wins, key=lambda x: x.win_prob, reverse=True):
        if w.win_prob < 0.55:
            continue
        picks.append(
            Pick(
                category="moneyline",
                title=f"{w.favorite_abbr} to win ({w.matchup})",
                detail=f"{w.win_prob:.0%} model win probability — {w.confidence}. " + "; ".join(w.rationale[:2]),
                score=w.win_prob,
                confidence=w.confidence,
                game_pk=w.game_pk,
            )
        )

    for h in sorted(hrs, key=lambda x: x.hr_prob, reverse=True)[:12]:
        if h.hr_prob < 0.12:
            continue
        lineup_note = "confirmed lineup" if h.in_lineup else "projected from HR leaders"
        picks.append(
            Pick(
                category="hr",
                title=f"{h.name} ({h.team_abbr}) HR",
                detail=(
                    f"{h.hr_prob:.0%} chance of ≥1 HR in {h.matchup} vs {h.pitcher_name} "
                    f"({lineup_note}; park {h.park_factor:.2f})"
                ),
                score=h.hr_prob + (0.03 if h.in_lineup else 0.0),
                confidence=_confidence_label(h.hr_prob + 0.35),  # map HR probs into labels
                game_pk=h.game_pk,
            )
        )

    # Composite slate score: strong ML + strong HR in same game
    by_game: dict[int, list[HRPrediction]] = {}
    for h in hrs:
        by_game.setdefault(h.game_pk, []).append(h)
    win_by_game = {w.game_pk: w for w in wins}
    for game_pk, w in win_by_game.items():
        top_hr = (by_game.get(game_pk) or [None])[0]
        if not top_hr:
            continue
        combo = (w.win_prob - 0.5) * 2 + top_hr.hr_prob
        if combo < 0.35:
            continue
        picks.append(
            Pick(
                category="slate",
                title=f"Stack: {w.favorite_abbr} ML + {top_hr.name} HR",
                detail=f"{w.matchup}: ML {w.win_prob:.0%} / HR {top_hr.hr_prob:.0%}",
                score=0.45 + combo * 0.4,
                confidence=_confidence_label(min(0.85, 0.5 + combo * 0.3)),
                game_pk=game_pk,
            )
        )

    picks.sort(key=lambda p: p.score, reverse=True)

    # Diversify categories a bit while keeping top scores
    selected: list[Pick] = []
    seen_titles: set[str] = set()
    cat_counts = {"moneyline": 0, "hr": 0, "slate": 0}
    for p in picks:
        if p.title in seen_titles:
            continue
        if cat_counts.get(p.category, 0) >= 4 and len(selected) < max_picks:
            # still allow if we don't have enough yet later
            pass
        if cat_counts.get(p.category, 0) >= 4:
            continue
        selected.append(p)
        seen_titles.add(p.title)
        cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
        if len(selected) >= max_picks:
            break
    return selected


def _team_strength(team: TeamSide, opposing_pitcher: Pitcher | None) -> float:
    # Base from win% centered at .500
    strength = (team.win_pct - 0.5) * 1.2

    # Pythagorean from runs if available
    if team.runs_scored and team.runs_allowed and team.runs_allowed > 0:
        pyth = team.runs_scored**2 / (team.runs_scored**2 + team.runs_allowed**2)
        strength = 0.55 * strength + 0.45 * ((pyth - 0.5) * 1.3)

    # Offense OPS vs league
    if team.team_ops is not None:
        strength += (team.team_ops - LEAGUE_OPS) * 0.55

    # Own pitching ERA
    if team.team_era is not None:
        strength += (LEAGUE_ERA - team.team_era) * 0.06

    # Opposing starter quality (better pitcher → harder for this team)
    if opposing_pitcher and opposing_pitcher.era is not None:
        # Lower ERA hurts this offense's chance
        strength -= (LEAGUE_ERA - opposing_pitcher.era) * 0.08
        if opposing_pitcher.whip is not None:
            strength -= (1.30 - opposing_pitcher.whip) * 0.12
    elif team.probable_pitcher and team.probable_pitcher.era is not None:
        # If we only know our pitcher, give a small boost for ace starts
        strength += (LEAGUE_ERA - team.probable_pitcher.era) * 0.05

    return strength


def _batter_hr_rate(batter: Batter) -> float:
    pa = batter.plate_appearances or batter.at_bats
    if pa and pa >= 40:
        return batter.home_runs / pa
    if pa and pa > 0:
        # Shrink small samples toward league average
        return (batter.home_runs + LEAGUE_HR_PER_PA * 80) / (pa + 80)
    return LEAGUE_HR_PER_PA


def _pitcher_hr_multiplier(pitcher: Pitcher | None) -> float:
    if not pitcher or pitcher.hr_per_9 is None:
        return 1.0
    # League HR/9 ~ 1.15
    return _clamp(pitcher.hr_per_9 / 1.15, 0.7, 1.4)


def _order_multiplier(order: int | None) -> float:
    if order is None:
        return 1.0
    # Middle of order gets more RBI opportunities / similar PA; top gets slightly more PA
    table = {1: 1.06, 2: 1.04, 3: 1.05, 4: 1.03, 5: 1.0, 6: 0.97, 7: 0.94, 8: 0.90, 9: 0.88}
    return table.get(order, 1.0)


def _weather_hr_multiplier(game: GameContext) -> float:
    mult = 1.0
    if game.weather_temp is not None:
        if game.weather_temp >= 85:
            mult += 0.08
        elif game.weather_temp >= 75:
            mult += 0.04
        elif game.weather_temp <= 55:
            mult -= 0.06
    wind = (game.weather_wind or "").lower()
    if "out" in wind:
        # Extract mph if present
        mph = _extract_mph(wind)
        if mph is None or mph >= 8:
            mult += 0.10
        elif mph >= 5:
            mult += 0.05
    elif "in" in wind:
        mph = _extract_mph(wind)
        if mph is None or mph >= 8:
            mult -= 0.10
        elif mph >= 5:
            mult -= 0.05
    condition = (game.weather_condition or "").lower()
    if "dome" in condition or "roof" in condition:
        mult = 1.0  # neutralize outdoor wind/temp quirks
    return _clamp(mult, 0.8, 1.25)


def _extract_mph(wind: str) -> int | None:
    match = re.search(r"(\d+)\s*mph", wind)
    return int(match.group(1)) if match else None


def _matchup_pairs(game: GameContext):
    for batter in game.away.batters:
        yield batter, game.home.probable_pitcher, game.home.abbreviation
    for batter in game.home.batters:
        yield batter, game.away.probable_pitcher, game.away.abbreviation


def _win_rationale(
    game: GameContext,
    home_prob: float,
    away_strength: float,
    home_strength: float,
) -> list[str]:
    notes: list[str] = []
    notes.append(
        f"Records {game.away.abbreviation} {game.away.wins}-{game.away.losses} "
        f"vs {game.home.abbreviation} {game.home.wins}-{game.home.losses}"
    )
    ap = game.away.probable_pitcher
    hp = game.home.probable_pitcher
    if ap and hp:
        notes.append(
            f"Starters: {ap.name} (ERA {ap.era if ap.era is not None else 'n/a'}) vs "
            f"{hp.name} (ERA {hp.era if hp.era is not None else 'n/a'})"
        )
    elif ap or hp:
        p = ap or hp
        assert p is not None
        notes.append(f"Probable: {p.name} (ERA {p.era if p.era is not None else 'n/a'})")
    if game.away.team_ops is not None and game.home.team_ops is not None:
        notes.append(f"Team OPS {game.away.abbreviation} {game.away.team_ops:.3f} / {game.home.abbreviation} {game.home.team_ops:.3f}")
    notes.append(f"Model home win% {home_prob:.0%} (strength {home_strength:+.3f} vs {away_strength:+.3f})")
    return notes


def _confidence_label(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.60:
        return "Medium"
    if prob >= 0.54:
        return "Lean"
    return "Toss-up"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
