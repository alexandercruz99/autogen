"""Approximate HR park factors (league average = 1.0)."""

from __future__ import annotations

# Relative home-run friendliness by venue name / common aliases.
# Sources: public park-factor composites (HR); approximate, for ranking only.
PARK_HR_FACTORS: dict[str, float] = {
    "Coors Field": 1.35,
    "Great American Ball Park": 1.22,
    "Yankee Stadium": 1.18,
    "Citizens Bank Park": 1.15,
    "Oriole Park at Camden Yards": 1.12,
    "Globe Life Field": 1.10,
    "American Family Field": 1.08,
    "Guaranteed Rate Field": 1.08,
    "Rate Field": 1.08,
    "Fenway Park": 1.05,
    "Dodger Stadium": 1.02,
    "Chase Field": 1.05,
    "Minute Maid Park": 1.04,
    "Daikin Park": 1.04,
    "Truist Park": 1.03,
    "loanDepot park": 1.02,
    "LoanDepot Park": 1.02,
    "Wrigley Field": 1.00,
    "Busch Stadium": 0.98,
    "Citi Field": 0.97,
    "Target Field": 0.96,
    "Progressive Field": 0.95,
    "Nationals Park": 0.95,
    "Rogers Centre": 0.94,
    "Angel Stadium": 0.93,
    "Angel Stadium of Anaheim": 0.93,
    "Kauffman Stadium": 0.90,
    "Petco Park": 0.88,
    "Oracle Park": 0.82,
    "T-Mobile Park": 0.85,
    "PNC Park": 0.88,
    "Comerica Park": 0.90,
    "Tropicana Field": 0.92,
    "Sutter Health Park": 1.05,  # Athletics temporary home
}

# Team abbreviation → typical home venue name (fallback if schedule lacks venue).
TEAM_HOME_PARK: dict[str, str] = {
    "ARI": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CWS": "Guaranteed Rate Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "DET": "Comerica Park",
    "HOU": "Daikin Park",
    "KC": "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "ATH": "Sutter Health Park",
    "OAK": "Sutter Health Park",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD": "Petco Park",
    "SF": "Oracle Park",
    "SEA": "T-Mobile Park",
    "STL": "Busch Stadium",
    "TB": "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
}


def park_hr_factor(venue_name: str | None, home_abbr: str | None = None) -> float:
    if venue_name and venue_name in PARK_HR_FACTORS:
        return PARK_HR_FACTORS[venue_name]
    if home_abbr and home_abbr in TEAM_HOME_PARK:
        return PARK_HR_FACTORS.get(TEAM_HOME_PARK[home_abbr], 1.0)
    return 1.0
