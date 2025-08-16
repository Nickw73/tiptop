"""
fixture_analysis.py
====================

This script provides utility functions to load Premier League match statistics
for multiple seasons and compute average performance metrics for individual
teams as well as head‑to‑head summaries between any two teams. The data used
in these calculations comes from the CSV files provided by
`football‑data.co.uk`. Each CSV contains match statistics with columns such as
home/away goals, shots on target, corners and card counts. For details on
the meaning of these columns, see the notes at
`http://www.football-data.co.uk/notes.txt`, which explain that fields like
``HY``/``AY`` are yellow cards for home/away teams, ``HR``/``AR`` are red
cards, ``HST``/``AST`` are shots on target and ``HC``/``AC`` are corners【415788315280667†L22-L43】.

Functions
---------
* ``load_premier_league_data(csv_files: list[str]) -> pd.DataFrame``: Load one
  or more season CSV files into a unified DataFrame with one record per
  team per match. Each record includes goals scored and conceded, shots on
  target, corners won/conceded and card counts.
* ``get_team_average(df: pd.DataFrame, team: str) -> pd.Series``: Compute
  average statistics for a given team across all matches in the provided
  DataFrame.
* ``get_head_to_head(df: pd.DataFrame, team1: str, team2: str) -> tuple``:
  Return both the individual match records where the two teams faced each
  other and the average statistics for each side in those encounters.

Example
-------

To compute average statistics for Everton and Newcastle across the past three
seasons and their head‑to‑head averages, run this module directly:

.. code-block:: bash

    python fixture_analysis.py --team1 Everton --team2 Newcastle \
        --csv E0.csv E0_2324.csv E0_2223.csv

This will print each team’s overall averages and their head‑to‑head
performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _prepare_match_records(df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper to convert a raw match DataFrame into team‑centric records.

    Each match in the raw CSV contains statistics for both the home and away
    sides. This function produces a long‑format DataFrame where each row
    represents a team’s view of a single match with columns for goals scored,
    goals conceded, cards, shots on target and corners. Missing columns in
    older seasons are filled with zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw match data with at least the columns documented in the notes file
        at football‑data.co.uk【415788315280667†L22-L43】.

    Returns
    -------
    pandas.DataFrame
        Long‑format DataFrame with one record per team per match.
    """
    # Ensure essential columns exist; fill missing ones with zeros.
    columns_needed = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "HY", "AY", "HR", "AR", "HST", "AST", "HC", "AC",
    ]
    for col in columns_needed:
        if col not in df.columns:
            df[col] = 0

    # Convert date column to datetime (errors='coerce' handles invalid dates)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    records = []
    for _, row in df.iterrows():
        # Home side perspective
        records.append({
            "Date": row["Date"],
            "Team": row["HomeTeam"],
            "Opponent": row["AwayTeam"],
            "HomeAway": "Home",
            "GoalsScored": row["FTHG"],
            "GoalsConceded": row["FTAG"],
            "YellowFor": row["HY"],
            "YellowAgainst": row["AY"],
            "RedFor": row["HR"],
            "RedAgainst": row["AR"],
            "ShotsOnTargetFor": row["HST"],
            "ShotsOnTargetAgainst": row["AST"],
            "CornersTotal": row["HC"] + row["AC"],
            "CornersWon": row["HC"],
            "CornersConceded": row["AC"],
        })
        # Away side perspective
        records.append({
            "Date": row["Date"],
            "Team": row["AwayTeam"],
            "Opponent": row["HomeTeam"],
            "HomeAway": "Away",
            "GoalsScored": row["FTAG"],
            "GoalsConceded": row["FTHG"],
            "YellowFor": row["AY"],
            "YellowAgainst": row["HY"],
            "RedFor": row["AR"],
            "RedAgainst": row["HR"],
            "ShotsOnTargetFor": row["AST"],
            "ShotsOnTargetAgainst": row["HST"],
            "CornersTotal": row["HC"] + row["AC"],
            "CornersWon": row["AC"],
            "CornersConceded": row["HC"],
        })

    return pd.DataFrame(records)


def load_premier_league_data(csv_files: List[str]) -> pd.DataFrame:
    """Load one or more Premier League CSV files and combine them.

    Parameters
    ----------
    csv_files : list of str
        Paths to CSV files downloaded from `football‑data.co.uk`. Each file
        should be from the same division (e.g. E0 for the Premier League) but
        can represent different seasons.

    Returns
    -------
    pandas.DataFrame
        Combined match records with one row per team per match.
    """
    all_records = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            # Skip files that are empty or contain no columns. This can happen
            # if an empty file was accidentally created or downloaded.
            continue
        match_records = _prepare_match_records(df)
        all_records.append(match_records)
    return pd.concat(all_records, ignore_index=True)


def get_team_average(df: pd.DataFrame, team: str) -> pd.Series:
    """Compute average statistics for a given team.

    The returned Series contains per‑match averages for goals scored,
    goals conceded, yellow/red cards, shots on target and corners.

    Parameters
    ----------
    df : pandas.DataFrame
        Long‑format DataFrame of match records produced by
        ``load_premier_league_data``.
    team : str
        Name of the team as it appears in the data.

    Returns
    -------
    pandas.Series
        Average statistics indexed by metric name.
    """
    team_df = df[df["Team"] == team]
    if team_df.empty:
        raise ValueError(f"Team '{team}' not found in dataset")
    return team_df[[
        "GoalsScored", "GoalsConceded", "YellowFor", "YellowAgainst",
        "RedFor", "RedAgainst", "ShotsOnTargetFor", "ShotsOnTargetAgainst",
        "CornersTotal", "CornersWon", "CornersConceded",
    ]].mean()


def get_head_to_head(df: pd.DataFrame, team1: str, team2: str, mode: str = "home_only") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve head‑to‑head matches and average statistics for two teams.

    Parameters
    ----------
    df : pandas.DataFrame
        Long‑format DataFrame of match records.
    team1 : str
        Name of the first team.
    team2 : str
        Name of the second team.

    Returns
    -------
    tuple
        A tuple containing:
        (0) ``pd.DataFrame`` of all matches where the two teams faced each other,
        sorted by date; and
        (1) ``pd.DataFrame`` with the average statistics for each team in those
            encounters (index will be the team names).
    """
    mask = ((df["Team"] == team1) & (df["Opponent"] == team2)) | \
           ((df["Team"] == team2) & (df["Opponent"] == team1))
    h2h_df = df[mask].sort_values("Date").reset_index(drop=True)
    if h2h_df.empty:
        raise ValueError(f"No head‑to‑head matches found for {team1} and {team2}")
    # Compute mean only on numeric columns to avoid errors if non‑numeric columns are present.
    h2h_avg = h2h_df.groupby("Team", numeric_only=True).mean()[[
        "GoalsScored", "GoalsConceded", "YellowFor", "YellowAgainst",
        "RedFor", "RedAgainst", "ShotsOnTargetFor", "ShotsOnTargetAgainst",
        "CornersTotal", "CornersWon", "CornersConceded",
    ]]
    return h2h_df, h2h_avg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average team and head‑to‑head statistics from Premier League data"
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Paths to Premier League CSV files (e.g. E0.csv E0_2324.csv E0_2223.csv)"
    )
    parser.add_argument(
        "--team1", required=True, help="Name of the first team"
    )
    parser.add_argument(
        "--team2", required=True, help="Name of the second team"
    )
    args = parser.parse_args()
    # Expand relative paths
    csv_paths = [str(Path(p).expanduser()) for p in args.csv]
    df = load_premier_league_data(csv_paths)
    # Overall averages
    team1_avg = get_team_average(df, args.team1)
    team2_avg = get_team_average(df, args.team2)
    print(f"Average stats for {args.team1} across provided seasons:\n")
    print(team1_avg.to_string())
    print("\n--------------------------------\n")
    print(f"Average stats for {args.team2} across provided seasons:\n")
    print(team2_avg.to_string())
    print("\n--------------------------------\n")
    # Head‑to‑head
    h2h_df, h2h_avg = get_head_to_head(df, args.team1, args.team2)
    print(f"Head‑to‑head match history between {args.team1} and {args.team2} (dates ascending):\n")
    print(h2h_df[["Date", "Team", "Opponent", "GoalsScored", "GoalsConceded", "YellowFor", "YellowAgainst", "RedFor", "RedAgainst", "ShotsOnTargetFor", "ShotsOnTargetAgainst", "CornersWon", "CornersConceded"]])
    print("\nHead‑to‑head averages:\n")
    print(h2h_avg)


if __name__ == "__main__":
    main()