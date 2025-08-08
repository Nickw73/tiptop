"""
fixture_app.py
================

This Streamlit web application provides an interactive interface for exploring
Premier League match statistics from the past three seasons. Users can select
any two teams to view their average performance metrics across all games as
well as their head‑to‑head match history and average statistics. The app
leverages the data processing functions defined in ``fixture_analysis.py``.

The underlying data is sourced from football‑data.co.uk, which documents
the meaning of columns such as shots on target (HST/AST), corners (HC/AC) and
yellow/red cards (HY/AY/HR/AR)【415788315280667†L22-L43】. This interface computes
metrics like goals scored, corners won and conceded, and card counts for each
team and match.

Usage
-----

1. Install the dependencies (e.g. ``pip install streamlit pandas``).
2. Ensure ``fixture_analysis.py`` and the CSV files for the seasons you wish to
   analyse (e.g. ``E0.csv``, ``E0_2324.csv``, ``E0_2223.csv``) are in the same
   directory.
3. Run the app with the command:

   ``streamlit run fixture_app.py``

   Streamlit will open a local web server; navigate to the displayed URL in
   your browser to interact with the interface.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import our analysis functions
from fixture_analysis import (
    load_premier_league_data,
    get_team_average,
    get_head_to_head,
)


def load_data(csv_paths):
    """Load the Premier League data for the given CSV files.

    This function simply calls ``load_premier_league_data`` from the analysis
    module. It does not use caching so that changes to the underlying CSV
    files or code are reflected immediately upon reload. If you notice stale
    data when running in Streamlit, you can use the ``--clear-cache`` flag
    when launching the app to remove any persisted caches.

    Parameters
    ----------
    csv_paths : list of str
        Paths to the CSV files to load.

    Returns
    -------
    pandas.DataFrame
        Long‑format DataFrame with one record per team per match.
    """
    return load_premier_league_data(csv_paths)


def main() -> None:
    st.title("Premier League Fixture Analysis")
    st.markdown(
        """
        Select two teams to compare their average performance over the last three
        Premier League seasons (2022‑23 to 2024‑25) and view their head‑to‑head
        match history and aggregated statistics.
        """
    )

    # Map friendly season names to their corresponding CSV files. Adjust these
    # filenames if your files are named differently.
    season_files = {
        "2024-25": "E0.csv",
        "2023-24": "E0_2324.csv",
        "2022-23": "E0_2223.csv",
    }
    season_options = list(season_files.keys())
    # Allow the user to select which seasons to include. By default, include all.
    selected_seasons = st.multiselect(
        "Select seasons to include", season_options, default=season_options
    )
    if not selected_seasons:
        st.warning("Please select at least one season to load data.")
        return
    # Convert selected seasons into CSV paths relative to this script file.
    csv_paths = [str(Path(__file__).parent / season_files[s]) for s in selected_seasons]
    # Optionally, you can print the file paths here for debugging.
    # (Removed during normal operation.)

    # Load the data from CSVs. If the files can't be read this will raise an error,
    # which Streamlit will display in the UI.
    df = load_data(csv_paths)

    # Load the raw CSV files again to obtain match‑level data for head‑to‑head display.
    # We only extract the columns needed for the raw head‑to‑head table.
    raw_dfs = []
    for p in csv_paths:
        try:
            raw = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        # Ensure necessary columns exist, fill missing with zeros
        needed = [
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HY", "AY",
            "HR", "AR", "HST", "AST", "HC", "AC",
        ]
        for col in needed:
            if col not in raw.columns:
                raw[col] = 0
        raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
        raw_dfs.append(raw[needed])
    raw_combined = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()

    # Debugging output for data shape and team names has been removed.

    # Attempt to derive the list of teams from the processed DataFrame. If the
    # expected 'Team' column is missing or empty, fall back to reading the
    # raw CSV files directly.
    try:
        if "Team" in df.columns and df["Team"].nunique() > 0:
            teams = sorted(df["Team"].dropna().unique())
        else:
            raise ValueError("Processed DataFrame missing 'Team' column or contains no teams")
    except Exception:
        raw_team_set = set()
        for p in csv_paths:
            try:
                raw_df = pd.read_csv(p)
                # Add unique home and away team names
                if "HomeTeam" in raw_df.columns:
                    raw_team_set.update(raw_df["HomeTeam"].dropna().unique())
                if "AwayTeam" in raw_df.columns:
                    raw_team_set.update(raw_df["AwayTeam"].dropna().unique())
            except Exception as read_err:
                st.error(f"Error reading {p}: {read_err}")
        teams = sorted(raw_team_set)

    col1, col2 = st.columns(2)
    with col1:
        # Always select the first team by default.
        team1 = st.selectbox("Select Team 1", teams, index=0)
    with col2:
        # Try to default to the second team if it exists; fall back to index 0.
        default_index = 1 if len(teams) > 1 else 0
        team2 = st.selectbox("Select Team 2", teams, index=default_index)

    if team1 == team2:
        st.warning("Please select two different teams to compare.")
        return

    # Compute average statistics for each selected team
    team1_avg = get_team_average(df, team1)
    team2_avg = get_team_average(df, team2)

    st.subheader("Average stats across all matches")
    avg_df = pd.DataFrame([team1_avg, team2_avg], index=[team1, team2])
    st.dataframe(
        avg_df.round(3).rename_axis("Team").reset_index(),
        use_container_width=True,
    )

    # Visual comparison with horizontal bar charts
    st.subheader("Visual comparison of key metrics")
    # Compute additional per‑game totals by combining own and opponent averages.
    # Total Goals per game = goals scored + goals conceded
    team1_total_goals_pg = team1_avg["GoalsScored"] + team1_avg["GoalsConceded"]
    team2_total_goals_pg = team2_avg["GoalsScored"] + team2_avg["GoalsConceded"]
    # Total Cards per game = yellow and red cards for and against
    team1_total_cards_pg = (
        team1_avg["YellowFor"] + team1_avg["YellowAgainst"] +
        team1_avg.get("RedFor", 0) + team1_avg.get("RedAgainst", 0)
    )
    team2_total_cards_pg = (
        team2_avg["YellowFor"] + team2_avg["YellowAgainst"] +
        team2_avg.get("RedFor", 0) + team2_avg.get("RedAgainst", 0)
    )
    # Total shots on target per game = shots on target for + shots on target against
    team1_total_shots_pg = team1_avg["ShotsOnTargetFor"] + team1_avg["ShotsOnTargetAgainst"]
    team2_total_shots_pg = team2_avg["ShotsOnTargetFor"] + team2_avg["ShotsOnTargetAgainst"]
    # Total corners per game = corners won + corners conceded
    team1_total_corners_pg = team1_avg["CornersWon"] + team1_avg["CornersConceded"]
    team2_total_corners_pg = team2_avg["CornersWon"] + team2_avg["CornersConceded"]

    # List of average metrics to visualize (excluding red cards) with readable labels.
    # Each entry corresponds to a key in team average or a derived per‑game total.
    metrics_to_show = [
        ("GoalsScored", "Goals Scored", team1_avg["GoalsScored"], team2_avg["GoalsScored"]),
        ("GoalsConceded", "Goals Conceded", team1_avg["GoalsConceded"], team2_avg["GoalsConceded"]),
        ("YellowFor", "Yellow Cards (For)", team1_avg["YellowFor"], team2_avg["YellowFor"]),
        ("YellowAgainst", "Yellow Cards (Against)", team1_avg["YellowAgainst"], team2_avg["YellowAgainst"]),
        ("ShotsOnTargetFor", "Shots on Target (For)", team1_avg["ShotsOnTargetFor"], team2_avg["ShotsOnTargetFor"]),
        ("ShotsOnTargetAgainst", "Shots on Target (Against)", team1_avg["ShotsOnTargetAgainst"], team2_avg["ShotsOnTargetAgainst"]),
        ("CornersWon", "Corners Won", team1_avg["CornersWon"], team2_avg["CornersWon"]),
        ("CornersConceded", "Corners Conceded", team1_avg["CornersConceded"], team2_avg["CornersConceded"]),
        ("CornersTotal", "Total Corners", team1_avg["CornersTotal"], team2_avg["CornersTotal"]),
        ("TotalGoalsPerGame", "Total Goals per Game", team1_total_goals_pg, team2_total_goals_pg),
        ("TotalCardsPerGame", "Total Cards per Game", team1_total_cards_pg, team2_total_cards_pg),
        ("TotalShotsPerGame", "Total Shots on Target per Game", team1_total_shots_pg, team2_total_shots_pg),
        ("TotalCornersPerGame", "Total Corners per Game", team1_total_corners_pg, team2_total_corners_pg),
    ]
    metric_names = [label for _, label, _, _ in metrics_to_show]
    team1_values = [val1 for _, _, val1, _ in metrics_to_show]
    team2_values = [val2 for _, _, _, val2 in metrics_to_show]
    y_pos = np.arange(len(metrics_to_show))
    bar_height = 0.35
    fig, ax = plt.subplots(figsize=(6, 3 + len(metrics_to_show) * 0.35))
    bars1 = ax.barh(y_pos - bar_height/2, team1_values, height=bar_height, label=team1, color="#1f77b4")
    bars2 = ax.barh(y_pos + bar_height/2, team2_values, height=bar_height, label=team2, color="#d62728")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.invert_yaxis()  # highest metric at top
    ax.set_xlabel("Average value per match")
    ax.set_title(f"{team1} vs {team2}: Average Metrics Comparison")
    ax.legend(loc="best")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    # Annotate bar values with two decimal places
    for bars, values in [(bars1, team1_values), (bars2, team2_values)]:
        ax.bar_label(bars, [f"{v:.2f}" for v in values], label_type="edge", padding=3)
    st.pyplot(fig, use_container_width=True)

    # Compute and visualize total metrics across selected seasons
    team1_records = df[df["Team"] == team1]
    team2_records = df[df["Team"] == team2]
    total_metrics = [
        ("Total Goals", team1_records["GoalsScored"].sum(), team2_records["GoalsScored"].sum()),
        ("Total Cards", (team1_records["YellowFor"].sum() + team1_records["RedFor"].sum()),
                        (team2_records["YellowFor"].sum() + team2_records["RedFor"].sum())),
        ("Total Shots on Target", team1_records["ShotsOnTargetFor"].sum(), team2_records["ShotsOnTargetFor"].sum()),
        ("Total Corners", team1_records["CornersWon"].sum(), team2_records["CornersWon"].sum()),
    ]
    total_names = [name for name, _, _ in total_metrics]
    team1_totals = [val1 for _, val1, _ in total_metrics]
    team2_totals = [val2 for _, _, val2 in total_metrics]
    y_pos_tot = np.arange(len(total_metrics))
    fig2, ax2 = plt.subplots(figsize=(6, 2 + len(total_metrics) * 0.35))
    bars1_tot = ax2.barh(y_pos_tot - bar_height/2, team1_totals, height=bar_height, label=team1, color="#1f77b4")
    bars2_tot = ax2.barh(y_pos_tot + bar_height/2, team2_totals, height=bar_height, label=team2, color="#d62728")
    ax2.set_yticks(y_pos_tot)
    ax2.set_yticklabels(total_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Total across selected seasons")
    ax2.set_title(f"{team1} vs {team2}: Total Metrics Comparison")
    ax2.legend(loc="best")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for bars, values in [(bars1_tot, team1_totals), (bars2_tot, team2_totals)]:
        ax2.bar_label(bars, [f"{v:.0f}" for v in values], label_type="edge", padding=3)
    st.pyplot(fig2, use_container_width=True)

    # Head‑to‑head match history and averages
    try:
        h2h_matches, h2h_avg = get_head_to_head(df, team1, team2)
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader(f"Head‑to‑head matches: {team1} vs {team2}")
    h2h_display = h2h_matches[[
        "Date", "Team", "Opponent", "HomeAway", "GoalsScored", "GoalsConceded",
        "YellowFor", "YellowAgainst", "RedFor", "RedAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst", "CornersWon", "CornersConceded",
    ]]
    st.dataframe(h2h_display.reset_index(drop=True), use_container_width=True)

    # Display raw match‑level data for the selected fixture history
    if not raw_combined.empty:
        mask_raw = (
            (raw_combined["HomeTeam"] == team1) & (raw_combined["AwayTeam"] == team2) |
            (raw_combined["HomeTeam"] == team2) & (raw_combined["AwayTeam"] == team1)
        )
        raw_h2h = raw_combined[mask_raw].sort_values("Date")
        if not raw_h2h.empty:
            st.subheader(f"Raw match data: {team1} vs {team2}")
            st.dataframe(
                raw_h2h[[
                    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HY", "AY",
                    "HR", "AR", "HST", "AST", "HC", "AC",
                ]].reset_index(drop=True),
                use_container_width=True,
            )

    st.subheader("Head‑to‑head averages")
    st.dataframe(
        h2h_avg.round(3).rename_axis("Team").reset_index(),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()