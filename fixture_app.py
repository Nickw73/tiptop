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
    # Display the resolved file paths and whether they exist to help debug missing files.
    for p in csv_paths:
        st.write(f"{p} exists: {Path(p).exists()}")

    # Load the data from CSVs. If the files can't be read this will raise an error,
    # which Streamlit will display in the UI.
    df = load_data(csv_paths)

    # Show the shape of the loaded data and the first few team names to verify it loaded properly.
    st.write("Data shape:", df.shape)
    # Show up to the first 10 unique team names
    st.write("Teams found:", sorted(df["Team"].unique())[:10])

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

    # Visual comparison with horizontal bar chart
    st.subheader("Visual comparison of key metrics")
    # Select a subset of metrics to visualize. You can modify this list to
    # include any metrics you care about.
    # Display all metrics except red cards (they are very rare). Each tuple is
    # (column_key, human‑readable label). Feel free to reorder or adjust labels.
    metrics_to_show = [
        ("GoalsScored", "Goals Scored"),
        ("GoalsConceded", "Goals Conceded"),
        ("YellowFor", "Yellow Cards (For)"),
        ("YellowAgainst", "Yellow Cards (Against)"),
        ("ShotsOnTargetFor", "Shots on Target (For)"),
        ("ShotsOnTargetAgainst", "Shots on Target (Against)"),
        ("CornersTotal", "Total Corners"),
        ("CornersWon", "Corners Won"),
        ("CornersConceded", "Corners Conceded"),
    ]
    metric_names = [label for _, label in metrics_to_show]
    team1_values = [team1_avg[m[0]] for m in metrics_to_show]
    team2_values = [team2_avg[m[0]] for m in metrics_to_show]

    y_pos = np.arange(len(metrics_to_show))
    bar_height = 0.35
    fig, ax = plt.subplots(figsize=(6, 3 + len(metrics_to_show) * 0.3))
    ax.barh(y_pos - bar_height/2, team1_values, height=bar_height, label=team1, color="#1f77b4")
    ax.barh(y_pos + bar_height/2, team2_values, height=bar_height, label=team2, color="#d62728")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.invert_yaxis()  # highest metric at top
    ax.set_xlabel("Average value per match")
    ax.set_title(f"{team1} vs {team2}: Key Metrics Comparison")
    ax.legend(loc="best")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    st.pyplot(fig, use_container_width=True)

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

    st.subheader("Head‑to‑head averages")
    st.dataframe(
        h2h_avg.round(3).rename_axis("Team").reset_index(),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()