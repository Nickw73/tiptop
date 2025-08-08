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
        team1 = st.selectbox("Select Home team", teams, index=0)
    with col2:
        # Try to default to the second team if it exists; fall back to index 0.
        default_index = 1 if len(teams) > 1 else 0
        team2 = st.selectbox("Select Away team", teams, index=default_index)

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

    # Visual comparison of key metrics
    st.subheader("Visual comparison of key metrics")

    # Build metrics in the desired order (no red cards)
    # Per-game averages for "for" and "against"
    metrics = [
        ("Goals Scored", float(team1_avg["GoalsScored"]), float(team2_avg["GoalsScored"])),
        ("Goals Conceded", float(team1_avg["GoalsConceded"]), float(team2_avg["GoalsConceded"])),
        ("Yellow Cards (For)", float(team1_avg["YellowFor"]), float(team2_avg["YellowFor"])),
        ("Yellow Cards (Against)", float(team1_avg["YellowAgainst"]), float(team2_avg["YellowAgainst"])),
        ("Shots on Target (For)", float(team1_avg["ShotsOnTargetFor"]), float(team2_avg["ShotsOnTargetFor"])),
        ("Shots on Target (Against)", float(team1_avg["ShotsOnTargetAgainst"]), float(team2_avg["ShotsOnTargetAgainst"])),
        ("Corners Won", float(team1_avg["CornersWon"]), float(team2_avg["CornersWon"])),
        ("Corners Conceded", float(team1_avg["CornersConceded"]), float(team2_avg["CornersConceded"])),
    ]

    # Derived totals per game
    total_corners_1 = float(team1_avg["CornersWon"] + team1_avg["CornersConceded"])
    total_corners_2 = float(team2_avg["CornersWon"] + team2_avg["CornersConceded"])
    total_goals_1 = float(team1_avg["GoalsScored"] + team1_avg["GoalsConceded"])
    total_goals_2 = float(team2_avg["GoalsScored"] + team2_avg["GoalsConceded"])
    total_yellows_1 = float(team1_avg["YellowFor"] + team1_avg["YellowAgainst"])
    total_yellows_2 = float(team2_avg["YellowFor"] + team2_avg["YellowAgainst"])
    total_shots_on_target_1 = float(team1_avg["ShotsOnTargetFor"] + team1_avg["ShotsOnTargetAgainst"])
    total_shots_on_target_2 = float(team2_avg["ShotsOnTargetFor"] + team2_avg["ShotsOnTargetAgainst"])

    metrics += [
        ("Total Corners", total_corners_1, total_corners_2),
        ("Total Goals per Game", total_goals_1, total_goals_2),
        ("Total Cards per Game", total_yellows_1, total_yellows_2),
        ("Total Shots on Target per Game", total_shots_on_target_1, total_shots_on_target_2),
        ("Total Corners per Game", total_corners_1, total_corners_2),
    ]

    # Plot
    names = [m[0] for m in metrics]
    team1_vals = [m[1] for m in metrics]
    team2_vals = [m[2] for m in metrics]

    y_pos = np.arange(len(names))
    bar_height = 0.4

    fig2, ax2 = plt.subplots(figsize=(8, 2 + len(names) * 0.35))

    bars1 = ax2.barh(y_pos - bar_height/2, team1_vals, height=bar_height, label=team1)
    bars2 = ax2.barh(y_pos + bar_height/2, team2_vals, height=bar_height, label=team2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Average value per match")
    ax2.set_title(f"{team1} vs {team2}: Average Metrics Comparison")
    ax2.legend(loc="upper right")

    # Show value labels on bar ends
    for bars, vals in [(bars1, team1_vals), (bars2, team2_vals)]:
        ax2.bar_label(bars, [f"{v:.2f}" for v in vals], label_type="edge", padding=3)

    # Light grid for readability
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    st.pyplot(fig2, use_container_width=True)

    # Head‑to‑head match history and averages
    try:
        # Use get_head_to_head to retrieve the list of matches. We will compute the
        # averages within this function to avoid issues with pandas versions.
        h2h_matches, _unused_h2h_avg = get_head_to_head(df, team1, team2)
    except ValueError as e:
        st.error(str(e))
        return
    # Compute head‑to‑head averages only on numeric columns. This avoids
    # triggering pandas type errors on object columns in different versions.
    numeric_cols = [
        "GoalsScored", "GoalsConceded", "YellowFor", "YellowAgainst",
        "RedFor", "RedAgainst", "ShotsOnTargetFor", "ShotsOnTargetAgainst",
        "CornersTotal", "CornersWon", "CornersConceded",
    ]
    h2h_avg = h2h_matches.groupby("Team")[numeric_cols].mean()

    
    # --- Head-to-head summary (Team 1 home vs Team 2 away) ---
    # Build a concise summary for previous meetings under the current season filter
    played = int(len(h2h_matches))
    if played > 0:
        wins1 = int((h2h_matches["GoalsScored"] > h2h_matches["GoalsConceded"]).sum())
        draws = int((h2h_matches["GoalsScored"] == h2h_matches["GoalsConceded"]).sum())
        losses1 = played - wins1 - draws
        gf1 = int(h2h_matches["GoalsScored"].sum())
        ga1 = int(h2h_matches["GoalsConceded"].sum())
        gd1 = gf1 - ga1

        summary_df = pd.DataFrame([
            {"Team": team1, "Venue": "Home", "Played": played, "Wins": wins1, "Draws": draws, "Losses": losses1, "GF": gf1, "GA": ga1, "GD": gd1},
            {"Team": team2, "Venue": "Away", "Played": played, "Wins": losses1, "Draws": draws, "Losses": wins1, "GF": ga1, "GA": gf1, "GD": -gd1},
        ])
        st.subheader("Head‑to‑head summary")
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.subheader("Head‑to‑head summary")
        st.info("No previous meetings in the selected seasons.")
    

    st.subheader("Head‑to‑head averages")
    st.dataframe(
        h2h_avg.round(3).rename_axis("Team").reset_index(),
        use_container_width=True,
    )

    # Visual tables (styled like other tables; excluding red card metrics)
    # Ensure avg_df is available for visual tables (recompute in case scope changed)
    avg_df = pd.DataFrame([team1_avg, team2_avg], index=[team1, team2])
    st.subheader("Visual table: Average metrics comparison")
    _avg_vis = avg_df.round(3).rename_axis("Team").reset_index().set_index("Team")
    # Drop red card columns if present
    _avg_vis = _avg_vis.drop(columns=[c for c in ["RedFor","RedAgainst"] if c in _avg_vis.columns], errors="ignore")
    _avg_vis = _avg_vis.style.format(precision=2).background_gradient(axis=None)
    st.dataframe(_avg_vis, use_container_width=True)

    # Ensure summary_df and played are defined for visual H2H summary
    played = int(len(h2h_matches))
    if played > 0:
        wins1 = int((h2h_matches["GoalsScored"] > h2h_matches["GoalsConceded"]).sum())
        draws = int((h2h_matches["GoalsScored"] == h2h_matches["GoalsConceded"]).sum())
        losses1 = played - wins1 - draws
        gf1 = int(h2h_matches["GoalsScored"].sum())
        ga1 = int(h2h_matches["GoalsConceded"].sum())
        gd1 = gf1 - ga1
        summary_df = pd.DataFrame([
            {"Team": team1, "Venue": "Home", "Played": played, "Wins": wins1, "Draws": draws, "Losses": losses1, "GF": gf1, "GA": ga1, "GD": gd1},
            {"Team": team2, "Venue": "Away", "Played": played, "Wins": losses1, "Draws": draws, "Losses": wins1, "GF": ga1, "GA": gf1, "GD": -gd1},
        ])
    st.subheader("Visual table: Head‑to‑head summary")
    # Recompute head-to-head summary numbers for visual table scope
    played = int(len(h2h_matches))
    if played > 0:
        wins1 = int((h2h_matches["GoalsScored"] > h2h_matches["GoalsConceded"]).sum())
        draws = int((h2h_matches["GoalsScored"] == h2h_matches["GoalsConceded"]).sum())
        losses1 = played - wins1 - draws
        gf1 = int(h2h_matches["GoalsScored"].sum())
        ga1 = int(h2h_matches["GoalsConceded"].sum())
        gd1 = gf1 - ga1
        summary_df = pd.DataFrame([
            {"Team": team1, "Venue": "Home", "Played": played, "Wins": wins1, "Draws": draws, "Losses": losses1, "GF": gf1, "GA": ga1, "GD": gd1},
            {"Team": team2, "Venue": "Away", "Played": played, "Wins": losses1, "Draws": draws, "Losses": wins1, "GF": ga1, "GA": gf1, "GD": -gd1},
        ])

    if played > 0:
        _sum_vis = summary_df.set_index("Team").style.format(precision=0).background_gradient(axis=None)
        st.dataframe(_sum_vis, use_container_width=True)
    else:
        st.info("No head‑to‑head data to visualize for the selected seasons.")
    st.subheader(f"Head‑to‑head matches: {team1} vs {team2}")
    h2h_display = h2h_matches[[
        "Date", "Team", "Opponent", "HomeAway", "GoalsScored", "GoalsConceded",
        "YellowFor", "YellowAgainst", "RedFor", "RedAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst", "CornersWon", "CornersConceded",
    ]]
    # Format Date column to YYYY-MM-DD (remove 00:00:00)
    if "Date" in h2h_display.columns:
        h2h_display = h2h_display.copy()
        h2h_display["Date"] = pd.to_datetime(h2h_display["Date"]).dt.strftime("%Y-%m-%d")
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
            _raw_df = raw_h2h[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HY", "AY",
                    "HR", "AR", "HST", "AST", "HC", "AC",
                ]].reset_index(drop=True)
            if "Date" in _raw_df.columns:
                _raw_df = _raw_df.copy()
                _raw_df["Date"] = pd.to_datetime(_raw_df["Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(
                _raw_df,
                use_container_width=True,
            )
if __name__ == "__main__":
    main()