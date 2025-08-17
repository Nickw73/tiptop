"""
Premier League Fixture Analysis & Predictions App
------------------------------------------------

This Streamlit application allows users to explore historical Premier League data and
generate simple per‑game predictions for upcoming fixtures in the 2025/26 season.

Features:

* **Team Analysis** – Compare any two teams across the selected seasons (2022‑23 to 2024‑25),
  view their average match statistics, head‑to‑head history, and visual comparisons.
* **Upcoming Fixtures (2025/26)** – Load the entire fixture list for the 2025/26 season from
  a local JSON file, fetch it automatically, or upload it manually. Then select a
  gameweek to see the fixtures and predictions based on historical averages.

Usage:

1. Place this file alongside:
   - `fixture_analysis.py` (providing `load_premier_league_data` and `get_team_average`).
   - Your historical CSV files (`E0.csv`, `E0_2324.csv`, `E0_2223.csv`).
   - Optionally, `epl-2025.json` (the official fixture list for 2025/26). If absent, the
     app will try to fetch the file or allow upload via the interface.
2. Install the necessary Python packages:
   ```bash
   pip install streamlit pandas numpy matplotlib requests
   ```
3. Run the app with:
   ```bash
   streamlit run fixture_app_updated.py --clear-cache
   ```
4. Open the URL displayed by Streamlit in your browser and interact with the app.

The fixture JSON format is compatible with the data from fixturedownload.com and
includes at least the following keys per entry: `RoundNumber`, `DateUtc`,
`HomeTeam`, `AwayTeam`, and optionally `Location`.
"""

from pathlib import Path
import json
from math import exp

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

from fixture_analysis import load_premier_league_data, get_team_average


def season_file_paths(selected_seasons, mapping):
    """Convert a list of season keys to absolute CSV file paths."""
    base = Path(__file__).parent
    return [str(base / mapping[s]) for s in selected_seasons if s in mapping]


NORMALISE_MAP = {
    "manchester united": "Man United",
    "manchester utd": "Man United",
    "man united": "Man United",
    "man utd": "Man United",
    "manchester city": "Man City",
    "man city": "Man City",
    "wolves": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "brighton & hove albion": "Brighton",
    "brighton": "Brighton",
    "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "sheffield united": "Sheffield United",
    "afc bournemouth": "Bournemouth",
    "liverpool": "Liverpool",
    "everton": "Everton",
    "aston villa": "Aston Villa",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "brentford": "Brentford",
    "crystal palace": "Crystal Palace",
    "fulham": "Fulham",
    "burnley": "Burnley",
    "southampton": "Southampton",
    "ipswich town": "Ipswich Town",
    "ipswich": "Ipswich Town",
}


def norm_team(name: str) -> str:
    """Normalise team names using the mapping."""
    key = (name or "").strip().lower()
    return NORMALISE_MAP.get(key, name)


def sigmoid_prob(home_val: float, away_val: float, k: float = 1.0) -> float:
    """Return probability that home > away based on difference."""
    diff = float(home_val) - float(away_val)
    return 1.0 / (1.0 + np.exp(-k * diff))


def metric_prediction(avg_home: dict, avg_away: dict):
    """Return predicted per-game metrics for home/away using historical averages."""
    def avg_pair(hkey_for, akey_against):
        return (float(avg_home.get(hkey_for, 0.0)) + float(avg_away.get(akey_against, 0.0))) / 2.0

    pred_goals_h = avg_pair("GoalsScored", "GoalsConceded")
    pred_goals_a = avg_pair("GoalsConceded", "GoalsScored")
    pred_corners_h = avg_pair("CornersWon", "CornersConceded")
    pred_corners_a = avg_pair("CornersConceded", "CornersWon")
    pred_sot_h = avg_pair("ShotsOnTargetFor", "ShotsOnTargetAgainst")
    pred_sot_a = avg_pair("ShotsOnTargetAgainst", "ShotsOnTargetFor")
    pred_yellow_h = avg_pair("YellowFor", "YellowAgainst")
    pred_yellow_a = avg_pair("YellowAgainst", "YellowFor")

    return {
        "GoalsH": pred_goals_h,
        "GoalsA": pred_goals_a,
        "CornersH": pred_corners_h,
        "CornersA": pred_corners_a,
        "SOTH": pred_sot_h,
        "SOTA": pred_sot_a,
        "YellowH": pred_yellow_h,
        "YellowA": pred_yellow_a,
    }


def load_fixtures_json():
    """Load the 2025/26 fixture list from local file, remote fetch, or upload."""
    base = Path(__file__).parent
    local_path = base / "epl-2025.json"
    # 1) local file
    if local_path.exists():
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error reading fixture file: {e}")
            return None
    # 2) remote fetch
    try:
        url = "https://fixturedownload.com/feed/json/epl-2025"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # save local copy
            try:
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception:
                pass
            return data
        else:
            st.warning(f"Failed to fetch fixtures: status {resp.status_code}. You can upload the file below.")
    except Exception as e:
        st.warning(f"Error fetching fixtures: {e}. You can upload the file below.")
    # 3) upload
    uploaded = st.file_uploader("Upload epl-2025.json", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            return data
        except Exception as e:
            st.error(f"Uploaded file is not valid JSON: {e}")
    return None


def main():
    st.set_page_config(page_title="PL Analysis & Predictions", layout="wide")
    page = st.sidebar.selectbox("Page", ["Team Analysis", "Upcoming Fixtures (2025/26)"])
    season_mapping = {
        "2024-25": "E0.csv",
        "2023-24": "E0_2324.csv",
        "2022-23": "E0_2223.csv",
    }
    default_seasons = list(season_mapping.keys())
    with st.sidebar.expander("Season filter"):
        selected_seasons = st.multiselect(
            "Include seasons",
            options=list(season_mapping.keys()),
            default=default_seasons,
        )
    csv_paths = season_file_paths(selected_seasons, season_mapping)
    if not csv_paths:
        st.error("No seasons selected.")
        return
    df = load_premier_league_data(csv_paths)
    if page == "Team Analysis":
        st.title("Team Analysis")
        teams = sorted(df["Team"].dropna().unique())
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", teams, index=0 if teams else None)
        with col2:
            default_index = 1 if len(teams) > 1 else 0
            team2 = st.selectbox("Select Team 2", teams, index=default_index)
        if not team1 or not team2:
            st.warning("Please select two teams to compare.")
            return
        if team1 == team2:
            st.warning("Please select two different teams.")
            return
        team1_avg = get_team_average(df, team1)
        team2_avg = get_team_average(df, team2)
        st.subheader("Average stats across selected seasons")
        avg_df = pd.DataFrame([team1_avg, team2_avg], index=[team1, team2])
        st.dataframe(avg_df.round(3).rename_axis("Team").reset_index(), use_container_width=True)
        st.subheader("Visual comparison (per-game averages)")
        metrics_to_show = [
            ("GoalsScored", "Goals Scored"),
            ("GoalsConceded", "Goals Conceded"),
            ("YellowFor", "Yellow Cards (For)"),
            ("YellowAgainst", "Yellow Cards (Against)"),
            ("ShotsOnTargetFor", "Shots on Target (For)"),
            ("ShotsOnTargetAgainst", "Shots on Target (Against)"),
            ("CornersWon", "Corners Won"),
            ("CornersConceded", "Corners Conceded"),
        ]
        team1_totals = {
            "Total Goals / Game": team1_avg["GoalsScored"] + team1_avg["GoalsConceded"],
            "Total Cards / Game": team1_avg["YellowFor"] + team1_avg["YellowAgainst"],
            "Total Shots on Target / Game": team1_avg["ShotsOnTargetFor"] + team1_avg["ShotsOnTargetAgainst"],
            "Total Corners / Game": team1_avg["CornersWon"] + team1_avg["CornersConceded"],
        }
        team2_totals = {
            "Total Goals / Game": team2_avg["GoalsScored"] + team2_avg["GoalsConceded"],
            "Total Cards / Game": team2_avg["YellowFor"] + team2_avg["YellowAgainst"],
            "Total Shots on Target / Game": team2_avg["ShotsOnTargetFor"] + team2_avg["ShotsOnTargetAgainst"],
            "Total Corners / Game": team2_avg["CornersWon"] + team2_avg["CornersConceded"],
        }
        team1_values = [team1_avg[k] for k, _ in metrics_to_show] + list(team1_totals.values())
        team2_values = [team2_avg[k] for k, _ in metrics_to_show] + list(team2_totals.values())
        labels = [lbl for _, lbl in metrics_to_show] + list(team1_totals.keys())
        y_pos = np.arange(len(labels))
        bar_height = 0.35
        fig, ax = plt.subplots(figsize=(7.0, 3 + len(labels) * 0.35))
        bars1 = ax.barh(y_pos - bar_height / 2, team1_values, height=bar_height, label=team1)
        bars2 = ax.barh(y_pos + bar_height / 2, team2_values, height=bar_height, label=team2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Per-game average")
        ax.legend()
        for bars in (bars1, bars2):
            for b in bars:
                w = b.get_width()
                ax.text(w + (0.01 if w >= 0 else -0.01), b.get_y() + b.get_height() / 2, f"{w:.2f}", va="center")
        st.pyplot(fig, clear_figure=True)
        st.subheader(f"Head-to-head matches: {team1} vs {team2}")
        h2h = df[((df["Team"] == team1) & (df["Opponent"] == team2)) | ((df["Team"] == team2) & (df["Opponent"] == team1))].copy()
        h2h = h2h.sort_values("Date")
        h2h_display = h2h[[
            "Date", "Team", "Opponent", "HomeAway",
            "GoalsScored", "GoalsConceded",
            "YellowFor", "YellowAgainst",
            "RedFor", "RedAgainst",
            "ShotsOnTargetFor", "ShotsOnTargetAgainst",
            "CornersWon", "CornersConceded",
        ]]
        st.dataframe(h2h_display.reset_index(drop=True), use_container_width=True)
    elif page == "Upcoming Fixtures (2025/26)":
        st.title("Upcoming Fixtures and Predictions (2025/26)")
        fixtures = load_fixtures_json()
        if not fixtures:
            return
        fx = pd.DataFrame(fixtures)
        required = {"RoundNumber", "DateUtc", "HomeTeam", "AwayTeam"}
        if not required.issubset(fx.columns):
            st.error("Fixture JSON missing required keys: RoundNumber, DateUtc, HomeTeam, AwayTeam")
            return
        fx["DateUtc"] = pd.to_datetime(fx["DateUtc"], utc=True, errors="coerce")
        fx = fx.dropna(subset=["DateUtc"])
        # Use pd.Timestamp.now with tz to avoid tz_localize errors on certain pandas versions
        now_utc = pd.Timestamp.now(tz="UTC")
        future_fx = fx[fx["DateUtc"] >= now_utc].copy()
        if future_fx.empty:
            st.info("No future fixtures found; showing all fixtures instead.")
            future_fx = fx.copy()
        rounds = sorted(future_fx["RoundNumber"].dropna().astype(int).unique().tolist())
        if not rounds:
            st.warning("No rounds found in fixture data.")
            return
        selected_round = st.selectbox("Select Gameweek (Round Number)", rounds, index=0)
        gw = future_fx[future_fx["RoundNumber"].astype(int) == int(selected_round)].copy()
        if gw.empty:
            gw = fx[fx["RoundNumber"].astype(int) == int(selected_round)].copy()
        rows = []
        for _, r in gw.iterrows():
            home = norm_team(str(r["HomeTeam"]))
            away = norm_team(str(r["AwayTeam"]))
            try:
                avg_home = get_team_average(df, home)
                avg_away = get_team_average(df, away)
            except Exception:
                avg_home = avg_away = None
            pred = None
            probs = {}
            # Only compute predictions if both averages are present (avoid pandas truth-value ambiguity)
            if avg_home is not None and avg_away is not None:
                pred = metric_prediction(avg_home, avg_away)
                probs = {
                    "P(Home more Goals)": sigmoid_prob(pred["GoalsH"], pred["GoalsA"]),
                    "P(Home more Corners)": sigmoid_prob(pred["CornersH"], pred["CornersA"]),
                    "P(Home more SOT)": sigmoid_prob(pred["SOTH"], pred["SOTA"]),
                    "P(Home more Yellows)": sigmoid_prob(pred["YellowH"], pred["YellowA"]),
                }
            rows.append({
                "Date (UTC)": r["DateUtc"].strftime("%Y-%m-%d %H:%M"),
                "Round": int(r["RoundNumber"]),
                "Home": home,
                "Away": away,
                "Pred Goals (H)": None if not pred else round(pred["GoalsH"], 2),
                "Pred Goals (A)": None if not pred else round(pred["GoalsA"], 2),
                "Pred Corners (H)": None if not pred else round(pred["CornersH"], 2),
                "Pred Corners (A)": None if not pred else round(pred["CornersA"], 2),
                "Pred SOT (H)": None if not pred else round(pred["SOTH"], 2),
                "Pred SOT (A)": None if not pred else round(pred["SOTA"], 2),
                "Pred Yellows (H)": None if not pred else round(pred["YellowH"], 2),
                "Pred Yellows (A)": None if not pred else round(pred["YellowA"], 2),
                "P(Home>Goals)%": None if not probs else int(round(probs["P(Home more Goals)"] * 100)),
                "P(Home>Corners)%": None if not probs else int(round(probs["P(Home more Corners)"] * 100)),
                "P(Home>SOT)%": None if not probs else int(round(probs["P(Home more SOT)"] * 100)),
                "P(Home>Yellows)%": None if not probs else int(round(probs["P(Home more Yellows)"] * 100)),
            })
        pred_df = pd.DataFrame(rows)
        st.subheader(f"Predictions for Round {selected_round}")
        st.dataframe(pred_df, use_container_width=True)
        st.caption(
            "Predictions use per-match averages from your selected seasons. "
            "If a team doesn’t appear in the historical data, predictions are omitted."
        )


if __name__ == "__main__":
    main()