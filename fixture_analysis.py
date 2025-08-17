"""
Premier League Fixture Analysis & Predictions App
- Team Analysis (multi-season averages, head-to-head table, charts)
- Upcoming Fixtures (2025/26) with per-game predictions & simple probabilities

Place this file alongside:
  - fixture_analysis.py (with load_premier_league_data, etc.)
  - Your historical CSVs (E0.csv, E0_2324.csv, E0_2223.csv)
  - epl-2025.json (optional; the app can fetch or accept upload if missing)
"""

from pathlib import Path
import os
import json
from datetime import datetime, timezone
from math import exp

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

# Local analysis helpers
from fixture_analysis import load_premier_league_data, get_team_average


# ---------- Utility: pretty title ----------
st.set_page_config(page_title="PL Analysis & Predictions", layout="wide")


# ---------- Sidebar Navigation ----------
page = st.sidebar.selectbox(
    "Page",
    ["Team Analysis", "Upcoming Fixtures (2025/26)"]
)


# ---------- Season selection helper ----------
SEASON_TO_FILE = {
    "2024-25": "E0.csv",
    "2023-24": "E0_2324.csv",
    "2022-23": "E0_2223.csv",
}
DEFAULT_SEASONS = ["2024-25", "2023-24", "2022-23"]


def season_file_paths(selected_seasons):
    base = Path(__file__).parent
    return [str(base / SEASON_TO_FILE[s]) for s in selected_seasons if s in SEASON_TO_FILE]


# ---------- Team name normalisation ----------
NORMALISE_MAP = {
    # common variants between feeds & football-data CSVs
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
    key = (name or "").strip().lower()
    return NORMALISE_MAP.get(key, name)


# ---------- Prediction helpers ----------
def sigmoid_prob(home_val: float, away_val: float, k: float = 1.0) -> float:
    """Return probability (0..1) that home > away, based on difference."""
    diff = float(home_val) - float(away_val)
    return 1.0 / (1.0 + exp(-k * diff))


def metric_prediction(avg_home: dict, avg_away: dict):
    """Return predicted per-game metrics for home/away (goals, corners, shots, yellows).
    Uses average of home 'for' and away 'against'. Excludes red cards by design.
    """
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
        "GoalsH": pred_goals_h, "GoalsA": pred_goals_a,
        "CornersH": pred_corners_h, "CornersA": pred_corners_a,
        "SOTH": pred_sot_h, "SOTA": pred_sot_a,
        "YellowH": pred_yellow_h, "YellowA": pred_yellow_a,
    }


# ---------- Load historical data ----------
with st.sidebar.expander("Season filter"):
    selected_seasons = st.multiselect(
        "Include seasons",
        options=list(SEASON_TO_FILE.keys()),
        default=DEFAULT_SEASONS
    )

csv_paths = season_file_paths(selected_seasons)
if not csv_paths:
    st.error("No seasons selected.")
    st.stop()

df = load_premier_league_data(csv_paths)  # team-centric records


# ---------- TEAM ANALYSIS PAGE ----------
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
        st.warning("Please select two teams.")
        st.stop()
    if team1 == team2:
        st.warning("Please select two different teams to compare.")
        st.stop()

    # Averages for both teams
    team1_avg = get_team_average(df, team1)
    team2_avg = get_team_average(df, team2)

    st.subheader("Average stats across selected seasons")
    avg_df = pd.DataFrame([team1_avg, team2_avg], index=[team1, team2])
    st.dataframe(avg_df.round(3).rename_axis("Team").reset_index(), use_container_width=True)

    # Visual comparison with bar chart (includes per-game totals; excludes red cards)
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
    # Add per-game totals (won+conceded) for goals, cards (yellow), shots on target, corners
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

    # Annotate bars with numeric values (2 dp)
    for bars in (bars1, bars2):
        for b in bars:
            w = b.get_width()
            ax.text(
                w + (0.01 if w >= 0 else -0.01),
                b.get_y() + b.get_height() / 2,
                f"{w:.2f}",
                va="center"
            )

    st.pyplot(fig, clear_figure=True)

    # Head-to-head table (raw rows)
    st.subheader(f"Head-to-head matches: {team1} vs {team2}")
    h2h = df[((df["Team"] == team1) & (df["Opponent"] == team2)) |
             ((df["Team"] == team2) & (df["Opponent"] == team1))].copy()
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

# ---------- UPCOMING FIXTURES PAGE ----------
elif page == "Upcoming Fixtures (2025/26)":
    st.title("Upcoming Fixtures (2025/26) & Predictions")

    def load_fixtures_json():
        """Load fixtures from local file, or fetch, or upload."""
        base = Path(__file__).parent
        local_path = base / "epl-2025.json"

        # 1) Local
        if local_path.exists():
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Error reading local fixtures: {e}")

        # 2) Try fetch (desktop UA)
        st.info("Local fixtures not found — attempting to fetch official feed…")
        try:
            url = "https://fixturedownload.com/feed/json/epl-2025"
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                # Save a local copy
                try:
                    with open(local_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    st.success("Fixtures fetched and saved to epl-2025.json.")
                except Exception:
                    # saving is optional; continue anyway
                    pass
                return data
            else:
                st.warning(f"Fetch failed (status {r.status_code}). You can upload the JSON below.")
        except Exception as e:
            st.warning(f"Fetch error: {e}. You can upload the JSON below.")

        # 3) Upload
        uploaded = st.file_uploader("Upload epl-2025.json", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                return data
            except Exception as e:
                st.error(f"Uploaded file isn't valid JSON: {e}")
                return None
        return None

    fixtures_data = load_fixtures_json()
    if not fixtures_data:
        st.stop()

    # Normalise into DataFrame
    fx = pd.DataFrame(fixtures_data)
    # Ensure required columns exist
    needed = {"RoundNumber", "DateUtc", "HomeTeam", "AwayTeam"}
    if not needed.issubset(set(fx.columns.str.strip())):
        st.error("Fixture JSON missing required keys: RoundNumber, DateUtc, HomeTeam, AwayTeam")
        st.stop()

    # Parse times and filter to future fixtures (UTC comparison)
    fx["DateUtc"] = pd.to_datetime(fx["DateUtc"], utc=True, errors="coerce")
    fx = fx.dropna(subset=["DateUtc"])
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    future_fx = fx[fx["DateUtc"] >= now_utc].copy()
    if future_fx.empty:
        st.info("No future fixtures found in the file. Showing all fixtures for reference.")
        future_fx = fx.copy()

    # Round selection
    rounds = sorted(future_fx["RoundNumber"].dropna().astype(int).unique().tolist())
    default_round = rounds[0] if rounds else 1
    selected_round = st.selectbox("Select Gameweek (RoundNumber)", rounds or [default_round], index=0)

    gw = future_fx[future_fx["RoundNumber"].astype(int) == int(selected_round)].copy()
    if gw.empty:
        st.warning("No fixtures for that round in the future filter; showing this round from full fixtures.")
        gw = fx[fx["RoundNumber"].astype(int) == int(selected_round)].copy()

    # Build predictions
    rows = []
    for _, row in gw.iterrows():
        home = norm_team(str(row["HomeTeam"]))
        away = norm_team(str(row["AwayTeam"]))

        # Compute averages from historical df
        # If a team is newly promoted and not in df, skip predictions
        try:
            avg_home = get_team_average(df, home)
            avg_away = get_team_average(df, away)
        except Exception:
            avg_home, avg_away = None, None

        pred = None
        probs = {}
        if avg_home and avg_away:
            pred = metric_prediction(avg_home, avg_away)
            # Simple probabilities: chance home > away for each metric
            probs = {
                "P(Home more Goals)": sigmoid_prob(pred["GoalsH"], pred["GoalsA"]),
                "P(Home more Corners)": sigmoid_prob(pred["CornersH"], pred["CornersA"]),
                "P(Home more SOT)": sigmoid_prob(pred["SOTH"], pred["SOTA"]),
                "P(Home more Yellows)": sigmoid_prob(pred["YellowH"], pred["YellowA"]),
            }

        rows.append({
            "Date (UTC)": row["DateUtc"].strftime("%Y-%m-%d %H:%M"),
            "Round": int(row["RoundNumber"]),
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
        "Notes: Predictions use per-match averages from your selected seasons. "
        "Team name normalisation is best-effort; if a name can’t be matched to historical data, "
        "predictions for that fixture are left blank."
    )
