
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
st.set_page_config(page_title="LaLiga 2025/26 Simulator", layout="wide")

# Load match dataset
@st.cache_data
def load_data():
    return pd.read_csv("matches_full_cleaned_final_standardized.csv")

matches_df = load_data()

# Extract team list
teams = sorted(list(set(matches_df["home_team"]).union(set(matches_df["away_team"]))))

# Compute confidence ratings
def compute_confidence(df, team, is_home):
    matches = df[df["home_team"] == team] if is_home else df[df["away_team"] == team]
    wins = np.sum((matches["home_team"] == team) & (matches["home_score"] > matches["away_score"])) if is_home else np.sum((matches["away_team"] == team) & (matches["away_score"] > matches["home_score"]))
    draws = np.sum(matches["home_score"] == matches["away_score"])
    total = len(matches)
    if total == 0:
        return 0.5
    return (wins + 0.5 * draws) / total

# Predict match score
def predict_score(home, away):
    home_conf = compute_confidence(matches_df, home, True)
    away_conf = compute_confidence(matches_df, away, False)

    base_home = random.gauss(1.5 + home_conf, 0.5)
    base_away = random.gauss(1.0 + away_conf, 0.5)

    score_home = int(np.clip(round(base_home), 0, 6))
    score_away = int(np.clip(round(base_away), 0, 6))
    return score_home, score_away

# Simulate a season
def simulate_season(teams):
    table = {team: {"W":0, "D":0, "L":0, "GF":0, "GA":0, "Pts":0} for team in teams}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            score_home, score_away = predict_score(home, away)
            table[home]["GF"] += score_home
            table[home]["GA"] += score_away
            table[away]["GF"] += score_away
            table[away]["GA"] += score_home
            if score_home > score_away:
                table[home]["W"] += 1
                table[home]["Pts"] += 3
                table[away]["L"] += 1
            elif score_home < score_away:
                table[away]["W"] += 1
                table[away]["Pts"] += 3
                table[home]["L"] += 1
            else:
                table[home]["D"] += 1
                table[away]["D"] += 1
                table[home]["Pts"] += 1
                table[away]["Pts"] += 1
    df = pd.DataFrame.from_dict(table, orient="index")
    df["GD"] = df["GF"] - df["GA"]
    df = df.sort_values(by=["Pts", "GD", "GF"], ascending=False)
    df.insert(0, "Position", range(1, len(df)+1))
    return df.reset_index().rename(columns={"index": "Team"})

# Heatmap simulation
def simulate_heatmap(teams, runs=1000):
    position_counts = {team: [0]*len(teams) for team in teams}
    for _ in range(runs):
        result = simulate_season(teams)
        for i, row in result.iterrows():
            position_counts[row["Team"]][i] += 1
    position_matrix = np.array([position_counts[team] for team in teams])
    probabilities = (position_matrix / runs) * 100
    return teams, probabilities

# --- Streamlit App ---

st.title("⚽ LaLiga 2025/26 Simulator")

tab1, tab2, tab3 = st.tabs(["📊 Season Simulator", "🔮 Match Predictor", "📈 Heatmap"])

with tab1:
    if st.button("Simulate Full Season"):
        season_table = simulate_season(teams)
        st.dataframe(season_table.set_index("Position"), use_container_width=True)

with tab2:
    st.subheader("Predict a Single Match")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams)
    with col2:
        away_team = st.selectbox("Away Team", teams)
    if home_team == away_team:
        st.warning("Select two different teams.")
    elif st.button("Predict Outcome"):
        score_home, score_away = predict_score(home_team, away_team)
        st.success(f"Predicted Score: {home_team} {score_home} - {score_away} {away_team}")

with tab3:
    if st.button("Run 1000 Simulations for Heatmap"):
        heatmap_teams, heatmap_probs = simulate_heatmap(teams, 100)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_probs,
            x=[f"{i+1}st" for i in range(len(teams))],
            y=heatmap_teams,
            colorscale="Reds",
            hoverongaps=False,
            zmin=0,
            zmax=100,
            colorbar=dict(title="% Probability")
        ))
        fig.update_layout(
            title="Probability of Each Team Finishing in Each Position",
            xaxis_title="Position",
            yaxis_title="Team",
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)
