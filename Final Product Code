
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

tier_map = {
  "Barcelona": 1,
  "Real Madrid": 1,
  "Atl\u00e9tico Madrid": 1,
  "Real Sociedad": 2,
  "Sevilla": 2,
  "Athletic Club": 2,
  "Villarreal": 2,
  "Betis": 2,
  "Valencia": 3,
  "Osasuna": 3,
  "Celta Vigo": 3,
  "Getafe": 3,
  "Mallorca": 3,
  "Rayo Vallecano": 3,
  "Almer\u00eda": 4,
  "Alav\u00e9s": 4,
  "C\u00e1diz": 4,
  "Granada": 4,
  "Las Palmas": 4
}

confidence_scores = {
  "Athletic Club": {
    "home_win_ratio": 0.5142857142857142,
    "home_draw_ratio": 0.21904761904761905,
    "home_loss_ratio": 0.26666666666666666,
    "away_win_ratio": 0.24299065420560748,
    "away_draw_ratio": 0.3364485981308411,
    "away_loss_ratio": 0.4205607476635514
  },
  "Barcelona": {
    "home_win_ratio": 0.7047619047619048,
    "home_draw_ratio": 0.14285714285714285,
    "home_loss_ratio": 0.1523809523809524,
    "away_win_ratio": 0.5688073394495413,
    "away_draw_ratio": 0.22935779816513763,
    "away_loss_ratio": 0.2018348623853211
  },
  "Celta Vigo": {
    "home_win_ratio": 0.3853211009174312,
    "home_draw_ratio": 0.23853211009174313,
    "home_loss_ratio": 0.3761467889908257,
    "away_win_ratio": 0.18691588785046728,
    "away_draw_ratio": 0.34579439252336447,
    "away_loss_ratio": 0.4672897196261682
  },
  "Real Madrid": {
    "home_win_ratio": 0.6981132075471698,
    "home_draw_ratio": 0.24528301886792453,
    "home_loss_ratio": 0.05660377358490566,
    "away_win_ratio": 0.6074766355140186,
    "away_draw_ratio": 0.22429906542056074,
    "away_loss_ratio": 0.16822429906542055
  },
  "Valencia": {
    "home_win_ratio": 0.37037037037037035,
    "home_draw_ratio": 0.32407407407407407,
    "home_loss_ratio": 0.3055555555555556,
    "away_win_ratio": 0.125,
    "away_draw_ratio": 0.28846153846153844,
    "away_loss_ratio": 0.5865384615384616
  },
  "Real Sociedad": {
    "home_win_ratio": 0.4392523364485981,
    "home_draw_ratio": 0.308411214953271,
    "home_loss_ratio": 0.2523364485981308,
    "away_win_ratio": 0.38095238095238093,
    "away_draw_ratio": 0.2571428571428571,
    "away_loss_ratio": 0.3619047619047619
  },
  "Atl\u00e9tico Madrid": {
    "home_win_ratio": 0.7411764705882353,
    "home_draw_ratio": 0.18823529411764706,
    "home_loss_ratio": 0.07058823529411765,
    "away_win_ratio": 0.46987951807228917,
    "away_draw_ratio": 0.24096385542168675,
    "away_loss_ratio": 0.2891566265060241
  },
  "Getafe": {
    "home_win_ratio": 0.3490566037735849,
    "home_draw_ratio": 0.3490566037735849,
    "home_loss_ratio": 0.3018867924528302,
    "away_win_ratio": 0.16666666666666666,
    "away_draw_ratio": 0.3425925925925926,
    "away_loss_ratio": 0.49074074074074076
  },
  "Granada": {
    "home_win_ratio": 0.41333333333333333,
    "home_draw_ratio": 0.28,
    "home_loss_ratio": 0.30666666666666664,
    "away_win_ratio": 0.14666666666666667,
    "away_draw_ratio": 0.2,
    "away_loss_ratio": 0.6533333333333333
  },
  "Sevilla": {
    "home_win_ratio": 0.4857142857142857,
    "home_draw_ratio": 0.26666666666666666,
    "home_loss_ratio": 0.24761904761904763,
    "away_win_ratio": 0.32075471698113206,
    "away_draw_ratio": 0.33962264150943394,
    "away_loss_ratio": 0.33962264150943394
  },
  "Mallorca": {
    "home_win_ratio": 0.3763440860215054,
    "home_draw_ratio": 0.3010752688172043,
    "home_loss_ratio": 0.3225806451612903,
    "away_win_ratio": 0.1590909090909091,
    "away_draw_ratio": 0.22727272727272727,
    "away_loss_ratio": 0.6136363636363636
  },
  "Betis": {
    "home_win_ratio": 0.4588235294117647,
    "home_draw_ratio": 0.29411764705882354,
    "home_loss_ratio": 0.24705882352941178,
    "away_win_ratio": 0.3058823529411765,
    "away_draw_ratio": 0.29411764705882354,
    "away_loss_ratio": 0.4
  },
  "Alav\u00e9s": {
    "home_win_ratio": 0.3088235294117647,
    "home_draw_ratio": 0.27941176470588236,
    "home_loss_ratio": 0.4117647058823529,
    "away_win_ratio": 0.11428571428571428,
    "away_draw_ratio": 0.24285714285714285,
    "away_loss_ratio": 0.6428571428571429
  },
  "Osasuna": {
    "home_win_ratio": 0.37383177570093457,
    "home_draw_ratio": 0.2336448598130841,
    "home_loss_ratio": 0.3925233644859813,
    "away_win_ratio": 0.3018867924528302,
    "away_draw_ratio": 0.20754716981132076,
    "away_loss_ratio": 0.49056603773584906
  },
  "C\u00e1diz": {
    "home_win_ratio": 0.22950819672131148,
    "home_draw_ratio": 0.3442622950819672,
    "home_loss_ratio": 0.4262295081967213,
    "away_win_ratio": 0.18032786885245902,
    "away_draw_ratio": 0.2786885245901639,
    "away_loss_ratio": 0.5409836065573771
  },
  "Rayo Vallecano": {
    "home_win_ratio": 0.3698630136986301,
    "home_draw_ratio": 0.3150684931506849,
    "home_loss_ratio": 0.3150684931506849,
    "away_win_ratio": 0.14084507042253522,
    "away_draw_ratio": 0.29577464788732394,
    "away_loss_ratio": 0.5633802816901409
  },
  "Girona": {
    "home_win_ratio": 0.5961538461538461,
    "home_draw_ratio": 0.15384615384615385,
    "home_loss_ratio": 0.25,
    "away_win_ratio": 0.2807017543859649,
    "away_draw_ratio": 0.2807017543859649,
    "away_loss_ratio": 0.43859649122807015
  },
  "Almer\u00eda": {
    "home_win_ratio": 0.2727272727272727,
    "home_draw_ratio": 0.30303030303030304,
    "home_loss_ratio": 0.42424242424242425,
    "away_win_ratio": 0.09090909090909091,
    "away_draw_ratio": 0.21212121212121213,
    "away_loss_ratio": 0.696969696969697
  },
  "Las Palmas": {
    "home_win_ratio": 0.20588235294117646,
    "home_draw_ratio": 0.4117647058823529,
    "home_loss_ratio": 0.38235294117647056,
    "away_win_ratio": 0.22857142857142856,
    "away_draw_ratio": 0.17142857142857143,
    "away_loss_ratio": 0.6
  }
}

st.set_page_config(page_title="LaLiga 2025/26 Simulator", layout="wide")

# Load match dataset
@st.cache_data
def load_data():
    return pd.read_csv("matches_2025_26_laliga_19_teams_only.csv")

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
