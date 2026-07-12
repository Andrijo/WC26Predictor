"""
Day 3: Interfaz interactiva en Streamlit para explorar predicciones
del ensemble Elo + Poisson.
"""

import sys
from pathlib import Path

# Permite importar desde src/ al correr streamlit desde la raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_processed
from src.ensemble import EnsemblePredictor
from src.utils import get_team_list, head_to_head

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")


@st.cache_data
def get_matches() -> pd.DataFrame:
    return load_processed()


@st.cache_resource
def get_model(elo_weight: float, poisson_weight: float, _matches: pd.DataFrame) -> EnsemblePredictor:
    return EnsemblePredictor(elo_weight=elo_weight, poisson_weight=poisson_weight).fit(_matches)


st.title("World Cup 2026 Match Predictor")

matches = get_matches()
teams = get_team_list(matches)

with st.sidebar:
    st.header("Configuración")
    elo_weight = st.slider("Peso del modelo Elo", 0.0, 1.0, 0.5, 0.05)
    poisson_weight = round(1 - elo_weight, 2)
    st.write(f"Peso del modelo Poisson: **{poisson_weight}**")

model = get_model(elo_weight, poisson_weight, matches)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo local", teams, index=teams.index("France") if "France" in teams else 0)
with col2:
    away_team = st.selectbox("Equipo visitante", teams, index=teams.index("Spain") if "Spain" in teams else 1)

if home_team == away_team:
    st.warning("Selecciona dos equipos diferentes.")
else:
    result = model.predict(home_team, away_team)

    st.subheader(f"{home_team} vs. {away_team}")

    probs_df = pd.DataFrame({
        "Resultado": [f"Gana {home_team}", "Empate", f"Gana {away_team}"],
        "Probabilidad": [
            result["ensemble"]["home_win"],
            result["ensemble"]["draw"],
            result["ensemble"]["away_win"],
        ],
    })

    fig = px.bar(probs_df, x="Resultado", y="Probabilidad", text_auto=".1%", color="Resultado")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {home_team}", f"{result['ensemble']['home_win']:.1%}")
    c2.metric("Empate", f"{result['ensemble']['draw']:.1%}")
    c3.metric(f"Gana {away_team}", f"{result['ensemble']['away_win']:.1%}")

    st.info(f"Marcador más probable (modelo Poisson): **{result['most_likely_score']}**")

    with st.expander("Ver desglose por modelo (Elo vs Poisson)"):
        st.json({"Elo": result["elo"], "Poisson": result["poisson"]})

    with st.expander("Historial de enfrentamientos directos"):
        h2h = head_to_head(matches, home_team, away_team)
        if h2h.empty:
            st.write("Sin enfrentamientos previos registrados.")
        else:
            st.dataframe(
                h2h[["date", "home_team", "away_team", "home_score", "away_score"]],
                use_container_width=True,
            )
