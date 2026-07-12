"""
Day 2: Modelo de goles esperados (attack/defense strength) + distribución
de Poisson para simular marcadores exactos.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson


class PoissonGoalModel:
    def __init__(self, max_goals: int = 6):
        self.max_goals = max_goals
        self.avg_home_goals = None
        self.avg_away_goals = None
        self.team_attack: dict[str, float] = {}
        self.team_defense: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame):
        """Calcula fuerza de ataque/defensa relativa a la media de la liga."""
        self.avg_home_goals = matches["home_score"].mean()
        self.avg_away_goals = matches["away_score"].mean()

        teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())

        for team in teams:
            home_games = matches[matches["home_team"] == team]
            away_games = matches[matches["away_team"] == team]

            # Ataque: cuántos goles mete el equipo vs. la media
            goals_for = pd.concat([home_games["home_score"], away_games["away_score"]])
            attack = goals_for.mean() / ((self.avg_home_goals + self.avg_away_goals) / 2) if len(goals_for) else 1.0

            # Defensa: cuántos goles recibe vs. la media (valores bajos = buena defensa)
            goals_against = pd.concat([home_games["away_score"], away_games["home_score"]])
            defense = goals_against.mean() / ((self.avg_home_goals + self.avg_away_goals) / 2) if len(goals_against) else 1.0

            self.team_attack[team] = attack if not np.isnan(attack) else 1.0
            self.team_defense[team] = defense if not np.isnan(defense) else 1.0

        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)

        exp_home_goals = home_attack * away_defense * self.avg_home_goals
        exp_away_goals = away_attack * home_defense * self.avg_away_goals

        return exp_home_goals, exp_away_goals

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Matriz de probabilidad para cada marcador posible (home_goals x away_goals)."""
        exp_home, exp_away = self.expected_goals(home_team, away_team)

        home_probs = [poisson.pmf(i, exp_home) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(i, exp_away) for i in range(self.max_goals + 1)]

        return np.outer(home_probs, away_probs)

    def predict(self, home_team: str, away_team: str) -> dict:
        """Probabilidades 1X2 y marcador más probable."""
        matrix = self.score_matrix(home_team, away_team)

        p_home_win = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away_win = np.triu(matrix, 1).sum()

        most_likely_idx = np.unravel_index(matrix.argmax(), matrix.shape)

        return {
            "home_win": round(float(p_home_win), 4),
            "draw": round(float(p_draw), 4),
            "away_win": round(float(p_away_win), 4),
            "most_likely_score": f"{most_likely_idx[0]}-{most_likely_idx[1]}",
        }


if __name__ == "__main__":
    from src.data_loader import load_processed

    matches = load_processed()
    model = PoissonGoalModel().fit(matches)

    print("Predicción France vs Spain:")
    print(model.predict("France", "Spain"))
