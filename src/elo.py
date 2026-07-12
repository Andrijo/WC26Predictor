"""
Day 1: Sistema de rating Elo para selecciones de fútbol.
"""

from collections import defaultdict
import pandas as pd

DEFAULT_RATING = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 60  # bonus de puntos Elo por jugar de local


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidad esperada de que 'a' le gane a 'b' según Elo."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


class EloModel:
    def __init__(self, k: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def predict(self, home_team: str, away_team: str) -> dict:
        """Devuelve probabilidades de victoria local / empate / victoria visitante."""
        home_r = self.get_rating(home_team) + self.home_advantage
        away_r = self.get_rating(away_team)

        p_home = expected_score(home_r, away_r)

        # Aproximación simple para separar el empate de la probabilidad continua de Elo
        p_draw = 0.28 - 0.10 * abs(p_home - 0.5) * 2
        p_draw = max(p_draw, 0.05)

        p_home_win = p_home - p_draw / 2
        p_away_win = 1 - p_home_win - p_draw

        return {
            "home_win": round(max(p_home_win, 0), 4),
            "draw": round(max(p_draw, 0), 4),
            "away_win": round(max(p_away_win, 0), 4),
        }

    def update(self, home_team: str, away_team: str, home_score: int, away_score: int):
        """Actualiza los ratings tras un partido real."""
        home_r = self.get_rating(home_team) + self.home_advantage
        away_r = self.get_rating(away_team)

        expected_home = expected_score(home_r, away_r)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Margen de victoria como multiplicador (opcional pero mejora el modelo)
        margin = abs(home_score - away_score)
        margin_mult = 1 + (margin - 1) * 0.15 if margin > 1 else 1

        delta = self.k * margin_mult * (actual_home - expected_home)

        self.ratings[home_team] = self.get_rating(home_team) + delta
        self.ratings[away_team] = self.get_rating(away_team) - delta

    def fit(self, matches: pd.DataFrame):
        """Entrena el modelo procesando el histórico partido a partido, en orden cronológico."""
        for _, row in matches.iterrows():
            self.update(row["home_team"], row["away_team"], row["home_score"], row["away_score"])
        return self

    def top_teams(self, n: int = 10) -> pd.Series:
        return pd.Series(self.ratings).sort_values(ascending=False).head(n)


if __name__ == "__main__":
    from src.data_loader import load_processed

    matches = load_processed()
    model = EloModel().fit(matches)

    print("Top 10 selecciones por rating Elo:")
    print(model.top_teams(10))

    print("\nEjemplo de predicción:")
    print(model.predict("Brazil", "Argentina"))
