"""
ensemble.py
-----------
Day 3: Combina el modelo Elo (Day 1) y el modelo Poisson (Day 2)
en una predicción única, ponderando ambos enfoques.
"""

import pandas as pd

from src.elo import EloModel
from src.poisson_model import PoissonGoalModel
from src.league_config import LeagueConfig, WORLD_CUP


class EnsemblePredictor:
    def __init__(
        self,
        elo_weight: float = 0.5,
        poisson_weight: float = 0.5,
        league: LeagueConfig = WORLD_CUP,
    ):
        assert abs(elo_weight + poisson_weight - 1.0) < 1e-6, "Los pesos deben sumar 1"
        self.elo_weight = elo_weight
        self.poisson_weight = poisson_weight
        self.league = league
        self.elo_model = EloModel(
            home_advantage=league.home_advantage,
            altitude_teams=league.altitude_teams,
            altitude_bonus_per_1000m=league.altitude_bonus_per_1000m,
        )
        self.poisson_model = PoissonGoalModel()

    def fit(self, matches: pd.DataFrame):
        self.elo_model.fit(matches)
        self.poisson_model.fit(matches)
        return self

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> dict:
        """neutral=True para partidos en sede neutral (ej. semifinales de un Mundial,
        como Argentina vs Inglaterra en Atlanta o Francia vs España en Texas)."""
        elo_pred = self.elo_model.predict(home_team, away_team, neutral=neutral)
        poisson_pred = self.poisson_model.predict(home_team, away_team, neutral=neutral)

        combined = {
            "home_win": round(
                self.elo_weight * elo_pred["home_win"] + self.poisson_weight * poisson_pred["home_win"], 4
            ),
            "draw": round(
                self.elo_weight * elo_pred["draw"] + self.poisson_weight * poisson_pred["draw"], 4
            ),
            "away_win": round(
                self.elo_weight * elo_pred["away_win"] + self.poisson_weight * poisson_pred["away_win"], 4
            ),
        }

        return {
            "ensemble": combined,
            "elo": elo_pred,
            "poisson": poisson_pred,
            "most_likely_score": poisson_pred["most_likely_score"],
        }


if __name__ == "__main__":
    from src.data_loader import load_processed

    matches = load_processed()
    model = EnsemblePredictor().fit(matches)

    result = model.predict("Brazil", "Argentina")
    print(result)
