"""
poisson_model.py
-----------------
Day 2: Modelo de goles esperados (attack/defense strength) + distribución
de Poisson para simular marcadores exactos.

Incluye la corrección Dixon-Coles (1997): el Poisson "puro" asume que los
goles de local y visitante son independientes, pero en la práctica los
marcadores bajos (0-0, 1-0, 0-1, 1-1) están correlacionados de forma que el
modelo puro los sub/sobre-estima sistemáticamente. Dixon-Coles corrige esas
4 celdas con un factor tau(rho) ajustado por máxima verosimilitud sobre el
histórico real.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar


class PoissonGoalModel:
    def __init__(self, max_goals: int = 6, rho: float = 0.0):
        self.max_goals = max_goals
        self.rho = rho  # 0.0 = sin corrección Dixon-Coles (Poisson puro)
        self.avg_home_goals = None
        self.avg_away_goals = None
        self.team_attack: dict[str, float] = {}
        self.team_defense: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame, fit_rho: bool = True):
        """Calcula fuerza de ataque/defensa relativa a la media de la liga.
        Si fit_rho=True (default), también ajusta el parámetro Dixon-Coles
        rho por máxima verosimilitud sobre los marcadores bajos del histórico.

        Usa groupby() en vez de filtrar el dataframe completo por cada equipo
        (`matches[matches["home_team"] == team]` recorre las 49k+ filas una
        vez POR EQUIPO, ~300 pasadas completas al dataset). groupby() hace
        el mismo cálculo en una sola pasada, mismo resultado, mucho más rápido."""
        self.avg_home_goals = matches["home_score"].mean()
        self.avg_away_goals = matches["away_score"].mean()
        league_avg = (self.avg_home_goals + self.avg_away_goals) / 2

        # Goles a favor: como local (home_score) + como visitante (away_score)
        home_scored = matches.groupby("home_team")["home_score"].agg(["sum", "count"])
        away_scored = matches.groupby("away_team")["away_score"].agg(["sum", "count"])
        scored = home_scored.add(away_scored, fill_value=0)

        # Goles en contra: como local (away_score recibido) + como visitante (home_score recibido)
        home_conceded = matches.groupby("home_team")["away_score"].agg(["sum", "count"])
        away_conceded = matches.groupby("away_team")["home_score"].agg(["sum", "count"])
        conceded = home_conceded.add(away_conceded, fill_value=0)

        attack = (scored["sum"] / scored["count"]) / league_avg
        defense = (conceded["sum"] / conceded["count"]) / league_avg

        self.team_attack = attack.fillna(1.0).to_dict()
        self.team_defense = defense.fillna(1.0).to_dict()

        if fit_rho:
            self.fit_rho(matches)

        return self

    @staticmethod
    def _dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Factor de corrección de Dixon-Coles para las 4 celdas de marcador bajo.
        Para cualquier otro marcador, tau = 1 (sin corrección)."""
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit_rho(self, matches: pd.DataFrame) -> float:
        """
        Ajusta rho por máxima verosimilitud. Como tau=1 fuera de las 4 celdas
        de marcador bajo, la log-verosimilitud solo depende de rho a través de
        esos partidos — filtramos a ellos y optimizamos de forma vectorizada
        (rápido incluso con decenas de miles de partidos).
        """
        home_attack = matches["home_team"].map(self.team_attack).fillna(1.0).to_numpy()
        home_defense = matches["home_team"].map(self.team_defense).fillna(1.0).to_numpy()
        away_attack = matches["away_team"].map(self.team_attack).fillna(1.0).to_numpy()
        away_defense = matches["away_team"].map(self.team_defense).fillna(1.0).to_numpy()

        if "neutral" in matches.columns:
            neutral = matches["neutral"].to_numpy(dtype=bool)
        else:
            neutral = np.zeros(len(matches), dtype=bool)

        avg_neutral = (self.avg_home_goals + self.avg_away_goals) / 2
        lam = np.where(neutral, home_attack * away_defense * avg_neutral, home_attack * away_defense * self.avg_home_goals)
        mu = np.where(neutral, away_attack * home_defense * avg_neutral, away_attack * home_defense * self.avg_away_goals)

        x = matches["home_score"].to_numpy()
        y = matches["away_score"].to_numpy()

        mask = ((x == 0) & (y == 0)) | ((x == 0) & (y == 1)) | ((x == 1) & (y == 0)) | ((x == 1) & (y == 1))
        lam_m, mu_m, x_m, y_m = lam[mask], mu[mask], x[mask], y[mask]

        if len(lam_m) == 0:
            self.rho = 0.0
            return self.rho

        idx00 = (x_m == 0) & (y_m == 0)
        idx01 = (x_m == 0) & (y_m == 1)
        idx10 = (x_m == 1) & (y_m == 0)
        idx11 = (x_m == 1) & (y_m == 1)

        def neg_log_likelihood(rho: float) -> float:
            tau = np.ones_like(lam_m)
            tau[idx00] = 1 - lam_m[idx00] * mu_m[idx00] * rho
            tau[idx01] = 1 + lam_m[idx01] * rho
            tau[idx10] = 1 + mu_m[idx10] * rho
            tau[idx11] = 1 - rho
            tau = np.clip(tau, 1e-6, None)  # tau debe ser > 0 para poder sacar log
            return -np.sum(np.log(tau))

        result = minimize_scalar(neg_log_likelihood, bounds=(-0.3, 0.3), method="bounded")
        self.rho = float(result.x)
        return self.rho

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)

        if neutral:
            # En sede neutral no tiene sentido aplicar el sesgo "local mete más
            # goles que visitante" que el modelo aprendió de la liga regular.
            # Usamos el promedio general de goles por partido para ambos lados.
            avg_goals = (self.avg_home_goals + self.avg_away_goals) / 2
            exp_home_goals = home_attack * away_defense * avg_goals
            exp_away_goals = away_attack * home_defense * avg_goals
        else:
            exp_home_goals = home_attack * away_defense * self.avg_home_goals
            exp_away_goals = away_attack * home_defense * self.avg_away_goals

        return exp_home_goals, exp_away_goals

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = False) -> np.ndarray:
        """Matriz de probabilidad para cada marcador posible (home_goals x away_goals),
        con la corrección Dixon-Coles aplicada a las 4 celdas de marcador bajo si rho != 0."""
        exp_home, exp_away = self.expected_goals(home_team, away_team, neutral=neutral)

        home_probs = [poisson.pmf(i, exp_home) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(i, exp_away) for i in range(self.max_goals + 1)]

        matrix = np.outer(home_probs, away_probs)

        if self.rho != 0.0:
            for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                matrix[x, y] *= self._dixon_coles_tau(x, y, exp_home, exp_away, self.rho)
            matrix = matrix / matrix.sum()  # renormalizar: la corrección redistribuye masa

        return matrix

    def top_scores(self, home_team: str, away_team: str, neutral: bool = False, n: int = 3) -> list[tuple[str, float]]:
        """Devuelve los n marcadores más probables como [('1-1', 0.14), ('1-0', 0.11), ...].
        Más informativo que un solo 'marcador más probable': con goles esperados
        bajos (típico en fútbol), la moda de Poisson concentra mucha masa en 1-1
        aunque un equipo sea claramente favorito, así que un solo marcador
        aislado puede ser engañoso sobre quién es favorito."""
        matrix = self.score_matrix(home_team, away_team, neutral=neutral)
        flat_indices = np.argsort(matrix.ravel())[::-1][:n]
        scores = []
        for idx in flat_indices:
            h, a = np.unravel_index(idx, matrix.shape)
            scores.append((f"{h}-{a}", round(float(matrix[h, a]), 4)))
        return scores

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> dict:
        """Probabilidades 1X2, marcador más probable y top-3 de marcadores.
        neutral=True para partidos en sede neutral (ej. semifinales de un Mundial)."""
        matrix = self.score_matrix(home_team, away_team, neutral=neutral)

        p_home_win = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away_win = np.triu(matrix, 1).sum()

        most_likely_idx = np.unravel_index(matrix.argmax(), matrix.shape)
        top_scores = self.top_scores(home_team, away_team, neutral=neutral, n=3)

        return {
            "home_win": round(float(p_home_win), 4),
            "draw": round(float(p_draw), 4),
            "away_win": round(float(p_away_win), 4),
            "most_likely_score": f"{most_likely_idx[0]}-{most_likely_idx[1]}",
            "top_scores": top_scores,
        }


if __name__ == "__main__":
    from src.data_loader import load_processed

    matches = load_processed()
    model = PoissonGoalModel().fit(matches)

    print(f"Rho (Dixon-Coles) ajustado: {model.rho:.4f}")
    print("Predicción Brazil vs Argentina:")
    print(model.predict("Brazil", "Argentina"))
