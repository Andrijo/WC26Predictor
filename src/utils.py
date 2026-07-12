"""
Funciones auxiliares compartidas entre módulos.
"""

import pandas as pd


def get_team_list(matches: pd.DataFrame) -> list[str]:
    """Lista ordenada de todos los equipos presentes en el dataset."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    return sorted(teams.tolist())


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    """Historial de enfrentamientos directos entre dos equipos."""
    mask = (
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b))
        | ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    )
    return matches[mask].sort_values("date", ascending=False)


def recent_form(matches: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Últimos n partidos jugados por un equipo (local o visitante)."""
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    return matches[mask].sort_values("date", ascending=False).head(n)
