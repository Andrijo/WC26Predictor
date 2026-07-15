""""
Configuración por liga. Permite que el mismo motor
se reutilice para distintos contextos.
"""

from dataclasses import dataclass, field


# Altitud aproximada (metros sobre el nivel del mar) de las sedes de equipos
# de Liga MX que juegan en altitud significativa. Se usa como bonus extra
# de local, más allá del home_advantage genérico.
# IMPORTANTE: los nombres deben coincidir EXACTO con los que trae la fuente
# de datos (footballcsv/cache.wfb usa nombres como "CF América", no "América").
# Verifica con get_team_list(matches) tras cargar los datos.
LIGA_MX_ALTITUDE_TEAMS = {
    "Deportivo Toluca": 2680,
    "Cruz Azul": 2240,           # CDMX
    "CF América": 2240,          # CDMX
    "Pumas UNAM": 2240,          # CDMX
    "CF Pachuca": 2426,
    "Club León": 1815,
    "Atlas Guadalajara": 1566,   # Guadalajara
    "Deportivo Guadalajara": 1566,  # Chivas, Guadalajara
}


@dataclass
class LeagueConfig:
    name: str
    data_source: str | list[str]    
    date_col: str = "date"
    home_team_col: str = "home_team"
    away_team_col: str = "away_team"
    home_score_col: str = "home_score"
    away_score_col: str = "away_score"
    home_advantage: float = 60       # bonus base de Elo por jugar de local
    altitude_teams: dict = field(default_factory=dict)  # equipo -> metros
    altitude_bonus_per_1000m: float = 25  # puntos Elo extra por cada 1000m de altitud
    has_playoffs: bool = False        # liguilla / playoffs tipo bracket
    season_split: bool = False        # ligas con Apertura/Clausura (temporadas cortas)


WORLD_CUP = LeagueConfig(
    name="World Cup / Internacional",
    data_source="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    home_advantage=60,
    has_playoffs=False,
    season_split=False,
)

# footballcsv/cache.wfb no tiene un CSV único con todo el histórico de Liga MX:
# cada archivo mx.1.csv cubre UN torneo (Apertura o Clausura) de UNA temporada,
# y solo hay cobertura consistente desde 2020-21. Por eso combinamos varias
# temporadas en una lista; data_loader.py las concatena automáticamente.
_LIGA_MX_BASE = "https://raw.githubusercontent.com/footballcsv/cache.wfb/master"
LIGA_MX_SEASONS = [
    f"{_LIGA_MX_BASE}/2020-21/mx.1.csv",
    f"{_LIGA_MX_BASE}/2021-22/mx.1.csv",
    f"{_LIGA_MX_BASE}/2022-23/mx.1.csv",
    f"{_LIGA_MX_BASE}/2023-24/mx.1.csv",
    f"{_LIGA_MX_BASE}/2024-25/mx.1.csv",
    # Nota: mx.2.csv en este repo es Liga de Expansión MX (segunda división).
]

LIGA_MX = LeagueConfig(
    name="Liga MX",
    data_source=LIGA_MX_SEASONS,
    date_col="Date",
    home_team_col="Team 1",
    away_team_col="Team 2",
    home_advantage=45,               # el "local" pesa distinto en formato de liga vs. seleccionados
    altitude_teams=LIGA_MX_ALTITUDE_TEAMS,
    altitude_bonus_per_1000m=25,
    has_playoffs=True,
    season_split=True,
)

LEAGUES = {
    "world_cup": WORLD_CUP,
    "liga_mx": LIGA_MX,
}


def get_league(key: str) -> LeagueConfig:
    if key not in LEAGUES:
        raise ValueError(f"Liga desconocida: {key}. Opciones: {list(LEAGUES)}")
    return LEAGUES[key]
