"""
Simulador de liguilla (fase de eliminación directa a ida y vuelta) para
Liga MX. 

Usa el EnsemblePredictor ya entrenado para simular cada serie ida/vuelta
y hacer avanzar a los equipos por el bracket.
"""

import random
from src.ensemble import EnsemblePredictor


def simulate_single_leg(model: EnsemblePredictor, home: str, away: str) -> tuple[int, int]:
    """Simula el marcador de un partido usando la matriz de Poisson del ensemble."""
    matrix = model.poisson_model.score_matrix(home, away)
    flat_probs = matrix.flatten()
    flat_probs = flat_probs / flat_probs.sum()  # normalizar por seguridad

    max_goals = model.poisson_model.max_goals + 1
    idx = random.choices(range(len(flat_probs)), weights=flat_probs, k=1)[0]
    home_goals, away_goals = divmod(idx, max_goals)
    return home_goals, away_goals


def simulate_series(model: EnsemblePredictor, team_a: str, team_b: str) -> str:
    """
    Simula una serie de ida y vuelta (formato Liga MX: el mejor posicionado
    define local en la vuelta). Si hay empate en el global, se define por
    goles de visitante y, si persiste, por penales (coin-flip ponderado).
    """
    # Ida: team_b de local; Vuelta: team_a de local (mejor posicionado juega la vuelta en casa)
    away_leg_a, away_leg_b = simulate_single_leg(model, team_b, team_a)  # ida
    home_leg_a, home_leg_b = simulate_single_leg(model, team_a, team_b)  # vuelta

    global_a = home_leg_a + away_leg_b
    global_b = home_leg_b + away_leg_a

    if global_a != global_b:
        return team_a if global_a > global_b else team_b

    # Desempate por gol de visitante (regla histórica en Liga MX hasta hace unos años)
    away_goals_a = away_leg_b   # goles de A jugando de visitante (en la ida)
    away_goals_b = home_leg_b   # goles de B jugando de visitante (en la vuelta)
    if away_goals_a != away_goals_b:
        return team_a if away_goals_a > away_goals_b else team_b

    # Penales: aproximamos con 50/50 (opcionalmente podrías ponderar por Elo)
    return random.choice([team_a, team_b])


def simulate_liguilla(model: EnsemblePredictor, seeded_teams: list[str], n_simulations: int = 1000) -> dict:
    """
    seeded_teams: lista de 8 equipos clasificados, en orden de posición
    (1ro al 8vo), como en el formato real de Liga MX (1 vs 8, 2 vs 7, etc.)

    Devuelve la probabilidad de campeonar de cada equipo tras N simulaciones.
    """
    assert len(seeded_teams) == 8, "La liguilla de Liga MX se juega con 8 equipos"

    champion_counts = {team: 0 for team in seeded_teams}

    for _ in range(n_simulations):
        # Cuartos de final: 1v8, 4v5, 3v6, 2v7 (seeding estándar)
        pairs = [
            (seeded_teams[0], seeded_teams[7]),
            (seeded_teams[3], seeded_teams[4]),
            (seeded_teams[2], seeded_teams[5]),
            (seeded_teams[1], seeded_teams[6]),
        ]
        semifinalists = [simulate_series(model, a, b) for a, b in pairs]

        finalists = [
            simulate_series(model, semifinalists[0], semifinalists[1]),
            simulate_series(model, semifinalists[2], semifinalists[3]),
        ]

        champion = simulate_series(model, finalists[0], finalists[1])
        champion_counts[champion] += 1

    return {team: round(count / n_simulations, 4) for team, count in champion_counts.items()}


if __name__ == "__main__":
    from src.data_loader import load_processed
    from src.league_config import LIGA_MX

    matches = load_processed(LIGA_MX)
    model = EnsemblePredictor(league=LIGA_MX).fit(matches)

    # Ejemplo: reemplaza por los 8 equipos clasificados reales, en orden de posición
    top8 = ["CF América", "Deportivo Toluca", "Cruz Azul", "CF Monterrey",
            "UANL Tigres", "CF Pachuca", "Deportivo Guadalajara", "Pumas UNAM"]
    probs = simulate_liguilla(model, top8, n_simulations=2000)

    print("Probabilidad de ser campeón:")
    for team, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {team}: {p:.1%}")
