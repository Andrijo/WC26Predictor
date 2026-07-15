"""
model_cache.py
---------------
Persiste el EnsemblePredictor ya entrenado en disco para que el dashboard
no tenga que reentrenar desde cero cada vez que arranca.

La caché se invalida automáticamente si cambian los datos de entrada
(comparamos un hash del contenido del CSV procesado) o los parámetros
del modelo (pesos del ensemble, liga), así nunca sirve un modelo obsoleto.
"""

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from src.ensemble import EnsemblePredictor
from src.league_config import LeagueConfig, WORLD_CUP

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "model_cache"


def _cache_key(matches: pd.DataFrame, league: LeagueConfig, elo_weight: float, poisson_weight: float) -> str:
    """Hash determinístico de los datos + parámetros, para invalidar la caché
    automáticamente si cualquiera de los dos cambia."""
    data_fingerprint = pd.util.hash_pandas_object(matches, index=False).sum()
    raw = f"{league.name}|{elo_weight}|{poisson_weight}|{data_fingerprint}|{len(matches)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_fit(
    matches: pd.DataFrame,
    league: LeagueConfig = WORLD_CUP,
    elo_weight: float = 0.5,
    poisson_weight: float = 0.5,
) -> EnsemblePredictor:
    """
    Carga el EnsemblePredictor desde caché en disco si existe uno válido para
    estos datos y parámetros exactos; si no, entrena uno nuevo y lo guarda
    para la próxima vez.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(matches, league, elo_weight, poisson_weight)
    cache_path = CACHE_DIR / f"{key}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    model = EnsemblePredictor(elo_weight=elo_weight, poisson_weight=poisson_weight, league=league).fit(matches)

    with open(cache_path, "wb") as f:
        pickle.dump(model, f)

    return model


def clear_cache():
    """Borra toda la caché de modelos (útil si algo se corrompe o para forzar reentreno)."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()


if __name__ == "__main__":
    import time
    from src.data_loader import load_processed

    matches = load_processed()

    clear_cache()
    t0 = time.time()
    load_or_fit(matches)
    print(f"Primera carga (entrena y guarda en caché): {time.time()-t0:.3f}s")

    t0 = time.time()
    load_or_fit(matches)
    print(f"Segunda carga (desde caché): {time.time()-t0:.3f}s")
