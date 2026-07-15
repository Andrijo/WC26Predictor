"""
data_loader.py
--------------
Descarga y limpia el histórico de partidos para cualquier liga soportada
en league_config.py (World Cup/internacional, Liga MX, etc.).

El dataframe de salida siempre queda normalizado a las columnas:
date, home_team, away_team, home_score, away_score, result
sin importar cómo se llamen las columnas en la fuente original.
"""

from pathlib import Path
import pandas as pd
import requests

from src.league_config import LeagueConfig, WORLD_CUP

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def download_data(url: str, filename: str) -> Path:
    """Descarga el CSV crudo y lo guarda en data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / filename

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    print(f"Datos descargados en: {out_path}")
    return out_path


def load_raw(league: LeagueConfig, filename: str | None = None) -> pd.DataFrame:
    """
    Carga el CSV crudo desde data/raw/, descargándolo si hace falta.
    Si league.data_source es una lista (multi-temporada), descarga cada URL
    por separado y concatena los resultados en un solo dataframe.
    """
    sources = league.data_source if isinstance(league.data_source, list) else [league.data_source]

    frames = []
    for i, url in enumerate(sources):
        part_filename = filename or f"{_safe_name(league.name)}_{i}.csv"
        path = RAW_DIR / part_filename
        if not path.exists():
            path = download_data(url, part_filename)
        frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "")


def clean_data(df: pd.DataFrame, league: LeagueConfig) -> pd.DataFrame:
    """
    Limpieza y normalización de columnas según la config de la liga.
    Renombra las columnas propias de cada fuente a un esquema común.
    """
    df = df.copy()

    rename_map = {
        league.date_col: "date",
        league.home_team_col: "home_team",
        league.away_team_col: "away_team",
        league.home_score_col: "home_score",
        league.away_score_col: "away_score",
    }
    df = df.rename(columns=rename_map)

    # Algunas fuentes (ej. footballcsv) traen el marcador como "2-1" en una sola
    # columna "FT" en vez de dos columnas separadas. Lo separamos si aplica.
    if "home_score" not in df.columns and "FT" in df.columns:
        scores = df["FT"].str.split("-", expand=True)
        df["home_score"] = scores[0]
        df["away_score"] = scores[1]

    # Partidos aún no jugados (marcador vacío, ej. jornadas futuras cacheadas
    # de antemano) se descartan junto con cualquier otro nulo.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    # Algunas fuentes anotan marcadores especiales (ej. "0 (*)" = partido dado
    # por perdido administrativamente, walkover, etc.). Extraemos solo el
    # número inicial; si no hay número, la fila se descarta.
    df["home_score"] = pd.to_numeric(
        df["home_score"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    df["away_score"] = pd.to_numeric(
        df["away_score"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["result"] = df.apply(
        lambda r: 1 if r["home_score"] > r["away_score"]
        else (-1 if r["home_score"] < r["away_score"] else 0),
        axis=1,
    )

    # Conservamos 'neutral' si la fuente la trae (ej. World Cup dataset).
    # Si no existe (ej. Liga MX, donde casi todo se juega en casa del local),
    # se asume False para todos los partidos.
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    else:
        df["neutral"] = False

    keep_cols = ["date", "home_team", "away_team", "home_score", "away_score", "result", "neutral"]
    return df[keep_cols].sort_values("date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Guarda el dataframe limpio en data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Datos procesados guardados en: {out_path}")
    return out_path


def load_processed(league: LeagueConfig = WORLD_CUP, filename: str | None = None) -> pd.DataFrame:
    """Carga el dataset ya limpio para la liga dada; si no existe, lo genera desde cero."""
    filename = filename or f"{league.name.lower().replace(' ', '_').replace('/', '')}_clean.csv"
    path = PROCESSED_DIR / filename
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    raw = load_raw(league)
    clean = clean_data(raw, league)
    save_processed(clean, filename)
    return clean


if __name__ == "__main__":
    import argparse
    from src.league_config import LEAGUES

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(LEAGUES), default="world_cup")
    args = parser.parse_args()

    league_cfg = LEAGUES[args.league]
    raw_df = load_raw(league_cfg)
    clean_df = clean_data(raw_df, league_cfg)
    save_processed(clean_df, f"{args.league}_clean.csv")
    print(clean_df.head())
    print(f"\nTotal de partidos: {len(clean_df):,}")
