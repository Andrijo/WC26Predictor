"""
Descarga y limpia el histórico de partidos internacionales.
"""

from pathlib import Path
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def download_data(url: str = DATA_URL, filename: str = "results.csv") -> Path:
    """Descarga el CSV crudo y lo guarda en data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / filename

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    print(f"Datos descargados en: {out_path}")
    return out_path


def load_raw(filename: str = "results.csv") -> pd.DataFrame:
    """Carga el CSV crudo desde data/raw/."""
    path = RAW_DIR / filename
    if not path.exists():
        path = download_data(filename=filename)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: tipos de dato, columnas relevantes, nulos."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Columna auxiliar de resultado: 1 = gana local, 0 = empate, -1 = gana visitante
    df["result"] = df.apply(
        lambda r: 1 if r["home_score"] > r["away_score"]
        else (-1 if r["home_score"] < r["away_score"] else 0),
        axis=1,
    )

    return df.sort_values("date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str = "matches_clean.csv") -> Path:
    """Guarda el dataframe limpio en data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Datos procesados guardados en: {out_path}")
    return out_path


def load_processed(filename: str = "matches_clean.csv") -> pd.DataFrame:
    """Carga el dataset ya limpio; si no existe, lo genera desde cero."""
    path = PROCESSED_DIR / filename
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    raw = load_raw()
    clean = clean_data(raw)
    save_processed(clean, filename)
    return clean


if __name__ == "__main__":
    raw_df = load_raw()
    clean_df = clean_data(raw_df)
    save_processed(clean_df)
    print(clean_df.head())
    print(f"\nTotal de partidos: {len(clean_df):,}")
