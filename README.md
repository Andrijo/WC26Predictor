# World Cup 2026 Match Predictor

Proyecto de práctica diaria en Python: predicción de partidos del Mundial 2026
usando datos históricos de más de 49,000 partidos internacionales desde 1872.

## Estructura del proyecto

```
worldcup2026-predictor/
├── data/
│   ├── raw/              # datos crudos descargados
│   └── processed/        # datos limpios listos para modelar
├── src/
│   ├── data_loader.py    # descarga y limpieza de datos
│   ├── elo.py             # Day 1 - modelo Elo
│   ├── poisson_model.py   # Day 2 - modelo de goles Poisson (Dixon-Coles)
│   ├── ensemble.py        # Day 3 - combina Elo + Poisson
│   ├── league_config.py   # configuración por liga (World Cup / Liga MX)
│   ├── backtest.py        # validación walk-forward fuera de muestra
│   ├── liguilla.py        # simulador de liguilla Monte Carlo (Liga MX)
│   ├── model_cache.py     # persistencia de modelos entrenados en disco
│   └── utils.py           # utilidades generales
├── app/
│   └── dashboard.py       # interfaz interactiva con Streamlit
├── tests/                 # tests unitarios
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### 1. Descargar y preparar los datos

```bash
python -m src.data_loader
```

### 2. Ejecutar el dashboard interactivo

```bash
streamlit run app/dashboard.py
```

Si el comando `streamlit` no se reconoce (error `command not found` porque el
script no quedó en el `PATH`, algo común con `pip install --user` o en algunos
entornos virtuales), usa el módulo directamente con el mismo intérprete de
Python donde lo instalaste:

```bash
python -m streamlit run app/dashboard.py
```

Esto aplica a cualquier paquete con CLI (pytest, jupyter, etc.): si el
ejecutable no está en el `PATH`, casi siempre puedes invocarlo como módulo:

```bash
python -m pytest tests/ -v
python -m pip install -r requirements.txt
```

Para confirmar qué intérprete y qué paquetes está usando tu shell:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show streamlit
```

### 3. Backtesting walk-forward

```bash
python -m src.backtest --league world_cup --start-year 2015
python -m src.backtest --league liga_mx --start-year 2021
```

Valida el modelo fuera de muestra usando walk-forward por año: entrena con todo
lo anterior al año Y, predice los partidos de Y, y repite. Calcula log-loss,
Brier score y accuracy comparando Elo, Poisson, ensemble y un baseline ingenuo.

### 4. Simulación de liguilla (Liga MX)

```bash
python -m src.liguilla
```

Simula la fase de eliminación directa de Liga MX (cuartos → semis → final)
usando Monte Carlo con la matriz de Poisson del ensemble. Devuelve la
probabilidad de campeonar de cada equipo clasificado.

### 5. Caché de modelos

El dashboard usa `model_cache.py` para persistir el modelo entrenado en disco.
La caché se invalida automáticamente si cambian los datos o los parámetros.

```bash
python -m src.model_cache   # benchmark: primera carga vs. desde caché
```

## Soporte multi-liga (World Cup / Liga MX)

El proyecto ahora soporta más de una competición mediante `src/league_config.py`.
Cada liga define su fuente de datos, nombres de columnas, ventaja de local,
altitud (Liga MX) y si tiene fase de liguilla.

```bash
# descargar y procesar datos de Liga MX
python -m src.data_loader --league liga_mx

# descargar y procesar datos de World Cup / internacional (default)
python -m src.data_loader --league world_cup
```

```python
from src.data_loader import load_processed
from src.league_config import LIGA_MX
from src.ensemble import EnsemblePredictor

matches = load_processed(LIGA_MX)
model = EnsemblePredictor(league=LIGA_MX).fit(matches)
model.predict("América", "Chivas")
```

## Fuente de datos

Partidos internacionales históricos (ej. dataset de Mark Rowan /
"International football results from 1872 to 2024" en Kaggle, o cualquier
fuente equivalente vía API/CSV).
