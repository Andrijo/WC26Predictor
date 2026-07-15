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
│   ├── poisson_model.py   # Day 2 - modelo de goles Poisson
│   ├── ensemble.py        # Day 3 - combina Elo + Poisson
│   └── utils.py
├── notebooks/             # exploración y prototipos
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

## Roadmap diario

- [x] **Day 1** — Modelo Elo básico
- [x] **Day 2** — Modelo Poisson de goles
- [x] **Day 3** — Ensemble (Elo + Poisson) + dashboard interactivo con Streamlit
- [ ] **Day 4+** — ideas futuras: simulación Monte Carlo del torneo completo,
      ajuste por local/visitante/neutral, ponderar partidos recientes, etc.

### Fuente de datos de Liga MX: lo que encontramos al investigarla

El repo `footballcsv/cache.wfb` (el que puede que hayas visto en GitHub) **no
trae un solo CSV con todo el histórico**, a diferencia del dataset de World
Cup. Esto es lo que confirmamos revisándolo directamente:

- `mx.1.csv` = Liga MX (primera división), **un archivo por torneo/temporada**
- `mx.2.csv` = ⚠️ **no es el Clausura**, es la Liga de Expansión MX (segunda
  división) — equipos como Leones Negros, Tapatío, Venados FC
- Cobertura consistente solo **desde 2020-21 en adelante**; antes hay huecos
  (2019-20 falta por completo, probablemente por la pandemia)
- Algunos marcadores traen anotaciones especiales como `"0 (*)"` (partidos
  dados por perdido administrativamente); el loader ya los filtra
- Las últimas jornadas de la temporada más reciente pueden venir sin marcador
  (partidos aún no jugados al momento del scrape); el loader los descarta

`league_config.py` ya combina 5 temporadas (2020-21 a 2024-25, ~1,440
partidos) en `LIGA_MX_SEASONS`. Si quieres ampliar la ventana histórica,
tendrías que buscar otra fuente para años anteriores a 2020, o aceptar que
el Elo/Poisson solo tendrá memoria de los últimos 5 años.

**Importante sobre nombres de equipos:** esta fuente usa nombres como
`"CF América"`, `"Deportivo Toluca"`, `"Deportivo Guadalajara"` (no
`"América"`, `"Toluca"`, `"Chivas"`). El diccionario `LIGA_MX_ALTITUDE_TEAMS`
ya está ajustado a estos nombres exactos — si cambias de fuente, corre
`get_team_list(matches)` primero para verificar los nombres reales antes de
tocar ese diccionario.

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

### Diferencias que sí importan al adaptar a Liga MX

- **Altitud**: equipos como Toluca, Cruz Azul, América, Pumas, Pachuca juegan
  a gran altitud. `EloModel` acepta un bonus extra de local proporcional a la
  altitud (`altitude_teams`, `altitude_bonus_per_1000m`), aparte del
  `home_advantage` genérico.
- **Liguilla**: el campeón se define por bracket de eliminación directa
  (ida y vuelta), no por posición final. `src/liguilla.py` simula la fase
  final completa (cuartos → semis → final) usando la matriz de Poisson del
  ensemble ya entrenado, con miles de simulaciones Monte Carlo para estimar
  probabilidad de campeonar por equipo.
- **Temporadas cortas (Apertura/Clausura)**: a diferencia de partidos
  internacionales que abarcan décadas continuas, Liga MX se juega en dos
  torneos cortos por año. Si quieres que el Elo "resetee" parcialmente entre
  temporadas (common practice: regresión a la media entre Apertura y
  Clausura), puedes añadir una función que aplique un decay a los ratings
  al inicio de cada temporada nueva — no está implementado por defecto para
  mantener el modelo simple, pero es la extensión natural del Day 4+.

El dashboard (`app/dashboard.py`) incluye un selector de liga en la barra
lateral; al elegir una liga con `has_playoffs=True` (como Liga MX) aparece
automáticamente la sección de simulación de liguilla.

## Fuente de datos

Partidos internacionales históricos (ej. dataset de Mark Rowan /
"International football results from 1872 to 2024" en Kaggle, o cualquier
fuente equivalente vía API/CSV).
