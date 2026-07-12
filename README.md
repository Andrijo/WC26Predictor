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

## Roadmap diario

- [x] **Day 1** — Modelo Elo básico
- [x] **Day 2** — Modelo Poisson de goles
- [x] **Day 3** — Ensemble (Elo + Poisson) + dashboard interactivo con Streamlit
- [ ] **Day 4+** — ideas futuras: simulación Monte Carlo del torneo completo,
      ajuste por local/visitante/neutral, ponderar partidos recientes, etc.

## Fuente de datos

Partidos internacionales históricos (ej. dataset de Mark Rowan /
"International football results from 1872 to 2024" en Kaggle, o cualquier
fuente equivalente vía API/CSV).
