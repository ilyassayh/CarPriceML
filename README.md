## CarPriceML — Estimation du prix de voitures d’occasion

Projet pédagogique complet: pipeline ML, API FastAPI, UI Streamlit, et déploiement Docker.

### Fonctionnalités
- Prétraitement et entraînement d’un `RandomForestRegressor` (OneHotEncoder + StandardScaler)
- Sauvegarde d’un pipeline complet `joblib`
- API FastAPI: `/health`, `/predict`
- Frontend Streamlit: formulaire, appel API, affichage du prix estimé
- Docker Compose: backend + frontend
- Tests Pytest: endpoints `/health` et `/predict`

### Structure du projet
```
.
├─ app/
│  ├─ api/
│  │  ├─ __init__.py
│  │  └─ main.py
│  └─ frontend/
│     └─ streamlit_app.py
├─ models/
│  └─ .gitkeep
├─ pipeline/
│  ├─ __init__.py
│  └─ train.py
├─ tests/
│  └─ test_api.py
├─ Dockerfile            (backend)
├─ Dockerfile.frontend   (frontend)
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

### Prérequis
- Python 3.11+
- Pip / virtualenv (ou conda)
- Docker & Docker Compose (pour déploiement)

### Données
Spécifiez le chemin du CSV. Exemple fourni par l’utilisateur:
`C:\\Users\\Ilyass\\OneDrive - SUP RH Ecole Supérieure de Management et de Gestion des Ressources Humaines\\Desktop\\CarPriceML\\car-details.csv`

Note devise: si les prix ne sont pas en MAD, utilisez `--currency-rate` pour convertir.

### Installation locale
```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### Entraînement
```bash
.\.venv\Scripts\python pipeline/train.py \
  --csv "C:\\Users\\Ilyass\\OneDrive - SUP RH Ecole Supérieure de Management et de Gestion des Ressources Humaines\\Desktop\\CarPriceML\\car-details.csv" \
  --target price \
  --test-size 0.3 \
  --currency-rate 1.0 \
  --out-model models/rf_model.joblib \
  --out-meta models/metadata.json
```

Le script détecte automatiquement variables numériques et catégorielles (dtype object). Les features sont standardisées/encodées via `ColumnTransformer`.

### Lancement API (local)
```bash
.\.venv\Scripts\uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Lancement frontend (local)
```bash
set API_URL=http://localhost:8000
.\.venv\Scripts\streamlit run app/frontend/streamlit_app.py
```

### Docker
Construire et lancer:
```bash
docker compose up --build
```

L’API sera accessible sur `http://localhost:8000`, le frontend sur `http://localhost:8501`.

Assurez-vous que `models/` contient `rf_model.joblib` et `metadata.json` (générés par l’entraînement). Vous pouvez monter le CSV local si vous souhaitez réentraîner dans un conteneur.

### Tests
```bash
.\.venv\Scripts\pytest -q
```

### Endpoints
- GET `/health`: statut du service
- POST `/predict`: JSON d’attributs véhicule → prix estimé `{ "price": float }`

### Notes techniques
- Le pipeline sauvegardé inclut prétraitements + modèle pour une prédiction reproductible.
- La conversion de devise se fait à l’entraînement via `--currency-rate`.
- `metadata.json` contient la liste des colonnes d’entrée attendues.


