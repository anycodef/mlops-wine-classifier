# Demo — MLOps de punta a punta (≈ 2 min)

Guía de mano. Sigue el orden. Cada paso: **dónde · qué hacer · qué decir**.

---

## Antes de empezar (fuera del cronómetro)

```bash
cd /home/darkuser/Documents/anycodef/mlflow
source .venv/bin/activate

docker compose up -d mlflow
until curl -sf http://localhost:5000/health >/dev/null; do sleep 1; done

docker compose up -d api
sleep 3
curl -s http://localhost:8000/health          # -> "model_loaded":true
```

Deja **3 pestañas** abiertas en el navegador:

- MLflow UI → http://localhost:5000
- Web de predicción → http://localhost:8000/
- Swagger (API docs) → http://localhost:8000/docs

Y una **terminal** con el venv activo.

---

## Los 2 minutos

### 1 · Componentes  ·  *terminal*  ·  ~0:00
```bash
docker compose ps
```
> "Dos componentes, contenedorizados: el **servidor MLflow** y la **API REST**."

### 2 · Pipeline reproducible (DVC)  ·  *terminal*  ·  ~0:15
```bash
dvc dag
```
> "Un **pipeline reproducible**: prepara → entrena → evalúa. Con `dvc repro` se reejecuta solo lo que cambió."

### 3 · Experimentos (MLflow tracking)  ·  *navegador :5000*  ·  ~0:35
Experimento **`wine_classification`** → selecciona los 4 runs → **Compare**.
> "Cada entrenamiento queda **trazado** con sus parámetros y métricas. Comparo y elijo el mejor. Nada se pierde en un notebook."

### 4 · Registro de modelos (MLflow Registry)  ·  *navegador :5000 → Models*  ·  ~0:55
**`wine_classifier`** → alias **`@champion`**.
> "El mejor modelo, **versionado** y promovido con el alias `champion`, aparte del código."

### 5 · Servir + explicar  ·  *web :8000/*  ·  ~1:10
**Ejemplo Clase 1** → **Predecir y explicar**.
> "La API sirve ese `champion`. No solo predice: **explica con SHAP** por qué. Abajo, la importancia global del modelo."

### 6 · API REST  ·  *Swagger :8000/docs*  ·  ~1:40
Muestra `POST /predict` y `POST /explain`.
> "Todo expuesto como **API REST**, lista para integrarse en cualquier sistema."

### Cierre  ·  ~1:50
> "De un experimento a un servicio **trazable, versionado, reproducible y explicable**. Eso es MLOps."

---

## Extra (solo si sobra tiempo)

**Rollback de modelo** *(terminal)* — mover el alias, sin tocar código:
```bash
python -c "import os; os.environ['MLFLOW_TRACKING_URI']='http://localhost:5000'; \
from mlflow import MlflowClient; MlflowClient().set_registered_model_alias('wine_classifier','champion',1)"
docker compose restart api
```

**Monitoreo / degradación** *(terminal)*:
```bash
python -m src.monitor
```

---

## Si algo falla

- `model_loaded: false` → `docker compose restart api`
- Terminal sin `dvc`/`python` → `source .venv/bin/activate`
- Apagar al terminar (conserva el modelo) → `docker compose down`
