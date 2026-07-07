"""FastAPI service that serves the promoted model from the MLflow registry.

Endpoints:
  GET  /                  single-page web UI (predict + explain)
  GET  /health            liveness + model status
  POST /predict           batch class predictions
  POST /explain           prediction + local SHAP feature attributions
  GET  /example/{label}   a real sample of the requested class (to prefill the UI)
  GET  /reports/{name}    pipeline-generated images (global SHAP, confusion matrix)

The champion model is loaded once at startup from
``models:/wine_classifier@champion``. Rolling out a new model is a registry
alias move - no code change and no redeploy required.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sklearn.datasets import load_wine

from api.schemas import (
    CANONICAL_COLUMNS,
    SAFE_FIELDS,
    ExplainResponse,
    FeatureContribution,
    PredictRequest,
    PredictResponse,
    WineSample,
)
from src.config import CHAMPION_ALIAS, REGISTERED_MODEL, REPORTS_DIR, TRACKING_URI

MODEL_URI = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
UI_HTML = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
ALLOWED_REPORTS = {"shap_summary.png", "confusion_matrix.png", "model_decay.png"}

state: dict = {"model": None, "sk_model": None, "explainer": None}


def _build_explainer(sk_model):
    """Prefer exact TreeExplainer; fall back to a model-agnostic explainer."""
    import shap

    try:
        return shap.TreeExplainer(sk_model)
    except Exception:  # noqa: BLE001 - pipelines / non-tree models
        frame = load_wine(as_frame=True).frame[CANONICAL_COLUMNS]
        background = shap.sample(frame, min(40, len(frame)), random_state=0)
        return shap.Explainer(sk_model.predict_proba, background)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    # Cache one real sample per class for the UI's preset buttons.
    frame = load_wine(as_frame=True).frame
    state["examples"] = {
        int(label): dict(zip(SAFE_FIELDS,
                             frame[frame["target"] == label].iloc[0][CANONICAL_COLUMNS]))
        for label in sorted(frame["target"].unique())
    }
    try:
        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        state["sk_model"] = mlflow.sklearn.load_model(MODEL_URI)
        state["explainer"] = _build_explainer(state["sk_model"])
    except Exception as exc:  # noqa: BLE001 - surfaced at /health
        print(f"[api] could not load {MODEL_URI}: {exc}")
    yield
    state.clear()


app = FastAPI(title="Wine Classifier API", version="1.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state.get("model") is not None,
            "explain_ready": state.get("explainer") is not None,
            "model_uri": MODEL_URI}


@app.get("/example/{label}")
def example(label: int) -> dict:
    examples = state.get("examples", {})
    if label not in examples:
        raise HTTPException(status_code=404, detail="unknown class label")
    return {k: float(v) for k, v in examples[label].items()}


@app.get("/reports/{name}")
def report(name: str) -> FileResponse:
    if name not in ALLOWED_REPORTS:
        raise HTTPException(status_code=404, detail="unknown report")
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not generated yet")
    return FileResponse(path, media_type="image/png")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame([s.to_row() for s in request.samples], columns=CANONICAL_COLUMNS)
    preds = model.predict(frame)
    return PredictResponse(predictions=[int(p) for p in preds], model_alias=CHAMPION_ALIAS)


@app.post("/explain", response_model=ExplainResponse)
def explain(sample: WineSample) -> ExplainResponse:
    sk_model = state.get("sk_model")
    explainer = state.get("explainer")
    if sk_model is None or explainer is None:
        raise HTTPException(status_code=503, detail="model/explainer not loaded")

    frame = pd.DataFrame([sample.to_row()], columns=CANONICAL_COLUMNS)
    proba = sk_model.predict_proba(frame)[0]
    pred = int(proba.argmax())

    # Unified SHAP call -> values shaped (1, n_features, n_classes) for multiclass.
    values = explainer(frame).values
    per_feature = values[0, :, pred] if values.ndim == 3 else values[0]

    contributions = sorted(
        (FeatureContribution(feature=CANONICAL_COLUMNS[i],
                             value=float(frame.iloc[0, i]),
                             shap=float(per_feature[i]))
         for i in range(len(CANONICAL_COLUMNS))),
        key=lambda c: abs(c.shap), reverse=True,
    )
    return ExplainResponse(
        prediction=pred,
        probabilities=[float(p) for p in proba],
        contributions=contributions,
        model_alias=CHAMPION_ALIAS,
    )
