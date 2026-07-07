"""FastAPI service that serves the promoted model from the MLflow registry.

The champion model is loaded once at startup from
``models:/wine_classifier@champion``. Rolling out a new model is a registry
alias move - no code change and no redeploy required.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CANONICAL_COLUMNS, PredictRequest, PredictResponse
from src.config import CHAMPION_ALIAS, REGISTERED_MODEL, TRACKING_URI

MODEL_URI = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"

state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:  # noqa: BLE001 - surface load failure at /health
        print(f"[api] could not load {MODEL_URI}: {exc}")
        state["model"] = None
    yield
    state.clear()


app = FastAPI(title="Wine Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    frame = pd.DataFrame(
        [s.to_row() for s in request.samples], columns=CANONICAL_COLUMNS
    )
    preds = model.predict(frame)
    return PredictResponse(
        predictions=[int(p) for p in preds],
        model_alias=CHAMPION_ALIAS,
    )
