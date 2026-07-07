"""Deploy stage: register the best run and promote it with the champion alias.

MLflow model *stages* (Staging/Production) are deprecated since MLflow 2.9, so
this project uses the current mechanism: a **registered model version** plus an
**alias** (``champion``). Serving code loads ``models:/wine_classifier@champion``,
which makes promotion and rollback a one-line alias move.
"""
from __future__ import annotations

import json

import mlflow
from mlflow import MlflowClient

from src.config import (
    BEST_RUN_JSON,
    CHAMPION_ALIAS,
    REGISTERED_MODEL,
    TRACKING_URI,
)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    with open(BEST_RUN_JSON, "r", encoding="utf-8") as fh:
        best = json.load(fh)

    result = mlflow.register_model(
        model_uri=f"runs:/{best['run_id']}/model",
        name=REGISTERED_MODEL,
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias=CHAMPION_ALIAS,
        version=result.version,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL,
        version=result.version,
        key="source_run_name",
        value=best["run_name"],
    )

    print(f"[register] {REGISTERED_MODEL} v{result.version} "
          f"<- run {best['run_name']} ({best['run_id']})")
    print(f"[register] alias '@{CHAMPION_ALIAS}' -> v{result.version}")


if __name__ == "__main__":
    main()
