"""Train stage: fit several model configurations and log each as an MLflow run.

Every configuration is trained inside its own MLflow run so that parameters,
metrics and the packaged model artifact are tracked and comparable in the
Tracking UI. The best run (by the configured selection metric) is written to
``artifacts/best_run.json`` as the hand-off to the evaluate / register stages.
"""
from __future__ import annotations

import json

import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    BEST_RUN_JSON,
    EXPERIMENT_NAME,
    TRACKING_URI,
    ensure_dirs,
    load_params,
)
from src.data import load_test, load_train

RANDOM_STATE = 42


def model_configs():
    """The four configurations compared in the experiment.

    RandomForest is unscaled (tree models are scale-invariant); the linear /
    kernel models are wrapped with a StandardScaler so the comparison is fair.
    """
    return [
        ("rf_10", RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE),
         {"model": "RandomForest", "n_estimators": 10}),
        ("rf_100", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
         {"model": "RandomForest", "n_estimators": 100}),
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
         {"model": "LogisticRegression", "max_iter": 1000}),
        ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
         {"model": "SVM", "kernel": "rbf", "C": 1.0}),
    ]


def main() -> None:
    ensure_dirs()
    params = load_params()["train"]
    selection_metric = params["selection_metric"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train = load_train()
    X_test, y_test = load_test()

    best = None
    for run_name, estimator, log_params in model_configs():
        with mlflow.start_run(run_name=run_name) as run:
            estimator.fit(X_train, y_train)
            preds = estimator.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1_macro": f1_score(y_test, preds, average="macro"),
            }

            mlflow.log_params(log_params)
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_test, preds)
            mlflow.sklearn.log_model(
                estimator,
                name="model",
                signature=signature,
                input_example=X_test.head(2),
            )

            score = metrics[selection_metric]
            print(f"[train] {run_name:8s} acc={metrics['accuracy']:.4f} "
                  f"f1_macro={metrics['f1_macro']:.4f}")

            candidate = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                **metrics,
            }
            if best is None or score > best[selection_metric]:
                best = candidate

    with open(BEST_RUN_JSON, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    print(f"[train] best run -> {best['run_name']} ({best['run_id']})")


if __name__ == "__main__":
    main()
