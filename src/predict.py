"""Batch prediction demo: consume the promoted model from the registry.

Loads ``models:/wine_classifier@champion`` as a generic pyfunc model and scores
a handful of held-out samples, printing predictions next to the true labels.
"""
from __future__ import annotations

import mlflow

from src.config import CHAMPION_ALIAS, REGISTERED_MODEL, TRACKING_URI
from src.data import load_test

N_SAMPLES = 20


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
    )

    X_test, y_test = load_test()
    sample = X_test.head(N_SAMPLES)
    preds = model.predict(sample)

    print("Predictions:", list(map(int, preds)))
    print("True labels:", list(map(int, y_test.head(N_SAMPLES))))


if __name__ == "__main__":
    main()
