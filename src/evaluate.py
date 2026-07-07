"""Evaluate stage: score the best run and explain it with SHAP.

Reads the hand-off produced by the train stage, loads the packaged model back
from MLflow, and writes DVC-tracked outputs: a metrics file, a confusion matrix
and a SHAP feature-importance summary.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from src.config import (  # noqa: E402
    BEST_RUN_JSON,
    CONFUSION_PNG,
    METRICS_JSON,
    SHAP_PNG,
    TRACKING_URI,
    ensure_dirs,
    load_params,
)
from src.data import load_test, load_train  # noqa: E402


def _shap_summary(model, X_train, X_test, sample_size: int) -> bool:
    """Model-agnostic SHAP summary. Returns False if SHAP is unavailable."""
    try:
        import shap
    except ImportError:
        print("[evaluate] shap not installed - skipping explainability plot")
        return False

    if not hasattr(model, "predict_proba"):
        print("[evaluate] best model has no predict_proba - skipping SHAP")
        return False

    background = shap.sample(X_train, min(50, len(X_train)), random_state=0)
    sample = X_test.head(sample_size)
    explainer = shap.Explainer(model.predict_proba, background)
    shap_values = explainer(sample)

    plt.figure()
    shap.summary_plot(shap_values[..., 1], sample, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] SHAP summary -> {SHAP_PNG}")
    return True


def main() -> None:
    ensure_dirs()
    params = load_params()["evaluate"]
    mlflow.set_tracking_uri(TRACKING_URI)

    with open(BEST_RUN_JSON, "r", encoding="utf-8") as fh:
        best = json.load(fh)

    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")

    X_train, _ = load_train()
    X_test, y_test = load_test()
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "best_run_name": best["run_name"],
        "best_run_id": best["run_id"],
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[evaluate] metrics -> {METRICS_JSON} "
          f"(acc={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f})")

    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=120)
    plt.close()
    print(f"[evaluate] confusion matrix -> {CONFUSION_PNG}")

    _shap_summary(model, X_train, X_test, params["shap_sample_size"])


if __name__ == "__main__":
    main()
