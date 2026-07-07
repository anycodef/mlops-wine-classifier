"""Central configuration: paths, names and parameter loading.

Every script imports from here so that the tracking URI, experiment name and
registered-model name are defined in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# --- Repository layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
BEST_RUN_JSON = ARTIFACTS_DIR / "best_run.json"

METRICS_JSON = REPORTS_DIR / "metrics.json"
CONFUSION_PNG = REPORTS_DIR / "confusion_matrix.png"
SHAP_PNG = REPORTS_DIR / "shap_summary.png"
DECAY_PNG = REPORTS_DIR / "model_decay.png"

TARGET = "target"

# --- MLflow identifiers ------------------------------------------------------
# Override the tracking URI via env var so the same code works locally
# (SQLite file) and against a remote MLflow server (docker-compose / CI).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = "wine_classification"
REGISTERED_MODEL = "wine_classifier"
CHAMPION_ALIAS = "champion"


def load_params() -> dict:
    """Load params.yaml as a plain dict."""
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
