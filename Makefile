.PHONY: help setup repro ui register predict monitor serve test docker-up docker-down clean

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

repro:  ## Run the reproducible pipeline (prepare -> train -> evaluate)
	dvc repro

ui:  ## Launch the MLflow Tracking UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

register:  ## Register and promote the best run (@champion)
	$(PYTHON) -m src.register_model

predict:  ## Run the batch prediction demo against @champion
	$(PYTHON) -m src.predict

monitor:  ## Simulate post-deployment model decay
	$(PYTHON) -m src.monitor

serve:  ## Serve the champion model with FastAPI at http://localhost:8000
	uvicorn api.main:app --reload

test:  ## Run the unit tests
	$(PYTHON) -m pytest -q

docker-up:  ## Start the MLflow server + API stack
	docker compose up -d --build

docker-down:  ## Stop the Docker stack
	docker compose down

clean:  ## Remove generated artifacts and local tracking data
	rm -rf data reports artifacts mlruns mlartifacts mlflow.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
