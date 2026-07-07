# Architecture & design notes

This document explains *why* the project is split across tools and how the
hand-offs work. For the quickstart and commands, see the [README](../README.md).

## The three things that get versioned

A machine-learning project mixes three artifacts with very different lifecycles.
Trying to version all of them in Git fails: data and models are large and binary,
and experiment runs are not files at all.

| Artifact | Lifecycle owner | In Git? |
| --- | --- | --- |
| Code | Git / GitHub | yes |
| Data & pipeline outputs | DVC | only the `.dvc` / `dvc.lock` pointers |
| Experiments & packaged models | MLflow | no — lives in the tracking backend |

## Hand-offs between stages

The pipeline is a chain of small scripts wired together by `dvc.yaml`. Each
hand-off is an explicit, inspectable artifact:

1. **prepare → train** — `data/processed/{train,test}.csv` (DVC-tracked output).
2. **train → evaluate / register** — `artifacts/best_run.json`, containing the
   winning run id and its metrics. This decouples "which run won" from MLflow's
   internal run identifiers.
3. **register → serve** — the registry alias `wine_classifier@champion`. The API
   only knows this URI; it has no idea which run or version currently backs it.

Because the hand-offs are explicit, any stage can be re-run in isolation and the
tools never reach into each other's internals.

## Why DVC when the data comes from scikit-learn?

The Wine dataset ships inside scikit-learn, so there is no external raw file to
version. DVC still earns its place: it turns the scripts into a **dependency
graph** (`dvc repro` only re-runs stages whose code, params or inputs changed)
and it versions the *outputs* of that graph. On a project with real, external
data, the same `dvc.yaml` would additionally track the raw dataset via a
`.dvc` pointer and a remote.

## MLflow: tracking vs. registry

- **Tracking** answers "what did each experiment do?" — every training run logs
  its parameters, metrics and a packaged model artifact, comparable in the UI.
- **Registry** answers "which model is live?" — the best run is registered as a
  named model and promoted with the `champion` alias.

### Stages vs. aliases

MLflow model *stages* (`Staging`/`Production`) are deprecated since MLflow 2.9 in
favour of **aliases** and **tags**. This project uses an alias (`champion`):

- Promotion is `set_registered_model_alias(..., version=N)`.
- Rollback is the same call pointing back at the previous version.
- Serving code (`models:/wine_classifier@champion`) never changes.

This gives instant, code-free promotion and rollback — the operational property
that makes a registry worth having.

## Reproducibility model

`prepare` is deterministic (fixed `random_state`), so the split is bit-for-bit
reproducible. `train` intentionally creates *new* MLflow runs each execution, so
`best_run.json` (and therefore `dvc.lock`) changes between runs — this is
expected. CI does not assert byte-identical outputs; it asserts that the whole
pipeline **still runs** and the tests still pass.

## Serving topology

Locally the API reads the model straight from the SQLite-backed store. In the
containerized setup (`docker-compose.yml`) an MLflow server runs with
`--serve-artifacts`, so the API downloads the model over HTTP instead of relying
on shared file paths — the same pattern used when the tracking server and the
serving service live on different hosts.

## How this maps to a larger system

The `prepare → train → evaluate → register → serve → monitor` flow is the same
one production teams run at scale; only the components grow: an orchestrator
(Airflow, Kubeflow) replaces `dvc repro`, a model server (KServe, Seldon)
replaces the FastAPI container, and monitoring reads real traffic metrics instead
of a simulated series. The responsibilities — and the hand-offs — stay identical.
