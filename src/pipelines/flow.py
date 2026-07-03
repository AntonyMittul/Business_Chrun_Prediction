"""Prefect orchestration: the full retrain pipeline as one flow.

Prefect over Airflow here because it runs natively on Windows (Airflow needs
Docker/WSL2) and the DAG is code, not YAML. Same orchestration concepts:
tasks, dependencies, retries, observability.

Run once:            python -m src.pipelines.flow
Inspect runs (UI):   prefect server start   ->  http://127.0.0.1:4200
"""
from __future__ import annotations

from prefect import flow, task


@task(retries=1)
def clean():
    from src.data.clean_data import main

    main()


@task(retries=1)
def features():
    from src.features.build_features import main

    main()


@task
def train():
    from src.models.train import main

    main()


@task
def drift_report():
    from src.monitoring.drift import run_drift_report

    return run_drift_report()


@flow(name="churn-retrain", log_prints=True)
def retrain_pipeline():
    """clean -> features -> train -> drift, with explicit dependencies."""
    c = clean(return_state=True)
    f = features(wait_for=[c], return_state=True)
    train(wait_for=[f])
    drift_report(wait_for=[f])


if __name__ == "__main__":
    retrain_pipeline()
