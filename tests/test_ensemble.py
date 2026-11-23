"""End-to-end tests for training ``EnsembleClassifier`` from the server.

The library module is intentionally left untouched: the sandbox prepares native
hyperparameters and one-hot targets, and the training layer saves the model in a
``bundle`` directory (``EnsembleClassifier.save`` strips a trailing ``l`` from
``model``).  These tests use the real child interpreter, so they are slower than
the mocked store tests.

Run with::

    python3 -m pytest tests/test_ensemble.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import training_core  # noqa: E402


def wait_for(store: training_core.TrainingStore, job_id: str, timeout: float = 180.0) -> dict:
    """Poll ``store`` until the job leaves the pending/running states."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.job(job_id)
        if job and job["status"] not in ("pending", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish in {timeout:g} s")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """One real fit job (one member, one epoch) shared by the assertions."""
    directory = tmp_path_factory.mktemp("ensemble-store")
    store = training_core.TrainingStore(directory)
    frame = pd.DataFrame(
        {
            "a": [float(index) for index in range(12)],
            "b": [float(index % 3) for index in range(12)],
            "target": [index % 3 for index in range(12)],
        }
    )
    dataset = store.datasets.put("ensemble.csv", frame, "ensemble.csv", "csv")
    job = store.start(
        {
            "model_id": "EnsembleClassifier",
            "parameters": {"n_models": 1},
            "arguments": {"epochs": 1, "batch_size": 4},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "xy",
                "task": "classification",
                "features": ["a", "b"],
                "target": "target",
            },
        }
    )
    finished = wait_for(store, job["job_id"])
    try:
        yield store, finished
    finally:
        store.shutdown()


def test_save_filename_avoids_the_ensemble_rstrip_bug():
    """``model`` would become ``mode`` inside ``EnsembleClassifier.save``."""
    assert (
        training_core.save_filename_for(training_core.core.MODELS_BY_ID["EnsembleClassifier"])
        == "bundle"
    )
    assert (
        training_core.save_filename_for(training_core.core.MODELS_BY_ID["AutoClassifier"])
        == "model"
    )


def test_fit_trains_and_saves_a_loadable_bundle(fitted):
    """The sandbox fixes numpy configs/one-hot targets and saves natively."""
    _, finished = fitted
    assert finished["status"] == "succeeded", finished.get("error")
    assert finished["result"]["history"]["loss"]

    artifact = finished["artifact"]
    assert artifact["save_method"] == "save"
    assert Path(artifact["load_path"]).name == "bundle"
    assert (Path(artifact["path"]) / "bundle" / "meta.json").is_file()


def test_bundle_can_be_evaluated(fitted):
    """The saved directory loads through ``EnsembleClassifier.load``."""
    store, finished = fitted
    job = store.start(
        {
            "kind": "evaluate",
            "model_id": "EnsembleClassifier",
            "artifact_id": finished["artifact"]["artifact_id"],
            "method": "predict",
            "arguments": {"X": [[0.1, 0.2], [0.5, 0.4]]},
            "timeout": 180,
        }
    )
    evaluated = wait_for(store, job["job_id"])
    assert evaluated["status"] == "succeeded", evaluated.get("error")
    assert evaluated["result_type"] == "ndarray"
