"""Tests for the training job registry and its sandbox payloads.

The tests patch :func:`server_core.run_invoke` so they exercise the job
lifecycle, the constructor resolution and the artifact bookkeeping without
paying the cost of a real fitting call.  A regression test covers the original
failure: ``Arima`` requires ``datapoints`` in its constructor and a job created
with an empty ``parameters`` object must still be able to start.

Run with::

    python3 -m pytest tests/test_training.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import training_core  # noqa: E402


def wait_for(store: training_core.TrainingStore, job_id: str, timeout: float = 10.0) -> dict:
    """Poll ``store`` until the job leaves the pending/running states."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.job(job_id)
        if job and job["status"] not in ("pending", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in {timeout:g} s")


@pytest.fixture
def store(tmp_path: Path):
    """A training store backed by an ephemeral directory."""
    instance = training_core.TrainingStore(tmp_path)
    try:
        yield instance
    finally:
        instance.shutdown()


@pytest.fixture
def captured(monkeypatch):
    """Record the payloads that would reach the child interpreter."""
    calls = []

    def fake_run_invoke(model, method, parameters, arguments, **kwargs):
        save_path = kwargs.get("save_path")
        calls.append(
            {
                "model": model["id"],
                "method": method["name"],
                "parameters": dict(parameters),
                "arguments": dict(arguments),
                "load": kwargs.get("load"),
                "save_path": save_path,
            }
        )
        if save_path:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"pickled-model")
        return {
            "type": model["id"],
            "module": model["module"],
            "repr": f"<{model['id']}>",
            "compute_seconds": 0.01,
            "result_type": "NoneType",
            "result_repr": "None",
            "save_bytes": 13 if save_path else None,
        }

    monkeypatch.setattr(training_core.core, "run_invoke", fake_run_invoke)
    return calls


def test_arima_fit_resolves_the_required_constructor_data(store, captured):
    """An empty ``parameters`` object no longer raises ``datapoints`` is missing."""
    job = store.start({"model_id": "Arima"})

    assert job["status"] in ("pending", "running")
    assert job["method"] == "train"  # the first fit-like method declared by Arima

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["parameters"]["datapoints"]
    assert finished["arguments"] == {}

    call = captured[0]
    assert call["parameters"]["datapoints"]
    assert call["arguments"] == {}

    artifact = finished["artifact"]
    assert Path(artifact["path"]).is_file()
    assert Path(artifact["path"]).is_relative_to(store.directory)


def test_generic_fit_method_falls_back_to_the_declared_train_method(store, captured):
    """``method: fit`` works even when the model only declares ``train``."""
    job = store.start({"model_id": "Arima", "method": "fit"})

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["method"] == "train"
    assert captured[0]["method"] == "train"


def test_training_data_sent_as_arguments_moves_to_the_constructor(store, captured):
    """``arguments: {datapoints: ...}`` is accepted for models such as ``Arima``."""
    job = store.start(
        {
            "model_id": "Arima",
            "method": "train",
            "arguments": {"datapoints": [1, 2, 3, 4]},
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["parameters"]["datapoints"] == [1, 2, 3, 4]
    assert finished["arguments"] == {}


def test_evaluate_without_an_artifact_builds_the_model_from_the_constructor(store, captured):
    """The declared optional artifact keeps working instead of raising 404."""
    job = store.start({"kind": "evaluate", "model_id": "Arima", "method": "predict"})

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["artifact"] is None
    assert captured[0]["load"] is None


def test_jobs_and_artifacts_are_persisted_in_the_store_directory(store, captured):
    """A second store reading the same directory sees the finished job."""
    job = store.start({"model_id": "Arima"})
    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]

    reloaded = training_core.TrainingStore(store.directory)
    try:
        restored = reloaded.job(job["job_id"])
        assert restored is not None
        assert restored["status"] == "succeeded"
        assert Path(restored["artifact"]["path"]).is_file()
    finally:
        reloaded.shutdown()
