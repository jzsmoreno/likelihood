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

import json
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
                "compile": kwargs.get("compile_options"),
                "load": kwargs.get("load"),
                "save_path": save_path,
            }
        )
        saved = None
        if save_path:
            target = Path(save_path)
            target.mkdir(parents=True, exist_ok=True)
            saved = target / "model.pkl"
            saved.write_bytes(b"pickled-model")
        history = {"loss": [0.5], "accuracy": [0.8]} if method["name"] == "fit" else None
        return {
            "type": model["id"],
            "module": model["module"],
            "repr": f"<{model['id']}>",
            "compute_seconds": 0.01,
            "result": {"history": history} if history else None,
            "result_type": "History" if history else "NoneType",
            "result_repr": "History" if history else "None",
            "save_path": str(saved) if saved else None,
            "load_path": str(saved) if saved else None,
            "save_method": "pickle" if saved else None,
            "save_bytes": 13 if saved else None,
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
    assert Path(artifact["path"]).is_dir()
    assert Path(artifact["load_path"]).is_file()
    assert (Path(artifact["path"]) / "artifact.json").is_file()
    assert Path(artifact["path"]).is_relative_to(store.directory)


def test_generic_fit_method_falls_back_to_the_declared_train_method(store, captured):
    """``method: fit`` works even when the model only declares ``train``."""
    job = store.start({"model_id": "Arima", "method": "fit"})

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["method"] == "train"
    assert captured[0]["method"] == "train"


def test_artifact_savers_are_detected_for_serving():
    """Native savers and the AutoClassifier component layout are selected."""
    assert training_core.save_method_for(training_core.core.MODELS_BY_ID["GANRegressor"]) == (
        "save_model",
        "path",
    )
    assert training_core.save_method_for(training_core.core.MODELS_BY_ID["Arima"]) == (
        "save_model",
        "name",
    )
    assert training_core.save_layout_for(training_core.core.MODELS_BY_ID["AutoClassifier"]) == (
        "components"
    )
    assert training_core.save_layout_for(training_core.core.MODELS_BY_ID["Arima"]) == ""


def test_keras_models_expose_an_inherited_fit_adapter():
    """Keras models get a synthetic fit because the AST catalog cannot see it."""
    model = training_core.core.MODELS_BY_ID["AutoClassifier"]
    fit = training_core.training_fit_method(model)

    assert fit is not None
    assert fit["name"] == "fit"
    assert fit["framework"] == "keras"
    assert [parameter["name"] for parameter in fit["parameters"]][:2] == ["x", "y"]


def test_keras_fit_job_uses_the_server_compile_defaults(store, captured):
    """A Keras fit job compiles the model before calling the inherited fit."""
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "parameters": {
                "input_shape_parm": 4,
                "num_classes": 3,
                "units": 8,
                "activation": "relu",
            },
            "arguments": {"x": [[0.1, 0.2, 0.3, 0.4]] * 4, "y": [0, 1, 2, 0], "epochs": 1},
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["adapter"] == "keras"
    assert finished["method"] == "fit"
    assert finished["compile"]["optimizer"] == "adam"
    assert finished["compile"]["loss"] == "sparse_categorical_crossentropy"
    assert finished["compile"]["metrics"] == ["accuracy"]
    assert finished["result"]["history"]["loss"]

    call = captured[0]
    assert call["method"] == "fit"
    assert call["compile"] == finished["compile"]


def test_keras_compile_options_can_be_overridden(store, captured):
    """The request's compile object wins over the server default."""
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "parameters": {
                "input_shape_parm": 4,
                "num_classes": 4,
                "units": 8,
                "activation": "relu",
            },
            "arguments": {"x": [[0.1, 0.2, 0.3, 0.4]] * 4, "y": [0, 1, 2, 3]},
            "compile": {"optimizer": "sgd", "loss": "categorical_crossentropy"},
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["compile"]["optimizer"] == "sgd"
    assert finished["compile"]["loss"] == "categorical_crossentropy"
    assert finished["compile"]["metrics"] == ["accuracy"]


def test_dataset_backed_job_keeps_only_a_compact_reference(store, captured):
    """A dataset job rebuilds X/Y in the worker instead of storing them in jobs.json."""
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "species": ["a", "b", "a", "c"],
        }
    )
    dataset = store.datasets.put("iris.csv", frame, "iris.csv", "csv")
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "xy",
                "task": "classification",
                "features": ["x1", "x2"],
                "target": "species",
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["dataset"]["rows"] == 4
    assert finished["dataset"]["classes"] == {"a": 0, "b": 1, "c": 2}
    assert finished["parameters"]["input_shape_parm"] == 2
    assert finished["parameters"]["num_classes"] == 3
    metadata = json.loads(
        (Path(finished["artifact"]["path"]) / "artifact.json").read_text(encoding="utf-8")
    )
    assert metadata["parameters"]["input_shape_parm"] == 2
    assert metadata["parameters"]["num_classes"] == 3
    assert "x" not in finished["arguments"]
    assert "y" not in finished["arguments"]

    call = captured[0]
    assert call["arguments"]["x"] == [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]]
    assert call["arguments"]["y"] == [0, 1, 0, 2]


def test_dataset_datapoints_mode_rebuilds_the_sequence(store, captured):
    """A datapoints selection is removed from the stored parameters and rebuilt."""
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    dataset = store.datasets.put("series.csv", frame, "series.csv", "csv")
    job = store.start(
        {
            "model_id": "Arima",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "target": "value",
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert "datapoints" not in finished["parameters"]
    assert captured[0]["parameters"]["datapoints"] == [1.0, 2.0, 3.0, 4.0]


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
        assert Path(restored["artifact"]["path"]).is_dir()
        assert Path(restored["artifact"]["load_path"]).is_file()
    finally:
        reloaded.shutdown()
