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
                "training_controls": kwargs.get("training_controls"),
                "limits": kwargs.get("limits"),
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


def test_keras_target_metric_control_is_normalized_and_forwarded(store, captured):
    """A target metric reaches the child as EarlyStopping/ReduceLROnPlateau options."""
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "arguments": {
                "x": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]],
                "y": [0, 1],
                "epochs": 2,
            },
            "control": {
                "metric": "val_loss",
                "action": "stop",
                "mode": "min",
                "value": 0.2,
                "patience": 2,
                "min_delta": 0.01,
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["control"]["metric"] == "val_loss"
    assert finished["control"]["action"] == "stop"
    assert finished["control"]["value"] == 0.2
    assert finished["control"]["max_epochs"] == 50
    assert captured[0]["training_controls"]["patience"] == 2
    assert captured[0]["arguments"]["epochs"] == 50


def test_training_normalization_scales_features_without_touching_targets():
    """Standard, 0..1 and -1..1 scaling apply only to feature arguments."""
    arguments = {"x": [[0.0, 1.0], [2.0, 3.0]], "y": [1, 2]}
    standard_args, _ = training_core.apply_normalization(arguments, {}, "standard")
    assert standard_args["x"] == [[-1.0, -1.0], [1.0, 1.0]]
    assert standard_args["y"] == [1, 2]

    minmax_args, _ = training_core.apply_normalization(arguments, {}, "minmax")
    assert minmax_args["x"] == [[0.0, 0.0], [1.0, 1.0]]

    symmetric_args, _ = training_core.apply_normalization(arguments, {}, "minmax_symmetric")
    assert symmetric_args["x"] == [[-1.0, -1.0], [1.0, 1.0]]
    assert symmetric_args["y"] == [1, 2]


def test_training_normalization_detrends_series_rows():
    """detrend_minmax removes the line per series and keeps the [-1,1] range."""
    arguments = {"datapoints": [[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 0.0, 2.0]]}
    scaled, _ = training_core.apply_normalization(arguments, {}, "detrend_minmax")

    assert scaled["datapoints"][0] == [0.0, 0.0, 0.0, 0.0]
    assert all(-1.0001 <= value <= 1.0001 for value in scaled["datapoints"][1])
    assert scaled["datapoints"][1] != [0.0, 0.0, 0.0, 0.0]
    assert training_core.normalization_method("detrend") == "detrend_minmax"


def test_target_normalization_does_not_detrend_y():
    """A detrended y could not be inverted, so auto falls back to none."""
    method = training_core.target_normalization_method("auto", "regression", "detrend_minmax")
    assert method == "none"
    with pytest.raises(training_core.core.ValidationError):
        training_core.target_normalization_method("detrend_minmax", "regression", "none")


def test_training_normalization_rejects_unknown_methods():
    with pytest.raises(training_core.core.ValidationError):
        training_core.normalization_method("unknown")


def test_training_normalization_is_applied_to_job_arguments(store, captured):
    """The normalized values reach the sandbox, not just the UI preview."""
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "arguments": {
                "x": [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]],
                "y": [0, 1],
            },
            "normalization": "minmax",
        }
    )
    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["normalization"] == "minmax"
    assert captured[0]["arguments"]["x"] == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    assert captured[0]["arguments"]["y"] == [0, 1]


def test_dataset_normalization_is_applied_by_the_worker(store, captured):
    """A dataset-backed job rebuilds and scales ``x`` inside the worker."""
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "y": [0, 1, 0, 1],
        }
    )
    dataset = store.datasets.put("training.csv", frame, "training.csv", "csv")
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "xy",
                "task": "classification",
                "features": ["x1", "x2"],
                "target": "y",
            },
            "normalization": "minmax",
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert captured[0]["arguments"]["x"] == [
        [0.0, 0.0],
        [1 / 3, 1 / 3],
        [2 / 3, 2 / 3],
        [1.0, 1.0],
    ]
    assert captured[0]["arguments"]["y"] == [0, 1, 0, 1]


def test_target_normalization_auto_only_applies_to_regression():
    """``auto`` mirrors the feature method for regression, never for labels."""
    assert training_core.target_normalization_method("auto", "regression", "standard") == "standard"
    assert training_core.target_normalization_method("auto", "classification", "standard") == "none"
    assert training_core.target_normalization_method("none", "regression", "standard") == "none"
    assert training_core.target_normalization_method("minmax", "regression", "none") == "minmax"


def test_classification_target_normalization_is_rejected(store):
    """Class codes are not a continuous quantity, so scaling them is an error."""
    with pytest.raises(training_core.core.ValidationError):
        store.start(
            {
                "model_id": "AutoClassifier",
                "arguments": {"x": [[0.1, 0.2], [0.2, 0.1]], "y": [0, 1]},
                "target_normalization": "standard",
            }
        )


def test_direct_regression_arguments_normalize_the_target(store, captured):
    """``target_normalization: auto`` scales ``y`` for a regressor model."""
    job = store.start(
        {
            "model_id": "GANRegressor",
            "arguments": {
                "x": [[0.0, 0.0], [1.0, 1.0]],
                "y": [[10.0], [20.0]],
                "epochs": 1,
            },
            "normalization": "minmax",
        }
    )
    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["target_normalization"]["method"] == "minmax"
    assert finished["target_normalization"]["columns"][0]["min"] == 10.0
    assert finished["target_normalization"]["columns"][0]["max"] == 20.0
    assert captured[0]["arguments"]["y"] == [[0.0], [1.0]]
    assert captured[0]["arguments"]["x"] == [[0.0, 0.0], [1.0, 1.0]]


def test_regression_dataset_target_scaler_is_written_to_the_artifact(store, captured):
    """A dataset regression job scales y and persists the scaler for inference."""
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )
    dataset = store.datasets.put("regression.csv", frame, "regression.csv", "csv")
    job = store.start(
        {
            "model_id": "GANRegressor",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "xy",
                "task": "regression",
                "features": ["x1", "x2"],
                "target": "y",
            },
            "normalization": "minmax",
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["target_normalization"] == {
        "method": "minmax",
        "min": 10.0,
        "max": 40.0,
    }
    assert captured[0]["arguments"]["y"] == [0.0, 1 / 3, 2 / 3, 1.0]
    metadata = json.loads(
        (Path(finished["artifact"]["path"]) / "artifact.json").read_text(encoding="utf-8")
    )
    assert metadata["target_normalization"] == {
        "method": "minmax",
        "min": 10.0,
        "max": 40.0,
    }


def test_training_jobs_bypass_the_playground_cpu_cap(store, captured):
    """Multi-threaded training must not die from the playground's 30 s CPU cap.

    ``RLIMIT_CPU`` counts CPU time across every thread, so a TensorFlow fit can
    exhaust it long before its wall-clock timeout (the symptom is exit code
    ``-24``, ``SIGXCPU``).  Training therefore sends an explicit CPU budget.
    """
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "arguments": {"x": [[0.1, 0.2], [0.2, 0.1]], "y": [0, 1], "epochs": 1},
        }
    )
    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert captured[0]["limits"] == {"cpu": training_core.TRAIN_CPU_SECONDS}


def test_run_invoke_merges_sandbox_limit_overrides(monkeypatch):
    """``run_invoke(limits=...)`` overrides one limit and keeps the rest."""
    model = training_core.core.MODELS_BY_ID["AutoClassifier"]
    method = training_core.framework_fit_method(model)
    sent = {}

    def fake_run(argv, **kwargs):
        sent.update(json.loads(kwargs["input"]))
        marker = training_core.core._PLAYGROUND_MARKER
        payload = {"ok": True, "type": model["id"], "module": model["module"]}
        stdout = "\n" + marker + json.dumps(payload) + "\n"
        return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(training_core.core.subprocess, "run", fake_run)
    result = training_core.core.run_invoke(
        model,
        method,
        {},
        {"x": [[0.1, 0.2], [0.2, 0.1]], "y": [0, 1]},
        limits={"cpu": 0},
    )

    assert result["method"] == "fit"
    assert sent["limits"]["cpu"] == 0
    expected = training_core.core.playground_limits()
    assert sent["limits"]["memory"] == expected["memory"]
    assert sent["limits"]["files"] == expected["files"]


def test_non_keras_models_reject_target_metric_controls(store):
    """The callback contract is Keras-specific and fails clearly on other models."""
    with pytest.raises(training_core.core.ValidationError):
        store.start(
            {
                "model_id": "Arima",
                "control": {"metric": "loss", "action": "stop", "value": 0.1},
            }
        )


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
    assert training_core.save_method_for(training_core.core.MODELS_BY_ID["FourierRegression"]) == (
        "",
        "",
    )


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
    assert finished["compile"]["metrics"] == list(training_core.CLASSIFICATION_METRICS)
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
    assert finished["compile"]["metrics"] == list(training_core.CLASSIFICATION_METRICS)


def test_regression_default_compile_includes_popular_metrics():
    """Regression logs MAE, MSE, RMSE and R² by default; MAPE stays opt-in."""
    options = training_core.default_compile_options({}, {})
    assert options["optimizer"] == "adam"
    assert options["loss"] == "mse"
    assert options["metrics"] == list(training_core.REGRESSION_METRICS)
    assert list(training_core.REGRESSION_METRICS) == [
        "mae",
        "mse",
        "root_mean_squared_error",
        "r2_score",
    ]


def test_classification_defaults_include_popular_metrics():
    """Classifiers log accuracy, precision, recall, F1, AUC and Cohen's kappa."""
    expected = ["accuracy", "precision", "recall", "f1_score", "auc", "kappa"]
    multiclass = training_core.default_compile_options(
        {}, {"classifier_activation": "softmax", "num_classes": 3}
    )
    binary = training_core.default_compile_options(
        {}, {"classifier_activation": "sigmoid", "num_classes": 1}
    )
    assert list(training_core.CLASSIFICATION_METRICS) == expected
    assert multiclass["loss"] == "sparse_categorical_crossentropy"
    assert multiclass["metrics"] == expected
    assert binary["loss"] == "binary_crossentropy"
    assert binary["metrics"] == expected


def test_target_metric_control_offers_cohens_kappa():
    """The control accepts Cohen's kappa for early stopping / LR scheduling."""
    assert "kappa" in training_core.TARGET_METRICS
    assert "val_kappa" in training_core.TARGET_METRICS
    control = training_core.normalize_training_control(
        {"metric": "val_kappa", "action": "stop", "value": 0.8}
    )
    assert control["metric"] == "val_kappa"
    assert control["value"] == 0.8


def test_supported_optimizer_names_are_forwarded():
    """Every optimizer offered by the wizard is accepted by the compile merge."""
    for name in (
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "nadam",
        "adagrad",
        "adadelta",
        "adamax",
        "ftrl",
    ):
        options = training_core.resolve_compile_options({}, {}, {"optimizer": name})
        assert options["optimizer"] == name


def test_keras_compile_options_accept_optimizer_and_learning_rate(store, captured):
    """The optimizer selector and learning rate reach the sandbox."""
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
            "compile": {"optimizer": "sgd", "learning_rate": 0.01},
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["compile"]["optimizer"] == "sgd"
    assert finished["compile"]["learning_rate"] == 0.01
    assert captured[0]["compile"]["optimizer"] == "sgd"
    assert captured[0]["compile"]["learning_rate"] == 0.01


def test_compile_rejects_invalid_learning_rate(store):
    """A non-positive or non-numeric learning rate fails before the job starts."""
    for invalid in (-0.1, 0, "fast", True):
        with pytest.raises(training_core.core.ValidationError):
            store.start(
                {
                    "model_id": "AutoClassifier",
                    "parameters": {
                        "input_shape_parm": 4,
                        "num_classes": 3,
                        "units": 8,
                        "activation": "relu",
                    },
                    "arguments": {
                        "x": [[0.1, 0.2, 0.3, 0.4]] * 4,
                        "y": [0, 1, 2, 0],
                    },
                    "compile": {"learning_rate": invalid},
                }
            )


def test_regression_compile_accepts_mape_loss_and_metric(store, captured):
    """MAPE can be selected as the regression loss and/or metric."""
    job = store.start(
        {
            "model_id": "GANRegressor",
            "arguments": {
                "x": [[0.0, 0.0], [1.0, 1.0]],
                "y": [[10.0], [20.0]],
                "epochs": 1,
            },
            "compile": {
                "optimizer": "adam",
                "loss": "mean_absolute_percentage_error",
                "metrics": ["mae", "mean_absolute_percentage_error"],
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["compile"]["loss"] == "mean_absolute_percentage_error"
    assert finished["compile"]["metrics"] == ["mae", "mean_absolute_percentage_error"]
    assert captured[0]["compile"]["loss"] == "mean_absolute_percentage_error"
    assert captured[0]["compile"]["metrics"] == ["mae", "mean_absolute_percentage_error"]


def test_target_metric_control_offers_mape():
    """The control accepts MAPE under both Keras spellings."""
    assert "mape" in training_core.TARGET_METRICS
    assert "mean_absolute_percentage_error" in training_core.TARGET_METRICS
    control = training_core.normalize_training_control(
        {"metric": "val_mean_absolute_percentage_error", "action": "stop", "value": 5.0}
    )
    assert control["metric"] == "val_mean_absolute_percentage_error"
    assert control["action"] == "stop"


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


def test_ensemble_dataset_uses_the_fit_argument_spelling(store, captured):
    """``EnsembleClassifier.fit(X, y)`` receives the dataset as ``X``/``y``.

    Regression test: the dataset builder always sent lowercase ``x``, so the
    training form rejected the job with ``Argumentos desconocidos: x``.
    """
    model = training_core.core.MODELS_BY_ID["EnsembleClassifier"]
    assert training_core.training_dataset_mode(model) == "xy"
    assert training_core.xy_argument_names(training_core.training_fit_method(model)) == ("X", "y")

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
            "arguments": {"epochs": 2},
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
    assert finished["status"] == "succeeded", finished["error"]
    call = captured[0]
    assert set(call["arguments"]) == {"X", "y", "epochs"}
    assert call["arguments"]["X"][0] == [0.0, 0.0]
    assert call["arguments"]["y"] == [index % 3 for index in range(12)]
    assert call["arguments"]["epochs"] == 2


def test_ensemble_training_example_uses_capital_x():
    """The wizard example matches ``EnsembleClassifier.fit(X, y)``."""
    model = training_core.core.MODELS_BY_ID["EnsembleClassifier"]
    example = training_core.training_example(model)
    assert example["method"] == "fit"
    assert set(example["arguments"]) >= {"X", "y", "epochs"}
    assert len(example["arguments"]["X"]) == 12
    assert example["parameters"]["n_models"] == 2


def test_target_normalization_accepts_the_declared_target_name():
    """The scaler edits ``Y`` when the fit method capitalises the target."""
    arguments, stats = training_core.apply_target_normalization(
        {"X": [[1.0], [2.0]], "Y": [1.0, 3.0]}, "standard", "Y"
    )
    assert stats["mean"] == 2.0
    assert arguments["Y"] == [-1.0, 1.0]
    assert arguments["X"] == [[1.0], [2.0]]


def test_fourier_dataset_builds_a_series_matrix(store, captured):
    """A multi-column datapoints job sends a ``(series, points)`` matrix."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "a": [float(index) for index in range(10)],
            "b": [float(index) * 2 for index in range(10)],
        }
    )
    dataset = store.datasets.put("series.csv", frame, "series.csv", "csv")
    job = store.start(
        {
            "model_id": "FourierRegression",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "task": "auto",
                "series": ["a", "b"],
                "sample_rows": 8,
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["method"] == "fit"
    assert finished["dataset"]["series"] == ["a", "b"]
    assert finished["dataset"]["series_mode"] == "matrix"
    assert finished["dataset"]["rows"] == 8
    assert finished["dataset"]["label_column"] == "date"

    matrix = captured[0]["parameters"]["datapoints"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 8
    assert matrix[0] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert matrix[1][0] == 4.0


def test_datapoints_normalization_scales_each_series_row():
    """``datapoints`` (series × points) is scaled per row, not per time step."""
    _, parameters = training_core.apply_normalization(
        {}, {"datapoints": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]}, "standard"
    )
    matrix = parameters["datapoints"]
    expected = [-1.224744871391589, 0.0, 1.224744871391589]
    assert matrix[0] == pytest.approx(expected)
    assert matrix[1] == pytest.approx(expected)

    single_row = training_core.apply_normalization(
        {}, {"datapoints": [[1.0, 2.0, 3.0]]}, "standard"
    )[1]["datapoints"][0]
    assert single_row == pytest.approx(expected)

    flat_arguments, _ = training_core.apply_normalization(
        {"datapoints": [1.0, 2.0, 3.0]}, {}, "standard"
    )
    assert flat_arguments["datapoints"] == pytest.approx(expected)

    x_arguments, _ = training_core.apply_normalization(
        {"x": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]}, {}, "standard"
    )
    assert x_arguments["x"][0] == pytest.approx([expected[0], expected[0]])
    assert x_arguments["x"][2][0] == pytest.approx(expected[2])


def test_fourier_datapoints_normalization_uses_series_rows(store, captured):
    """The job normalization must not collapse a single series to zeros."""
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )
    dataset = store.datasets.put("series.csv", frame, "series.csv", "csv")
    job = store.start(
        {
            "model_id": "FourierRegression",
            "normalization": "standard",
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "task": "auto",
                "series": ["a", "b"],
                "sample_rows": 8,
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    matrix = captured[0]["parameters"]["datapoints"]
    for row in matrix:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        assert abs(mean) < 1e-9
        assert abs(variance - 1.0) < 1e-6
    assert len(set(matrix[0])) > 1


def test_dataset_sample_rows_limits_the_training_data(store, captured):
    """``sample_rows`` sends a reproducible random portion to the fit call."""
    frame = pd.DataFrame(
        {
            "x1": [float(index) for index in range(20)],
            "x2": [float(index % 5) for index in range(20)],
            "target": ["a", "b"] * 10,
        }
    )
    dataset = store.datasets.put("sample.csv", frame, "sample.csv", "csv")
    job = store.start(
        {
            "model_id": "AutoClassifier",
            "parameters": {"input_shape_parm": 2, "num_classes": 2},
            "arguments": {"epochs": 1},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "xy",
                "task": "classification",
                "features": ["x1", "x2"],
                "target": "target",
                "sample_rows": 6,
            },
        }
    )

    finished = wait_for(store, job["job_id"])
    assert finished["status"] == "succeeded", finished["error"]
    assert finished["dataset"]["rows"] == 6
    assert finished["dataset"]["sample_rows"] == 6
    call = captured[0]
    assert len(call["arguments"]["x"]) == 6
    assert len(call["arguments"]["y"]) == 6


def test_dataset_sample_rows_must_be_positive(store):
    """A zero/negative sample size is rejected before queueing the job."""
    frame = pd.DataFrame({"x1": [1.0, 2.0], "target": ["a", "b"]})
    dataset = store.datasets.put("sample.csv", frame, "sample.csv", "csv")
    with pytest.raises(training_core.core.ValidationError):
        store.start(
            {
                "model_id": "AutoClassifier",
                "dataset": {
                    "dataset_id": dataset["dataset_id"],
                    "mode": "xy",
                    "task": "classification",
                    "features": ["x1"],
                    "target": "target",
                    "sample_rows": 0,
                },
            }
        )


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
    assert finished["dataset"]["series_mode"] == "sequence"
    assert finished["dataset"]["series"] == ["value"]
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
