"""End-to-end tests for the FourierRegression series analysis.

A synthetic two-series dataset is fitted with the real sandbox and then the
hold-out validation and future forecast of ``TrainingStore.analyse_series``
are exercised.  The series analysis is slower than the mocked store tests
because it imports the model and runs the ARIMA walkers in a child process.

Run with::

    python3 -m pytest tests/test_training_series.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import server_core  # noqa: E402
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
    """A real FourierRegression fit job over two synthetic series."""
    directory = tmp_path_factory.mktemp("series-store")
    store = training_core.TrainingStore(directory)
    points = 60
    time_axis = np.arange(points)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=points, freq="MS").astype(str),
            "alpha": 10 + np.sin(2 * np.pi * time_axis / 12) * 3,
            "beta": 20 + np.cos(2 * np.pi * time_axis / 6) * 2,
        }
    )
    dataset = store.datasets.put("series.csv", frame, "series.csv", "csv")
    job = store.start(
        {
            "model_id": "FourierRegression",
            "arguments": {"sigma": 0, "mov": 20, "mode": False},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "task": "auto",
                "series": ["alpha", "beta"],
            },
        }
    )
    finished = wait_for(store, job["job_id"])
    try:
        yield store, finished
    finally:
        store.shutdown()


@pytest.fixture(scope="module")
def fitted_arima(tmp_path_factory):
    """A real Arima fit job over a single synthetic series."""
    directory = tmp_path_factory.mktemp("arima-store")
    store = training_core.TrainingStore(directory)
    points = 60
    time_axis = np.arange(points)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=points, freq="MS").astype(str),
            "value": 10 + np.sin(2 * np.pi * time_axis / 12) * 3,
        }
    )
    dataset = store.datasets.put("arima.csv", frame, "arima.csv", "csv")
    job = store.start(
        {
            "model_id": "Arima",
            "arguments": {"nwalkers": 1, "mov": 20, "weights": False},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "task": "auto",
                "target": "value",
            },
        }
    )
    finished = wait_for(store, job["job_id"])
    try:
        yield store, finished
    finally:
        store.shutdown()


@pytest.fixture(scope="module")
def fitted_arima_multi(tmp_path_factory):
    """A real Arima fit job over two series (one model per series)."""
    directory = tmp_path_factory.mktemp("arima-multi-store")
    store = training_core.TrainingStore(directory)
    points = 60
    time_axis = np.arange(points)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=points, freq="MS").astype(str),
            "alpha": 10 + np.sin(2 * np.pi * time_axis / 12) * 3,
            "beta": 20 + np.cos(2 * np.pi * time_axis / 6) * 2,
        }
    )
    dataset = store.datasets.put("arima-multi.csv", frame, "arima-multi.csv", "csv")
    job = store.start(
        {
            "model_id": "Arima",
            "arguments": {"nwalkers": 1, "mov": 20, "weights": False},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "mode": "datapoints",
                "task": "auto",
                "series": ["alpha", "beta"],
            },
        }
    )
    finished = wait_for(store, job["job_id"])
    try:
        yield store, finished
    finally:
        store.shutdown()


def test_arima_multi_series_fit_and_analysis(fitted_arima_multi):
    """Arima trains one univariate model per selected series."""
    store, job = fitted_arima_multi
    assert job["status"] == "succeeded", job.get("error")
    assert job["dataset"]["series_mode"] == "sequence"
    assert job["dataset"]["series"] == ["alpha", "beta"]
    artifact = job["artifact"] or {}
    assert artifact.get("save_method") == "pickle-batch"
    names = {Path(name).name for name in artifact.get("files") or []}
    assert {"artifact.json", "series_0.pkl", "series_1.pkl"} <= names

    first = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 5,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
            "seed": 3,
            "scaling": "detrend_minmax",
        },
    )
    assert first["scaling"] == "detrend_minmax"
    assert first["series"] == ["alpha", "beta"]
    assert len(first["actual"]) == 2
    assert len(first["fitted"]) == 2
    assert len(first["validation"]) == 2
    assert len(first["forecast"]) == 2
    assert len(first["metrics"]) == 2
    assert all(len(row) == 15 for row in first["validation"])
    assert all(len(row) == 5 for row in first["forecast"])

    symmetric = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 5,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
            "seed": 3,
            "scaling": "minmax_symmetric",
        },
    )
    assert symmetric["scaling"] == "minmax_symmetric"
    assert symmetric["scale"] is True
    # Without detrending the inputs differ, so the predictions do too.
    assert symmetric["validation"] != first["validation"]

    raw = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 2,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
            "seed": 3,
            "scaling": "none",
        },
    )
    assert raw["scaling"] == "none"
    assert raw["scale"] is False
    assert len(raw["validation"]) == 2


def test_series_analysis_retries_after_walker_early_convergence(tmp_path, monkeypatch):
    """An early-converging walker is retried with a single walker."""
    (tmp_path / "lm_flaky_series.py").write_text(
        "import numpy as np\n"
        "\n"
        "class FlakySeries:\n"
        "    def __init__(self, datapoints):\n"
        "        self.datapoints = np.asarray(datapoints, dtype=float)\n"
        "\n"
        "    def train(self, nwalkers=10, mov=200):\n"
        "        if nwalkers > 1:\n"
        "            raise IndexError('list index out of range')\n"
        "        self.n = len(self.datapoints)\n"
        "\n"
        "    def predict(self, n_steps=0):\n"
        "        return np.full(len(self.datapoints) + n_steps, float(self.n))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    model = {
        "id": "FlakySeries",
        "name": "FlakySeries",
        "module": "lm_flaky_series",
        "parameters": [{"name": "datapoints", "type": "np.ndarray"}],
    }
    result = server_core.run_series_analysis(
        model,
        {"datapoints": [[float(index) for index in range(10)]]},
        {"nwalkers": 5, "mov": 10},
        {},
        8,
        2,
        fit_method="train",
        predict_method="predict",
        series_mode="sequence",
        seed=1,
        timeout=60,
    )

    assert len(result["validation_full"]) == 1
    assert len(result["validation_full"][0]) == 10
    assert len(result["validation"][0]) == 2
    assert len(result["forecast"][0]) == 2
    # The holdout model saw 8 points; the forecast model must see all 10.
    assert result["validation_training_rows"] == 8
    assert result["forecast_training_rows"] == 10
    assert result["validation"][0] == [8.0, 8.0]
    assert result["forecast"][0] == [10.0, 10.0]


def test_arima_series_analysis_uses_the_sequence_pipeline(fitted_arima):
    """Arima gets the same validation/forecast and original-unit scaling."""
    store, job = fitted_arima
    assert job["status"] == "succeeded", job.get("error")
    assert job["dataset"]["series_mode"] == "sequence"
    assert job["dataset"]["series"] == ["value"]

    analysis = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 5,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
        },
    )

    assert analysis["series_mode"] == "sequence"
    assert analysis["series"] == ["value"]
    assert analysis["rows"] == 60
    assert analysis["split"] == 45
    assert analysis["holdout"] == 15
    assert analysis["scale"] is True
    assert analysis["scaling"] == "detrend_minmax"
    assert analysis["validation_mode"] == "holdout"
    assert analysis["validation_training_rows"] == 45
    assert analysis["forecast_training_rows"] == 60
    assert len(analysis["fitted"][0]) == 45
    assert len(analysis["validation"][0]) == 15
    assert len(analysis["forecast"][0]) == 5
    assert len(analysis["forecast_fitted"][0]) == 60
    assert analysis["metrics"][0]["points"] == 15
    assert analysis["metrics"][0]["rmse"] is not None


def test_series_analysis_can_use_one_full_model(fitted_arima):
    """``validation_mode='full'`` fits once and reuses it for both charts."""
    store, job = fitted_arima
    analysis = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 5,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
            "validation_mode": "full",
        },
    )

    assert analysis["validation_mode"] == "full"
    assert analysis["validation_training_rows"] == 60
    assert analysis["forecast_training_rows"] == 60
    # The in-sample tail and the historical fitted line come from the same fit.
    assert np.allclose(analysis["fitted"][0], analysis["forecast_fitted"][0][:45])
    assert np.allclose(analysis["validation"][0], analysis["forecast_fitted"][0][45:])
    assert len(analysis["forecast"][0]) == 5


def test_series_analysis_does_not_rewrite_fit_artifact(fitted_arima):
    """Analizar series is read-only: the fit artifact is not touched."""
    store, job = fitted_arima
    before = dict(job["artifact"] or {})
    store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.2,
            "n_steps": 2,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
            "seed": 13,
            "validation_mode": "full",
        },
    )
    after = store.job(job["job_id"])["artifact"]
    assert after["files"] == before["files"]
    assert after["size_bytes"] == before["size_bytes"]
    assert after["load_path"] == before["load_path"]


def test_series_analysis_is_reproducible(fitted_arima):
    """The same request/seed returns identical curves (ARIMA walkers are random)."""
    store, job = fitted_arima
    options = {
        "validation_ratio": 0.25,
        "n_steps": 5,
        "sigma": 0,
        "mov": 20,
        "n_walkers": 2,
        "seed": 7,
    }
    first = store.analyse_series(job["job_id"], options)
    second = store.analyse_series(job["job_id"], options)

    assert first["validation"] == second["validation"]
    assert first["forecast"] == second["forecast"]
    assert first["fitted"] == second["fitted"]
    assert first["seed"] == 7
    assert first["cached"] is False
    assert second["cached"] is True


def test_series_analysis_cache_reuses_forecast_when_ratio_changes(fitted_arima):
    """Changing only the hold-out size retrains validation, not the forecast."""
    store, job = fitted_arima
    options = {
        "n_steps": 5,
        "sigma": 0,
        "mov": 20,
        "n_walkers": 1,
        "seed": 11,
    }
    first = store.analyse_series(job["job_id"], {**options, "validation_ratio": 0.2})
    second = store.analyse_series(job["job_id"], {**options, "validation_ratio": 0.3})

    assert first["cached"] is False
    assert first["cached_forecast"] is False
    assert second["cached"] is False
    assert second["cached_forecast"] is True
    assert first["split"] != second["split"]
    assert first["forecast"] == second["forecast"]
    assert first["forecast_fitted"] == second["forecast_fitted"]
    assert first["validation"] != second["validation"]
    assert second["forecast_training_rows"] == 60


def test_full_model_cache_avoids_retraining_when_ratio_changes(fitted_arima):
    """In full mode the same fit is simply re-sliced for a new validation size."""
    store, job = fitted_arima
    options = {
        "n_steps": 5,
        "sigma": 0,
        "mov": 20,
        "n_walkers": 1,
        "seed": 12,
        "validation_mode": "full",
    }
    first = store.analyse_series(job["job_id"], {**options, "validation_ratio": 0.2})
    second = store.analyse_series(job["job_id"], {**options, "validation_ratio": 0.3})

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["split"] != first["split"]
    assert first["forecast"] == second["forecast"]
    assert first["fitted"][0] == second["forecast_fitted"][0][: first["split"]]
    assert second["fitted"][0] == first["forecast_fitted"][0][: second["split"]]
    assert first["validation"] != second["validation"]
    assert second["validation_training_rows"] == 60
    assert second["forecast_training_rows"] == 60


def test_series_analysis_returns_validation_and_forecast(fitted):
    """The hold-out tail and the future steps come from the same sandbox run."""
    store, job = fitted
    assert job["status"] == "succeeded", job.get("error")
    assert job["dataset"]["series_mode"] == "matrix"

    analysis = store.analyse_series(
        job["job_id"],
        {
            "validation_ratio": 0.25,
            "n_steps": 5,
            "sigma": 0,
            "mov": 20,
            "n_walkers": 1,
        },
    )

    assert analysis["series"] == ["alpha", "beta"]
    assert analysis["series_mode"] == "matrix"
    assert analysis["rows"] == 60
    assert analysis["split"] == 45
    assert analysis["holdout"] == 15
    assert analysis["scale"] is True
    assert analysis["scaling"] == "detrend_minmax"
    assert len(analysis["fitted"]) == 2
    assert len(analysis["fitted"][0]) == 45
    assert len(analysis["validation"]) == 2
    assert len(analysis["validation"][0]) == 15
    assert len(analysis["forecast"]) == 2
    assert len(analysis["forecast"][0]) == 5
    assert len(analysis["forecast_fitted"]) == 2
    assert len(analysis["forecast_fitted"][0]) == 60
    assert analysis["labels"][0] == "2019-01-01"
    assert analysis["future_labels"][0] > analysis["labels"][-1]
    assert analysis["metrics"][0]["points"] == 15
    assert analysis["metrics"][0]["rmse"] is not None
    assert analysis["compute_seconds"] is not None
