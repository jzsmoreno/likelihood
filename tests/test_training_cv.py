"""Cross-validation orchestration inside the training store.

The sandbox call is patched, so the test exercises the fold splitting, the
``validation_data`` wiring, the best-fold selection and the artifact promotion
without paying the cost of a real Keras fit.

Run with::

    python3 -m pytest tests/test_training_cv.py -v
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

import server_core as core  # noqa: E402
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
    instance = training_core.TrainingStore(tmp_path)
    try:
        yield instance
    finally:
        instance.shutdown()


def make_dataset(store: training_core.TrainingStore, rows: int = 30) -> dict:
    """Dataset whose first feature is the unique row index."""
    frame = pd.DataFrame(
        {
            "x1": [float(index) for index in range(rows)],
            "x2": [float(index % 5) for index in range(rows)],
            "target": [index % 3 for index in range(rows)],
        }
    )
    return store.datasets.put("cv.csv", frame, "cv.csv", "csv")


def job_spec(dataset: dict, **cross_validation) -> dict:
    return {
        "model_id": "AutoClassifier",
        "parameters": {"input_shape_parm": 2, "num_classes": 3, "units": 4},
        "arguments": {"epochs": 2},
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "mode": "xy",
            "features": ["x1", "x2"],
            "target": "target",
            "task": "classification",
        },
        "cross_validation": {"folds": 3, **cross_validation},
    }


def test_cross_validation_runs_one_child_per_fold(store, monkeypatch):
    dataset = make_dataset(store)
    calls = []

    def fake_run_invoke(model, method, parameters, arguments, **kwargs):
        calls.append({"arguments": dict(arguments), "save_path": kwargs.get("save_path")})
        index = len(calls)
        target = Path(kwargs["save_path"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.pkl").write_bytes(("fold-%d" % index).encode())
        score = {1: 0.4, 2: 0.2, 3: 0.3}[index]
        return {
            "type": model["id"],
            "module": model["module"],
            "repr": "<%s>" % model["id"],
            "compute_seconds": 0.5,
            "result": {
                "history": {
                    "loss": [0.9, 0.7],
                    "accuracy": [0.5, 0.6],
                    "kappa": [0.0, 0.1],
                    "val_loss": [0.5, score],
                    "val_accuracy": [0.4, 0.5],
                    "val_kappa": [0.0, 0.05],
                }
            },
            "save_path": str(target),
            "load_path": str(target),
            "save_method": "pickle",
            "save_bytes": 6,
        }

    monkeypatch.setattr(training_core.core, "run_invoke", fake_run_invoke)
    job = store.start(job_spec(dataset))
    finished = wait_for(store, job["job_id"])

    assert finished["status"] == "succeeded", finished["error"]
    assert len(calls) == 3
    for call in calls:
        arguments = call["arguments"]
        assert "validation_data" in arguments
        assert len(arguments["validation_data"]) == 2
        assert len(arguments["x"]) == len(arguments["y"])
        assert len(arguments["x"]) + len(arguments["validation_data"][0]) == 30
        assert len(arguments["validation_data"][0]) == len(arguments["validation_data"][1])
    # ``x1`` is the row index, so the union of validation sets proves coverage.
    covered = sorted(row[0] for call in calls for row in call["arguments"]["validation_data"][0])
    assert covered == [float(index) for index in range(30)]

    result = finished["result"]
    cv = result["cross_validation"]
    assert cv["folds"] == 3
    assert cv["metric"] == "val_loss"
    assert cv["mode"] == "min"
    assert cv["best_fold"] == 2
    assert cv["best_score"] == pytest.approx(0.2)
    assert cv["mean"] == pytest.approx(0.3)
    assert cv["std"] == pytest.approx(0.08165, abs=1e-6)
    assert [record["fold"] for record in result["folds"]] == [1, 2, 3]
    assert [record["score"] for record in result["folds"]] == pytest.approx([0.4, 0.2, 0.3])
    assert result["history"]["val_loss"][-1] == pytest.approx(0.2)
    assert finished["compute_seconds"] == pytest.approx(1.5)

    artifact = finished["artifact"]
    root = Path(artifact["path"])
    assert (root / "model.pkl").read_bytes() == b"fold-2"
    assert not (root / "folds").exists()
    assert artifact["size_bytes"] > 0
    assert artifact["load_path"] == str(root)
    metadata = json.loads((root / "artifact.json").read_text())
    assert metadata["cross_validation"]["folds"] == 3
    assert metadata["selection"]["best_fold"] == 2
    assert metadata["selection"]["best_score"] == pytest.approx(0.2)


def test_cross_validation_requires_a_dataset(store):
    with pytest.raises(core.ValidationError):
        store.start(
            {
                "model_id": "AutoClassifier",
                "parameters": {"input_shape_parm": 2, "num_classes": 3},
                "arguments": {"epochs": 1},
                "cross_validation": {"folds": 3},
            }
        )


def test_cross_validation_rejects_more_folds_than_rows(store):
    dataset = make_dataset(store, rows=4)
    with pytest.raises(core.ValidationError):
        store.start(job_spec(dataset, folds=5))


def test_cross_validation_rejects_an_explicit_validation_data_argument(store):
    dataset = make_dataset(store)
    spec = job_spec(dataset)
    spec["arguments"]["validation_data"] = [[[0.0, 0.0]], [0]]
    with pytest.raises(core.ValidationError):
        store.start(spec)
