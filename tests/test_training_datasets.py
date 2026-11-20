"""Tests for CSV/XLSX/Parquet loading in Model Training."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import training_datasets  # noqa: E402


def sample_frame() -> pd.DataFrame:
    """Small classification dataset used by the parser tests."""
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "target": ["a", "b", "a", "b"],
        }
    )


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("sample.csv", lambda frame: frame.to_csv(index=False).encode()),
        (
            "sample.xlsx",
            lambda frame: _excel_bytes(frame),
        ),
        (
            "sample.parquet",
            lambda frame: _parquet_bytes(frame),
        ),
    ],
)
def test_load_bytes_supports_csv_xlsx_and_parquet(filename, payload):
    """The three requested formats parse into the same DataFrame."""
    frame, name, source, fmt = training_datasets.load_bytes(payload(sample_frame()), filename)
    assert name == filename
    assert source == filename
    assert fmt == Path(filename).suffix.lstrip(".")
    assert list(frame.columns) == ["x1", "x2", "target"]
    assert len(frame) == 4


def _excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_classification_encodes_labels_and_reports_classes():
    """Text labels become integer codes and the mapping is returned."""
    result = training_datasets.build_xy(sample_frame(), ["x1", "x2"], "target", "classification")
    assert result["task"] == "classification"
    assert result["classes"] == {"a": 0, "b": 1}
    assert result["y"] == [0, 1, 0, 1]
    assert result["x"][0] == [1.0, 0.1]


def test_regression_and_datapoints_modes():
    """Numeric targets stay numeric and datapoints flatten to one sequence."""
    frame = sample_frame()
    frame["value"] = [0.5, 0.6, 0.7, 0.8]
    regression = training_datasets.build_xy(frame, ["x1", "x2"], "value", "regression")
    assert regression["task"] == "regression"
    assert regression["y"] == [0.5, 0.6, 0.7, 0.8]
    datapoints = training_datasets.build_datapoints(frame, "value")
    assert datapoints["datapoints"] == [0.5, 0.6, 0.7, 0.8]
    assert datapoints["target"] == "value"


def test_parameter_overrides_follow_the_selected_dataset():
    """Keras shape and class parameters are derived from the selection."""
    model = {
        "parameters": [
            {"name": "input_shape_parm"},
            {"name": "num_classes"},
            {"name": "output_shape_parm"},
        ]
    }
    overrides = training_datasets.parameter_overrides(
        model, ["x1", "x2"], "classification", {"a": 0, "b": 1}
    )
    assert overrides["input_shape_parm"] == 2
    assert overrides["num_classes"] == 2
    assert "output_shape_parm" not in overrides


def test_dataset_store_exposes_preview_and_columns():
    """The store returns JSON metadata without exposing the DataFrame."""
    store = training_datasets.DatasetStore(max_items=2)
    metadata = store.put("sample.csv", sample_frame(), "sample.csv", "csv")
    assert metadata["dataset_id"].startswith("ds-")
    assert metadata["rows"] == 4
    assert [column["name"] for column in metadata["columns"]] == ["x1", "x2", "target"]
    assert metadata["preview"][0]["x1"] == 1.0
    assert "frame" not in metadata
    assert store.get(metadata["dataset_id"]) is not None


def test_invalid_numeric_feature_is_rejected():
    """Categorical columns cannot silently become NaN features."""
    frame = sample_frame()
    frame["bad"] = ["a", "b", "c", "d"]
    with pytest.raises(Exception):
        training_datasets.build_xy(frame, ["bad"], "target", "classification")
