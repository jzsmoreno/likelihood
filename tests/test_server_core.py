"""Tests for the framework-independent domain layer (:mod:`server_core`).

The HTTP surface lives in ``tests/test_server_fastapi.py``; this module covers
the catalog/introspection helpers, the JSON coercion rules, pagination, static
asset resolution, the rate limiter, the metrics and the invoke bootstrap
contracts that are easy to regress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import server_core as core  # noqa: E402


@pytest.mark.parametrize(
    ("kind", "type_hint", "value", "expected"),
    [
        ("int", "int", "3", 3),
        ("int", "int", 3.0, 3),
        ("int", "int", 3, 3),
        ("float", "float", "2.5", 2.5),
        ("float", "float", 2, 2.0),
        ("bool", "bool", "true", True),
        ("bool", "bool", "no", False),
        ("bool", "bool", 1, True),
        ("str", "str", "softmax", "softmax"),
        ("text", "Path", "models", "models"),
        ("raw", "list", [1, 2], [1, 2]),
        ("raw", "dict", {"a": 1}, {"a": 1}),
    ],
)
def test_coerce_playground_value_accepts_json(kind, type_hint, value, expected):
    field = {"name": "value", "kind": kind, "type": type_hint, "required": True}
    assert core.coerce_playground_value(field, value) == expected


@pytest.mark.parametrize(
    ("kind", "type_hint", "value"),
    [
        ("int", "int", "abc"),
        ("int", "int", True),
        ("int", "int", 1.5),
        ("float", "float", "inf"),
        ("bool", "bool", "maybe"),
        ("str", "str", 3),
        ("raw", "np.ndarray", "matrix"),
        ("raw", "np.ndarray", []),
    ],
)
def test_coerce_playground_value_rejects_invalid_values(kind, type_hint, value):
    field = {"name": "value", "kind": kind, "type": type_hint, "required": True}
    with pytest.raises(ValueError):
        core.coerce_playground_value(field, value)


def test_parse_pagination():
    assert core.parse_pagination({}) == (None, 0)
    assert core.parse_pagination({"limit": ["2"], "offset": ["3"]}) == (2, 3)
    with pytest.raises(ValueError):
        core.parse_pagination({"limit": ["0"]})
    with pytest.raises(ValueError):
        core.parse_pagination({"limit": ["9999"]})
    with pytest.raises(ValueError):
        core.parse_pagination({"offset": ["-1"]})


def test_constructor_samples_include_the_auto_classifier_shape():
    model = next(item for item in core.CATALOG if item["id"] == "AutoClassifier")
    samples = core.constructor_samples(model)
    assert samples["input_shape_parm"] == 4
    assert samples["num_classes"] >= 2
    assert samples["activation"]


def test_invoke_spec_is_json_ready_for_every_model():
    for model in core.CATALOG:
        spec = core.invoke_spec(model)
        json.dumps(spec)
        for method in spec["methods"]:
            for field in method["fields"]:
                assert field["name"]
                assert "type" in field and "sample" in field


def test_dummy_json_value_shapes_payloads_from_the_constructor():
    matrix = core.dummy_json_value("X", "np.ndarray", {"window": 4, "n_features": 3})
    assert len(matrix) == 4
    assert all(len(row) == 3 for row in matrix)
    sequence = core.dummy_json_value("sequence", "list", {"n_observations": 3})
    assert sequence and all(isinstance(item, int) and 0 <= item < 3 for item in sequence)
    shape = core.dummy_json_value("input_shape", "tuple", {})
    assert shape and all(isinstance(item, int) for item in shape)


def test_resolve_asset_blocks_traversal():
    assert core.resolve_asset("/") is not None
    assert core.resolve_asset("/../server_core.py") is None
    assert core.resolve_asset("/.env") is None


def test_content_type_for():
    assert core.content_type_for(core.INDEX_FILE) == "text/html; charset=utf-8"
    assert core.content_type_for(Path("favicon.svg")) == "image/svg+xml; charset=utf-8"
    assert core.content_type_for(Path("model.ipynb")) == ("application/x-ipynb+json; charset=utf-8")


def test_rate_limiter_and_metrics():
    limiter = core.RateLimiter(2)
    assert limiter.hit("client") is None
    assert limiter.hit("client") is None
    assert limiter.hit("client") is not None
    limiter.reset()
    assert limiter.hit("client") is None

    metrics = core.Metrics()
    metrics.record("GET /api/health", 200, 0.01)
    metrics.record("GET /api/health", 500, 0.02)
    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 2
    assert snapshot["total_errors"] == 1
    assert snapshot["endpoints"]["GET /api/health"]["count"] == 2


def test_parameters_signature_is_order_independent():
    # The session rebuild compares constructor values, never raw payload order.
    assert core.parameters_signature({"input_shape_parm": 4, "units": 8}) == (
        core.parameters_signature({"units": 8, "input_shape_parm": 4})
    )
    assert core.parameters_signature({"input_shape_parm": 4}) != (
        core.parameters_signature({"input_shape_parm": 6})
    )


def test_invoke_bootstrap_contracts():
    # The child interpreter must keep the shape/tuple/rebuild behaviour in sync.
    assert "constructor_parameters" in core._SANDBOX_PRELUDE
    assert "tuple(value)" in core._SANDBOX_PRELUDE
    assert "_NO_AUTO_BUILD" in core._SANDBOX_PRELUDE
    assert "_mark_prepared" in core._SANDBOX_PRELUDE
    assert '"numpy"' in core._SANDBOX_PRELUDE
    assert "instance = _build()" in core._SESSION_BOOTSTRAP
