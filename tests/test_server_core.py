"""Tests for the framework-independent domain layer (:mod:`server_core`).

The HTTP surface lives in ``tests/test_server_fastapi.py``; this module covers the
catalog/introspection helpers, the JSON coercion rules, pagination, static asset resolution, the
rate limiter, the metrics and the invoke bootstrap contracts that are easy to regress.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import model_catalog  # noqa: E402
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


def test_generated_notebook_does_not_append_the_source():
    import notebook_generator

    notebook = notebook_generator.build_notebook(core.CATALOG[0])
    tags = [cell.get("metadata", {}).get("tags", []) for cell in notebook["cells"]]
    assert not any("source" in cell_tags for cell_tags in tags)
    assert notebook["cells"][-1]["cell_type"] == "code"


def test_auto_classifier_exposes_the_modern_loader():
    model = next(item for item in core.CATALOG if item["id"] == "AutoClassifier")
    loader = core.resolve_loader(model, "load")
    assert loader["requires_instance"] is False
    assert loader["path_parameter"] == "filepath"
    assert loader.get("module", "").endswith("_autoencoders")
    assert core.load_spec(model)["loadable"] is True


def test_catalog_exposes_a_loader_for_every_persistable_model():
    """Every model with a checkpoint API is advertised as loadable."""
    loaders = {
        model["id"]: [loader["name"] for loader in model["loaders"]] for model in core.CATALOG
    }
    assert loaders["HMM"] == ["load_model"]
    assert loaders["Arima"] == ["load_model"]
    assert loaders["FourierRegression"] == ["load_model"]
    assert loaders["AutoClassifier"] == ["load"]
    assert loaders["EnsembleClassifier"] == ["load"]
    assert loaders["SimulationEngine"] == ["load"]
    assert loaders["GANRegressor"] == ["load_model"]
    assert loaders["MultiBanditNet"] == ["load"]
    assert loaders["ActionSpace"] == ["load"]


def test_models_without_a_checkpoint_are_not_advertised_as_loadable():
    """Callbacks, live objects and dataset-derived state are not checkpoints."""
    models = {model["id"]: model for model in core.CATALOG}
    for model_id in ("EarlyStopping", "AutoQL", "Env", "GetInsights", "OptionCriticEnv"):
        assert models[model_id]["loaders"] == [], model_id
        assert core.load_spec(models[model_id])["loadable"] is False


def test_dummy_json_value_shapes_payloads_from_the_constructor():
    matrix = core.dummy_json_value("X", "np.ndarray", {"window": 4, "n_features": 3})
    assert len(matrix) == 4
    assert all(len(row) == 3 for row in matrix)
    sequence = core.dummy_json_value("sequence", "list", {"n_observations": 3})
    assert sequence and all(isinstance(item, int) and 0 <= item < 3 for item in sequence)
    shape = core.dummy_json_value("input_shape", "tuple", {})
    assert shape and all(isinstance(item, int) for item in shape)


def test_dummy_json_value_reads_the_input_shape_parameter():
    """``input_shape_parm`` is the name several models use for their feature count."""
    matrix = core.dummy_json_value("X", "np.ndarray", {"input_shape_parm": 4})
    assert matrix and all(len(row) == 4 for row in matrix)


def test_tensor_arguments_are_jsonable_and_sampled_as_matrices():
    """``torch.Tensor``/``tf.Tensor`` are containers: the sandbox converts the list."""
    parameter = _parameter("state", "torch.Tensor", None, required=True, sample="None")
    assert core.parameter_is_jsonable(parameter) is True
    matrix = core.dummy_json_value("state", "torch.Tensor", {"state_dim": 4})
    assert len(matrix) >= 2
    assert all(len(row) == 4 for row in matrix)
    method = {
        "name": "forward",
        "signature": "forward(state: torch.Tensor, multiple_option: bool = False)",
        "parameters": [parameter],
        "kind": "method",
        "inherited": False,
    }
    assert core.method_is_jsonable(method) is True
    arguments = core.sample_arguments({"parameters": [parameter]}, method, {"state_dim": 4})
    assert arguments["state"] == matrix


def test_multi_bandit_network_exposes_forward():
    """The torch model is invokable now that tensor arguments can travel as JSON."""
    entry = next(model for model in core.CATALOG if model["id"] == "MultiBanditNet")
    names = [method["name"] for method in core.invokable_methods(entry)]
    assert "forward" in names
    assert "save" not in names  # side effects stay blocked
    forward = core.find_method(entry, "forward")
    arguments = core.sample_arguments(entry, forward, core.constructor_samples(entry))
    state = arguments["state"]
    assert len(state) >= 2 and all(len(row) == 4 for row in state)


@pytest.mark.parametrize(
    "type_hint,default,expected",
    [
        ("str", "softmax", "softmax"),
        ("str", "leaky_relu", "leaky_relu"),
        ("str", "''", ""),
        ("str", "nn.SELU()", None),
        ("list", "np.array([])", None),
    ],
)
def test_source_expression_defaults_are_ignored(type_hint, default, expected):
    """A default that JSON cannot carry falls back to the generated sample."""
    parameter = _parameter("activation", type_hint, default)
    assert core.playground_fields({"parameters": [parameter]})[0]["default"] == expected


def _displayed(field):
    """Value the Playground renders for ``field`` (models/preview.js:fieldControl)."""
    value = field["default"]
    if value is None and field["required"]:
        value = field["sample"]
    return value


def _parameter(name, type_hint, default, required=False, sample="9"):
    return {
        "name": name,
        "type": type_hint,
        "default": default,
        "required": required,
        "kind": "positional" if required else "keyword_only",
        "variadic": False,
        "description": "",
        "widget": core.guess_widget(type_hint),
        "sample": sample,
    }


@pytest.mark.parametrize(
    "type_hint,default,expected",
    [
        ("int", "4", 4),
        ("int", "0", 0),
        ("float", "2.5", 2.5),
        ("bool", "False", False),
        ("bool", "True", True),
        ("str", "''", ""),
        ("list", "[]", []),
        ("dict", "{}", {}),
    ],
)
def test_declared_defaults_are_never_overridden(type_hint, default, expected):
    """Neither the hardcoded name samples nor the recorded sample win over a default."""
    parameter = _parameter("n_states", type_hint, default)
    field = core.playground_fields({"parameters": [parameter]})[0]
    assert field["default"] == expected
    assert field["sample"] == expected


def test_optional_parameters_without_a_default_are_left_empty():
    """The Playground shows nothing, so no invented value is suggested."""
    parameter = _parameter("dropout", "float", None, sample="0.1")
    field = core.playground_fields({"parameters": [parameter]})[0]
    assert field["default"] is None
    assert field["sample"] is None
    method = {
        "name": "train",
        "signature": "train(dropout: float = 0.1)",
        "parameters": [parameter],
        "kind": "method",
        "inherited": False,
    }
    assert core.sample_arguments({"parameters": [parameter]}, method) == {}


def test_catalog_examples_mirror_the_playground_defaults():
    """Every dummy argument equals the value shown for the same parameter."""
    for model in core.CATALOG:
        context = core.constructor_samples(model)
        playground = {field["name"]: _displayed(field) for field in core.playground_fields(model)}
        for method in core.invokable_methods(model):
            arguments = core.sample_arguments(model, method, context)
            for field in core.method_fields(method, context):
                expected = _displayed(field)
                name = field["name"]
                if expected is None:
                    assert name not in arguments, (model["id"], method["name"], name)
                else:
                    assert arguments[name] == expected, (model["id"], method["name"], name)
                if name in playground:
                    assert playground[name] == expected, (model["id"], method["name"], name)


def test_resolve_asset_blocks_traversal():
    assert core.resolve_asset("/") is not None
    assert core.resolve_asset("/../server_core.py") is None
    assert core.resolve_asset("/.env") is None


def test_content_type_for():
    assert core.content_type_for(core.INDEX_FILE) == "text/html; charset=utf-8"
    assert core.content_type_for(Path("favicon.svg")) == "image/svg+xml; charset=utf-8"
    assert core.content_type_for(Path("model.ipynb")) == ("application/x-ipynb+json; charset=utf-8")


def test_tag_links_point_to_the_catalogue_filter():
    assert core.tag_url("Deep Learning") == "/?tag=Deep%20Learning"
    assert core.tag_url("Time Series", "/?category=x") == "/?tag=Time%20Series"
    html = core.tags_html({"tags": ["Regression", "Time Series"]})
    assert html.count('class="tag"') == 2
    assert 'href="/?tag=Regression"' in html


def test_related_section_explains_the_match():
    model = {
        "id": "Base",
        "icon": "🧪",
        "title": "Base",
        "category_label": "Testing",
        "short_description": "Sonda de prueba",
    }
    html = core.related_html([{"model": model, "score": 3, "shared_tags": ["Regression"]}])
    assert 'id="relacionados"' in html
    assert 'class="related-card" href="/preview/Base"' in html
    assert "Comparte: Regression" in html
    assert core.related_html([]) == ""


def test_model_brief_exposes_the_import_statement():
    model = next(item for item in core.CATALOG if item["id"] == "Arima")
    brief = core.model_brief(model)
    assert brief["import_statement"] == model["import_statement"]


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


def test_metrics_prometheus_exposition():
    metrics = core.Metrics()
    metrics.record("GET /api/models/<model_id>", 200, 0.02)
    metrics.record("GET /api/models/<model_id>", 500, 0.01)
    text = metrics.prometheus()
    assert "# TYPE likelihood_uptime_seconds gauge" in text
    assert "# TYPE likelihood_requests_total counter" in text
    assert 'likelihood_requests_total{endpoint="GET /api/models/<model_id>"} 2' in text
    assert 'likelihood_request_errors_total{endpoint="GET /api/models/<model_id>"} 1' in text
    assert 'likelihood_responses_total{status="500"} 1' in text
    assert 'quantile="0.5"' in text and 'quantile="0.95"' in text
    assert text.endswith("\n")

    escaped = core.Metrics()
    escaped.record('GET "/x\\y"', 200, 0.01)
    # Backslashes and quotes of the label value are escaped for Prometheus.
    assert 'endpoint="GET \\"/x\\\\y\\""' in escaped.prometheus()


def test_related_models_scores_shared_tags_above_the_category():
    def entry(identifier, category, tags):
        return {"id": identifier, "title": identifier, "category": category, "tags": tags}

    base = entry("Base", "regression", ["Time Series"])
    sibling = entry("Sibling", "regression", [])
    cousin = entry("Cousin", "deep", ["Time Series"])

    ranked = model_catalog.related_models([base, sibling, cousin], base)
    assert [item["model"]["id"] for item in ranked] == ["Cousin", "Sibling"]
    assert ranked[0]["score"] == 2 and ranked[0]["shared_tags"] == ["Time Series"]
    assert ranked[1]["score"] == 1 and ranked[1]["shared_tags"] == []
    assert (
        model_catalog.related_models([base, sibling, cousin], base, limit=1)[0]["model"] == cousin
    )
    assert model_catalog.related_models([base], base) == []


def test_related_models_of_the_real_catalog():
    model = next(item for item in core.CATALOG if item["id"] == "Arima")
    related = model_catalog.related_models(core.CATALOG, model, limit=4)
    assert all(item["model"]["id"] != "Arima" for item in related)
    scores = [item["score"] for item in related]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= 1 for score in scores)


def test_job_queue_cancel_stops_the_remaining_runs(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_run(model, parameters):
        started.set()
        release.wait(10)
        return {"type": "Fake", "repr": "Fake()"}

    monkeypatch.setattr(core, "run_playground", slow_run)
    queue = core.JobQueue(workers=1)
    runs = [{"model": {"id": "Fake"}, "parameters": {}} for _ in range(3)]
    job = queue.submit(runs)
    try:
        assert started.wait(10)
        cancelled = queue.cancel(job.id)
        assert cancelled is job
        assert job.cancel_requested is True
        release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not job.done:
            time.sleep(0.02)
        assert job.status == "cancelled"
        assert len(job.results) == 1  # the run in flight finishes, the rest never start
        assert job.active is False and job.to_dict()["cancel_requested"] is True
        assert queue.forget(job.id) is job
        assert queue.get(job.id) is None
        assert queue.cancel("missing") is None
    finally:
        release.set()
        queue.shutdown()


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
