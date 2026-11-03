"""Tests for the catalog web server (:mod:`server`).

The suite starts the real :class:`server.CatalogServer` on an ephemeral port and
exercises it over HTTP, so routing, headers, JSON projections, HTML rendering and
the playground subprocess are covered end to end.  Only the standard library and
pytest are required.

Run with::

    python3 -m pytest tests/test_server.py -v
"""

from __future__ import annotations

import copy
import http.client
import json
import re
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import server

TOKEN = re.compile(r"\{\{[A-Z_]+\}\}")


@pytest.fixture(scope="module")
def base_url():
    """A running server bound to a free port for the whole module."""
    httpd = server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def request(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
):
    """Perform one HTTP request and return ``(status, headers, body)``."""
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    req = Request(base_url + path, data=data, method=method, headers=request_headers)
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def json_of(base_url: str, path: str, **kwargs: Any):
    status, headers, body = request(base_url, path, **kwargs)
    return status, headers, json.loads(body.decode("utf-8"))


def first_model_id() -> str:
    return server.CATALOG[0]["id"]


class _NoRedirect(HTTPRedirectHandler):
    """Turn redirects into :class:`HTTPError` so the status can be asserted."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def request_without_redirect(base_url: str, path: str):
    opener = build_opener(_NoRedirect)
    try:
        with opener.open(base_url + path, timeout=15) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


@pytest.fixture
def fake_catalog(monkeypatch):
    """Minimal catalog with an importable, dependency free class."""
    fake = {
        "id": "Counter",
        "title": "Counter",
        "name": "Counter",
        "module": "collections",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "Contador de pruebas",
        "full_description": "Contador de pruebas",
        "import_statement": "from collections import Counter",
        "signature": "Counter()",
        "parameters": [],
        "methods": [],
        "attributes": [],
        "tags": ["Testing"],
        "source": "class Counter:\n    ...",
        "example": "model = Counter()\n",
        "safe_calls": [],
        "bases": [],
        "kind": "class",
        "path": "collections/__init__.py",
        "lineno": 1,
        "line_count": 1,
    }
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [fake])
    return fake


def test_health(base_url):
    status, _, data = json_of(base_url, "/api/health")
    assert status == 200
    assert data["success"] is True
    assert data["status"] == "ok"
    assert data["models"] == len(server.CATALOG)
    assert data["summary"]["total"] == len(server.CATALOG)


def test_categories(base_url):
    status, _, data = json_of(base_url, "/api/categories")
    assert status == 200
    assert data["count"] == len(data["categories"])
    assert sum(item["count"] for item in data["categories"]) == len(server.CATALOG)
    assert {"id", "label", "icon", "count"} <= set(data["categories"][0])


def test_models_list_is_backwards_compatible(base_url):
    status, headers, data = json_of(base_url, "/api/models")
    assert status == 200
    assert "max-age" in headers["Cache-Control"]
    assert headers["ETag"]
    assert headers["X-API-Version"] == server.API_VERSION
    assert data["count"] == len(server.CATALOG)
    assert data["total"] == len(server.CATALOG)
    assert data["matched"] == len(server.CATALOG)
    assert data["limit"] is None
    assert data["offset"] == 0
    assert data["has_more"] is False
    assert data["next"] is None
    assert "source" not in data["models"][0]


def test_models_pagination(base_url):
    status, headers, first = json_of(base_url, "/api/models?limit=2")
    assert status == 200
    assert first["count"] == 2
    assert first["has_more"] is True
    assert first["limit"] == 2
    assert first["next"] == "/api/models?offset=2&limit=2"
    assert headers["Link"] == f'<{first["next"]}>; rel="next"'

    status, _, second = json_of(base_url, first["next"])
    assert status == 200
    assert second["offset"] == 2
    assert {m["id"] for m in first["models"]}.isdisjoint({m["id"] for m in second["models"]})

    status, _, tail = json_of(base_url, f"/api/models?offset={len(server.CATALOG)}")
    assert status == 200
    assert tail["count"] == 0
    assert tail["has_more"] is False


def test_models_conditional_request(base_url):
    _, headers, _ = json_of(base_url, "/api/models?limit=1")
    etag = headers["ETag"]
    status, cached, body = request(base_url, "/api/models?limit=1", headers={"If-None-Match": etag})
    assert status == 304
    assert body == b""
    assert cached["ETag"] == etag


def test_models_rejects_unknown_field_names(base_url):
    """The listing only exposes snake_case keys."""
    _, _, data = json_of(base_url, "/api/models")
    brief = data["models"][0]
    assert "parameter_count" in brief and "method_count" in brief
    assert "parameterCount" not in brief and "methodCount" not in brief


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=abc", "limit=1000", "offset=-1", "offset=abc"],
)
def test_models_rejects_invalid_pagination(base_url, query):
    status, _, data = json_of(base_url, f"/api/models?{query}")
    assert status == 400
    assert data["success"] is False


def test_models_filters(base_url):
    status, _, data = json_of(base_url, "/api/models?category=regression")
    assert status == 200
    assert data["models"]
    assert all(model["category"] == "regression" for model in data["models"])
    assert data["matched"] == data["count"]
    assert data["total"] == len(server.CATALOG)

    status, _, searched = json_of(base_url, "/api/models?q=hmm")
    assert status == 200
    assert any(model["id"] == "HMM" for model in searched["models"])


def test_model_detail(base_url):
    model_id = first_model_id()
    status, _, data = json_of(base_url, f"/api/models/{model_id}")
    assert status == 200
    model = data["model"]
    assert model["id"] == model_id
    assert model["source"]
    assert model["source_url"] == f"/api/models/{model_id}/source"
    assert model["source_download_url"].endswith("?download=1")
    assert model["playground_url"] == f"/api/playground/{model_id}"
    assert model["playground"]["model_id"] == model_id


def test_model_detail_unknown(base_url):
    status, _, data = json_of(base_url, "/api/models/does-not-exist")
    assert status == 404
    assert data["success"] is False


def test_model_source(base_url):
    model_id = first_model_id()
    status, headers, body = request(base_url, f"/api/models/{model_id}/source")
    assert status == 200
    assert headers["Content-Type"].startswith("text/x-python")
    assert "Content-Disposition" not in headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert f"class {server.MODELS_BY_ID[model_id]['name']}" in body.decode("utf-8")


def test_model_source_download(base_url):
    model_id = first_model_id()
    status, headers, _ = request(base_url, f"/api/models/{model_id}/source?download=1")
    assert status == 200
    assert headers["Content-Disposition"] == f'attachment; filename="{model_id}.py"'


def test_model_source_unknown(base_url):
    status, _, data = json_of(base_url, "/api/models/nope/source")
    assert status == 404
    assert data["success"] is False


def test_search_get(base_url):
    status, _, data = json_of(base_url, "/api/search?q=hmm")
    assert status == 200
    assert data["query"] == "hmm"
    assert data["suggestions"]
    assert {"id", "title", "category", "category_label", "icon"} <= set(data["suggestions"][0])
    assert "source" not in data["results"][0]

    status, _, limited = json_of(base_url, "/api/search?q=a&limit=1")
    assert status == 200
    assert limited["count"] <= 1


@pytest.mark.parametrize("path", ["/api/search", "/api/search?q=", "/api/search?q=a&limit=0"])
def test_search_get_rejects_bad_requests(base_url, path):
    status, _, data = json_of(base_url, path)
    assert status == 400
    assert data["success"] is False


def test_search_post_is_deprecated_but_works(base_url):
    status, headers, data = json_of(
        base_url, "/api/search", method="POST", payload={"query": "hmm"}
    )
    assert status == 200
    assert headers["Deprecation"] == "true"
    assert "successor-version" in headers["Link"]
    assert data["deprecated"] is True
    assert data["results"]


def test_notebook_download(base_url):
    model_id = first_model_id()
    status, headers, body = request(base_url, f"/api/notebook/{model_id}")
    assert status == 200
    assert headers["Content-Type"].startswith("application/x-ipynb+json")
    assert headers["Content-Disposition"] == f'attachment; filename="{model_id}.ipynb"'
    notebook = json.loads(body.decode("utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_notebook_post(base_url):
    model_id = first_model_id()
    status, _, data = json_of(
        base_url, "/api/notebook", method="POST", payload={"model_id": model_id}
    )
    assert status == 200
    assert data["model_id"] == model_id
    assert data["filename"] == f"{model_id}.ipynb"
    assert data["notebook"]["cells"]

    status, _, error = json_of(base_url, "/api/notebook", method="POST", payload={})
    assert status == 400
    assert error["success"] is False


def test_index_page(base_url):
    status, headers, body = request(base_url, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "Catálogo de Modelos" in body.decode("utf-8")
    assert "max-age" in headers["Cache-Control"]
    assert headers["ETag"]


def test_index_conditional_request(base_url):
    _, headers, _ = request(base_url, "/")
    etag = headers["ETag"]
    status, cached_headers, body = request(base_url, "/", headers={"If-None-Match": etag})
    assert status == 304
    assert body == b""
    assert cached_headers["ETag"] == etag


def test_preview_page(base_url):
    model_id = first_model_id()
    status, headers, body = request(base_url, f"/preview/{model_id}")
    text = body.decode("utf-8")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store"
    assert TOKEN.search(text) is None
    assert 'id="playground-data"' in text
    assert 'id="invoke-data"' in text
    assert "/api/playground" in text
    assert f"/api/models/{model_id}/invoke" in text
    assert f"/api/models/{model_id}/source?download=1" in text


def test_preview_initialises_the_invoke_panel():
    """The invoke tab must be rendered and wired, not just declared.

    ``renderInvoke()`` fills the method selector; without it the tab shows an
    empty dropdown and the default ``sin sesión`` badge forever.
    """
    template = server.load_template()
    assert re.search(r"^\s*renderInvoke\(\);", template, re.MULTILINE)
    assert "invokeDom.select.addEventListener('change', renderInvokeMethod)" in template
    assert "invokeDom.run.addEventListener('click', runInvoke)" in template
    assert "invokeDom.closeSession.addEventListener('click'" in template
    assert "invokeDom.sample.addEventListener('click', fillSampleArguments)" in template
    assert "invokeDom.deploy.addEventListener('click', deployEndpoint)" in template
    assert re.search(r'id="invoke-use-session" checked', template)
    assert 'id="sample-invoke"' in template
    assert 'id="deploy-invoke"' in template
    assert 'id="invoke-endpoint"' in template


def test_preview_marks_models_that_need_python(base_url):
    status, _, body = request(base_url, "/preview/AutoQL")
    text = body.decode("utf-8")
    assert status == 200
    assert '"buildable": false' in text
    assert "tf.keras.Model" in text


def test_invoke_spec_is_embedded_in_the_preview():
    model = next(item for item in server.CATALOG if server.invokable_methods(item))
    document = server.render_preview(model, server.load_template())
    spec = server.invoke_spec(model)
    assert spec["model_name"] == model["name"]
    assert spec["import_statement"] == model["import_statement"]
    for method in spec["methods"]:
        assert f'"{method["name"]}"' in document


def test_preview_legacy_two_part_url(base_url):
    model = server.CATALOG[0]
    status, _, body = request(base_url, f"/preview/{model['category']}/{model['id']}")
    assert status == 200
    assert server._escape(model["title"]) in body.decode("utf-8")


def test_preview_unknown_model(base_url):
    status, headers, body = request(base_url, "/preview/does-not-exist")
    assert status == 404
    assert headers["Content-Type"].startswith("text/html")
    assert "Modelo no encontrado" in body.decode("utf-8")


def test_render_preview_escapes_html():
    model = copy.deepcopy(next(item for item in server.CATALOG if item["parameters"]))
    model["title"] = "<script>alert(1)</script>"
    document = server.render_preview(model, server.load_template())
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document


def test_playground_json_is_script_safe():
    model = copy.deepcopy(next(item for item in server.CATALOG if item["parameters"]))
    model["parameters"][0]["description"] = "</script><b>boom</b>"
    document = server.render_preview(model, server.load_template())
    assert "</script><b>boom</b>" not in document
    assert "\\u003c/script\\u003e" in document


def test_fallback_template_renders():
    model = server.CATALOG[0]
    document = server.render_preview(model, server._FALLBACK_TEMPLATE)
    assert TOKEN.search(document) is None
    assert server._escape(model["title"]) in document


def test_favicon(base_url):
    status, headers, body = request(base_url, "/favicon.svg")
    assert status == 200
    assert headers["Content-Type"] == "image/svg+xml; charset=utf-8"
    assert body.lstrip().startswith(b"<svg")


def test_favicon_ico_redirects(base_url):
    status, headers, _ = request_without_redirect(base_url, "/favicon.ico")
    assert status == 308
    assert headers["Location"] == "/favicon.svg"


def test_static_traversal_is_blocked(base_url):
    status, _, body = request(base_url, "/..%2Fserver.py")
    assert status == 404
    assert b"import argparse" not in body


def test_head_request(base_url):
    status, headers, body = request(base_url, "/api/health", method="HEAD")
    assert status == 200
    assert body == b""
    assert int(headers["Content-Length"]) > 0


def test_method_rejection(base_url):
    status, headers, data = json_of(base_url, "/api/models", method="PUT")
    assert status == 405
    assert headers["Allow"] == "GET, HEAD, POST, DELETE, OPTIONS"
    assert headers["X-Request-Id"]
    assert data["success"] is False


def test_body_too_large(base_url):
    host, port = base_url[len("http://") :].split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=15)
    connection.putrequest("POST", "/api/notebook")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(server.MAX_BODY_BYTES + 1))
    connection.putheader("Connection", "close")
    connection.endheaders()
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 413
    assert payload["success"] is False


@pytest.fixture
def fake_parameter_catalog(monkeypatch):
    """Catalog entry with one required integer argument."""
    fake = {
        "id": "Counter",
        "title": "Counter",
        "name": "Counter",
        "module": "collections",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "",
        "full_description": "",
        "import_statement": "from collections import Counter",
        "signature": "Counter(value: int)",
        "parameters": [
            {
                "name": "value",
                "type": "int",
                "default": None,
                "required": True,
                "kind": "positional",
                "variadic": False,
                "description": "Valor inicial",
                "widget": "int",
                "sample": "1",
            }
        ],
        "methods": [],
        "attributes": [],
        "tags": [],
        "source": "class Counter:\n    ...",
        "example": "",
        "safe_calls": [],
        "bases": [],
        "kind": "class",
        "path": "collections/__init__.py",
        "lineno": 1,
        "line_count": 1,
    }
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [fake])
    return fake


def test_playground_spec_endpoint(base_url):
    model_id = first_model_id()
    status, _, data = json_of(base_url, f"/api/playground/{model_id}")
    assert status == 200
    assert data["enabled"] is True
    spec = data["playground"]
    assert spec["model_id"] == model_id
    assert spec["endpoint"] == "/api/playground"
    assert spec["fields"]
    assert {"name", "type", "kind", "required", "default"} <= set(spec["fields"][0])


def test_playground_spec_unknown(base_url):
    status, _, data = json_of(base_url, "/api/playground/does-not-exist")
    assert status == 404
    assert data["success"] is False


def test_playground_get_is_rejected(base_url):
    status, headers, data = json_of(base_url, "/api/playground")
    assert status == 405
    assert headers["Allow"] == "POST"
    assert data["success"] is False


def test_run_playground_helper():
    result = server.run_playground({"module": "collections", "name": "Counter"}, {})
    assert result["type"] == "Counter"
    assert result["module"] == "collections"
    assert "Counter" in result["repr"]
    assert result["elapsed"] >= 0


def test_run_playground_reports_constructor_errors():
    with pytest.raises(server.PlaygroundError) as info:
        server.run_playground({"module": "collections", "name": "NoSuchThing"}, {})
    assert info.value.status == 422
    assert "AttributeError" in str(info.value)
    assert info.value.detail


def test_playground_http_run(base_url, fake_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    assert status == 200
    assert data["success"] is True
    assert data["type"] == "Counter"
    assert data["parameters"] == {}


def test_playground_rejects_unknown_parameters(base_url, fake_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {"missing": 1}},
    )
    assert status == 400
    assert data["unknown"] == ["missing"]


def test_playground_requires_known_model(base_url):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "does-not-exist", "parameters": {}},
    )
    assert status == 404
    assert data["success"] is False


def test_playground_requires_model_id(base_url):
    status, _, data = json_of(base_url, "/api/playground", method="POST", payload={})
    assert status == 400
    assert data["success"] is False


def test_playground_requires_parameters_object(base_url, fake_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": [1, 2]},
    )
    assert status == 400
    assert data["success"] is False


def test_playground_reports_missing_required_parameters(base_url, fake_parameter_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    assert status == 400
    assert data["missing"] == ["value"]


def test_playground_reports_invalid_types(base_url, fake_parameter_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {"value": "not-a-number"}},
    )
    assert status == 400
    assert data["invalid"]


def test_playground_can_be_disabled(base_url, fake_catalog, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "playground_enabled", False)
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    assert status == 403
    assert data["success"] is False


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        ("int", "12", 12),
        ("int", 3.0, 3),
        ("float", "0.5", 0.5),
        ("float", 2, 2.0),
        ("bool", "true", True),
        ("bool", "no", False),
        ("bool", 1, True),
        ("str", "softmax", "softmax"),
        ("raw", [1, 2], [1, 2]),
        ("raw", {"a": 1}, {"a": 1}),
    ],
)
def test_coerce_playground_value(kind, value, expected):
    field = {"name": "parameter", "kind": kind}
    assert server.coerce_playground_value(field, value) == expected


@pytest.mark.parametrize(
    "kind, value",
    [("int", "x"), ("int", 1.5), ("int", True), ("float", None), ("bool", "maybe"), ("str", 3)],
)
def test_coerce_playground_value_rejects(kind, value):
    field = {"name": "parameter", "kind": kind}
    with pytest.raises(ValueError):
        server.coerce_playground_value(field, value)


@pytest.mark.parametrize(
    "query, expected",
    [
        ({}, (None, 0)),
        ({"limit": ["5"]}, (5, 0)),
        ({"offset": ["3"]}, (None, 3)),
        ({"limit": ["2"], "offset": ["4"]}, (2, 4)),
    ],
)
def test_parse_pagination(query, expected):
    assert server.parse_pagination(query) == expected


def test_playground_spec_includes_samples():
    model = next(item for item in server.CATALOG if any(p["required"] for p in item["parameters"]))
    fields = server.playground_fields(model)
    required = [field for field in fields if field["required"]]
    assert required
    assert any(field["sample"] is not None for field in required)


def test_default_value_parsing():
    assert server._default_value({"default": "1", "type": "int", "widget": "int"}) == 1
    assert server._default_value({"default": "'softmax'", "type": "str", "widget": "str"}) == (
        "softmax"
    )
    assert server._default_value({"default": "softmax", "type": "str", "widget": "str"}) == (
        "softmax"
    )
    assert server._default_value({"default": "(0, 100)", "type": "Tuple", "widget": "raw"}) == [
        0,
        100,
    ]
    assert server._default_value({"default": None, "type": "int", "widget": "int"}) is None
    # Containers cannot be expressed as JSON literals: a dummy payload is generated.
    assert server._sample_value(
        {"name": "X", "sample": "np.zeros(3)", "type": "np.ndarray", "widget": "raw"}
    ) == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    assert server._sample_value(
        {"name": "sequence", "type": "list", "widget": "raw"}, {"n_observations": 2}
    ) == [0, 1, 0]
    assert server._sample_value({"sample": "1", "type": "int", "widget": "int"}) == 1
    # Method parameters carry no sample: the default is used instead.
    assert (
        server._sample_value({"name": "epochs", "type": "int", "widget": "int", "default": "5"})
        == 5
    )


def test_guess_widget_unwraps_optional_and_union():
    assert server.guess_widget("Optional[int]") == "int"
    assert server.guess_widget("int | None") == "int"
    assert server.guess_widget("Union[int, float]") == "float"
    assert server.guess_widget("Optional[Dict[str, Any]]") == "raw"
    assert server.guess_widget("Optional[Path]") == "text"


def test_constructor_samples_are_model_friendly():
    catalog = {model["id"]: model for model in server.CATALOG}
    assert server.constructor_samples(catalog["HMM"]) == {
        "n_states": 2,
        "n_observations": 3,
    }
    arima = server.constructor_samples(catalog["Arima"])
    assert arima["datapoints"] == [1.0, 2.0, 3.0]
    classifier = server.constructor_samples(catalog["AutoClassifier"])
    assert classifier["input_shape_parm"] == 4
    assert classifier["num_classes"] == 3
    assert classifier["units"] == 8


def test_models_that_need_python_objects_are_marked():
    catalog = {model["id"]: model for model in server.CATALOG}
    autoql = server.playground_spec(catalog["AutoQL"])
    assert autoql["buildable"] is False
    assert {item["name"] for item in autoql["unsupported"]} == {"env", "model"}
    insights = server.playground_spec(catalog["GetInsights"])
    assert [item["name"] for item in insights["unsupported"]] == ["model"]
    critic = server.playground_spec(catalog["OptionCriticEnv"])
    assert "claves enteras" in critic["unsupported"][0]["reason"]
    assert server.playground_spec(catalog["HMM"])["buildable"] is True
    assert server.invoke_spec(catalog["HMM"])["buildable"] is True


def test_script_json_escapes_markup():
    payload = server._script_json({"value": "</script><b>&"})
    assert "</script>" not in payload
    assert "\\u003c/script\\u003e" in payload
    assert json.loads(payload) == {"value": "</script><b>&"}


# ---------------------------------------------------------------------------
# Seguridad: token, CORS, rate limiting y lectura del cuerpo
# ---------------------------------------------------------------------------


def test_api_token_is_enforced(base_url, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "api_token", "sesamo")

    status, headers, data = json_of(base_url, "/api/models")
    assert status == 401
    assert headers["WWW-Authenticate"].startswith("Bearer")
    assert data["success"] is False

    # ``/api/health`` and the static assets stay open.
    assert json_of(base_url, "/api/health")[0] == 200
    assert request(base_url, "/")[0] == 200

    status, _, _ = json_of(base_url, "/api/models", headers={"Authorization": "Bearer sesamo"})
    assert status == 200
    status, _, _ = json_of(base_url, "/api/models", headers={"X-API-Token": "sesamo"})
    assert status == 200
    status, _, _ = json_of(base_url, "/api/models", headers={"Authorization": "Bearer otro"})
    assert status == 401


def test_cors_headers_and_preflight(base_url, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "cors_origins", ("http://app.test",))

    status, headers, _ = request(base_url, "/api/models", headers={"Origin": "http://app.test"})
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "http://app.test"
    assert headers["Vary"] == "Origin"

    _, headers, _ = request(base_url, "/api/models", headers={"Origin": "http://evil.test"})
    assert "Access-Control-Allow-Origin" not in headers

    status, headers, body = request(
        base_url, "/api/models", method="OPTIONS", headers={"Origin": "http://app.test"}
    )
    assert status == 204
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert "Authorization" in headers["Access-Control-Allow-Headers"]
    assert body == b""

    status, _, _ = request(base_url, "/api/models", method="OPTIONS")
    assert status == 405


def test_wildcard_cors_origin(base_url, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "cors_origins", ("*",))
    status, headers, _ = request(base_url, "/api/models", headers={"Origin": "http://any.test"})
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_rate_limit(base_url, monkeypatch):
    limiter = server.RateLimiter(2)
    monkeypatch.setattr(server.ModelAPIHandler, "rate_limiter", limiter)

    assert json_of(base_url, "/api/categories")[0] == 200
    assert json_of(base_url, "/api/categories")[0] == 200
    status, headers, data = json_of(base_url, "/api/categories")
    assert status == 429
    assert int(headers["Retry-After"]) >= 1
    assert data["success"] is False

    limiter.reset()
    assert json_of(base_url, "/api/categories")[0] == 200


def test_unsupported_media_type(base_url, fake_catalog):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
        headers={"Content-Type": "text/plain"},
    )
    assert status == 415
    assert data["success"] is False


def test_chunked_body_is_accepted(base_url):
    """``Transfer-Encoding: chunked`` works without a ``Content-Length``."""
    host, port = base_url[len("http://") :].split(":")
    payload = json.dumps({"model_id": first_model_id()}).encode("utf-8")
    connection = http.client.HTTPConnection(host, int(port), timeout=15)
    connection.putrequest("POST", "/api/notebook")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Transfer-Encoding", "chunked")
    connection.endheaders()
    for index in range(0, len(payload), 8):
        chunk = payload[index : index + 8]
        connection.send(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
    connection.send(b"0\r\n\r\n")
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    assert response.status == 200
    assert body["model_id"] == first_model_id()


def test_incomplete_body_is_rejected(base_url):
    """A ``Content-Length`` larger than the body must answer ``400``, not hang."""
    host, port = base_url[len("http://") :].split(":")
    with socket.create_connection((host, int(port)), timeout=15) as connection:
        connection.sendall(
            b"POST /api/notebook HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
            b'{"model_id"'
        )
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = connection.recv(4096)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks).decode("utf-8", "replace")
    assert raw.startswith("HTTP/1.1 400")
    assert "incompleto" in raw


# ---------------------------------------------------------------------------
# Documentación, métricas y búsqueda facetada
# ---------------------------------------------------------------------------


def test_openapi_document(base_url):
    status, headers, data = json_of(base_url, "/api/openapi.json")
    assert status == 200
    assert headers["ETag"]
    assert data["openapi"].startswith("3.1")
    assert {"/api/models", "/api/playground", "/api/search"} <= set(data["paths"])
    assert {"/api/models/{model_id}/invoke", "/api/sessions/{session_id}/invoke"} <= set(
        data["paths"]
    )
    schemas = data["components"]["schemas"]
    assert data["components"]["schemas"]["PlaygroundRequest"]["properties"]["model_id"]["enum"]
    assert {"InvokeRequest", "InvokeResult", "Session", "SessionRequest"} <= set(schemas)
    assert {"SampleRequest", "SampleResponse"} <= set(schemas)
    assert "/api/models/{model_id}/sample" in data["paths"]
    parameter_schemas = [name for name in schemas if name.endswith("Parameters")]
    assert parameter_schemas


def test_openapi_parameters_mirror_the_catalog():
    model = next(item for item in server.CATALOG if any(p["required"] for p in item["parameters"]))
    schema = server.model_request_schema(model)
    required = [p["name"] for p in model["parameters"] if p["required"] and not p["variadic"]]
    assert schema["required"] == required
    assert set(schema["properties"]) == {
        p["name"] for p in model["parameters"] if not p["variadic"]
    }


def test_docs_page(base_url):
    status, headers, body = request(base_url, "/docs")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"/api/openapi.json" in body


def test_health_reports_metrics_and_jobs(base_url):
    json_of(base_url, "/api/categories")
    status, _, data = json_of(base_url, "/api/health")
    assert status == 200
    assert data["api_version"] == server.API_VERSION
    assert data["metrics"]["total_requests"] >= 1
    endpoint = data["metrics"]["endpoints"]["GET /api/categories"]
    assert {"p50", "p95", "max"} <= set(endpoint["latency_ms"])
    assert data["jobs"]["workers"] >= 1
    assert data["playground"]["timeout_seconds"] == server.PLAYGROUND_TIMEOUT
    assert data["catalog"]["reloads"] >= 1


def test_metrics_record_error_statuses():
    metrics = server.Metrics()
    metrics.record("GET /x", 200, 0.01)
    metrics.record("GET /x", 500, 0.03)
    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 2
    assert snapshot["total_errors"] == 1
    assert snapshot["endpoints"]["GET /x"]["statuses"] == {"200": 1, "500": 1}
    assert snapshot["endpoints"]["GET /x"]["latency_ms"]["max"] == 30.0


def test_metrics_cap_the_number_of_endpoints():
    metrics = server.Metrics(max_endpoints=2)
    for index in range(5):
        metrics.record(f"GET /custom/{index}", 200, 0.001)
    assert len(metrics.endpoints) == 3  # two real labels plus "other"
    assert metrics.snapshot()["endpoints"]["other"]["count"] == 3


def test_endpoint_label_groups_model_ids():
    assert server.endpoint_label("GET", "/api/models/HMM") == "GET /api/models/<model_id>"
    assert (
        server.endpoint_label("GET", "/api/models/HMM/source")
        == "GET /api/models/<model_id>/source"
    )
    assert (
        server.endpoint_label("GET", "/api/playground/jobs/abc")
        == "GET /api/playground/jobs/<job_id>"
    )
    assert server.endpoint_label("GET", "/api/models") == "GET /api/models"


def test_search_facets_and_tag_filter(base_url):
    status, _, data = json_of(base_url, "/api/search?q=a")
    assert status == 200
    assert data["facets"]["categories"]
    assert data["facets"]["tags"]
    tag = data["facets"]["tags"][0]["tag"]

    status, _, filtered = json_of(base_url, f"/api/search?q=a&tags={quote(tag)}")
    assert status == 200
    assert filtered["tags"] == [tag]
    assert filtered["total"] <= data["total"]

    status, _, unknown = json_of(base_url, "/api/search?q=a&tags=no-existe")
    assert status == 200
    assert unknown["total"] == 0


def test_models_accepts_tags_filter(base_url):
    model = next(item for item in server.CATALOG if item["tags"])
    tag = model["tags"][0]
    status, _, data = json_of(base_url, f"/api/models?tags={quote(tag)}")
    assert status == 200
    assert data["count"] >= 1
    assert all(tag.lower() in {t.lower() for t in item["tags"]} for item in data["models"])


def test_search_post_keeps_deprecation_headers(base_url):
    status, headers, data = json_of(
        base_url, "/api/search", method="POST", payload={"query": "hmm", "tags": []}
    )
    assert status == 200
    assert headers["Deprecation"] == "true"
    assert headers["Sunset"] == server.SEARCH_ALIAS_SUNSET
    assert data["facets"]["categories"]


def test_normalize_tags():
    assert server.normalize_tags("Deep Learning") == ["deep learning"]
    assert server.normalize_tags(["a, b", "C"]) == ["a", "b", "c"]
    assert server.normalize_tags(None) == []


# ---------------------------------------------------------------------------
# Historial de las fuentes
# ---------------------------------------------------------------------------


def test_model_versions(base_url):
    model_id = first_model_id()
    status, _, data = json_of(base_url, f"/api/models/{model_id}/versions?limit=2")
    if status == 503:
        pytest.skip("git no está disponible en este entorno")
    assert status == 200
    assert data["model_id"] == model_id
    assert data["path"].endswith(".py")
    assert 0 < len(data["versions"]) <= 2
    assert {"commit", "date", "author", "subject"} <= set(data["versions"][0])


def test_model_source_diff(base_url):
    model_id = first_model_id()
    status, headers, data = json_of(base_url, f"/api/models/{model_id}/source?diff=HEAD")
    if status == 503:
        pytest.skip("git no está disponible en este entorno")
    assert status == 200
    assert data["ref"] == "HEAD"
    assert data["changes"] == data["added"] + data["removed"]
    assert int(headers["X-Diff-Changes"]) == data["changes"]
    assert data["diff"].startswith("--- ") or data["changes"] == 0


def test_source_diff_helper_reports_no_changes():
    model = server.CATALOG[0]
    diff = server.source_diff(model["path"], model["source"], "HEAD")
    assert diff["ref"] == "HEAD"
    assert diff["changes"] == diff["added"] + diff["removed"]


def test_model_source_diff_rejects_bad_ref(base_url):
    status, _, data = json_of(base_url, f"/api/models/{first_model_id()}/source?diff=--help")
    assert status == 400
    assert data["success"] is False


def test_model_source_diff_unknown_ref(base_url):
    status, _, data = json_of(base_url, f"/api/models/{first_model_id()}/source?diff=no-such-ref")
    if status == 503:
        pytest.skip("git no está disponible en este entorno")
    assert status == 404
    assert data["success"] is False


def test_git_helpers_are_safe():
    with pytest.raises(server.ValidationError):
        server.git_blob("likelihood/models/hmm/base.py", "HEAD; rm -rf /")


# ---------------------------------------------------------------------------
# Notebooks con los valores del playground
# ---------------------------------------------------------------------------


def test_notebook_uses_playground_values(base_url, fake_parameter_catalog):
    status, _, data = json_of(
        base_url,
        "/api/notebook",
        method="POST",
        payload={"model_id": "Counter", "parameters": {"value": 7}},
    )
    assert status == 200
    assert data["parameters"] == {"value": 7}
    assert data["notebook"]["metadata"]["likelihood"]["parameters"] == {"value": 7}
    interactive = next(
        cell
        for cell in data["notebook"]["cells"]
        if "interactive" in cell["metadata"].get("tags", [])
    )
    assert "IntSlider(value=7" in interactive["source"]


def test_notebook_download_accepts_parameters(base_url, fake_parameter_catalog):
    status, _, body = request(
        base_url, f"/api/notebook/Counter?parameters={quote(json.dumps({'value': 3}))}"
    )
    notebook = json.loads(body.decode("utf-8"))
    assert status == 200
    assert notebook["metadata"]["likelihood"]["parameters"] == {"value": 3}


def test_notebook_rejects_unknown_parameters(base_url, fake_parameter_catalog):
    status, _, data = json_of(
        base_url,
        "/api/notebook",
        method="POST",
        payload={"model_id": "Counter", "parameters": {"nope": 1}},
    )
    assert status == 400
    assert data["unknown"] == ["nope"]


def test_apply_parameter_values_keeps_the_catalog_intact():
    model = copy.deepcopy(next(item for item in server.CATALOG if item["parameters"]))
    name = model["parameters"][0]["name"]
    patched = server.notebook_generator.apply_parameter_values(model, {name: "valor"})
    assert patched["parameters"][0]["default"] == "'valor'"
    assert model["parameters"][0]["default"] != "'valor'"


# ---------------------------------------------------------------------------
# Playground asíncrono (jobs)
# ---------------------------------------------------------------------------


@pytest.fixture
def job_queue(monkeypatch):
    """Isolated job queue so every test polls its own jobs."""
    queue = server.JobQueue(workers=1, retention=10, maximum=5)
    monkeypatch.setattr(server.ModelAPIHandler, "jobs", queue)
    try:
        yield queue
    finally:
        queue.shutdown()


def wait_for_job(base_url: str, job_id: str, timeout: float = 60.0) -> Dict[str, Any]:
    """Poll ``GET /api/playground/jobs/<id>`` until the job finishes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _, data = json_of(base_url, f"/api/playground/jobs/{job_id}")
        assert status == 200
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"El job {job_id} no terminó en {timeout} segundos")


def test_playground_job_single(base_url, fake_catalog, job_queue):
    status, headers, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    assert status == 202
    assert headers["Location"] == f"/api/playground/jobs/{data['job_id']}"
    assert data["poll_url"] == headers["Location"]
    assert data["batch"] is False

    job = wait_for_job(base_url, data["job_id"])
    assert job["status"] == "succeeded"
    assert job["runs"] == 1
    assert job["results"][0]["type"] == "Counter"
    assert job["elapsed"] is not None


def test_playground_job_batch(base_url, fake_catalog, job_queue):
    status, _, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload={
            "jobs": [
                {"model_id": "Counter", "parameters": {}},
                {"model_id": "Counter", "parameters": {}},
            ]
        },
    )
    assert status == 202
    assert data["batch"] is True
    assert data["runs"] == 2

    job = wait_for_job(base_url, data["job_id"])
    assert job["status"] == "succeeded"
    assert [result["type"] for result in job["results"]] == ["Counter", "Counter"]


def test_playground_job_accepts_a_bare_list(base_url, fake_catalog, job_queue):
    status, _, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload=[{"model_id": "Counter", "parameters": {}}],
    )
    assert status == 202
    assert data["runs"] == 1
    wait_for_job(base_url, data["job_id"])


def test_playground_job_reports_invalid_runs(base_url, fake_catalog, job_queue):
    status, _, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload={"jobs": [{"model_id": "Counter"}, {"model_id": "nope"}]},
    )
    assert status == 404
    assert data["run_index"] == 1


def test_playground_job_rejects_empty_batches(base_url, fake_catalog, job_queue):
    status, _, data = json_of(base_url, "/api/playground/jobs", method="POST", payload={"jobs": []})
    assert status == 400
    assert data["success"] is False


def test_playground_job_rejects_oversized_batches(base_url, fake_catalog, job_queue, monkeypatch):
    monkeypatch.setattr(server, "PLAYGROUND_MAX_BATCH", 1)
    status, _, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload=[
            {"model_id": "Counter", "parameters": {}},
            {"model_id": "Counter", "parameters": {}},
        ],
    )
    assert status == 400
    assert "lote" in data["error"].lower()


def test_playground_job_unknown_id(base_url, job_queue):
    status, _, data = json_of(base_url, "/api/playground/jobs/does-not-exist")
    assert status == 404
    assert data["success"] is False


def test_playground_job_list(base_url, fake_catalog, job_queue):
    json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    status, _, data = json_of(base_url, "/api/playground/jobs")
    assert status == 200
    assert data["count"] >= 1
    assert "results" not in data["jobs"][0]
    assert {"job_id", "status", "runs"} <= set(data["jobs"][0])


def test_playground_jobs_disabled_with_the_playground(
    base_url, fake_catalog, job_queue, monkeypatch
):
    monkeypatch.setattr(server.ModelAPIHandler, "playground_enabled", False)
    status, _, data = json_of(
        base_url,
        "/api/playground/jobs",
        method="POST",
        payload={"model_id": "Counter", "parameters": {}},
    )
    assert status == 403
    assert data["success"] is False


# ---------------------------------------------------------------------------
# Sandbox del proceso hijo y recarga del catálogo
# ---------------------------------------------------------------------------


def test_playground_limits_are_positive():
    limits = server.playground_limits()
    assert limits["cpu"] >= 1
    assert limits["memory"] >= 1
    assert limits["files"] >= 1


def test_playground_blocks_dangerous_imports(tmp_path, monkeypatch):
    module = tmp_path / "lm_sandbox_probe.py"
    module.write_text(
        textwrap.dedent("""
            import subprocess


            class Probe:
                \"\"\"Modelo que intenta escapar del sandbox.\"\"\"

                def __init__(self, value: int = 1) -> None:
                    self.value = value
            """),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(server.PlaygroundError) as info:
        server.run_playground({"module": "lm_sandbox_probe", "name": "Probe"}, {})
    assert "ImportError" in str(info.value)
    assert "subprocess" in str(info.value)
    assert "LM_PLAYGROUND_BLOCKED_MODULES" in str(info.value)


def test_playground_allows_imports_made_by_libraries(tmp_path, monkeypatch):
    """A library may import ``subprocess`` internally (matplotlib, torch, ...).

    The guard is frame aware: only the imports performed by the catalog code are
    rejected, so a model can still be built when one of its dependencies pulls a
    "blocked" module in.
    """
    (tmp_path / "lm_sandbox_library.py").write_text(
        "import subprocess\n\nVALUE = subprocess.__name__\n", encoding="utf-8"
    )
    (tmp_path / "lm_sandbox_model.py").write_text(
        textwrap.dedent("""
            import lm_sandbox_library


            class Probe:
                \"\"\"Modelo que usa una librería que importa subprocess.\"\"\"

                def __init__(self, value: int = 1) -> None:
                    self.value = value

                def library(self) -> str:
                    return lm_sandbox_library.VALUE
            """),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = server.run_playground({"module": "lm_sandbox_model", "name": "Probe"}, {})
    assert result["type"] == "Probe"


def test_playground_guard_is_frame_aware():
    """The child guard knows the catalog package and the repository root."""
    assert "_catalog_frame" in server._PLAYGROUND_BOOTSTRAP
    assert "project_root" in server._PLAYGROUND_BOOTSTRAP
    assert "LM_PLAYGROUND_BLOCKED_MODULES" in server._PLAYGROUND_BOOTSTRAP


# ---------------------------------------------------------------------------
# Invocación de métodos: POST /api/models/<id>/invoke
# ---------------------------------------------------------------------------

INVOKE_MODULE = '''
"""Modelo de prueba para la invocación de métodos."""


class Calculator:
    """Calculadora de prueba con estado."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def add(self, amount: int = 1) -> int:
        """Suma ``amount`` al valor actual."""
        self.value += amount
        return self.value

    def set_value(self, value: int) -> int:
        """Fija el valor actual."""
        self.value = value
        return self.value

    def history(self) -> list:
        """Devuelve una lista (para comprobar la serialización)."""
        return [1, 2, 3]

    def fail(self) -> None:
        """Falla siempre."""
        raise ValueError("boom")

    def save(self, filename: str = "x") -> None:
        """Método con efectos secundarios: nunca debe invocarse."""
        raise AssertionError("no debería llamarse")

    @property
    def total(self) -> int:
        """Propiedad: no es invocable."""
        return self.value
'''


def method_parameter(name, annotation, required=False, default=None, widget=None):
    """Parameter record as produced by ``model_catalog``."""
    return {
        "name": name,
        "type": annotation,
        "default": default,
        "required": required,
        "kind": "positional" if required else "keyword_only",
        "variadic": False,
        "description": "",
        "widget": widget or server.guess_widget(annotation),
        "sample": "1",
    }


def calculator_entry():
    """Catalog entry of the probe model used by the invoke tests."""
    return {
        "id": "Calculator",
        "title": "Calculator",
        "name": "Calculator",
        "module": "lm_invoke_probe",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "Calculadora de prueba",
        "full_description": "Calculadora de prueba",
        "import_statement": "from lm_invoke_probe import Calculator",
        "signature": "Calculator(value: int = 0)",
        "parameters": [method_parameter("value", "int", default="0")],
        "methods": [
            {
                "name": "add",
                "signature": "add(amount: int = 1) -> int",
                "description": "Suma",
                "parameters": [method_parameter("amount", "int", default="1")],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "set_value",
                "signature": "set_value(value: int) -> int",
                "description": "Fija el valor",
                "parameters": [method_parameter("value", "int", required=True)],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "history",
                "signature": "history() -> list",
                "description": "Historial",
                "parameters": [],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "fail",
                "signature": "fail()",
                "description": "Falla",
                "parameters": [],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "save",
                "signature": "save(filename: str = 'x')",
                "description": "Escribe en disco",
                "parameters": [method_parameter("filename", "str", default="'x'")],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "total",
                "signature": "total",
                "description": "Propiedad",
                "parameters": [],
                "kind": "property",
                "inherited": False,
            },
            {
                "name": "`ghost`",
                "signature": "`ghost`",
                "description": "Entrada inventada por el docstring",
                "parameters": [],
                "kind": "method",
                "inherited": False,
            },
        ],
        "attributes": [],
        "tags": ["Testing"],
        "source": "",
        "example": "",
        "safe_calls": [],
        "bases": [],
        "kind": "class",
        "path": "lm_invoke_probe.py",
        "lineno": 1,
        "line_count": 1,
    }


@pytest.fixture
def calculator(tmp_path, monkeypatch):
    """Probe model importable by the child processes of the server."""
    (tmp_path / "lm_invoke_probe.py").write_text(INVOKE_MODULE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    entry = calculator_entry()
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [entry])
    return entry


LAZY_MODULE = '''
"""Modelo que solo funciona después de ``build``."""


class LazyModel:
    """Modelo de prueba con preparación perezosa (como un Keras Model)."""

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        self.engine = None
        self.built = False
        self.builds = 0

    def build(self, input_shape: tuple = (None, 1)) -> None:
        """Crea el motor interno (no marca ``built``, como el modelo real)."""
        self.builds += 1
        self.engine = lambda value: [item * self.scale for item in value]

    def build_count(self) -> int:
        """Cuántas veces se construyó el motor."""
        return self.builds

    def call(self, x: list) -> list:
        """Multiplica cada fila por ``scale`` (falla si no se construyó)."""
        return [self.engine(value) for value in x]

    def scale_value(self, value: int = 1) -> list:
        """Usa el motor interno: también requiere ``build``."""
        return self.engine([value])

    def explode(self, value: int = 1) -> None:
        """Reproduce el error posicional de Keras."""
        raise ValueError(
            "Only input tensors may be passed as positional arguments. The "
            "following argument value should be passed as a keyword argument: "
            "1 (of type <class 'int'>)"
        )

    def explode_tensor(self, x: list) -> None:
        """Reproduce el error posicional de Keras con un argumento tensor."""
        raise ValueError(
            "Only input tensors may be passed as positional arguments. The "
            "following argument value should be passed as a keyword argument: "
            "1 (of type <class 'int'>)"
        )
'''


def lazy_entry():
    """Catalog entry of the lazily built probe model."""
    return {
        "id": "LazyModel",
        "title": "LazyModel",
        "name": "LazyModel",
        "module": "lm_lazy_probe",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "Preparación perezosa",
        "full_description": "Preparación perezosa",
        "import_statement": "from lm_lazy_probe import LazyModel",
        "signature": "LazyModel(scale: int = 2)",
        "parameters": [method_parameter("scale", "int", default="2")],
        "methods": [
            {
                "name": "build",
                "signature": "build(input_shape: tuple = (None, 1))",
                "description": "Construye el motor",
                "parameters": [method_parameter("input_shape", "tuple", default="(None, 1)")],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "call",
                "signature": "call(x: list) -> list",
                "description": "Multiplica la entrada",
                "parameters": [method_parameter("x", "list", required=True)],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "scale_value",
                "signature": "scale_value(value: int = 1) -> list",
                "description": "Usa el motor interno",
                "parameters": [method_parameter("value", "int", default="1")],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "build_count",
                "signature": "build_count() -> int",
                "description": "Veces que se construyó",
                "parameters": [],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "explode",
                "signature": "explode(value: int = 1)",
                "description": "Error posicional de Keras",
                "parameters": [method_parameter("value", "int", default="1")],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "explode_tensor",
                "signature": "explode_tensor(x: list)",
                "description": "Error posicional de Keras con tensor",
                "parameters": [method_parameter("x", "list", required=True)],
                "kind": "method",
                "inherited": False,
            },
        ],
        "attributes": [],
        "tags": ["Testing"],
        "source": "",
        "example": "",
        "safe_calls": [],
        "bases": [],
        "kind": "class",
        "path": "lm_lazy_probe.py",
        "lineno": 1,
        "line_count": 1,
    }


@pytest.fixture
def lazy_model(tmp_path, monkeypatch):
    """Probe model that must be built before calling ``call``."""
    (tmp_path / "lm_lazy_probe.py").write_text(LAZY_MODULE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    entry = lazy_entry()
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [entry])
    return entry


SESSION_MODULE = '''
"""Modelo que informa el tipo recibido (con un numpy simulado)."""

import numpy as np


class SessionProbe:
    """Devuelve el tipo del argumento recibido."""

    def __init__(self, scale: int = 1) -> None:
        self.scale = scale

    def kind(self, x: np.ndarray) -> str:
        """Tipo de ``x`` tras la conversión del servidor."""
        return type(x).__name__
'''


NUMPY_SHIM = '''
"""numpy mínimo para probar la conversión de argumentos."""


class ndarray(list):
    """Lista que imita el nombre de ``numpy.ndarray``."""

    @property
    def shape(self):
        rows = len(self)
        columns = len(self[0]) if self and isinstance(self[0], list) else 1
        return (rows, columns)


def asarray(value):
    """Convierte una lista en ``ndarray``."""
    return ndarray(value)
'''


def session_probe_entry():
    """Catalog entry of the argument conversion probe."""
    return {
        "id": "SessionProbe",
        "title": "SessionProbe",
        "name": "SessionProbe",
        "module": "lm_session_probe",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "Conversión de argumentos",
        "full_description": "Conversión de argumentos",
        "import_statement": "from lm_session_probe import SessionProbe",
        "signature": "SessionProbe(scale: int = 1)",
        "parameters": [method_parameter("scale", "int", default="1")],
        "methods": [
            {
                "name": "kind",
                "signature": "kind(x: np.ndarray) -> str",
                "description": "Tipo recibido",
                "parameters": [method_parameter("x", "np.ndarray", required=True)],
                "kind": "method",
                "inherited": False,
            }
        ],
        "attributes": [],
        "tags": ["Testing"],
        "source": "",
        "example": "",
        "safe_calls": [],
        "bases": [],
        "kind": "class",
        "path": "lm_session_probe.py",
        "lineno": 1,
        "line_count": 1,
    }


@pytest.fixture
def session_probe(tmp_path, monkeypatch):
    """Probe module plus a tiny ``numpy`` shim for the child process."""
    (tmp_path / "lm_session_probe.py").write_text(SESSION_MODULE, encoding="utf-8")
    (tmp_path / "numpy.py").write_text(NUMPY_SHIM, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    entry = session_probe_entry()
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [entry])
    return entry


@pytest.fixture
def session_store(monkeypatch):
    """Isolated session store so every test owns its live workers."""
    store = server.SessionStore(maximum=2, ttl=30, timeout=30)
    monkeypatch.setattr(server.ModelAPIHandler, "sessions", store)
    try:
        yield store
    finally:
        store.shutdown()


def test_invoke_spec_lists_the_allowed_methods(base_url, calculator):
    status, _, data = json_of(base_url, "/api/models/Calculator/invoke")
    assert status == 200
    assert data["enabled"] is True
    assert data["endpoint"] == "/api/models/Calculator/invoke"
    assert data["sample_endpoint"] == "/api/models/Calculator/sample"
    assert data["model_name"] == "Calculator"
    assert data["import_statement"] == "from lm_invoke_probe import Calculator"
    assert data["session_ttl_seconds"] > 0
    assert data["max_sessions"] >= 1
    names = [method["name"] for method in data["methods"]]
    assert names == ["add", "set_value", "history", "fail"]
    add = data["methods"][0]
    assert add["fields"][0]["name"] == "amount"
    assert add["fields"][0]["default"] == 1


def test_dummy_json_value_shapes_payloads_from_the_constructor():
    matrix = server.dummy_json_value("X", "np.ndarray", {"window": 4, "n_features": 3})
    assert len(matrix) == 4
    assert all(len(row) == 3 for row in matrix)
    sequence = server.dummy_json_value("sequence", "list", {"n_observations": 3})
    assert sequence and all(isinstance(item, int) and 0 <= item < 3 for item in sequence)
    shape = server.dummy_json_value("input_shape", "tuple", {})
    assert shape and all(isinstance(item, int) for item in shape)
    frame = server.dummy_json_value("df", "pd.DataFrame", {})
    assert isinstance(frame, list) and isinstance(frame[0], dict)
    assert server.dummy_json_value("source_model", "Any", {}) is None
    assert server.dummy_json_value("epsilon", "Any", {}) == 0.1
    assert server.dummy_json_value("save_path", "Any", {}) == "model"


def test_invoke_spec_samples_are_json_ready():
    for model in server.CATALOG:
        spec = server.invoke_spec(model)
        json.dumps(spec)  # every sample must be JSON serialisable
        for method in spec["methods"]:
            for field in method["fields"]:
                if field["kind"] in ("int", "float", "bool", "str", "text"):
                    assert field["sample"] is not None, (model["id"], method["name"], field)
                if field["kind"] == "raw" and (
                    field["name"] in server.MATRIX_ARGUMENTS
                    or field["name"] in server.SEQUENCE_ARGUMENTS
                ):
                    assert isinstance(field["sample"], list), (model["id"], method["name"], field)


def test_constructor_samples_fill_required_containers():
    catalog = {model["id"]: model for model in server.CATALOG}
    # ``Arima`` is a time series: its ``datapoints`` is a 1-D series, not a matrix.
    series = server.constructor_samples(catalog["Arima"])["datapoints"]
    assert isinstance(series, list) and isinstance(series[0], (int, float))
    matrix = server.constructor_samples(catalog["GetInsights"])["inputs"]
    assert isinstance(matrix, list) and isinstance(matrix[0], list)


def test_sample_endpoint_generates_arguments(base_url, calculator):
    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/sample",
        method="POST",
        payload={"method": "add", "parameters": {"value": 4}},
    )
    assert status == 200
    assert data["parameters"] == {"value": 4}
    assert data["arguments"] == {"amount": 1}


def test_sample_endpoint_shapes_sequences_with_the_constructor(base_url):
    status, _, data = json_of(
        base_url,
        "/api/models/HMM/sample",
        method="POST",
        payload={"method": "viterbi", "parameters": {"n_states": 2, "n_observations": 3}},
    )
    assert status == 200
    sequence = data["arguments"]["sequence"]
    assert sequence and all(isinstance(item, int) and 0 <= item < 3 for item in sequence)


def test_sample_endpoint_rejects_unknown_methods(base_url, calculator):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/sample", method="POST", payload={"method": "nope"}
    )
    assert status == 404
    assert data["success"] is False


def test_sample_endpoint_respects_the_method_policy(base_url, calculator):
    status, _, _ = json_of(
        base_url, "/api/models/Calculator/sample", method="POST", payload={"method": "save"}
    )
    assert status == 403


def test_sample_endpoint_requires_known_parameters(base_url, calculator):
    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/sample",
        method="POST",
        payload={"method": "add", "parameters": {"nope": 1}},
    )
    assert status == 400
    assert "nope" in data["error"]


def test_sample_endpoint_can_be_disabled(base_url, calculator, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "invoke_enabled", False)
    status, _, _ = json_of(
        base_url, "/api/models/Calculator/sample", method="POST", payload={"method": "add"}
    )
    assert status == 403


def test_playground_reports_parameters_that_need_python(base_url):
    status, _, data = json_of(
        base_url,
        "/api/playground",
        method="POST",
        payload={"model_id": "AutoQL", "parameters": {}},
    )
    assert status == 400
    assert "'env'" in data["error"]
    assert "objetos Python" in data["error"]
    assert data["missing"] == ["env", "model"]


def test_invoke_spec_marks_models_that_need_python(base_url):
    status, _, data = json_of(base_url, "/api/models/AutoQL/invoke")
    assert status == 200
    assert data["buildable"] is False
    assert {item["name"] for item in data["unsupported"]} == {"env", "model"}


def test_call_builds_the_model_before_invoking(base_url, lazy_model):
    status, _, data = json_of(
        base_url,
        "/api/models/LazyModel/invoke",
        method="POST",
        payload={
            "method": "call",
            "arguments": {"x": [[1, 2], [3, 4]]},
            "parameters": {"scale": 3},
        },
    )
    assert status == 200
    assert data["result"] == [[3, 6], [9, 12]]


def test_call_builds_the_instance_once_in_a_session(base_url, lazy_model, session_store):
    created = json_of(
        base_url,
        "/api/models/LazyModel/sessions",
        method="POST",
        payload={"parameters": {"scale": 2}},
    )[2]
    session_id = created["session_id"]
    payload = {"method": "call", "arguments": {"x": [[1, 2]]}}
    first = json_of(base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload=payload)[
        2
    ]
    second = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload=payload
    )[2]
    assert first["result"] == [[2, 4]]
    assert second["result"] == [[2, 4]]
    state = json_of(base_url, f"/api/sessions/{session_id}")[2]
    assert state["calls"] == 2
    count = json_of(
        base_url,
        f"/api/sessions/{session_id}/invoke",
        method="POST",
        payload={"method": "build_count"},
    )[2]
    assert count["result"] == 1


def test_methods_other_than_call_also_build_the_instance(base_url, lazy_model):
    status, _, data = json_of(
        base_url,
        "/api/models/LazyModel/invoke",
        method="POST",
        payload={"method": "scale_value", "arguments": {"value": 4}, "parameters": {"scale": 3}},
    )
    assert status == 200
    assert data["result"] == [12]


def test_invoke_rejects_scalars_for_tensor_arguments(base_url, lazy_model):
    status, _, data = json_of(
        base_url,
        "/api/models/LazyModel/invoke",
        method="POST",
        payload={"method": "call", "arguments": {"x": 1}, "parameters": {"scale": 2}},
    )
    assert status == 400
    assert "lista o matriz JSON" in data["error"]
    assert "x" in data["error"]


def test_stateless_invocation_converts_arguments_with_method_hints(base_url, session_probe):
    status, _, data = json_of(
        base_url,
        "/api/models/SessionProbe/invoke",
        method="POST",
        payload={"method": "kind", "arguments": {"x": [[1, 2]]}},
    )
    assert status == 200
    assert data["result"] == "ndarray"


def test_session_invocation_converts_arguments_with_method_hints(
    base_url, session_probe, session_store
):
    created = json_of(
        base_url,
        "/api/models/SessionProbe/sessions",
        method="POST",
        payload={"parameters": {}},
    )[2]
    session_id = created["session_id"]
    status, _, data = json_of(
        base_url,
        f"/api/sessions/{session_id}/invoke",
        method="POST",
        payload={"method": "kind", "arguments": {"x": [[1, 2], [3, 4]]}},
    )
    assert status == 200
    assert data["result"] == "ndarray"


def test_keras_positional_error_names_the_tensor_argument(base_url, lazy_model):
    status, _, data = json_of(
        base_url,
        "/api/models/LazyModel/invoke",
        method="POST",
        payload={"method": "explode_tensor", "arguments": {"x": [[1]]}},
    )
    assert status == 422
    assert "Keras solo acepta tensores" in data["error"]
    assert "'x' (list)" in data["error"]


def test_keras_positional_error_gets_a_hint(base_url, lazy_model):
    status, _, data = json_of(
        base_url,
        "/api/models/LazyModel/invoke",
        method="POST",
        payload={"method": "explode", "arguments": {"value": 1}},
    )
    assert status == 422
    assert "Keras solo acepta tensores como argumento posicional" in data["error"]


def test_invoke_spec_unknown_model(base_url, calculator):
    status, _, data = json_of(base_url, "/api/models/nope/invoke")
    assert status == 404
    assert data["success"] is False


def test_invoke_runs_a_method(base_url, calculator):
    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/invoke",
        method="POST",
        payload={"method": "add", "arguments": {"amount": 3}},
    )
    assert status == 200
    assert data["method"] == "add"
    assert data["result"] == 3
    assert data["result_type"] == "int"
    assert data["arguments"] == {"amount": 3}
    assert data["session_id"] is None
    assert data["elapsed"] >= 0


def test_invoke_serialises_containers(base_url, calculator):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "history"}
    )
    assert status == 200
    assert data["result"] == [1, 2, 3]
    assert data["result_type"] == "list"


def test_invoke_is_stateless(base_url, calculator):
    payload = {"method": "add", "arguments": {"amount": 5}}
    _, _, first = json_of(base_url, "/api/models/Calculator/invoke", method="POST", payload=payload)
    _, _, second = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload=payload
    )
    assert first["result"] == second["result"] == 5


def test_invoke_blocks_side_effect_methods(base_url, calculator):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "save"}
    )
    assert status == 403
    assert data["method"] == "save"


@pytest.mark.parametrize("method", ["total", "`ghost`"])
def test_invoke_rejects_properties_and_docstring_entries(base_url, calculator, method):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": method}
    )
    assert status == 403
    assert data["success"] is False


def test_invoke_unknown_method(base_url, calculator):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "nope"}
    )
    assert status == 404
    assert data["success"] is False


def test_invoke_requires_the_method_field(base_url, calculator):
    status, _, data = json_of(base_url, "/api/models/Calculator/invoke", method="POST", payload={})
    assert status == 400
    assert "'method'" in data["error"]


def test_invoke_validates_arguments(base_url, calculator):
    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/invoke",
        method="POST",
        payload={"method": "add", "arguments": {"nope": 1}},
    )
    assert status == 400
    assert data["unknown"] == ["nope"]

    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/invoke",
        method="POST",
        payload={"method": "add", "arguments": {"amount": "x"}},
    )
    assert status == 400
    assert data["invalid"]

    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "set_value"}
    )
    assert status == 400
    assert data["missing"] == ["value"]


def test_invoke_reports_model_errors(base_url, calculator):
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "fail"}
    )
    assert status == 422
    assert "ValueError" in data["error"]
    assert "boom" in data["error"]
    assert data["detail"]


def test_invoke_can_be_disabled(base_url, calculator, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "invoke_enabled", False)
    status, _, data = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "add"}
    )
    assert status == 403
    assert data["success"] is False


def test_invoke_concurrency_is_capped(base_url, calculator, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(server.ModelAPIHandler, "invoke_slots", slots)
    assert slots.acquire(blocking=False)
    try:
        status, headers, _ = json_of(
            base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "add"}
        )
        assert status == 429
        assert headers["Retry-After"] == "5"
    finally:
        slots.release()
    status, _, _ = json_of(
        base_url, "/api/models/Calculator/invoke", method="POST", payload={"method": "add"}
    )
    assert status == 200


def test_invoke_urls_are_exposed_in_the_model_payload(base_url, calculator):
    status, _, data = json_of(base_url, "/api/models/Calculator")
    assert status == 200
    model = data["model"]
    assert model["invoke_url"] == "/api/models/Calculator/invoke"
    assert model["sessions_url"] == "/api/models/Calculator/sessions"
    assert model["invokable_methods"] == ["add", "set_value", "history", "fail"]


def test_method_policy_honours_the_allowlist(monkeypatch):
    method = {"name": "save", "kind": "method", "parameters": [], "inherited": False}
    assert server.method_allowed("save", method) is False
    monkeypatch.setattr(server, "INVOKE_ALLOWED_METHODS", frozenset({"save"}))
    assert server.method_allowed("save", method) is True
    monkeypatch.setattr(server, "INVOKE_BLOCKED_METHODS", frozenset({"save"}))
    assert server.method_allowed("save", method) is False


def test_method_policy_can_allow_side_effects(monkeypatch):
    method = {"name": "plot_all", "kind": "method", "parameters": [], "inherited": False}
    assert server.method_allowed("plot_all", method) is False
    monkeypatch.setattr(server, "INVOKE_ALLOW_SIDE_EFFECTS", True)
    assert server.method_allowed("plot_all", method) is True


def test_endpoint_label_groups_invocations():
    assert (
        server.endpoint_label("POST", "/api/models/HMM/invoke")
        == "POST /api/models/<model_id>/invoke"
    )
    assert (
        server.endpoint_label("POST", "/api/sessions/abc/invoke")
        == "POST /api/sessions/<session_id>/invoke"
    )
    assert (
        server.endpoint_label("DELETE", "/api/sessions/abc") == "DELETE /api/sessions/<session_id>"
    )


# ---------------------------------------------------------------------------
# Sesiones persistentes: POST /api/models/<id>/sessions + /api/sessions/<id>/invoke
# ---------------------------------------------------------------------------


def test_session_create_and_invoke_keeps_state(base_url, calculator, session_store):
    status, headers, created = json_of(
        base_url,
        "/api/models/Calculator/sessions",
        method="POST",
        payload={"parameters": {"value": 10}},
    )
    assert status == 201
    session_id = created["session_id"]
    assert headers["Location"] == f"/api/sessions/{session_id}"
    assert created["status"] == "open"
    assert created["type"] == "Calculator"
    assert created["parameters"] == {"value": 10}
    assert created["invoke_url"] == f"/api/sessions/{session_id}/invoke"

    payload = {"method": "add", "arguments": {"amount": 5}}
    _, _, first = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload=payload
    )
    _, _, second = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload=payload
    )
    assert first["result"] == 15
    assert second["result"] == 20
    assert second["calls"] == 2
    assert second["session_id"] == session_id


def test_session_lifecycle(base_url, calculator, session_store):
    _, _, created = json_of(base_url, "/api/models/Calculator/sessions", method="POST", payload={})
    session_id = created["session_id"]

    status, _, data = json_of(base_url, "/api/sessions")
    assert status == 200
    assert data["count"] == 1
    assert data["max_sessions"] == 2

    status, _, data = json_of(base_url, f"/api/sessions/{session_id}")
    assert status == 200
    assert data["session_id"] == session_id
    assert "stdout" in data

    status, _, data = json_of(base_url, f"/api/sessions/{session_id}", method="DELETE")
    assert status == 200
    assert data["status"] == "closed"

    status, _, _ = json_of(base_url, f"/api/sessions/{session_id}")
    assert status == 404
    status, _, _ = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload={"method": "add"}
    )
    assert status == 404
    status, _, _ = json_of(base_url, f"/api/sessions/{session_id}", method="DELETE")
    assert status == 404


def test_session_validates_parameters(base_url, calculator, session_store):
    status, _, data = json_of(
        base_url,
        "/api/models/Calculator/sessions",
        method="POST",
        payload={"parameters": {"nope": 1}},
    )
    assert status == 400
    assert data["unknown"] == ["nope"]


def test_session_validates_the_method_and_arguments(base_url, calculator, session_store):
    _, _, created = json_of(base_url, "/api/models/Calculator/sessions", method="POST", payload={})
    session_id = created["session_id"]

    status, _, _ = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload={"method": "save"}
    )
    assert status == 403

    status, _, data = json_of(
        base_url,
        f"/api/sessions/{session_id}/invoke",
        method="POST",
        payload={"method": "add", "arguments": {"amount": "x"}},
    )
    assert status == 400
    assert data["invalid"]

    status, _, data = json_of(
        base_url, f"/api/sessions/{session_id}/invoke", method="POST", payload={"method": "fail"}
    )
    assert status == 422
    assert "ValueError" in data["error"]


def test_sessions_can_be_disabled(base_url, calculator, session_store, monkeypatch):
    monkeypatch.setattr(server.ModelAPIHandler, "sessions_enabled", False)
    status, _, data = json_of(
        base_url, "/api/models/Calculator/sessions", method="POST", payload={}
    )
    assert status == 403
    assert data["success"] is False
    assert json_of(base_url, "/api/sessions")[0] == 403


def test_session_store_evicts_the_oldest_worker(calculator):
    store = server.SessionStore(maximum=2, ttl=30, timeout=30)
    try:
        first = store.create(calculator, {})
        second = store.create(calculator, {})
        third = store.create(calculator, {})
        assert first.alive() is False
        assert {session.id for session in store.recent()} == {second.id, third.id}
    finally:
        store.shutdown()


def test_session_store_expires_idle_sessions(calculator):
    store = server.SessionStore(maximum=2, ttl=1, timeout=30)
    try:
        session = store.create(calculator, {})
        assert store.get(session.id) is not None
        session.last_used -= 5  # simulate an idle session
        assert store.get(session.id) is None
        assert session.alive() is False
    finally:
        store.shutdown()


def test_session_store_reports_counters(calculator):
    store = server.SessionStore(maximum=3, ttl=30, timeout=30)
    try:
        store.create(calculator, {})
        snapshot = store.snapshot()
        assert snapshot["open"] == 1
        assert snapshot["created"] == 1
        assert snapshot["max_sessions"] == 3
    finally:
        store.shutdown()


def test_session_creation_reports_constructor_errors(calculator):
    store = server.SessionStore(maximum=1, ttl=30, timeout=30)
    try:
        with pytest.raises(server.InvocationError) as info:
            store.create(calculator, {"nope": 1})
        assert info.value.status == 422
        assert "TypeError" in str(info.value)
        assert store.snapshot()["open"] == 0
    finally:
        store.shutdown()


def test_health_reports_invoke_and_sessions(base_url, calculator, session_store):
    json_of(base_url, "/api/models/Calculator/sessions", method="POST", payload={})
    status, _, data = json_of(base_url, "/api/health")
    assert status == 200
    assert data["invoke"]["enabled"] is True
    assert data["invoke"]["max_concurrency"] >= 1
    assert data["sessions"]["open"] == 1
    assert data["sessions"]["max_sessions"] == 2


def test_playground_import_guard_can_be_disabled(tmp_path, monkeypatch):
    module = tmp_path / "lm_sandbox_probe_ok.py"
    module.write_text(
        "import subprocess\n\n\nclass Probe:\n"
        "    def __init__(self, value: int = 1) -> None:\n"
        "        self.value = value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(server, "PLAYGROUND_BLOCKED_MODULES", frozenset())
    result = server.run_playground({"module": "lm_sandbox_probe_ok", "name": "Probe"}, {})
    assert result["type"] == "Probe"


def test_catalog_store_detects_changes(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = textwrap.dedent('''
        """Modelos de prueba."""


        class Dummy:
            """Modelo de prueba.

            Parameters
            ----------
            value : int
                Un valor cualquiera.
            """

            def __init__(self, value: int = 1) -> None:
                self.value = value
        ''')
    (package / "dummy.py").write_text(source, encoding="utf-8")

    store = server.CatalogStore(package, ttl=0.01)
    assert [model["id"] for model in store.load()] == ["Dummy"]
    fingerprint = store.fingerprint
    assert store.changed() is False
    assert store.refresh() is False  # nothing changed yet

    (package / "extra.py").write_text(source.replace("Dummy", "Extra"), encoding="utf-8")
    assert store.changed() is True
    time.sleep(0.02)
    assert store.refresh() is True
    assert {model["id"] for model in store.catalog} == {"Dummy", "Extra"}
    assert store.fingerprint != fingerprint
    assert store.reloads >= 2


def test_catalog_store_can_be_disabled(tmp_path):
    package = tmp_path / "models"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "dummy.py").write_text("class Dummy:\n    pass\n", encoding="utf-8")
    store = server.CatalogStore(package, ttl=0)
    store.load()
    (package / "other.py").write_text("class Other:\n    pass\n", encoding="utf-8")
    assert store.refresh() is False
    assert store.refresh(force=True) is True
