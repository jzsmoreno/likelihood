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
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError
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
        "categoryLabel": "Testing",
        "icon": "🧪",
        "shortDescription": "Contador de pruebas",
        "fullDescription": "Contador de pruebas",
        "importStatement": "from collections import Counter",
        "signature": "Counter()",
        "parameters": [],
        "methods": [],
        "attributes": [],
        "tags": ["Testing"],
        "source": "class Counter:\n    ...",
        "example": "model = Counter()\n",
        "safeCalls": [],
        "bases": [],
        "kind": "class",
        "path": "collections/__init__.py",
        "lineno": 1,
        "lineCount": 1,
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
    assert headers["Cache-Control"] == "no-store"
    assert data["count"] == len(server.CATALOG)
    assert data["total"] == len(server.CATALOG)
    assert data["matched"] == len(server.CATALOG)
    assert data["limit"] is None
    assert data["offset"] == 0
    assert data["hasMore"] is False
    assert "source" not in data["models"][0]


def test_models_pagination(base_url):
    status, _, first = json_of(base_url, "/api/models?limit=2")
    assert status == 200
    assert first["count"] == 2
    assert first["hasMore"] is True
    assert first["limit"] == 2

    status, _, second = json_of(base_url, "/api/models?limit=2&offset=2")
    assert status == 200
    assert second["offset"] == 2
    assert {m["id"] for m in first["models"]}.isdisjoint({m["id"] for m in second["models"]})

    status, _, tail = json_of(base_url, f"/api/models?offset={len(server.CATALOG)}")
    assert status == 200
    assert tail["count"] == 0
    assert tail["hasMore"] is False


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
    assert model["sourceUrl"] == f"/api/models/{model_id}/source"
    assert model["sourceDownloadUrl"].endswith("?download=1")
    assert model["playgroundUrl"] == f"/api/playground/{model_id}"
    assert model["playground"]["modelId"] == model_id


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
    assert {"id", "title", "category", "categoryLabel", "icon"} <= set(data["suggestions"][0])
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
    assert "/api/playground" in text
    assert f"/api/models/{model_id}/source?download=1" in text


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
    assert headers["Allow"] == "GET, HEAD, POST"
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
        "categoryLabel": "Testing",
        "icon": "🧪",
        "shortDescription": "",
        "fullDescription": "",
        "importStatement": "from collections import Counter",
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
        "safeCalls": [],
        "bases": [],
        "kind": "class",
        "path": "collections/__init__.py",
        "lineno": 1,
        "lineCount": 1,
    }
    monkeypatch.setattr(server.ModelAPIHandler, "catalog", [fake])
    return fake


def test_playground_spec_endpoint(base_url):
    model_id = first_model_id()
    status, _, data = json_of(base_url, f"/api/playground/{model_id}")
    assert status == 200
    assert data["enabled"] is True
    spec = data["playground"]
    assert spec["modelId"] == model_id
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
    assert server._sample_value(
        {"sample": "np.zeros(3)", "type": "np.ndarray", "widget": "raw"}
    ) is (None)
    assert server._sample_value({"sample": "1", "type": "int", "widget": "int"}) == 1


def test_script_json_escapes_markup():
    payload = server._script_json({"value": "</script><b>&"})
    assert "</script>" not in payload
    assert "\\u003c/script\\u003e" in payload
    assert json.loads(payload) == {"value": "</script><b>&"}
