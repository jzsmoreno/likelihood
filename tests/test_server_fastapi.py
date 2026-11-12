"""Tests for the FastAPI implementation of the catalogue server.

The suite builds the real application with :func:`server_fastapi.create_app` and
exercises it through ``fastapi.testclient.TestClient`` (which needs ``httpx``).
It reuses the shared domain helpers from :mod:`server_core`, so the playground,
invocation and sessions run in real child processes.

Run with::

    python3 -m pytest tests/test_server_fastapi.py -v
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import model_catalog  # noqa: E402
import server_core as server  # noqa: E402
import server_fastapi  # noqa: E402

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

    @staticmethod
    def load(filename: str = "./calculator") -> "Calculator":
        """Carga una calculadora serializada con pickle."""
        import pickle

        with open(filename, "rb") as handle:
            return pickle.load(handle)
'''


def method_parameter(name, annotation, required=False, default=None):
    """Parameter record as produced by ``model_catalog``."""
    return {
        "name": name,
        "type": annotation,
        "default": default,
        "required": required,
        "kind": "positional" if required else "keyword_only",
        "variadic": False,
        "description": "",
        "widget": server.guess_widget(annotation),
        "sample": "1",
    }


def calculator_entry() -> Dict[str, Any]:
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
        ],
        "attributes": [],
        "tags": ["Testing"],
        "source": "",
        "example": "",
        "safe_calls": [],
        "bases": [],
        "loaders": [
            {
                "name": "load",
                "signature": "load(filename: str = './calculator') -> 'Calculator'",
                "kind": "static",
                "description": "Carga una calculadora serializada con pickle.",
                "path_parameter": "filename",
                "path_type": "str",
                "default_path": "./calculator",
                "requires_instance": False,
            }
        ],
        "kind": "class",
        "path": "lm_invoke_probe.py",
        "lineno": 1,
        "line_count": 1,
    }


SHAPE_MODULE = '''
"""Modelo de prueba con un ``build`` que registra la forma recibida."""


class ShapeProbe:
    """Registra el tipo y el valor de ``input_shape``."""

    def __init__(self, input_shape_parm: int) -> None:
        self.input_shape_parm = input_shape_parm
        self.built = False
        self.shape = None
        self.shape_type = None

    def build(self, input_shape: tuple) -> None:
        """Guarda la forma recibida y marca la instancia como construida."""
        self.shape = list(input_shape)
        self.shape_type = type(input_shape).__name__
        self.built = True

    def describe(self) -> dict:
        """Devuelve el estado observable de la instancia."""
        return {
            "input_shape_parm": self.input_shape_parm,
            "shape": self.shape,
            "shape_type": self.shape_type,
            "built": self.built,
        }
'''


def shape_probe_entry() -> Dict[str, Any]:
    """Catalog entry of the ``build`` shape probe."""
    return {
        "id": "ShapeProbe",
        "title": "ShapeProbe",
        "name": "ShapeProbe",
        "module": "lm_shape_probe",
        "category": "testing",
        "category_label": "Testing",
        "icon": "🧪",
        "short_description": "Sonda de input_shape",
        "full_description": "Sonda de input_shape",
        "import_statement": "from lm_shape_probe import ShapeProbe",
        "signature": "ShapeProbe(input_shape_parm: int)",
        "parameters": [method_parameter("input_shape_parm", "int", required=True)],
        "methods": [
            {
                "name": "build",
                "signature": "build(input_shape: tuple) -> None",
                "description": "Construye con una forma",
                "parameters": [method_parameter("input_shape", "tuple", required=True)],
                "kind": "method",
                "inherited": False,
            },
            {
                "name": "describe",
                "signature": "describe() -> dict",
                "description": "Estado de la instancia",
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
        "path": "lm_shape_probe.py",
        "lineno": 1,
        "line_count": 1,
    }


@pytest.fixture
def shape_probe(tmp_path, monkeypatch, settings):
    """Probe model whose ``build`` exposes the shape it received."""
    (tmp_path / "lm_shape_probe.py").write_text(SHAPE_MODULE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    entry = shape_probe_entry()
    monkeypatch.setattr(settings, "catalog", [entry])
    return entry


@pytest.fixture(scope="module")
def client():
    """Create a running FastAPI application backed by an ephemeral test client."""
    app = server_fastapi.create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def settings(client):
    """Return the mutable runtime state of the application under test."""
    return client.app.state.settings


@pytest.fixture
def calculator(tmp_path, monkeypatch, settings):
    """Probe model importable by the child processes of the server."""
    (tmp_path / "lm_invoke_probe.py").write_text(INVOKE_MODULE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    entry = calculator_entry()
    monkeypatch.setattr(settings, "catalog", [entry])
    return entry


@pytest.fixture
def session_store(monkeypatch, settings):
    """Isolated session store so every test owns its live workers."""
    store = server.SessionStore(maximum=2, ttl=30, timeout=30)
    monkeypatch.setattr(settings, "sessions", store)
    try:
        yield store
    finally:
        store.shutdown()


def first_model_id() -> str:
    """Id of the first model of the real catalog."""
    return server.CATALOG[0]["id"]


# ---------------------------------------------------------------------------
# Health, catalog and search
# ---------------------------------------------------------------------------


def test_health_reports_the_catalog(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["api_version"] == server_fastapi.API_VERSION
    assert data["models"] == len(server.CATALOG)
    assert data["summary"]["total"] == len(server.CATALOG)
    assert response.headers["X-API-Version"] == server_fastapi.API_VERSION
    assert response.headers["X-Request-Id"]


def test_categories(client):
    response = client.get("/api/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(data["categories"])
    assert sum(item["count"] for item in data["categories"]) == len(server.CATALOG)


def test_models_list_and_conditional_request(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(server.CATALOG)
    assert data["total"] == len(server.CATALOG)
    assert data["matched"] == len(server.CATALOG)
    etag = response.headers["ETag"]
    assert etag
    cached = client.get("/api/models", headers={"If-None-Match": etag})
    assert cached.status_code == 304
    assert cached.content == b""


def test_models_pagination(client):
    first = client.get("/api/models?limit=1")
    assert first.status_code == 200
    data = first.json()
    assert data["count"] == 1
    assert data["has_more"] is True
    assert data["next"].startswith("/api/models?")
    assert first.headers["Link"].startswith("<" + data["next"])
    second = client.get(data["next"])
    assert second.status_code == 200
    assert second.json()["offset"] == 1


@pytest.mark.parametrize("query", ["limit=abc", "limit=0", "limit=999", "offset=-1"])
def test_models_rejects_invalid_pagination(client, query):
    response = client.get("/api/models?" + query)
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_models_filters(client):
    response = client.get("/api/models?category=all")
    assert response.status_code == 200
    assert response.json()["total"] == len(server.CATALOG)
    model = server.CATALOG[0]
    if model["tags"]:
        tagged = client.get("/api/models?tags=" + model["tags"][0])
        assert tagged.status_code == 200
        assert all(model["tags"][0] in item["tags"] for item in tagged.json()["models"])


def test_model_detail_and_unknown(client):
    model_id = first_model_id()
    response = client.get("/api/models/" + model_id)
    assert response.status_code == 200
    assert response.json()["model"]["id"] == model_id
    missing = client.get("/api/models/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["success"] is False


def test_related_models(client):
    model = server.CATALOG[0]
    response = client.get(f"/api/models/{model['id']}/related")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == model["id"]
    assert data["count"] == len(data["related"])
    assert all(entry["id"] != model["id"] for entry in data["related"])
    assert all(entry["score"] > 0 for entry in data["related"])
    scores = [entry["score"] for entry in data["related"]]
    assert scores == sorted(scores, reverse=True)
    limited = client.get(f"/api/models/{model['id']}/related?limit=1")
    assert limited.json()["count"] <= 1
    assert client.get("/api/models/does-not-exist/related").status_code == 404
    assert client.get(f"/api/models/{model['id']}/related?limit=0").status_code == 400


def test_models_sorting(client):
    descending = client.get("/api/models?sort=parameters&order=desc")
    assert descending.status_code == 200
    counts = [model["parameter_count"] for model in descending.json()["models"]]
    assert counts == sorted(counts, reverse=True)

    by_title = client.get("/api/models?sort=title").json()["models"]
    titles = [model["title"].lower() for model in by_title]
    assert titles == sorted(titles)

    assert client.get("/api/models?sort=does-not-exist").status_code == 400
    assert client.get("/api/models?sort=id&order=sideways").status_code == 400


def test_model_source_and_download(client):
    model_id = first_model_id()
    response = client.get("/api/models/" + model_id + "/source")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/x-python")
    assert response.text
    download = client.get("/api/models/" + model_id + "/source?download=1")
    assert download.headers["Content-Disposition"] == f'attachment; filename="{model_id}.py"'


def test_search_get_requires_q(client):
    missing = client.get("/api/search")
    assert missing.status_code == 400
    response = client.get("/api/search?q=auto")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "auto"
    assert "facets" in data
    assert "suggestions" in data


def test_search_post_is_deprecated_but_works(client):
    response = client.post("/api/search", json={"query": "auto"})
    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
    assert response.json()["deprecated"] is True


# ---------------------------------------------------------------------------
# Documentation, pages and static assets
# ---------------------------------------------------------------------------


def test_openapi_document(client):
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    document = response.json()
    assert document["openapi"].startswith("3.1")
    assert {"/api/models", "/api/playground", "/api/search"} <= set(document["paths"])
    assert {
        "/api/models/{model_id}/invoke",
        "/api/sessions/{session_id}/invoke",
        "/api/models/{model_id}/sample",
    } <= set(document["paths"])
    schemas = document["components"]["schemas"]
    assert {"PlaygroundRequest", "InvokeRequest", "SessionRequest"} <= set(schemas)


def test_docs_page(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/html")
    assert "/api/openapi.json" in response.text


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/html")


def test_catalog_page_uses_the_shared_design_system(client):
    """The catalog is an app shell (top bar, sidebar, drawer) driven by catalog.js."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'href="/app.css"' in html
    assert 'src="/catalog.js"' in html
    assert 'id="sidebar"' in html
    assert 'id="drawer"' in html
    assert 'id="results"' in html
    assert 'id="view-table"' in html
    assert "api/models" in html


def test_preview_page_uses_the_shared_design_system(client):
    """The preview keeps the server rendered sections and adds its own script."""
    response = client.get("/preview/" + first_model_id())
    assert response.status_code == 200
    html = response.text
    assert 'href="/app.css"' in html
    assert 'href="/preview.css"' in html
    assert 'src="/preview.js"' in html
    assert 'id="sidebar"' in html
    assert 'id="toc"' in html
    assert 'id="parametros"' in html
    assert 'id="laboratorio"' in html


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        ("/app.css", "text/css"),
        ("/preview.css", "text/css"),
        ("/catalog.js", "text/javascript"),
        ("/preview.js", "text/javascript"),
    ],
)
def test_design_system_assets_are_served(client, path, prefix):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith(prefix)
    assert response.text.strip()
    assert response.headers["ETag"]
    assert response.headers["Cache-Control"]


def test_favicon(client):
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/svg+xml; charset=utf-8"


def test_favicon_ico_redirects(client):
    response = client.get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 308
    assert response.headers["Location"] == "/favicon.svg"


def test_static_traversal_is_blocked(client):
    response = client.get("/..%2Fserver.py")
    assert response.status_code == 404
    assert b"import argparse" not in response.content


def test_preview_pages(client):
    model = server.CATALOG[0]
    response = client.get("/preview/" + model["id"])
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/html")
    legacy_url = client.get(f"/preview/{model['category']}/{model['id']}")
    assert legacy_url.status_code == 200
    missing = client.get("/preview/does-not-exist")
    assert missing.status_code == 404
    assert missing.headers["Content-Type"].startswith("text/html")


def test_preview_links_the_tags_and_the_related_models(client):
    """The preview page navigates back to the catalogue instead of dead ends."""
    probe = None
    for candidate in server.CATALOG:
        entries = model_catalog.related_models(server.CATALOG, candidate, limit=4)
        if entries:
            probe = (candidate, entries)
            break
    assert probe is not None, "the catalog should relate at least one model"
    model, entries = probe

    response = client.get("/preview/" + model["id"])
    assert response.status_code == 200
    html = response.text
    # Etiquetas -> catálogo filtrado por etiqueta.
    assert 'class="tag" href="/?tag=' in html
    # Modelos relacionados con enlaces a sus vistas previas.
    assert 'id="relacionados"' in html
    for entry in entries:
        assert f'/preview/{entry["model"]["id"]}' in html
    # El panel de invocación y el playground siguen embebidos.
    assert 'id="playground-data"' in html
    assert 'id="invoke-data"' in html
    # Ningún marcador de plantilla queda sin sustituir.
    assert "{{" not in html


def test_notebook_download_and_post(client):
    model_id = first_model_id()
    download = client.get("/api/notebook/" + model_id)
    assert download.status_code == 200
    assert download.headers["Content-Type"].startswith("application/x-ipynb+json")
    created = client.post("/api/notebook", json={"model_id": model_id})
    assert created.status_code == 200
    data = created.json()
    assert data["model_id"] == model_id
    assert data["notebook"]["nbformat"]


# ---------------------------------------------------------------------------
# Transport, caching and observability
# ---------------------------------------------------------------------------


def test_head_requests_use_the_get_handlers(client):
    response = client.head("/api/models")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.content == b""
    assert client.head("/api/health").status_code == 200
    assert client.head("/").status_code == 200


def test_responses_are_compressed_when_accepted(client):
    compressed = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert compressed.headers["Content-Encoding"] == "gzip"
    assert b"<!DOCTYPE html>" in compressed.content  # httpx already inflated it
    plain = client.get("/", headers={"Accept-Encoding": "identity"})
    assert "Content-Encoding" not in plain.headers


def test_every_response_carries_the_security_headers(client):
    for path in ("/", "/api/models", "/api/health", "/preview/" + first_model_id()):
        response = client.get(path)
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
        assert response.headers["Referrer-Policy"] == "no-referrer"


def test_conditional_requests_accept_wildcards_lists_and_weak_tags(client):
    response = client.get("/api/models?limit=1")
    etag = response.headers["ETag"]
    wildcard = client.get("/api/models?limit=1", headers={"If-None-Match": "*"})
    assert wildcard.status_code == 304
    listed = client.get("/api/models?limit=1", headers={"If-None-Match": f'"nope", W/{etag}'})
    assert listed.status_code == 304
    stale = client.get("/api/models?limit=1", headers={"If-None-Match": '"stale"'})
    assert stale.status_code == 200


def test_openapi_document_supports_conditional_requests(client):
    first = client.get("/api/openapi.json")
    assert first.status_code == 200
    etag = first.headers["ETag"]
    assert etag
    assert first.headers["Cache-Control"]
    cached = client.get("/api/openapi.json", headers={"If-None-Match": etag})
    assert cached.status_code == 304
    assert cached.content == b""


def test_static_assets_honour_if_modified_since(client):
    first = client.get("/favicon.svg")
    assert first.status_code == 200
    fresh = client.get(
        "/favicon.svg", headers={"If-Modified-Since": first.headers["Last-Modified"]}
    )
    assert fresh.status_code == 304
    stale = client.get(
        "/favicon.svg", headers={"If-Modified-Since": "Thu, 01 Jan 1970 00:00:00 GMT"}
    )
    assert stale.status_code == 200


def test_model_source_supports_conditional_requests(client):
    url = f"/api/models/{first_model_id()}/source"
    first = client.get(url)
    etag = first.headers["ETag"]
    assert etag
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 304


# ---------------------------------------------------------------------------
# Errors, authentication, CORS and limits
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.get("/api/models")
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/plain")
    assert response.headers["Cache-Control"] == "no-store"
    assert "# TYPE likelihood_uptime_seconds gauge" in response.text
    assert 'likelihood_requests_total{endpoint="GET /api/models"}' in response.text
    assert 'likelihood_responses_total{status="200"}' in response.text


def test_metrics_endpoint_requires_the_token(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    assert client.get("/api/metrics").status_code == 401
    allowed = client.get("/api/metrics", headers={"X-API-Token": "secret"})
    assert allowed.status_code == 200


def test_authenticated_answers_are_private(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    response = client.get("/api/models", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"].startswith("private")
    assert "X-API-Token" in response.headers["Vary"]


def test_health_hides_the_internals_from_anonymous_probes(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    anonymous = client.get("/api/health")
    assert anonymous.status_code == 200
    assert {"models", "status"} <= set(anonymous.json())
    assert "metrics" not in anonymous.json()
    full = client.get("/api/health", headers={"Authorization": "Bearer secret"}).json()
    assert full["metrics"]["total_requests"] >= 1
    assert full["server"]["trust_proxy"] is False


def test_forwarded_for_is_only_used_with_trust_proxy(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "rate_limiter", server.RateLimiter(2))
    monkeypatch.setattr(settings, "trust_proxy", True)
    headers = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
    assert client.get("/api/health", headers=headers).status_code == 200
    assert client.get("/api/health", headers=headers).status_code == 200
    assert client.get("/api/health", headers=headers).status_code == 429
    # A different forwarded address owns a different bucket.
    other = dict(headers, **{"X-Forwarded-For": "203.0.113.8"})
    assert client.get("/api/health", headers=other).status_code == 200


def test_job_can_be_cancelled_or_forgotten(client, calculator):
    created = client.post(
        "/api/playground/jobs", json={"model_id": "Calculator", "parameters": {"value": 1}}
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    cancelled = client.delete(f"/api/playground/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["job_id"] == job_id
    assert cancelled.json()["status"] in {"cancelled", "cancelling", "succeeded"}
    assert client.delete("/api/playground/jobs/does-not-exist").status_code == 404


def test_unknown_api_endpoint(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["success"] is False


@pytest.mark.parametrize("method", ["put", "patch"])
def test_method_rejection(client, method):
    response = getattr(client, method)("/api/models")
    assert response.status_code == 405
    assert response.headers["Allow"] == "GET, HEAD, POST, DELETE, OPTIONS"
    assert response.json()["success"] is False


def test_body_too_large(client):
    payload = b"x" * (server_fastapi.MAX_BODY_BYTES + 1)
    response = client.post(
        "/api/notebook", content=payload, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["success"] is False


def test_api_token_authentication(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "secret")
    denied = client.get("/api/models")
    assert denied.status_code == 401
    assert denied.headers["WWW-Authenticate"] == 'Bearer realm="likelihood"'
    assert client.get("/api/health").status_code == 200
    allowed = client.get("/api/models", headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200
    header = client.get("/api/models", headers={"X-API-Token": "secret"})
    assert header.status_code == 200


def test_rate_limit(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "rate_limiter", server.RateLimiter(2))
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 200
    limited = client.get("/api/health")
    assert limited.status_code == 429
    assert limited.headers["Retry-After"]
    assert limited.json()["success"] is False


def test_cors_preflight(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "cors_origins", ("*",))
    response = client.options(
        "/api/models",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    monkeypatch.setattr(settings, "cors_origins", ("http://allowed.test",))
    blocked = client.options(
        "/api/models",
        headers={"Origin": "http://evil.test", "Access-Control-Request-Method": "GET"},
    )
    assert blocked.status_code == 405


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------


def test_playground_spec(client):
    model_id = first_model_id()
    response = client.get("/api/playground/" + model_id)
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["playground"]["endpoint"] == "/api/playground"


def test_playground_get_is_rejected(client):
    response = client.get("/api/playground")
    assert response.status_code == 405
    assert response.headers["Allow"] == "POST"


def test_playground_requires_a_model(client):
    response = client.post("/api/playground", json={})
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_playground_can_be_disabled(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "playground_enabled", False)
    response = client.post("/api/playground", json={"model_id": first_model_id()})
    assert response.status_code == 403


def test_playground_runs_a_model(client, calculator):
    response = client.post(
        "/api/playground", json={"model_id": "Calculator", "parameters": {"value": 3}}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["type"] == "Calculator"
    assert data["parameters"] == {"value": 3}


def test_playground_jobs_single_and_listing(client, calculator):
    created = client.post(
        "/api/playground/jobs",
        json={"model_id": "Calculator", "parameters": {"value": 1}},
    )
    assert created.status_code == 202
    body = created.json()
    assert body["job_id"]
    assert created.headers["Location"] == body["poll_url"]
    job: Optional[Dict[str, Any]] = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        polled = client.get(body["poll_url"])
        assert polled.status_code == 200
        job = polled.json()
        if job["status"] not in {"pending", "running"}:
            break
        time.sleep(0.1)
    assert job is not None
    assert job["status"] == "succeeded"
    assert job["results"][0]["success"] is True
    listing = client.get("/api/playground/jobs")
    assert listing.status_code == 200
    assert listing.json()["count"] >= 1
    unknown = client.get("/api/playground/jobs/does-not-exist")
    assert unknown.status_code == 404


def test_playground_jobs_reject_empty_batches(client):
    response = client.post("/api/playground/jobs", json=[])
    assert response.status_code == 400
    assert response.json()["success"] is False


# ---------------------------------------------------------------------------
# Invocation and sessions
# ---------------------------------------------------------------------------


def test_invoke_spec(client, calculator):
    response = client.get("/api/models/Calculator/invoke")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["enabled"] is True
    assert {method["name"] for method in data["methods"]} >= {"add", "set_value"}


def test_invoke_runs_a_method(client, calculator):
    response = client.post(
        "/api/models/Calculator/invoke",
        json={"method": "add", "arguments": {"amount": 2}, "parameters": {"value": 1}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["method"] == "add"
    assert data["result"] == 3
    assert data["session_id"] is None


def test_invoke_requires_the_method(client, calculator):
    response = client.post("/api/models/Calculator/invoke", json={})
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_sample_endpoint(client, calculator):
    response = client.post("/api/models/Calculator/sample", json={"method": "add"})
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "add"
    assert "amount" in data["arguments"]


def test_load_spec_and_saved_model(client, calculator, tmp_path):
    spec = client.get("/api/models/Calculator/load")
    assert spec.status_code == 200
    body = spec.json()
    assert body["loadable"] is True
    assert body["loaders"][0]["name"] == "load"

    sys.path.insert(0, str(tmp_path))
    try:
        import lm_invoke_probe

        model_path = tmp_path / "saved_calculator.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(lm_invoke_probe.Calculator(value=10), handle)
    finally:
        sys.path.remove(str(tmp_path))

    loaded = client.post(
        "/api/models/Calculator/load",
        json={"path": str(model_path), "loader": "load"},
    )
    assert loaded.status_code == 200
    assert loaded.json()["type"] == "Calculator"

    invoked = client.post(
        "/api/models/Calculator/invoke",
        json={
            "method": "add",
            "arguments": {"amount": 5},
            "path": str(model_path),
            "loader": "load",
        },
    )
    assert invoked.status_code == 200
    assert invoked.json()["result"] == 15
    assert invoked.json()["source"] == "path"


def test_load_session_keeps_the_saved_instance(client, calculator, tmp_path, session_store):
    sys.path.insert(0, str(tmp_path))
    try:
        import lm_invoke_probe

        model_path = tmp_path / "session_calculator.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(lm_invoke_probe.Calculator(value=7), handle)
    finally:
        sys.path.remove(str(tmp_path))

    created = client.post(
        "/api/models/Calculator/sessions",
        json={"path": str(model_path), "loader": "load"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["source"] == "path"
    session_id = body["session_id"]

    # ``parameters`` is sent on purpose: a loaded session must never be rebuilt
    # from the constructor (that would discard the saved state).
    invoked = client.post(
        f"/api/sessions/{session_id}/invoke",
        json={"method": "add", "arguments": {"amount": 3}, "parameters": {"value": 999}},
    )
    assert invoked.status_code == 200
    assert invoked.json()["result"] == 10
    assert invoked.json()["source"] == "path"


def test_load_rejects_unknown_loader(client, calculator, tmp_path):
    response = client.post(
        "/api/models/Calculator/load",
        json={"path": str(tmp_path / "missing.pkl"), "loader": "nope"},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_invoke_can_be_disabled(client, settings, calculator, monkeypatch):
    monkeypatch.setattr(settings, "invoke_enabled", False)
    response = client.post("/api/models/Calculator/invoke", json={"method": "add"})
    assert response.status_code == 403
    spec = client.get("/api/models/Calculator/invoke")
    assert spec.status_code == 200  # the spec stays readable


def test_session_lifecycle(client, calculator, session_store):
    created = client.post("/api/models/Calculator/sessions", json={"parameters": {"value": 5}})
    assert created.status_code == 201
    body = created.json()
    session_id = body["session_id"]
    assert body["invoke_url"] == f"/api/sessions/{session_id}/invoke"
    invoked = client.post(
        f"/api/sessions/{session_id}/invoke",
        json={"method": "add", "arguments": {"amount": 2}},
    )
    assert invoked.status_code == 200
    assert invoked.json()["result"] == 7
    state = client.get("/api/sessions/" + session_id)
    assert state.status_code == 200
    assert state.json()["calls"] >= 1
    listing = client.get("/api/sessions")
    assert listing.status_code == 200
    assert any(item["session_id"] == session_id for item in listing.json()["sessions"])
    closed = client.delete("/api/sessions/" + session_id)
    assert closed.status_code == 200
    assert client.get("/api/sessions/" + session_id).status_code == 404


def test_session_build_uses_the_constructor_shape_and_converts_tuples(
    client, shape_probe, session_store
):
    """``input_shape_parm`` shapes the auto-build and ``build`` accepts tuples."""
    created = client.post(
        "/api/models/ShapeProbe/sessions", json={"parameters": {"input_shape_parm": 4}}
    )
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    try:
        # ``describe`` triggers the automatic build, which must use input_shape_parm.
        auto = client.post(f"/api/sessions/{session_id}/invoke", json={"method": "describe"})
        assert auto.status_code == 200
        state = auto.json()["result"]
        assert state["built"] is True
        assert state["shape"] == [None, 4]

        # An explicit ``build`` converts the JSON list to a tuple and rebuilds the
        # already prepared instance with the new shape.
        built = client.post(
            f"/api/sessions/{session_id}/invoke",
            json={"method": "build", "arguments": {"input_shape": [None, 6]}},
        )
        assert built.status_code == 200, built.json()
        described = client.post(f"/api/sessions/{session_id}/invoke", json={"method": "describe"})
        assert described.status_code == 200
        state = described.json()["result"]
        assert state["shape"] == [None, 6]
        assert state["shape_type"] == "tuple"
        assert state["built"] is True
    finally:
        client.delete(f"/api/sessions/{session_id}")


def test_autoclassifier_invoke_honours_input_shape_parm(client, session_store):
    """End-to-end check of the reported ``AutoClassifier`` inference scenario."""
    pytest.importorskip("tensorflow")
    assert any(model["id"] == "AutoClassifier" for model in server.CATALOG)
    parameters = {
        "input_shape_parm": 6,
        "num_classes": 5,
        "units": 8,
        "activation": "relu",
    }
    created = client.post("/api/models/AutoClassifier/sessions", json={"parameters": parameters})
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    try:
        config = client.post(f"/api/sessions/{session_id}/invoke", json={"method": "get_config"})
        assert config.status_code == 200
        assert config.json()["result"]["input_shape_parm"] == 6

        built = client.post(
            f"/api/sessions/{session_id}/invoke",
            json={"method": "build", "arguments": {"input_shape": [None, 6]}},
        )
        assert built.status_code == 200, built.json()

        called = client.post(
            f"/api/sessions/{session_id}/invoke",
            json={"method": "call", "arguments": {"x": [[1.0] * 6, [2.0] * 6]}},
        )
        assert called.status_code == 200, called.json()
        result = called.json()["result"]
        assert len(result) == 2
        assert len(result[0]) == 5
    finally:
        client.delete(f"/api/sessions/{session_id}")


def test_session_invoke_rebuilds_when_the_constructor_changes(client, shape_probe, session_store):
    """A session must not keep an instance built with stale constructor values."""
    opened = []
    created = client.post(
        "/api/models/ShapeProbe/sessions", json={"parameters": {"input_shape_parm": 4}}
    )
    assert created.status_code == 201
    opened.append(created.json()["session_id"])
    try:
        first = client.post(f"/api/sessions/{opened[0]}/invoke", json={"method": "describe"})
        assert first.status_code == 200
        assert first.json()["rebuilt"] is False
        assert first.json()["result"]["shape"] == [None, 4]

        changed = client.post(
            f"/api/sessions/{opened[0]}/invoke",
            json={"method": "describe", "parameters": {"input_shape_parm": 6}},
        )
        assert changed.status_code == 200, changed.json()
        body = changed.json()
        assert body["rebuilt"] is True
        assert body["session_id"] != opened[0]
        assert body["parameters"] == {"input_shape_parm": 6}
        assert body["result"]["shape"] == [None, 6]
        opened.append(body["session_id"])
        # The old worker is gone: its id can no longer be invoked.
        assert client.get(f"/api/sessions/{opened[0]}").status_code == 404

        # The same constructor values keep the live instance (and its counter).
        same = client.post(
            f"/api/sessions/{body['session_id']}/invoke",
            json={"method": "describe", "parameters": {"input_shape_parm": 6}},
        )
        assert same.status_code == 200
        assert same.json()["rebuilt"] is False
        assert same.json()["session_id"] == body["session_id"]
        assert same.json()["calls"] == 2
    finally:
        for session_id in opened:
            client.delete(f"/api/sessions/{session_id}")


def test_sessions_can_be_disabled(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "sessions_enabled", False)
    response = client.get("/api/sessions")
    assert response.status_code == 403


def test_unknown_session(client, session_store):
    assert client.get("/api/sessions/missing").status_code == 404
    assert client.delete("/api/sessions/missing").status_code == 404


# ---------------------------------------------------------------------------
# Command line helper
# ---------------------------------------------------------------------------


def test_cli_can_list_the_catalog(capsys):
    assert server_fastapi._main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "modelos" in output
