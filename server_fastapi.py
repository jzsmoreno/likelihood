#!/usr/bin/env python3
"""FastAPI implementation of the Likelihood catalogue server.

This module re-implements the HTTP surface of :mod:`server_core` on top of
`FastAPI <https://fastapi.tiangolo.com/>`_ while reusing every domain helper of
that module: the introspected catalog (:class:`server_core.CatalogStore`), the
sandboxed playground (:func:`server_core.run_playground`), method invocation
(:func:`server_core.run_invoke`), live sessions
(:class:`server_core.SessionStore`), asynchronous jobs
(:class:`server_core.JobQueue`), notebook generation, HTML previews and the
catalog helpers.

Endpoints
---------
The routes are the same as the previous standard-library implementation::

    GET    /                          the interactive catalogue (models/index.html)
    GET    /preview/<model_id>        server rendered HTML preview of a model
    GET    /api/health                liveness probe, counters, metrics and jobs
    GET    /api/categories            categories with their model counts
    GET    /api/models                model list (``?category=``, ``?q=``, ``?tags=``,
                                      ``?limit=``, ``?offset=``)
    GET    /api/models/<model_id>     everything known about one model
    GET    /api/models/<id>/source    raw source (``?download=1``, ``?diff=<ref>``)
    GET    /api/models/<id>/versions  git commits that touched the source of a model
    GET    /api/search                ``?q=`` search with facets (autocomplete friendly)
    GET    /api/playground/<model_id> playground specification of one model
    GET    /api/playground/jobs       recent asynchronous playground jobs
    GET    /api/playground/jobs/<id>  state and results of one job
    GET    /api/models/<id>/invoke    methods that may be called on one model
    GET    /api/sessions              open model sessions
    GET    /api/sessions/<id>         state of one session
    GET    /api/notebook/<model_id>   generated ``.ipynb`` download (``?parameters=``)
    POST   /api/playground            ``{"model_id": ..., "parameters": {...}}``
    POST   /api/playground/jobs       one run or a list of runs -> ``202`` and a ``job_id``
    POST   /api/models/<id>/invoke    ``{"method": ..., "arguments": {...}}``
    POST   /api/models/<id>/sample    dummy arguments for one method
    POST   /api/models/<id>/sessions  build the model once and keep it alive
    POST   /api/sessions/<id>/invoke  call a method on the live instance
    DELETE /api/sessions/<id>         close a session
    POST   /api/notebook              ``{"model_id": ..., "parameters": {...}}``
    POST   /api/search                deprecated alias of ``GET /api/search``

The interactive documentation is served by FastAPI itself at ``/docs`` and the
OpenAPI document at ``/api/openapi.json``.

Configuration
-------------
The same ``LM_*`` environment variables are honoured (see the README for the
complete list).  The server also supports ``--host``, ``--port``,
``--package-dir``, ``--no-playground``, ``--no-invoke``, ``--token``,
``--cors-origin`` and ``--rate-limit`` through its command line interface.

Running
-------
::

    python3 server_fastapi.py                       # uses uvicorn
    uvicorn server_fastapi:app --host 0.0.0.0 --port 8000

``uvicorn`` is an optional dependency: install it with ``pip install
likelihood[server]`` (or ``pip install "fastapi[standard]" uvicorn``).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import math
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from email.utils import formatdate
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

import notebook_generator
import server_core as core
from model_catalog import (
    category_index,
    facet_counts,
    filter_by_category,
    filter_by_tags,
    find_model,
    normalize_tags,
    search_models,
    summarize,
)

LOGGER = logging.getLogger("likelihood.server.fastapi")

#: Version exposed through ``X-API-Version`` and the OpenAPI document.
API_VERSION = core.API_VERSION
#: Largest JSON body accepted by the endpoints (1 MiB).
MAX_BODY_BYTES = core.MAX_BODY_BYTES
#: Content types accepted by the JSON endpoints.
_JSON_TYPES = ("application/json", "text/json")
#: MIME type of the generated notebooks.
_NOTEBOOK_TYPE = "application/x-ipynb+json"
#: Methods advertised by the ``Allow`` header of the method rejections.
_ALLOWED_METHODS = "GET, HEAD, POST, DELETE, OPTIONS"


class ServerSettings:
    """Runtime state shared by every request of a :func:`create_app` instance.

    The mutable switches (token, CORS origins, feature flags, limiters) live here
    instead of on a request handler class, so several apps can coexist in the
    same process -- which is what the test-suite relies on.
    """

    def __init__(self, package_dir: Optional[Path] = None) -> None:
        self.store = (
            core.CatalogStore(package_dir) if package_dir is not None else core.CATALOG_STORE
        )
        if package_dir is not None:
            self.store.load()
        self.catalog: List[Dict[str, Any]] = self.store.catalog
        self.metrics = core.Metrics()
        self.rate_limiter = core.RateLimiter(core.RATE_LIMIT)
        self.jobs = core.JobQueue()
        self.sessions = core.SessionStore()
        self.invoke_slots = threading.BoundedSemaphore(core.INVOKE_MAX_CONCURRENCY)
        self.playground_enabled = core.PLAYGROUND_ENABLED
        self.invoke_enabled = core.INVOKE_ENABLED
        self.sessions_enabled = core.SESSIONS_ENABLED
        self.api_token = core.API_TOKEN
        self.cors_origins: Tuple[str, ...] = core.CORS_ORIGINS
        self.template: str = ""

    def refresh(self) -> None:
        """Rebuild the catalog when its sources changed on disk."""
        if self.catalog is not self.store.catalog:
            return  # an explicitly injected catalog (tests, --package-dir) wins
        if not self.store.refresh():
            return
        self.catalog = self.store.catalog
        LOGGER.info(
            "Catálogo recargado desde %s: %d modelos", self.store.package_dir, len(self.catalog)
        )

    def preview_template(self) -> str:
        """The preview template, loaded lazily and cached per app."""
        if not self.template:
            self.template = core.load_template()
        return self.template


def get_settings(request: Request) -> ServerSettings:
    """FastAPI dependency returning the settings of the running app."""
    return request.app.state.settings


# ---------------------------------------------------------------------------
# Request bodies (Pydantic models, so FastAPI documents and validates them)
# ---------------------------------------------------------------------------


class _AliasedBody(BaseModel):
    """Base body that tolerates the legacy aliases (``id``, ``params``, ...)."""

    model_config = ConfigDict(extra="allow")

    def payload(self) -> Dict[str, Any]:
        """The request data as a plain dict, dropping the ``None`` placeholders."""
        data = {key: value for key, value in self.model_dump().items() if value is not None}
        data.update(self.model_extra or {})
        return data


class PlaygroundRequest(_AliasedBody):
    """Body of ``POST /api/playground`` and of each run of a batch job."""

    model_id: Optional[str] = Field(default=None, description="Identificador del modelo.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )


class InvokeRequest(_AliasedBody):
    """Body of ``POST /api/models/<id>/invoke``."""

    method: str = Field(default="", description="Nombre del método a invocar.")
    arguments: Optional[Dict[str, Any]] = Field(default=None, description="Argumentos del método.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )


class SessionRequest(_AliasedBody):
    """Body of ``POST /api/models/<id>/sessions``."""

    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )


class SampleRequest(_AliasedBody):
    """Body of ``POST /api/models/<id>/sample``."""

    method: str = Field(default="", description="Nombre del método a muestrear.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )


class NotebookRequest(_AliasedBody):
    """Body of ``POST /api/notebook``."""

    model_id: Optional[str] = Field(default=None, description="Identificador del modelo.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Valores por defecto de los widgets del notebook."
    )


class SearchRequest(_AliasedBody):
    """Body of the deprecated ``POST /api/search``."""

    query: Optional[str] = Field(default=None, description="Término de búsqueda.")
    tags: List[str] = Field(default_factory=list, description="Etiquetas que filtrar.")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def json_response(
    status: int,
    payload: Any,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """JSON response in the catalogue's original JSON envelope."""
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    final_headers = {"Content-Type": "application/json; charset=utf-8"}
    final_headers.update(headers or {})
    return Response(content=body, status_code=status, headers=final_headers)


def bytes_response(
    status: int,
    body: bytes,
    content_type: str,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """Raw response that always declares the content type."""
    final_headers = {"Content-Type": content_type}
    final_headers.update(headers or {})
    return Response(content=body, status_code=status, headers=final_headers)


def html_error(status: int, title: str, message: str) -> Response:
    """HTML error page for the browser facing endpoints."""
    document = (
        "<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
        f"<title>{core._escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#1f2937;color:#f9fafb;"
        "display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
        "div{text-align:center;max-width:32rem;padding:2rem}h1{font-size:3rem;margin:0 0 .5rem}"
        "p{color:#9ca3af;line-height:1.6}a{color:#8ab4f8}</style></head><body><div>"
        f"<h1>{status}</h1><p>{core._escape(message)}</p>"
        "<p><a href='/'>← Volver al catálogo de modelos</a></p></div></body></html>"
    )
    return bytes_response(
        status,
        document.encode("utf-8"),
        "text/html; charset=utf-8",
        {"Cache-Control": "no-store"},
    )


def api_error_response(exc: core.APIError, request_id: str = "") -> Response:
    """Serialise a mapped exception (status, headers and extra fields)."""
    payload: Dict[str, Any] = {"success": False, "error": str(exc)}
    payload.update(exc.extra)
    headers = {"Cache-Control": "no-store"}
    headers.update(exc.headers)
    if request_id:
        headers["X-Request-Id"] = request_id
    return json_response(exc.status, payload, headers)


def cacheable_json(
    settings: ServerSettings,
    request: Request,
    payload: Any,
    cache_key: str,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """JSON response with ``ETag``/``Cache-Control`` support.

    The catalog fingerprint plus the request path is a valid, cheap ``ETag``; a
    matching ``If-None-Match`` answers ``304`` without sending the body.
    """
    digest = hashlib.sha1(
        f"{cache_key}:{settings.store.fingerprint}:{API_VERSION}".encode("utf-8")
    ).hexdigest()[:20]
    etag = f'"{digest}"'
    final_headers: Dict[str, str] = {"Cache-Control": core.API_CACHE_CONTROL, "ETag": etag}
    final_headers.update(headers or {})
    cached = (request.headers.get("if-none-match") or "").strip()
    if cached in {etag, f"W/{etag}"}:
        return bytes_response(304, b"", "application/json; charset=utf-8", final_headers)
    return json_response(200, payload, final_headers)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


async def read_body(request: Request) -> bytes:
    """Read the raw body, enforcing the :data:`MAX_BODY_BYTES` ceiling."""
    declared = (request.headers.get("content-length") or "").strip()
    if declared:
        try:
            length = int(declared)
        except ValueError as exc:
            raise core.ValidationError("Cabecera Content-Length inválida") from exc
        if length < 0:
            raise core.ValidationError("Cabecera Content-Length inválida")
        if length > MAX_BODY_BYTES:
            raise core.RequestBodyTooLarge(
                f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
            )
    chunks: List[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise core.RequestBodyTooLarge(
                f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def read_json(request: Request, allow_list: bool = False) -> Any:
    """Parse the request body as JSON, mirroring the legacy error messages."""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in _JSON_TYPES:
        raise core.UnsupportedMediaType(
            f"Tipo de contenido no soportado: {content_type}; usa application/json"
        )
    raw = await read_body(request)
    if not raw.strip():
        return [] if allow_list else {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise core.ValidationError(f"Cuerpo JSON inválido: {exc}") from exc
    if isinstance(data, dict):
        return data
    if allow_list and isinstance(data, list):
        return data
    raise core.ValidationError("El cuerpo JSON debe ser un objeto")


def query_dict(request: Request) -> Dict[str, List[str]]:
    """The query string as ``{key: [values]}`` (blank values preserved)."""
    query: Dict[str, List[str]] = {}
    for key, value in request.query_params.multi_items():
        query.setdefault(key, []).append(value)
    return query


# ---------------------------------------------------------------------------
# Catalog helpers ported from the standard-library handlers
# ---------------------------------------------------------------------------


def model_or_404(settings: ServerSettings, model_id: str) -> Dict[str, Any]:
    """Resolve ``model_id`` of a path (``404`` when missing)."""
    model = find_model(settings.catalog, (model_id or "").strip())
    if model is None:
        raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
    return model


def find_payload_model(settings: ServerSettings, data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``model_id`` of a JSON payload (``400`` when missing)."""
    model_id = str(data.get("model_id") or data.get("id") or "").strip()
    if not model_id:
        raise core.ValidationError("El campo 'model_id' es obligatorio")
    model = find_model(settings.catalog, model_id)
    if model is None:
        raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
    return model


def prepare_run(
    settings: ServerSettings, data: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate one playground payload and return ``(model, parameters)``."""
    model = find_payload_model(settings, data)
    raw_parameters = data.get("parameters", data.get("params", {}))
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise core.ValidationError("El campo 'parameters' debe ser un objeto")
    return model, core.coerce_model_parameters(model, raw_parameters)


def pagination(query: Dict[str, List[str]]) -> Tuple[Optional[int], int]:
    """Validate ``limit``/``offset`` and translate ``ValueError`` to ``400``."""
    try:
        return core.parse_pagination(query)
    except ValueError as exc:
        raise core.ValidationError(str(exc)) from exc


def filtered_models(settings: ServerSettings, query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Apply the ``category``, ``q`` and ``tags`` filters to the catalog."""
    selected = settings.catalog
    category = (query.get("category") or [""])[0].strip()
    if category and category.lower() != "all":
        selected = filter_by_category(selected, category)
    term = (query.get("q") or query.get("query") or [""])[0].strip()
    if term:
        selected = search_models(selected, term)
    tags = query.get("tags") or []
    if tags:
        selected = filter_by_tags(selected, tags)
    return selected


def page_url(query: Dict[str, List[str]], offset: int, limit: Optional[int]) -> str:
    """Absolute path of the next page, preserving every other filter."""
    params: List[Tuple[str, str]] = []
    for key, values in query.items():
        if key in ("offset", "limit"):
            continue
        params.extend((key, value) for value in values)
    params.append(("offset", str(offset)))
    if limit is not None:
        params.append(("limit", str(limit)))
    return "/api/models?" + urlencode(params)


def notebook_parameters(settings: ServerSettings, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validated parameter values that become the widget defaults."""
    raw = data.get("parameters", data.get("params")) if isinstance(data, dict) else None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise core.ValidationError("El campo 'parameters' debe ser un objeto")
    model = find_payload_model(settings, data)
    return core.coerce_model_parameters(model, raw, require_all=False)


def require_invoke(settings: ServerSettings) -> None:
    if not settings.invoke_enabled:
        raise core.APIError(
            "La invocación de métodos está deshabilitada en este servidor", status=403
        )


def require_sessions(settings: ServerSettings) -> None:
    require_invoke(settings)
    if not settings.sessions_enabled:
        raise core.APIError("Las sesiones están deshabilitadas en este servidor", status=403)


def acquire_invoke_slot(settings: ServerSettings) -> None:
    """Bound the number of concurrent ``/invoke`` child processes."""
    if not settings.invoke_slots.acquire(blocking=False):
        raise core.RateLimitExceeded(
            "Hay demasiadas llamadas en curso; reintenta en unos segundos",
            headers={"Retry-After": "5"},
        )


def search_payload(
    settings: ServerSettings,
    term: str,
    tags: Sequence[str],
    limit: int,
    deprecated: bool = False,
) -> Dict[str, Any]:
    """Search results plus the category/tag facets of the same query."""
    matches = search_models(settings.catalog, term, tags)
    # The facets are computed *before* the tag filter, so the counters tell how
    # many models each tag would keep.
    base = search_models(settings.catalog, term)
    selected = matches[:limit] if limit > 0 else matches
    return {
        "success": True,
        "query": term,
        "tags": normalize_tags(tags),
        "count": len(selected),
        "total": len(matches),
        "deprecated": deprecated,
        "hint": "Usa GET /api/search?q=... en su lugar" if deprecated else "",
        "facets": facet_counts(base),
        "results": [core.model_brief(model) for model in selected],
        "suggestions": [
            {
                "id": model["id"],
                "title": model["title"],
                "category": model["category"],
                "category_label": model["category_label"],
                "icon": model["icon"],
            }
            for model in selected
        ],
    }


# ---------------------------------------------------------------------------
# Static assets and cross-cutting concerns
# ---------------------------------------------------------------------------


def serve_asset(asset: Path, request: Request) -> Response:
    """Serve one static asset with ``ETag`` and ``Last-Modified``."""
    try:
        body = asset.read_bytes()
        stat = asset.stat()
    except OSError as exc:
        LOGGER.warning("Cannot read asset %s: %s", asset, exc)
        return html_error(404, "No encontrado", f'No se pudo leer "{asset.name}".')
    content_type = core.content_type_for(asset)
    etag = f'"{stat.st_mtime_ns:x}-{len(body):x}"'
    headers = {
        "Cache-Control": core.STATIC_CACHE_CONTROL,
        "ETag": etag,
        "Last-Modified": formatdate(stat.st_mtime, usegmt=True),
    }
    if request.headers.get("if-none-match") == etag:
        return bytes_response(304, b"", content_type, headers)
    return bytes_response(200, body, content_type, headers)


def cors_headers(settings: ServerSettings, request: Request) -> Dict[str, str]:
    """``Access-Control-*`` headers for the current ``Origin``, when allowed."""
    origin = (request.headers.get("origin") or "").strip().rstrip("/")
    if not origin or not settings.cors_origins:
        return {}
    wildcard = "*" in settings.cors_origins
    if not wildcard and origin not in settings.cors_origins:
        return {}
    headers = {
        "Access-Control-Allow-Origin": "*" if wildcard else origin,
        "Access-Control-Expose-Headers": (
            "ETag, X-API-Version, X-Request-Id, Deprecation, Sunset, Link, Retry-After"
        ),
    }
    if not wildcard:
        headers["Vary"] = "Origin"
    return headers


def authenticate(settings: ServerSettings, request: Request) -> None:
    """Reject ``/api`` requests that do not carry the configured token."""
    path = request.url.path
    if not settings.api_token or not path.startswith("/api/"):
        return
    if path == "/api/health":
        return  # liveness probes stay open, they expose no model data
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
    else:
        token = (request.headers.get("x-api-token") or "").strip()
    if token and hmac.compare_digest(token, settings.api_token):
        return
    raise core.AuthenticationError(
        "Token de API inválido o ausente",
        headers={"WWW-Authenticate": 'Bearer realm="likelihood"'},
    )


def enforce_rate_limit(settings: ServerSettings, request: Request) -> None:
    """Apply the sliding window limiter, when one is configured."""
    client = request.client.host if request.client and request.client.host else "unknown"
    retry_after = settings.rate_limiter.hit(client)
    if retry_after is None:
        return
    wait = max(1, math.ceil(retry_after))
    raise core.RateLimitExceeded(
        f"Demasiadas peticiones; reintenta en {wait} segundos",
        headers={"Retry-After": str(wait)},
    )


def preflight_response(settings: ServerSettings, request: Request) -> Response:
    """Reply to the CORS preflight of the allowed origins, else ``405``."""
    origin = (request.headers.get("origin") or "").strip().rstrip("/")
    allowed = bool(origin) and ("*" in settings.cors_origins or origin in settings.cors_origins)
    if not settings.cors_origins or not allowed:
        return json_response(
            405,
            {
                "success": False,
                "error": f"Método no soportado: {request.method}",
                "allow": "GET, HEAD, POST",
            },
            {"Allow": "GET, HEAD, POST"},
        )
    headers = cors_headers(settings, request)
    headers.update(
        {
            "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
            "Access-Control-Allow-Headers": (
                "Content-Type, Authorization, X-API-Token, If-None-Match"
            ),
            "Access-Control-Max-Age": "86400",
        }
    )
    return Response(status_code=204, headers=headers)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def build_router() -> APIRouter:
    """Build the API router with every catalogue endpoint."""
    router = APIRouter()

    # -- catalogue metadata -------------------------------------------------

    @router.get("/api/health", tags=["metadata"])
    def health(settings: ServerSettings = Depends(get_settings)) -> Response:
        """Liveness probe, counters, metrics and the state of the background pools."""
        counts = summarize(settings.catalog)
        return json_response(
            200,
            {
                "success": True,
                "status": "ok",
                "version": core.package_version(),
                "api_version": API_VERSION,
                "models": counts["total"],
                "categories": len(category_index(settings.catalog)),
                "summary": counts,
                "playground": {
                    "enabled": bool(settings.playground_enabled),
                    "timeout_seconds": core.PLAYGROUND_TIMEOUT,
                    "blocked_modules": sorted(core.PLAYGROUND_BLOCKED_MODULES),
                },
                "jobs": settings.jobs.snapshot(),
                "sessions": settings.sessions.snapshot(),
                "invoke": {
                    "enabled": bool(settings.invoke_enabled),
                    "sessions_enabled": bool(settings.sessions_enabled),
                    "timeout_seconds": core.INVOKE_TIMEOUT,
                    "max_concurrency": core.INVOKE_MAX_CONCURRENCY,
                    "side_effects_blocked": not core.INVOKE_ALLOW_SIDE_EFFECTS,
                    "allowed_methods": sorted(core.INVOKE_ALLOWED_METHODS),
                    "blocked_methods": sorted(core.INVOKE_BLOCKED_METHODS),
                },
                "catalog": {"reloads": settings.store.reloads, "ttl_seconds": settings.store.ttl},
                "metrics": settings.metrics.snapshot(),
            },
        )

    @router.get("/api/categories", tags=["metadata"])
    def categories(request: Request, settings: ServerSettings = Depends(get_settings)) -> Response:
        """Categories with their model counts."""
        index = category_index(settings.catalog)
        return cacheable_json(
            settings,
            request,
            {"success": True, "count": len(index), "categories": index},
            "/api/categories",
        )

    @router.get("/api/models", tags=["models"])
    def list_models(request: Request, settings: ServerSettings = Depends(get_settings)) -> Response:
        """Model list with pagination, free text search and facets."""
        query = query_dict(request)
        limit, offset = pagination(query)
        models = filtered_models(settings, query)
        page = models[offset : offset + limit] if limit is not None else models[offset:]
        has_more = offset + len(page) < len(models)
        headers: Dict[str, str] = {}
        next_url: Optional[str] = None
        if has_more and page:
            next_url = page_url(query, offset + len(page), limit)
            headers["Link"] = f'<{next_url}>; rel="next"'
        cache_key = "/api/models?" + urlencode(query, doseq=True)
        return cacheable_json(
            settings,
            request,
            {
                "success": True,
                "count": len(page),
                "total": len(settings.catalog),
                "matched": len(models),
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "next": next_url,
                "categories": category_index(settings.catalog),
                "models": [core.model_brief(model) for model in page],
            },
            cache_key,
            headers,
        )

    @router.get("/api/search", tags=["models"])
    def search(request: Request, settings: ServerSettings = Depends(get_settings)) -> Response:
        """Search endpoint used by the catalog autocomplete."""
        query = query_dict(request)
        term = (query.get("q") or query.get("query") or [""])[0].strip()
        if not term:
            raise core.ValidationError("El parámetro 'q' es obligatorio")
        limit, _ = pagination({**query, "limit": query.get("limit") or ["10"]})
        tags = query.get("tags") or []
        return json_response(200, search_payload(settings, term, tags, limit=limit or 10))

    @router.post("/api/search", tags=["models"], deprecated=True)
    def search_alias(
        body: SearchRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Deprecated alias of ``GET /api/search`` (kept for old clients)."""
        data = body.payload()
        term = str(data.get("query") or data.get("q") or "").strip()
        if not term:
            raise core.ValidationError("El campo 'query' es obligatorio")
        tags = data.get("tags") or []
        headers = {
            "Deprecation": "true",
            "Sunset": core.SEARCH_ALIAS_SUNSET,
            "Link": '</api/search?q=...>; rel="successor-version"',
        }
        return json_response(200, search_payload(settings, term, tags, 0, deprecated=True), headers)

    # -- one model ----------------------------------------------------------

    @router.get("/api/models/{model_id}", tags=["models"])
    def get_model(
        model_id: str, request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Everything known about one model."""
        model = model_or_404(settings, model_id)
        return cacheable_json(
            settings,
            request,
            {"success": True, "model": core.model_payload(model)},
            f"/api/models/{model['id']}",
        )

    @router.get("/api/models/{model_id}/source", tags=["models"])
    def model_source(
        model_id: str, request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Raw source of one model, its diff against a git ref, or a download."""
        model = model_or_404(settings, model_id)
        query = query_dict(request)
        ref = (query.get("diff") or [""])[0].strip()
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        }
        if ref:
            diff = core.source_diff(model["path"], model["source"], ref)
            headers["X-Diff-Changes"] = str(diff["changes"])
            headers["X-Diff-Ref"] = ref
            return json_response(200, {"success": True, "model_id": model["id"], **diff}, headers)
        download = (query.get("download") or [""])[0].strip().lower()
        if download in {"1", "true", "yes", "download"}:
            headers["Content-Disposition"] = f'attachment; filename="{model["id"]}.py"'
        return bytes_response(
            200, model["source"].encode("utf-8"), "text/x-python; charset=utf-8", headers
        )

    @router.get("/api/models/{model_id}/versions", tags=["models"])
    def model_versions(
        model_id: str, request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """List the commits that touched the source file of ``model_id``."""
        model = model_or_404(settings, model_id)
        query = query_dict(request)
        limit, _ = pagination({**query, "limit": query.get("limit") or ["10"]})
        versions = core.git_versions(model["path"], limit or 10)
        return json_response(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "path": model["path"],
                "count": len(versions),
                "versions": versions,
            },
        )

    @router.get("/api/models/{model_id}/invoke", tags=["invoke"])
    def invoke_spec_endpoint(
        model_id: str, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """The methods a client may call on ``model_id``."""
        model = model_or_404(settings, model_id)
        return json_response(200, {"success": True, **core.invoke_spec(model)})

    # -- playground ---------------------------------------------------------

    @router.get("/api/playground", tags=["playground"])
    def playground_get() -> Response:
        """``GET /api/playground`` is not supported; the run uses ``POST``."""
        return json_response(
            405,
            {"success": False, "error": "Método no soportado: GET", "allow": "POST"},
            {"Allow": "POST"},
        )

    @router.post("/api/playground", tags=["playground"])
    def playground_run(
        body: PlaygroundRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Construct a model with the parameters sent by the browser."""
        if not settings.playground_enabled:
            raise core.APIError("El playground está deshabilitado en este servidor", status=403)
        model, values = prepare_run(settings, body.payload())
        try:
            result = core.run_playground(model, values)
        except core.PlaygroundError as exc:
            exc.extra.setdefault("model_id", model["id"])
            raise
        payload: Dict[str, Any] = {"success": True, "model_id": model["id"], "parameters": values}
        payload.update(result)
        return json_response(200, payload)

    @router.get("/api/playground/jobs", tags=["playground"])
    def job_list(request: Request, settings: ServerSettings = Depends(get_settings)) -> Response:
        """The most recent jobs, without their results."""
        query = query_dict(request)
        limit, _ = pagination({**query, "limit": query.get("limit") or ["20"]})
        jobs = settings.jobs.recent(limit or 20)
        return json_response(
            200,
            {
                "success": True,
                "count": len(jobs),
                "jobs": [job.to_dict(include_results=False) for job in jobs],
            },
        )

    @router.post("/api/playground/jobs", status_code=202, tags=["playground"])
    async def job_create(
        request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Queue one run or a batch and return ``202`` with a ``job_id``."""
        body = await read_json(request, allow_list=True)
        if isinstance(body, list):
            entries: Any = body
        elif isinstance(body, dict) and ("jobs" in body or "batch" in body):
            entries = body.get("jobs") or body.get("batch")
        elif isinstance(body, dict) and ("model_id" in body or "id" in body):
            entries = [body]
        else:
            entries = []
        if not isinstance(entries, list) or not entries:
            raise core.ValidationError("El campo 'jobs' debe ser una lista no vacía")
        if len(entries) > core.PLAYGROUND_MAX_BATCH:
            raise core.ValidationError(
                f"El lote supera el máximo de {core.PLAYGROUND_MAX_BATCH} ejecuciones"
            )
        runs: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise core.ValidationError(f"La ejecución {index} debe ser un objeto")
            try:
                model, values = prepare_run(settings, entry)
            except core.APIError as exc:
                exc.extra.setdefault("run_index", index)
                raise
            runs.append({"model": model, "parameters": values})
        job = settings.jobs.submit(runs)
        poll_url = f"/api/playground/jobs/{job.id}"
        return json_response(
            202,
            {
                "success": True,
                "job_id": job.id,
                "status": job.status,
                "batch": job.batch,
                "runs": len(runs),
                "poll_url": poll_url,
            },
            {"Location": poll_url, "Cache-Control": "no-store"},
        )

    @router.get("/api/playground/jobs/{job_id}", tags=["playground"])
    def job_get(job_id: str, settings: ServerSettings = Depends(get_settings)) -> Response:
        """State (and results) of one asynchronous job."""
        job = settings.jobs.get(job_id.strip())
        if job is None:
            raise core.ModelNotFound(f"Job no encontrado o expirado: {job_id}")
        payload: Dict[str, Any] = {"success": True}
        payload.update(job.to_dict())
        payload["poll_url"] = f"/api/playground/jobs/{job.id}"
        return json_response(200, payload)

    @router.get("/api/playground/{model_id}", tags=["playground"])
    def playground_spec_endpoint(
        model_id: str, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Return the playground specification of one model."""
        model = model_or_404(settings, model_id)
        return json_response(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "enabled": bool(settings.playground_enabled),
                "playground": core.playground_spec(model),
            },
        )

    # -- method invocation --------------------------------------------------

    @router.post("/api/models/{model_id}/invoke", tags=["invoke"])
    def invoke(
        model_id: str, body: InvokeRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Build the model and call one of its methods in a child process."""
        require_invoke(settings)
        model = model_or_404(settings, model_id)
        method, arguments, parameters = core.prepare_invocation(model, body.payload())
        acquire_invoke_slot(settings)
        try:
            result = core.run_invoke(model, method, parameters, arguments)
        except core.PlaygroundError as exc:
            exc.extra.setdefault("model_id", model["id"])
            raise
        finally:
            settings.invoke_slots.release()
        payload: Dict[str, Any] = {
            "success": True,
            "model_id": model["id"],
            "method": method["name"],
            "arguments": arguments,
            "parameters": parameters,
            "session_id": None,
        }
        payload.update(result)
        return json_response(200, payload)

    @router.post("/api/models/{model_id}/sample", tags=["invoke"])
    def sample(
        model_id: str, body: SampleRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Dummy arguments for one method of ``model_id``."""
        require_invoke(settings)
        model = model_or_404(settings, model_id)
        data = body.payload()
        method = core.method_for_invocation(model, str(data.get("method") or ""))
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise core.ValidationError("El campo 'parameters' debe ser un objeto")
        parameters = core.resolved_parameters(model, raw_parameters)
        return json_response(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "method": method["name"],
                "parameters": parameters,
                "arguments": core.sample_arguments(model, method, parameters),
            },
        )

    @router.get("/api/sessions", tags=["sessions"])
    def session_list(settings: ServerSettings = Depends(get_settings)) -> Response:
        """Every open session."""
        require_sessions(settings)
        sessions = settings.sessions.recent()
        summary = settings.sessions.snapshot()
        return json_response(
            200,
            {
                "success": True,
                "count": len(sessions),
                "max_sessions": summary["max_sessions"],
                "ttl_seconds": summary["ttl_seconds"],
                "sessions": [session.to_dict() for session in sessions],
            },
        )

    @router.post("/api/models/{model_id}/sessions", tags=["sessions"])
    def session_create(
        model_id: str, body: SessionRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Build the model once and keep the live instance in a child process."""
        require_sessions(settings)
        model = model_or_404(settings, model_id)
        data = body.payload()
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise core.ValidationError("El campo 'parameters' debe ser un objeto")
        parameters = core.coerce_model_parameters(model, raw_parameters)
        session = settings.sessions.create(model, parameters)
        location = f"/api/sessions/{session.id}"
        payload: Dict[str, Any] = {
            "success": True,
            "parameters": parameters,
            "invoke_url": f"{location}/invoke",
        }
        payload.update(session.to_dict())
        return json_response(201, payload, {"Location": location, "Cache-Control": "no-store"})

    @router.get("/api/sessions/{session_id}", tags=["sessions"])
    def session_get(session_id: str, settings: ServerSettings = Depends(get_settings)) -> Response:
        """State of one live session."""
        require_sessions(settings)
        session = settings.sessions.get(session_id.strip())
        if session is None:
            raise core.ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        payload: Dict[str, Any] = {"success": True}
        payload.update(session.to_dict(include_output=True))
        return json_response(200, payload)

    @router.post("/api/sessions/{session_id}/invoke", tags=["sessions"])
    def session_invoke(
        session_id: str, body: InvokeRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Call a method on the live instance of a session."""
        require_sessions(settings)
        session = settings.sessions.get(session_id.strip())
        if session is None:
            raise core.ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        method, arguments, _ = core.prepare_invocation(
            session.model, body.payload(), with_parameters=False
        )
        result = session.invoke(method["name"], arguments)
        payload: Dict[str, Any] = {
            "success": True,
            "model_id": session.model["id"],
            "session_id": session.id,
            "method": method["name"],
            "arguments": arguments,
            "calls": session.calls,
        }
        payload.update(result)
        return json_response(200, payload)

    @router.delete("/api/sessions/{session_id}", tags=["sessions"])
    def session_delete(
        session_id: str, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Stop the worker and forget the session."""
        require_sessions(settings)
        if not settings.sessions.drop(session_id.strip()):
            raise core.ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        return json_response(
            200, {"success": True, "session_id": session_id.strip(), "status": "closed"}
        )

    # -- notebooks ----------------------------------------------------------

    @router.post("/api/notebook", tags=["notebooks"])
    def notebook(
        body: NotebookRequest, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Generate the notebook of a model with the given widget defaults."""
        data = body.payload()
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            raise core.ValidationError("El campo 'model_id' es obligatorio")
        model = find_model(settings.catalog, model_id)
        if model is None:
            raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        parameters = notebook_parameters(settings, data)
        document = notebook_generator.build_notebook(model, parameters or None)
        return json_response(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "filename": f"{model['id']}.ipynb",
                "parameters": parameters,
                "notebook": document,
            },
        )

    @router.get("/api/notebook/{model_id}", tags=["notebooks"])
    def notebook_download(
        model_id: str, request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """Download the generated ``.ipynb`` of ``model_id``."""
        model = model_or_404(settings, model_id)
        query = query_dict(request)
        parameters: Dict[str, Any] = {}
        raw = (query.get("parameters") or [""])[0].strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise core.ValidationError(
                    f"El parámetro 'parameters' no es JSON válido: {exc}"
                ) from exc
            parameters = notebook_parameters(
                settings, {"model_id": model["id"], "parameters": parsed}
            )
        document = notebook_generator.build_notebook(model, parameters or None)
        payload = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
        return bytes_response(
            200,
            payload,
            f"{_NOTEBOOK_TYPE}; charset=utf-8",
            {"Content-Disposition": f'attachment; filename="{model["id"]}.ipynb"'},
        )

    # -- fallbacks and static assets ----------------------------------------

    def _unknown_api(request: Request) -> Response:
        """Unknown ``/api`` endpoints answer with the standard JSON envelope."""
        return json_response(
            404, {"success": False, "error": f"Endpoint desconocido: {request.url.path}"}
        )

    def _reject_method(request: Request) -> Response:
        """Answer ``405`` for the methods the catalog does not implement."""
        return json_response(
            405,
            {
                "success": False,
                "error": f"Método no soportado: {request.method}",
                "allow": _ALLOWED_METHODS,
            },
            {"Allow": _ALLOWED_METHODS},
        )

    router.add_api_route(
        "/api/{rest:path}",
        _unknown_api,
        methods=["GET"],
        tags=["metadata"],
        operation_id="unknown_api_get",
    )
    router.add_api_route(
        "/api/{rest:path}",
        _unknown_api,
        methods=["POST"],
        tags=["metadata"],
        operation_id="unknown_api_post",
    )
    router.add_api_route(
        "/api/{rest:path}",
        _unknown_api,
        methods=["DELETE"],
        tags=["metadata"],
        operation_id="unknown_api_delete",
    )
    router.add_api_route(
        "/api/{rest:path}",
        _reject_method,
        methods=["PUT"],
        tags=["metadata"],
        operation_id="reject_method_put",
    )
    router.add_api_route(
        "/api/{rest:path}",
        _reject_method,
        methods=["PATCH"],
        tags=["metadata"],
        operation_id="reject_method_patch",
    )

    @router.get("/", tags=["pages"])
    def index(request: Request) -> Response:
        """The interactive catalogue shipped in ``models/index.html``."""
        return serve_asset(core.INDEX_FILE, request)

    @router.get("/index.html", tags=["pages"])
    def index_html(request: Request) -> Response:
        """Alias of ``/`` kept for compatibility."""
        return serve_asset(core.INDEX_FILE, request)

    @router.get("/favicon.ico", tags=["pages"])
    def favicon() -> Response:
        """Redirect the legacy icon path to the shipped SVG favicon."""
        return RedirectResponse(
            url="/favicon.svg", status_code=308, headers={"Cache-Control": "public, max-age=86400"}
        )

    @router.get("/preview", tags=["pages"])
    def preview_missing() -> Response:
        """A preview without a model id is a client error."""
        return html_error(400, "Petición inválida", "Falta el identificador del modelo.")

    @router.get("/preview/{remainder:path}", tags=["pages"])
    def preview(
        remainder: str, request: Request, settings: ServerSettings = Depends(get_settings)
    ) -> Response:
        """HTML preview of one model (``/preview/<id>`` or ``/preview/<category>/<id>``)."""
        parts = [part for part in remainder.split("/") if part]
        if not parts:
            return html_error(400, "Petición inválida", "Falta el identificador del modelo.")
        model = find_model(settings.catalog, parts[-1])
        if model is None:
            return html_error(
                404, "Modelo no encontrado", f'No existe ningún modelo con el id "{parts[-1]}".'
            )
        host = str(request.base_url).rstrip("/")
        document = core.render_preview(model, settings.preview_template(), back_url="/", host=host)
        return bytes_response(
            200,
            document.encode("utf-8"),
            "text/html; charset=utf-8",
            {"Cache-Control": "no-store"},
        )

    @router.get("/{asset_path:path}", tags=["pages"])
    def static_asset(asset_path: str, request: Request) -> Response:
        """Any other path is resolved inside ``models/`` (traversal proof)."""
        asset = core.resolve_asset("/" + asset_path)
        if asset is None:
            return html_error(404, "No encontrado", f'No existe el recurso "/{asset_path}".')
        return serve_asset(asset, request)

    return router


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    package_dir: Optional[Path] = None,
    settings: Optional[ServerSettings] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    package_dir : pathlib.Path, optional
        Directory with the model sources to introspect (defaults to the package).
    settings : ServerSettings, optional
        Pre-built runtime state; when omitted a fresh one is created.
    """
    state = settings or ServerSettings(package_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Release the background pools when the server stops."""
        yield
        state.jobs.shutdown()
        state.sessions.shutdown()

    app = FastAPI(
        title="Likelihood catalogue",
        description="HTTP API that exposes the ``likelihood`` model catalogue.",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        redirect_slashes=False,
        lifespan=lifespan,
    )
    app.state.settings = state

    # -- exception handlers -------------------------------------------------

    @app.exception_handler(core.APIError)
    async def _api_error(request: Request, exc: core.APIError) -> Response:
        return api_error_response(exc, getattr(request.state, "request_id", ""))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        problems: List[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            message = str(error.get("msg") or "valor inválido")
            problems.append(f"{location}: {message}" if location else message)
        detail = "; ".join(problems) or "datos inválidos"
        return api_error_response(
            core.ValidationError(f"Datos de la petición inválidos: {detail}", invalid=problems),
            getattr(request.state, "request_id", ""),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        headers = dict(exc.headers or {})
        if exc.status_code == 405:
            allow = headers.get("allow") or headers.get("Allow") or _ALLOWED_METHODS
            headers["Allow"] = allow
            return json_response(
                405,
                {
                    "success": False,
                    "error": f"Método no soportado: {request.method}",
                    "allow": allow,
                },
                headers,
            )
        return json_response(exc.status_code, {"success": False, "error": str(exc.detail)}, headers)

    # -- middleware ---------------------------------------------------------

    @app.middleware("http")
    async def observability(request: Request, call_next):
        """Authentication, rate limiting, metrics, logging and CORS headers."""
        request.state.request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        route = core.endpoint_label(request.method, request.url.path)
        client = request.client.host if request.client and request.client.host else "-"
        try:
            if request.method == "OPTIONS":
                response = preflight_response(state, request)
            else:
                declared = (request.headers.get("content-length") or "").strip()
                if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
                    raise core.RequestBodyTooLarge(
                        f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
                    )
                authenticate(state, request)
                enforce_rate_limit(state, request)
                state.refresh()
                response = await call_next(request)
        except core.APIError as exc:
            LOGGER.warning(
                "[%s] %s %s -> %d: %s",
                request.state.request_id,
                client,
                route,
                exc.status,
                exc,
            )
            response = api_error_response(exc, request.state.request_id)
        except Exception:  # pragma: no cover - last resort, keep the server alive
            LOGGER.exception(
                "[%s] Error inesperado en %s (%.1f ms)",
                request.state.request_id,
                route,
                (time.monotonic() - started) * 1000.0,
            )
            response = json_response(
                500,
                {"success": False, "error": "Error interno del servidor"},
                {"X-Request-Id": request.state.request_id},
            )
        response.headers["X-API-Version"] = API_VERSION
        response.headers["X-Request-Id"] = request.state.request_id
        for key, value in cors_headers(state, request).items():
            response.headers.setdefault(key, value)
        elapsed = (time.monotonic() - started) * 1000.0
        state.metrics.record(route, response.status_code, elapsed / 1000.0)
        LOGGER.info(
            "[%s] %s %s %d %.1f ms",
            request.state.request_id,
            client,
            route,
            response.status_code,
            elapsed,
        )
        return response

    app.include_router(build_router())
    return app


#: ASGI application used by ``uvicorn server_fastapi:app``.
app = create_app()


def run(host: str = core.DEFAULT_HOST, port: int = core.DEFAULT_PORT, **options: Any) -> int:
    """Serve the catalogue with uvicorn until interrupted."""
    try:
        import uvicorn
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        print(
            "uvicorn no está instalado; ejecuta `pip install likelihood[server]` "
            "o `pip install uvicorn`.",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(create_app(**options), host=host, port=port, log_level="info")
    return 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sirve el catálogo de modelos de Likelihood.")
    parser.add_argument(
        "--host",
        default=core.DEFAULT_HOST,
        help=f"interfaz (por defecto: {core.DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=core.DEFAULT_PORT,
        help=f"puerto (por defecto: {core.DEFAULT_PORT})",
    )
    parser.add_argument(
        "--package-dir", type=Path, default=None, help="paquete de modelos a inspeccionar"
    )
    parser.add_argument("--list", action="store_true", help="muestra el catálogo y sale")
    parser.add_argument(
        "--no-playground",
        action="store_true",
        help="deshabilita POST /api/playground (no construye modelos desde el navegador)",
    )
    parser.add_argument(
        "--no-invoke",
        action="store_true",
        help="deshabilita /invoke y las sesiones (el catálogo queda de solo lectura)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="exige Authorization: Bearer <token> en /api (equivale a LM_API_TOKEN)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        metavar="ORIGIN",
        help="origen permitido para CORS; puede repetirse (equivale a LM_CORS_ORIGINS)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="peticiones por minuto y cliente (equivale a LM_RATE_LIMIT)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="registra cada petición")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    settings = ServerSettings(arguments.package_dir)
    if arguments.no_playground:
        settings.playground_enabled = False
    if arguments.no_invoke:
        settings.invoke_enabled = False
        settings.sessions_enabled = False
    if arguments.token is not None:
        settings.api_token = arguments.token.strip()
    if arguments.cors_origin:
        settings.cors_origins = tuple(
            origin.strip().rstrip("/") for origin in arguments.cors_origin if origin.strip()
        )
    if arguments.rate_limit is not None:
        settings.rate_limiter = core.RateLimiter(arguments.rate_limit)

    if arguments.list:
        for model in settings.catalog:
            print(f"{model['icon']} {model['id']:<22} {model['module']}")
        print(f"\n{len(settings.catalog)} modelos")
        return 0

    if not settings.catalog:
        print("No se encontraron modelos en el paquete indicado.", file=sys.stderr)
        return 1
    return run(arguments.host, arguments.port, settings=settings)


if __name__ == "__main__":
    raise SystemExit(_main())
