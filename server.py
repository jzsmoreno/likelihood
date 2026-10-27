#!/usr/bin/env python3
"""HTTP server that exposes the ``likelihood`` model catalog.

The server is intentionally small and dependency free (standard library only).
All model information comes from :mod:`model_catalog`, which introspects the
package sources, so there is nothing to keep in sync manually.

Endpoints
---------
``GET  /``                          the interactive catalogue (``models/index.html``)
``GET  /preview/<model_id>``        server rendered HTML preview of a model
``GET  /preview/<category>/<id>``   same, accepting the legacy two-part URL
``GET  /api/health``                liveness probe with a few counters
``GET  /api/categories``            categories with their model counts
``GET  /api/models``                model list (``?category=``, ``?q=``, ``?limit=``, ``?offset=``)
``GET  /api/models/<model_id>``     everything known about one model
``GET  /api/models/<id>/source``    raw source of the model (``?download=1`` attaches it)
``GET  /api/search``                ``?q=`` search (autocomplete friendly)
``GET  /api/playground/<model_id>`` playground specification of one model
``GET  /api/notebook/<model_id>``   generated ``.ipynb`` download
``POST /api/playground``            ``{"model_id": "...", "parameters": {...}}`` -> run the model
``POST /api/notebook``              ``{"model_id": "..."}`` -> notebook JSON
``POST /api/search``                deprecated alias of ``GET /api/search``

Any other path is resolved inside ``models/`` so that the HTML assets shipped
with the project (``index.html``, ``preview.html``, ``favicon.svg``) are served
as static files.

Configuration
-------------
``LM_HOST``                interface to bind (default ``127.0.0.1``)
``LM_PORT``                TCP port (default ``8000``)
``LM_PLAYGROUND``          set it to ``0``/``false`` to disable ``POST /api/playground``
``LM_PLAYGROUND_TIMEOUT``  seconds a playground run may take (default ``60``)

.. warning::
   The playground constructs real model objects in a child Python process.  Keep
   the server bound to ``127.0.0.1`` (the default) unless you trust every client
   that can reach the port.

Running
-------
::

    python3 server.py
    python3 server.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import notebook_generator
from model_catalog import (
    REPO_ROOT,
    build_catalog,
    category_index,
    category_meta,
    filter_by_category,
    find_model,
    guess_widget,
    search_models,
    summarize,
)

LOGGER = logging.getLogger("likelihood.server")
ASSETS_DIR = REPO_ROOT / "models"
PREVIEW_TEMPLATE = ASSETS_DIR / "preview.html"
INDEX_FILE = ASSETS_DIR / "index.html"

DEFAULT_HOST = os.environ.get("LM_HOST") or os.environ.get("SERVER_HOST") or "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("LM_PORT") or os.environ.get("SERVER_PORT") or 8000)

_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")

#: Largest request body accepted by the JSON endpoints (1 MiB).
MAX_BODY_BYTES = 1_048_576
#: Largest page returned by ``GET /api/models``.
MAX_PAGE_SIZE = 200
#: ``Cache-Control`` used for the static assets under ``models/``.
STATIC_CACHE_CONTROL = "public, max-age=300, must-revalidate"
#: Seconds a ``POST /api/playground`` run may take.
PLAYGROUND_TIMEOUT = float(os.environ.get("LM_PLAYGROUND_TIMEOUT") or 60)
#: Marker used to separate the playground result from anything the model prints.
_PLAYGROUND_MARKER = "__LIKELIHOOD_PLAYGROUND__"


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean environment variable (``0``/``false``/``no`` are false)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


#: Whether ``POST /api/playground`` may construct models (``--no-playground``).
PLAYGROUND_ENABLED = _env_flag("LM_PLAYGROUND", True)

# The catalog is static for the lifetime of the process: it is derived from the
# package sources, which do not change while the server is running.
CATALOG: List[Dict[str, Any]] = build_catalog()
MODELS_BY_ID: Dict[str, Dict[str, Any]] = {model["id"]: model for model in CATALOG}


def model_brief(model: Dict[str, Any]) -> Dict[str, Any]:
    """Small projection used by listings and search results."""
    return {
        "id": model["id"],
        "title": model["title"],
        "icon": model["icon"],
        "category": model["category"],
        "categoryLabel": model["categoryLabel"],
        "shortDescription": model["shortDescription"],
        "tags": model["tags"],
        "module": model["module"],
        "signature": model["signature"],
        "parameterCount": len(model["parameters"]),
        "methodCount": len(model["methods"]),
        "lineCount": model["lineCount"],
    }


def model_payload(model: Dict[str, Any], include_source: bool = True) -> Dict[str, Any]:
    """Full JSON representation of ``model``."""
    payload = dict(model)
    if not include_source:
        payload.pop("source", None)
    payload["previewUrl"] = f"/preview/{model['id']}"
    payload["notebookUrl"] = f"/api/notebook/{model['id']}"
    payload["sourceUrl"] = f"/api/models/{model['id']}/source"
    payload["sourceDownloadUrl"] = f"/api/models/{model['id']}/source?download=1"
    payload["playgroundUrl"] = f"/api/playground/{model['id']}"
    payload["playground"] = playground_spec(model)
    return payload


def payloads(models: Sequence[Dict[str, Any]], include_source: bool = True) -> List[Dict[str, Any]]:
    """Full JSON representation of a list of models."""
    return [model_payload(model, include_source) for model in models]


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Best effort conversion of a Python literal into something JSON can carry."""
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in list(value.items())[:100]}
    return None


def _literal_for_kind(parameter: Dict[str, Any], text: Optional[str], fallback: bool) -> Any:
    """Turn source text recorded by the catalog into a JSON friendly value."""
    if text is None:
        return None
    try:
        value: Any = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        # Bare words such as ``softmax`` are not valid literals but are still
        # useful as starting values; source expressions (``np.array(...)``) are not.
        return text if fallback else None
    value = _json_safe(value)
    kind = parameter.get("widget") or guess_widget(parameter["type"])
    if isinstance(value, str) and kind in ("int", "float"):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if kind == "int" and number.is_integer() else number
    if isinstance(value, str) and kind == "bool" and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _default_value(parameter: Dict[str, Any]) -> Any:
    """JSON friendly version of the recorded default of ``parameter``."""
    return _literal_for_kind(parameter, parameter.get("default"), fallback=True)


def _sample_value(parameter: Dict[str, Any]) -> Any:
    """JSON friendly version of the generated example value, when it is a literal."""
    return _literal_for_kind(parameter, parameter.get("sample"), fallback=False)


def playground_fields(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe the constructor arguments that the playground can tune."""
    fields: List[Dict[str, Any]] = []
    for parameter in model["parameters"]:
        if parameter.get("variadic"):
            # ``*args`` / ``**kwargs`` cannot be sent as a JSON object.
            continue
        fields.append(
            {
                "name": parameter["name"],
                "type": parameter["type"],
                "kind": parameter.get("widget") or guess_widget(parameter["type"]),
                "required": bool(parameter["required"]),
                "default": _default_value(parameter),
                "sample": _sample_value(parameter),
                "description": parameter["description"] or "",
            }
        )
    return fields


def playground_spec(model: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the browser needs to render the playground of ``model``."""
    return {
        "modelId": model["id"],
        "modelName": model["name"],
        "importStatement": model["importStatement"],
        "endpoint": "/api/playground",
        "fields": playground_fields(model),
    }


_BOOL_TRUE = {"true", "1", "yes", "y", "si", "sí", "on"}
_BOOL_FALSE = {"false", "0", "no", "n", "off"}


def coerce_playground_value(field: Dict[str, Any], value: Any) -> Any:
    """Validate ``value`` against the declared type of ``field``.

    The playground only accepts JSON values: it never evaluates Python code.
    Raises :class:`ValueError` with a user facing explanation when the value
    cannot be converted.
    """
    name = field["name"]
    kind = field["kind"]
    if kind == "int":
        if isinstance(value, bool) or value is None:
            raise ValueError(f"se esperaba un entero para '{name}'")
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ValueError(f"se esperaba un entero para '{name}': {value!r}") from exc
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"se esperaba un entero para '{name}': {value!r}")
            return int(value)
        if isinstance(value, int):
            return value
        raise ValueError(f"se esperaba un entero para '{name}'")
    if kind == "float":
        if isinstance(value, bool) or value is None:
            raise ValueError(f"se esperaba un número para '{name}'")
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"se esperaba un número para '{name}': {value!r}") from exc
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"se esperaba un número finito para '{name}'")
            return number
        raise ValueError(f"se esperaba un número para '{name}'")
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _BOOL_TRUE:
                return True
            if lowered in _BOOL_FALSE:
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(f"se esperaba true o false para '{name}': {value!r}")
    if kind in ("str", "text"):
        if isinstance(value, str):
            return value
        raise ValueError(f"se esperaba texto para '{name}': {value!r}")
    # ``raw`` fields receive the JSON value untouched (lists, objects, scalars).
    return value


def parse_pagination(query: Dict[str, List[str]]) -> Tuple[Optional[int], int]:
    """Validate the ``limit``/``offset`` query parameters.

    ``limit`` is ``None`` when the caller did not ask for pagination, which
    keeps the endpoint backwards compatible.
    """
    raw_limit = (query.get("limit") or [""])[0].strip()
    raw_offset = (query.get("offset") or ["0"])[0].strip()
    limit: Optional[int] = None
    if raw_limit:
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"El parámetro 'limit' debe ser un entero: {raw_limit!r}") from exc
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"El parámetro 'limit' debe estar entre 1 y {MAX_PAGE_SIZE}")
    try:
        offset = int(raw_offset or "0")
    except ValueError as exc:
        raise ValueError(f"El parámetro 'offset' debe ser un entero: {raw_offset!r}") from exc
    if offset < 0:
        raise ValueError("El parámetro 'offset' no puede ser negativo")
    return limit, offset


#: Runs in a child interpreter so a crashing or slow model cannot take the
#: server down.  The payload arrives through ``stdin``; the result is written to
#: ``stdout`` after a marker line, which keeps any ``print`` of the model apart.
_PLAYGROUND_BOOTSTRAP = r"""
import importlib
import json
import sys

payload = json.load(sys.stdin)
module = importlib.import_module(payload["module"])
target = getattr(module, payload["name"])
instance = target(**payload["parameters"])
try:
    attributes = sorted(name for name in vars(instance) if not name.startswith("_"))[:25]
except TypeError:
    attributes = []
result = {
    "type": type(instance).__name__,
    "module": type(instance).__module__,
    "repr": repr(instance)[:2000],
    "attributes": attributes,
}
sys.stdout.write("\n" + payload["marker"] + json.dumps(result, default=str) + "\n")
"""


class PlaygroundError(RuntimeError):
    """A playground run failed; ``status`` is the HTTP status to report."""

    def __init__(self, message: str, status: int = 422, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class RequestBodyTooLarge(ValueError):
    """The request body exceeded :data:`MAX_BODY_BYTES`."""


def _last_error_line(stderr: str) -> str:
    """The final meaningful line of a traceback (usually the exception)."""
    for line in reversed([line.strip() for line in stderr.splitlines()]):
        if line:
            return line
    return ""


def run_playground(
    model: Dict[str, Any],
    parameters: Dict[str, Any],
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Construct ``model`` in a child interpreter and describe the instance.

    Parameters
    ----------
    model : dict
        Catalog entry of the model to build.
    parameters : dict
        Keyword arguments already validated with :func:`coerce_playground_value`.
    timeout : float, optional
        Maximum number of seconds the child process may run.

    Returns
    -------
    dict
        ``type``, ``module``, ``repr``, ``attributes``, ``elapsed``, ``stdout``
        and ``stderr`` of the run.

    Raises
    ------
    PlaygroundError
        With status 504 on timeout and 422 when the constructor fails.
    """
    timeout = PLAYGROUND_TIMEOUT if timeout is None else timeout
    payload = json.dumps(
        {
            "module": model["module"],
            "name": model["name"],
            "parameters": parameters,
            "marker": _PLAYGROUND_MARKER,
        }
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), environment.get("PYTHONPATH", "")) if part
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PLAYGROUND_BOOTSTRAP],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=environment,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PlaygroundError(
            f"La ejecución superó el límite de {timeout:g} segundos", status=504
        ) from exc
    elapsed = time.monotonic() - started

    stdout = completed.stdout or ""
    result_line = ""
    for line in stdout.splitlines():
        if line.startswith(_PLAYGROUND_MARKER):
            result_line = line[len(_PLAYGROUND_MARKER) :]
    if completed.returncode != 0 or not result_line:
        stderr = (completed.stderr or "").strip()
        raise PlaygroundError(
            _last_error_line(stderr) or "No se pudo construir el modelo",
            detail=stderr[-4000:],
        )
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as exc:
        raise PlaygroundError(
            "El ejecutable devolvió una respuesta inválida", detail=stdout[-2000:]
        ) from exc
    result["elapsed"] = round(elapsed, 3)
    result["stdout"] = stdout.split(_PLAYGROUND_MARKER)[0][-4000:].strip()
    result["stderr"] = (completed.stderr or "").strip()[-4000:]
    return result


def _escape(value: Any) -> str:
    """HTML-escape ``value`` for use inside markup."""
    return html.escape(str(value), quote=True)


def tags_html(model: Dict[str, Any]) -> str:
    """``<span>`` badges for the model tags."""
    return "".join(f'<span class="tag">{_escape(tag)}</span>' for tag in model["tags"])


def parameters_html(model: Dict[str, Any]) -> str:
    """Cards describing every constructor parameter."""
    if not model["parameters"]:
        return '<p class="empty">Este modelo no declara parámetros de entrada.</p>'
    cards: List[str] = []
    for parameter in model["parameters"]:
        default = parameter["default"] if parameter["default"] is not None else "requerido"
        description = _escape(parameter["description"]) or "—"
        cards.append(
            '<div class="param-card">'
            f'<div class="param-name">{_escape(parameter["name"])}</div>'
            f'<div class="param-type">{_escape(parameter["type"])} · {_escape(default)}</div>'
            f'<div class="param-desc">{description}</div>'
            "</div>"
        )
    return "".join(cards)


def methods_html(model: Dict[str, Any]) -> str:
    """List items describing the public API."""
    if not model["methods"]:
        return '<li class="empty">Este modelo no expone métodos públicos.</li>'
    items: List[str] = []
    for method in model["methods"]:
        origin = "heredado" if method["inherited"] else "propio"
        description = _escape(method["description"]) or "—"
        items.append(
            "<li>"
            f'<span class="method-name">{_escape(method["signature"].split(" -> ")[0])}</span>'
            f'<span class="method-origin">{origin}</span>'
            f'<span class="method-desc">{description}</span>'
            "</li>"
        )
    return "".join(items)


def attributes_html(model: Dict[str, Any]) -> str:
    """Cards describing the documented attributes."""
    attributes = model.get("attributes") or []
    if not attributes:
        return ""
    cards = [
        '<div class="param-card">'
        f'<div class="param-name">{_escape(attribute["name"])}</div>'
        f'<div class="param-type">{_escape(attribute.get("type") or "—")}</div>'
        f'<div class="param-desc">{_escape(attribute.get("description")) or "—"}</div>'
        "</div>"
        for attribute in attributes
    ]
    return "".join(cards)


def attributes_section_html(model: Dict[str, Any]) -> str:
    """The complete ``Atributos`` section, or an empty string when absent.

    Rendering the heading together with its content means the template never
    has to hide empty sections with JavaScript.
    """
    cards = attributes_html(model)
    if not cards:
        return ""
    return f'<h2>Atributos</h2>\n<div class="grid">{cards}</div>'


def load_template(path: Path = PREVIEW_TEMPLATE) -> str:
    """Read the preview template, falling back to a minimal document."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - only when assets are missing
        LOGGER.warning("Preview template not found at %s; using the built-in one", path)
        return _FALLBACK_TEMPLATE


def _script_json(value: Any) -> str:
    """Serialise ``value`` so it can be embedded inside a ``<script>`` tag."""
    text = json.dumps(value, ensure_ascii=False)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render_preview(
    model: Dict[str, Any],
    template: str,
    back_url: str = "/",
    host: str = "",
) -> str:
    """Render the HTML preview of ``model`` from ``template``.

    Placeholders are plain ``{{NAME}}`` tokens replaced through ``str.replace``;
    this keeps the CSS/JS braces of the template untouched and avoids the
    ``str.format`` escaping problems of inline f-strings.
    """
    label, icon = category_meta(model["category"])
    values = {
        "MODEL_ID": _escape(model["id"]),
        "MODEL_TITLE": _escape(model["title"]),
        "MODEL_ICON": _escape(model["icon"] or icon),
        "MODEL_CATEGORY": _escape(label),
        "CATEGORY_CLASS": f"category-{_escape(model['category'])}",
        "MODEL_SHORT_DESCRIPTION": _escape(model["shortDescription"]),
        "MODEL_DESCRIPTION": _escape(model["fullDescription"]).replace("\n", "<br>"),
        "MODEL_TAGS": tags_html(model),
        "MODEL_PARAMETERS": parameters_html(model),
        "MODEL_ATTRIBUTES_SECTION": attributes_section_html(model),
        "MODEL_METHODS": methods_html(model),
        "MODEL_SOURCE": _escape(model["source"]),
        "MODEL_EXAMPLE": _escape(model["example"]),
        "MODEL_IMPORT": _escape(model["importStatement"]),
        "MODEL_MODULE": _escape(model["module"]),
        "MODEL_FILE": _escape(f"{model['path']}:{model['lineno']}"),
        "MODEL_SIGNATURE": _escape(model["signature"]),
        "BACK_URL": _escape(back_url),
        "NOTEBOOK_URL": _escape(f"/api/notebook/{model['id']}"),
        "API_URL": _escape(f"/api/models/{model['id']}"),
        "SOURCE_URL": _escape(f"/api/models/{model['id']}/source"),
        "SOURCE_DOWNLOAD_URL": _escape(f"/api/models/{model['id']}/source?download=1"),
        "PLAYGROUND_URL": _escape("/api/playground"),
        "MODEL_PLAYGROUND_JSON": _script_json(playground_spec(model)),
        "HOST_URL": _escape(host),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return _PLACEHOLDER.sub("", rendered)


_FALLBACK_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{{MODEL_TITLE}}</title>
</head>
<body>
<h1>{{MODEL_ICON}} {{MODEL_TITLE}}</h1>
<p>{{MODEL_SHORT_DESCRIPTION}}</p>
<h2>Firma</h2>
<div>{{MODEL_SIGNATURE}}</div>
<h2>Parámetros</h2>
<div>{{MODEL_PARAMETERS}}</div>
{{MODEL_ATTRIBUTES_SECTION}}
<h2>Métodos</h2>
<ul>{{MODEL_METHODS}}</ul>
<h2>Ejemplo</h2>
<pre>{{MODEL_EXAMPLE}}</pre>
<h2>Código fuente</h2>
<pre>{{MODEL_SOURCE}}</pre>
<p><a href="{{BACK_URL}}">← Volver</a></p>
</body>
</html>
"""


_JSON_TYPES = ("application/json", "text/json")
_NOTEBOOK_TYPE = "application/x-ipynb+json"


class ModelAPIHandler(BaseHTTPRequestHandler):
    """Serve the catalogue API, HTML previews and the static web assets."""

    server_version = "LikelihoodCatalog/1.0"
    protocol_version = "HTTP/1.1"

    catalog: List[Dict[str, Any]] = CATALOG
    template: str = ""
    playground_enabled: bool = PLAYGROUND_ENABLED

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _template(self) -> str:
        """The preview template, loaded lazily and cached per process."""
        if not ModelAPIHandler.template:
            ModelAPIHandler.template = load_template()
        return ModelAPIHandler.template

    def _host(self) -> str:
        """Absolute origin of this request, used for back links."""
        host = self.headers.get("Host")
        if host:
            return f"http://{host}"
        address, port = self.server.server_address[:2]
        return f"http://{address}:{port}"

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write a raw response, always declaring its length."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(
        self,
        status: int,
        payload: Any,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        headers = {"Cache-Control": "no-store"}
        headers.update(extra_headers or {})
        self._send_bytes(status, body, "application/json; charset=utf-8", headers)

    def _send_error_json(self, status: int, message: str, **extra: Any) -> None:
        payload = {"success": False, "error": message}
        payload.update(extra)
        self._send_json(status, payload)

    def _send_redirect(self, location: str, status: int = 302) -> None:
        """Answer with a body-less redirect (used by ``/favicon.ico``)."""
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()

    def _read_json_body(self) -> Dict[str, Any]:
        """Parse the request body as JSON.

        Returns an empty dict for an empty body.  Raises :class:`ValueError`
        when the payload is present but not a JSON object, or when it exceeds
        :data:`MAX_BODY_BYTES`.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Cabecera Content-Length inválida") from exc
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise RequestBodyTooLarge(
                f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
            )
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cuerpo JSON inválido: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("El cuerpo JSON debe ser un objeto")
        return data

    def _query(self) -> Dict[str, List[str]]:
        return parse_qs(urlparse(self.path).query, keep_blank_values=True)

    def _models(self, query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Apply the ``category`` and ``q`` filters to the catalog."""
        selected = self.catalog
        category = (query.get("category") or [""])[0].strip()
        if category and category.lower() != "all":
            selected = filter_by_category(selected, category)
        term = (query.get("q") or query.get("query") or [""])[0].strip()
        if term:
            selected = search_models(selected, term)
        return selected

    def do_HEAD(self) -> None:
        """HEAD reuses the GET routing; ``_send_bytes`` skips the body."""
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        query = self._query()
        try:
            if path in ("/", "/index.html"):
                self._serve_asset(INDEX_FILE)
            elif path == "/favicon.ico":
                self._send_redirect("/favicon.svg", 308)
            elif path == "/api/health":
                self._handle_health()
            elif path == "/api/categories":
                self._handle_categories()
            elif path == "/api/search":
                self._handle_search_query(query)
            elif path == "/api/models":
                self._handle_list_models(query)
            elif path.startswith("/api/models/"):
                self._route_model(path[len("/api/models/") :], query)
            elif path == "/api/playground":
                self._send_json(
                    405,
                    {
                        "success": False,
                        "error": "Método no soportado: GET",
                        "allow": "POST",
                    },
                    {"Allow": "POST"},
                )
            elif path.startswith("/api/playground/"):
                self._handle_playground_spec(path[len("/api/playground/") :])
            elif path.startswith("/api/notebook/"):
                self._handle_notebook_download(path[len("/api/notebook/") :])
            elif path.startswith("/preview/"):
                self._handle_preview(path[len("/preview/") :])
            else:
                self._serve_static(path)
        except BrokenPipeError:  # pragma: no cover - client closed the socket
            LOGGER.debug("Client disconnected while serving %s", path)
        except Exception:  # pragma: no cover - last resort, keep the server alive
            LOGGER.exception("Unhandled error while serving %s", path)
            self._send_error_json(500, "Error interno del servidor")

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            data = self._read_json_body()
        except RequestBodyTooLarge as exc:
            self.close_connection = True  # the unread body must not be parsed as a request
            self._send_error_json(413, str(exc))
            return
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        if path == "/api/notebook":
            self._handle_notebook(data)
        elif path == "/api/playground":
            self._handle_playground(data)
        elif path == "/api/search":
            self._handle_search(data, deprecated=True)
        else:
            self._send_error_json(404, f"Endpoint desconocido: {path}")

    def _route_model(self, remainder: str, query: Dict[str, List[str]]) -> None:
        """Dispatch ``/api/models/<id>`` and ``/api/models/<id>/source``."""
        parts = [part for part in remainder.split("/") if part]
        if len(parts) == 1:
            self._handle_get_model(parts[0])
        elif len(parts) == 2 and parts[1] == "source":
            self._handle_model_source(parts[0], query)
        else:
            self._send_error_json(404, f"Endpoint desconocido: /api/models/{remainder}")

    def _reject_method(self) -> None:
        self._send_json(
            405,
            {
                "success": False,
                "error": f"Método no soportado: {self.command}",
                "allow": "GET, HEAD, POST",
            },
            {"Allow": "GET, HEAD, POST"},
        )

    do_PUT = _reject_method
    do_PATCH = _reject_method
    do_DELETE = _reject_method
    do_OPTIONS = _reject_method

    def _handle_health(self) -> None:
        counts = summarize(self.catalog)
        self._send_json(
            200,
            {
                "success": True,
                "status": "ok",
                "version": package_version(),
                "models": counts["total"],
                "categories": len(category_index(self.catalog)),
                "summary": counts,
            },
        )

    def _handle_categories(self) -> None:
        categories = category_index(self.catalog)
        self._send_json(200, {"success": True, "count": len(categories), "categories": categories})

    def _handle_list_models(self, query: Dict[str, List[str]]) -> None:
        try:
            limit, offset = parse_pagination(query)
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        models = self._models(query)
        if limit is not None:
            page = models[offset : offset + limit]
        else:
            page = models[offset:] if offset else models
        self._send_json(
            200,
            {
                "success": True,
                "count": len(page),
                "total": len(self.catalog),
                "matched": len(models),
                "offset": offset,
                "limit": limit,
                "hasMore": offset + len(page) < len(models),
                "categories": category_index(self.catalog),
                "models": [model_brief(model) for model in page],
            },
        )

    def _handle_get_model(self, model_id: str) -> None:
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        self._send_json(200, {"success": True, "model": model_payload(model)})

    def _handle_model_source(self, model_id: str, query: Dict[str, List[str]]) -> None:
        """Serve the raw source of one model, optionally as an attachment."""
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        download = (query.get("download") or [""])[0].strip().lower()
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        }
        if download in {"1", "true", "yes", "download"}:
            headers["Content-Disposition"] = f'attachment; filename="{model["id"]}.py"'
        self._send_bytes(
            200,
            model["source"].encode("utf-8"),
            "text/x-python; charset=utf-8",
            headers,
        )

    def _handle_playground_spec(self, model_id: str) -> None:
        """Return the playground specification of one model."""
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "enabled": bool(self.playground_enabled),
                "playground": playground_spec(model),
            },
        )

    def _handle_playground(self, data: Dict[str, Any]) -> None:
        """Construct a model with the parameters sent by the browser."""
        if not self.playground_enabled:
            self._send_error_json(403, "El playground está deshabilitado en este servidor")
            return
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            self._send_error_json(400, "El campo 'model_id' es obligatorio")
            return
        model = find_model(self.catalog, model_id)
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            self._send_error_json(400, "El campo 'parameters' debe ser un objeto")
            return

        fields = {field["name"]: field for field in playground_fields(model)}
        unknown = sorted(name for name in raw_parameters if name not in fields)
        if unknown:
            self._send_error_json(
                400, f"Parámetros desconocidos: {', '.join(unknown)}", unknown=unknown
            )
            return

        values: Dict[str, Any] = {}
        problems: List[str] = []
        for name, value in raw_parameters.items():
            if value is None and not fields[name]["required"]:
                # ``None`` means "use the default" for optional parameters.
                continue
            try:
                values[name] = coerce_playground_value(fields[name], value)
            except ValueError as exc:
                problems.append(str(exc))
        if problems:
            self._send_error_json(400, "; ".join(problems), invalid=problems)
            return

        missing = [
            field["name"]
            for field in fields.values()
            if field["required"] and field["name"] not in values
        ]
        if missing:
            self._send_error_json(
                400, f"Faltan parámetros obligatorios: {', '.join(missing)}", missing=missing
            )
            return

        try:
            result = run_playground(model, values)
        except PlaygroundError as exc:
            self._send_error_json(exc.status, str(exc), model_id=model["id"], detail=exc.detail)
            return
        payload = {
            "success": True,
            "model_id": model["id"],
            "parameters": values,
        }
        payload.update(result)
        self._send_json(200, payload)

    def _handle_preview(self, remainder: str) -> None:
        parts = [part for part in remainder.split("/") if part]
        if not parts:
            self._send_html_error(400, "Petición inválida", "Falta el identificador del modelo.")
            return
        # Accept both /preview/<id> and the legacy /preview/<category>/<id>.
        model = find_model(self.catalog, parts[-1])
        if model is None:
            self._send_html_error(
                404, "Modelo no encontrado", f'No existe ningún modelo con el id "{parts[-1]}".'
            )
            return
        document = render_preview(model, self._template(), back_url="/", host=self._host())
        self._send_bytes(
            200,
            document.encode("utf-8"),
            "text/html; charset=utf-8",
            {"Cache-Control": "no-store"},
        )

    def _handle_notebook(self, data: Dict[str, Any]) -> None:
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            self._send_error_json(400, "El campo 'model_id' es obligatorio")
            return
        model = find_model(self.catalog, model_id)
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        notebook = notebook_generator.build_notebook(model)
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "filename": f"{model['id']}.ipynb",
                "notebook": notebook,
            },
        )

    def _handle_notebook_download(self, model_id: str) -> None:
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        notebook = notebook_generator.build_notebook(model)
        body = json.dumps(notebook, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(
            200,
            body,
            f"{_NOTEBOOK_TYPE}; charset=utf-8",
            {"Content-Disposition": f'attachment; filename="{model["id"]}.ipynb"'},
        )

    def _handle_search(self, data: Dict[str, Any], deprecated: bool = False) -> None:
        query = str(data.get("query") or data.get("q") or "").strip()
        if not query:
            self._send_error_json(400, "El campo 'query' es obligatorio")
            return
        matches = search_models(self.catalog, query)
        headers: Dict[str, str] = {}
        if deprecated:
            headers = {
                "Deprecation": "true",
                "Link": '</api/search?q=...>; rel="successor-version"',
            }
        self._send_json(
            200,
            {
                "success": True,
                "query": query,
                "count": len(matches),
                "deprecated": deprecated,
                "hint": "Usa GET /api/search?q=... en su lugar" if deprecated else "",
                "results": payloads(matches, include_source=False),
            },
            headers,
        )

    def _handle_search_query(self, query: Dict[str, List[str]]) -> None:
        """Search endpoint used by the catalog autocomplete."""
        term = (query.get("q") or query.get("query") or [""])[0].strip()
        if not term:
            self._send_error_json(400, "El parámetro 'q' es obligatorio")
            return
        try:
            limit, _ = parse_pagination({**query, "limit": query.get("limit") or ["10"]})
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        matches = search_models(self.catalog, term)
        selected = matches[: limit or 10]
        self._send_json(
            200,
            {
                "success": True,
                "query": term,
                "count": len(selected),
                "total": len(matches),
                "results": [model_brief(model) for model in selected],
                "suggestions": [
                    {
                        "id": model["id"],
                        "title": model["title"],
                        "category": model["category"],
                        "categoryLabel": model["categoryLabel"],
                        "icon": model["icon"],
                    }
                    for model in selected
                ],
            },
        )

    def _serve_static(self, path: str) -> None:
        asset = resolve_asset(path)
        if asset is None:
            self._send_html_error(404, "No encontrado", f'No existe el recurso "{path}".')
            return
        self._serve_asset(asset)

    def _serve_asset(self, asset: Path) -> None:
        try:
            body = asset.read_bytes()
            stat = asset.stat()
        except OSError as exc:
            LOGGER.warning("Cannot read asset %s: %s", asset, exc)
            self._send_html_error(404, "No encontrado", f'No se pudo leer "{asset.name}".')
            return
        content_type = content_type_for(asset)
        etag = f'"{stat.st_mtime_ns:x}-{len(body):x}"'
        headers = {
            "Cache-Control": STATIC_CACHE_CONTROL,
            "ETag": etag,
            "Last-Modified": self.date_time_string(stat.st_mtime),
        }
        if self.headers.get("If-None-Match") == etag:
            self._send_bytes(304, b"", content_type, headers)
            return
        self._send_bytes(200, body, content_type, headers)

    def _send_html_error(self, status: int, title: str, message: str) -> None:
        """Send an HTML error page (the browser facing endpoints)."""
        document = (
            "<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
            f"<title>{_escape(title)}</title>"
            "<style>body{font-family:system-ui,sans-serif;background:#1f2937;color:#f9fafb;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
            "div{text-align:center;max-width:32rem;padding:2rem}h1{font-size:3rem;margin:0 0 .5rem}"
            "p{color:#9ca3af;line-height:1.6}a{color:#8ab4f8}</style></head><body><div>"
            f"<h1>{status}</h1><p>{_escape(message)}</p>"
            "<p><a href='/'>← Volver al catálogo de modelos</a></p></div></body></html>"
        )
        self._send_bytes(
            status,
            document.encode("utf-8"),
            "text/html; charset=utf-8",
            {"Cache-Control": "no-store"},
        )


_MIME_TYPES: Dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".ipynb": _NOTEBOOK_TYPE,
}


def content_type_for(path: Path) -> str:
    """MIME type for ``path``, adding UTF-8 to textual types."""
    mime = _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    if mime.startswith("text/") or mime.endswith(("json", "javascript", "svg+xml")):
        return f"{mime}; charset=utf-8"
    return mime


def resolve_asset(path: str, root: Path = ASSETS_DIR) -> Optional[Path]:
    """Map a URL path onto an existing file inside ``root``.

    Returns ``None`` when the path escapes ``root``, points at a hidden file or
    does not resolve to a readable file.  This replaces the naive
    ``base / user_input`` joins that allowed directory traversal.
    """
    relative = unquote(path).lstrip("/")
    if not relative:
        relative = INDEX_FILE.name
    root = root.resolve()
    try:
        candidate = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root):
        return None
    if candidate.is_dir():
        candidate = candidate / INDEX_FILE.name
    if any(part.startswith(".") for part in candidate.relative_to(root).parts):
        return None
    if not candidate.is_file():
        return None
    return candidate


def package_version() -> str:
    """Version of the ``likelihood`` package, read from its ``VERSION`` file."""
    try:
        return (REPO_ROOT / "likelihood" / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover - only when the file is missing
        return "unknown"


class CatalogServer(ThreadingHTTPServer):
    """Threaded HTTP server with address reuse enabled."""

    daemon_threads = True
    allow_reuse_address = True


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> CatalogServer:
    """Build the HTTP server bound to ``host:port``."""
    return CatalogServer((host, port), ModelAPIHandler)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Serve the catalogue until interrupted."""
    counts = summarize(CATALOG)
    try:
        server = create_server(host, port)
    except OSError as exc:
        LOGGER.error("No se pudo abrir %s:%s -> %s", host, port, exc)
        return 1

    address, actual_port = server.server_address[:2]
    banner = [
        f"Catálogo Likelihood v{package_version()}",
        f"Modelos descubiertos : {counts['total']} en {len(category_index(CATALOG))} categorías",
        f"Catálogo web         : http://{address}:{actual_port}/",
        f"API                  : http://{address}:{actual_port}/api/models",
        "Playground           : "
        + (
            f"http://{address}:{actual_port}/api/playground"
            if ModelAPIHandler.playground_enabled
            else "deshabilitado"
        ),
        f"Assets               : {ASSETS_DIR}",
        "Ctrl+C para detener",
    ]
    print("\n".join(banner), flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo el servidor…", flush=True)
    finally:
        server.server_close()
    return 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sirve el catálogo de modelos de Likelihood.")
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"interfaz (por defecto: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"puerto (por defecto: {DEFAULT_PORT})"
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
    parser.add_argument("--verbose", "-v", action="store_true", help="registra cada petición")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    global CATALOG, MODELS_BY_ID
    if arguments.package_dir is not None:
        CATALOG = build_catalog(arguments.package_dir)
        MODELS_BY_ID = {model["id"]: model for model in CATALOG}
        ModelAPIHandler.catalog = CATALOG
    if arguments.no_playground:
        ModelAPIHandler.playground_enabled = False

    if arguments.list:
        for model in CATALOG:
            print(f"{model['icon']} {model['id']:<22} {model['module']}")
        print(f"\n{len(CATALOG)} modelos")
        return 0

    if not CATALOG:
        print("No se encontraron modelos en el paquete indicado.", file=sys.stderr)
        return 1
    return run_server(arguments.host, arguments.port)


if __name__ == "__main__":
    raise SystemExit(_main())
