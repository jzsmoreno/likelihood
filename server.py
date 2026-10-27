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
``GET  /api/models``                model list (``?category=``, ``?q=`` filters)
``GET  /api/models/<model_id>``     everything known about one model
``GET  /api/notebook/<model_id>``   generated ``.ipynb`` download
``POST /api/notebook``              ``{"model_id": "..."}`` -> notebook JSON
``POST /api/search``                ``{"query": "..."}``  -> matching models

Any other path is resolved inside ``models/`` so that the HTML assets shipped
with the project (``index.html``, ``preview.html``) are served as static files.

Configuration
-------------
``LM_HOST``       interface to bind (default ``127.0.0.1``)
``LM_PORT``       TCP port (default ``8000``)

Running
-------
::

    python3 server.py
    python3 server.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sys
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
    search_models,
    summarize,
)

LOGGER = logging.getLogger("likelihood.server")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ASSETS_DIR = REPO_ROOT / "models"
PREVIEW_TEMPLATE = ASSETS_DIR / "preview.html"
INDEX_FILE = ASSETS_DIR / "index.html"

DEFAULT_HOST = os.environ.get("LM_HOST") or os.environ.get("SERVER_HOST") or "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("LM_PORT") or os.environ.get("SERVER_PORT") or 8000)

_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")

# The catalog is static for the lifetime of the process: it is derived from the
# package sources, which do not change while the server is running.
CATALOG: List[Dict[str, Any]] = build_catalog()
MODELS_BY_ID: Dict[str, Dict[str, Any]] = {model["id"]: model for model in CATALOG}


# --------------------------------------------------------------------------
# JSON projections
# --------------------------------------------------------------------------


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
    return payload


def payloads(models: Sequence[Dict[str, Any]], include_source: bool = True) -> List[Dict[str, Any]]:
    """Full JSON representation of a list of models."""
    return [model_payload(model, include_source) for model in models]


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Request handler
# --------------------------------------------------------------------------

_JSON_TYPES = ("application/json", "text/json")
_NOTEBOOK_TYPE = "application/x-ipynb+json"


class ModelAPIHandler(BaseHTTPRequestHandler):
    """Serve the catalogue API, HTML previews and the static web assets."""

    server_version = "LikelihoodCatalog/1.0"
    protocol_version = "HTTP/1.1"

    catalog: List[Dict[str, Any]] = CATALOG
    template: str = ""

    # -- logging ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    # -- helpers ----------------------------------------------------------

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

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str, **extra: Any) -> None:
        payload = {"success": False, "error": message}
        payload.update(extra)
        self._send_json(status, payload)

    def _read_json_body(self) -> Dict[str, Any]:
        """Parse the request body as JSON.

        Returns an empty dict for an empty body.  Raises :class:`ValueError`
        when the payload is present but not a JSON object.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
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

    # -- routing ----------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """HEAD reuses the GET routing; ``_send_bytes`` skips the body."""
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        query = self._query()
        try:
            if path in ("/", "/index.html"):
                self._serve_asset(INDEX_FILE)
            elif path == "/api/health":
                self._handle_health()
            elif path == "/api/categories":
                self._handle_categories()
            elif path == "/api/models":
                self._handle_list_models(query)
            elif path.startswith("/api/models/"):
                self._handle_get_model(path[len("/api/models/") :])
            elif path.startswith("/api/notebook/"):
                self._handle_notebook_download(path[len("/api/notebook/") :])
            elif path.startswith("/preview/"):
                self._handle_preview(path[len("/preview/") :])
            elif path == "/favicon.ico":
                self._send_bytes(204, b"", "text/plain")
            else:
                self._serve_static(path)
        except BrokenPipeError:  # pragma: no cover - client closed the socket
            LOGGER.debug("Client disconnected while serving %s", path)
        except Exception:  # pragma: no cover - last resort, keep the server alive
            LOGGER.exception("Unhandled error while serving %s", path)
            self._send_error_json(500, "Error interno del servidor")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            data = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        if path == "/api/notebook":
            self._handle_notebook(data)
        elif path == "/api/search":
            self._handle_search(data)
        else:
            self._send_error_json(404, f"Endpoint desconocido: {path}")

    def _reject_method(self) -> None:
        self._send_error_json(405, f"Método no soportado: {self.command}", allow="GET, HEAD, POST")

    do_PUT = _reject_method
    do_PATCH = _reject_method
    do_DELETE = _reject_method
    do_OPTIONS = _reject_method

    # -- API handlers -----------------------------------------------------

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
        models = self._models(query)
        self._send_json(
            200,
            {
                "success": True,
                "count": len(models),
                "total": len(self.catalog),
                "categories": category_index(self.catalog),
                "models": [model_brief(model) for model in models],
            },
        )

    def _handle_get_model(self, model_id: str) -> None:
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            self._send_error_json(404, f'Modelo no encontrado: "{model_id}"')
            return
        self._send_json(200, {"success": True, "model": model_payload(model)})

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
        self._send_bytes(200, document.encode("utf-8"), "text/html; charset=utf-8")

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

    def _handle_search(self, data: Dict[str, Any]) -> None:
        query = str(data.get("query") or data.get("q") or "").strip()
        if not query:
            self._send_error_json(400, "El campo 'query' es obligatorio")
            return
        matches = search_models(self.catalog, query)
        self._send_json(
            200,
            {
                "success": True,
                "query": query,
                "count": len(matches),
                "results": payloads(matches, include_source=False),
            },
        )

    # -- static assets ----------------------------------------------------

    def _serve_static(self, path: str) -> None:
        asset = resolve_asset(path)
        if asset is None:
            self._send_html_error(404, "No encontrado", f'No existe el recurso "{path}".')
            return
        self._serve_asset(asset)

    def _serve_asset(self, asset: Path) -> None:
        try:
            body = asset.read_bytes()
        except OSError as exc:
            LOGGER.warning("Cannot read asset %s: %s", asset, exc)
            self._send_html_error(404, "No encontrado", f'No se pudo leer "{asset.name}".')
            return
        self._send_bytes(200, body, content_type_for(asset))

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
        self._send_bytes(status, document.encode("utf-8"), "text/html; charset=utf-8")


# --------------------------------------------------------------------------
# Path and MIME helpers
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


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
