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
``GET  /api/health``                liveness probe, counters, metrics and jobs
``GET  /api/openapi.json``          OpenAPI 3.1 description of this API
``GET  /docs``                      small viewer for ``/api/openapi.json``
``GET  /api/categories``            categories with their model counts
``GET  /api/models``                model list (``?category=``, ``?q=``, ``?tags=``,
                                    ``?limit=``, ``?offset=``)
``GET  /api/models/<model_id>``     everything known about one model
``GET  /api/models/<id>/source``    raw source (``?download=1``, ``?diff=<ref>``)
``GET  /api/models/<id>/versions``  git commits that touched the source of a model
``GET  /api/search``                ``?q=`` search with facets (autocomplete friendly)
``GET  /api/playground/<model_id>`` playground specification of one model
``GET  /api/playground/jobs``       recent asynchronous playground jobs
``GET  /api/playground/jobs/<id>``  state and results of one job
``GET  /api/models/<id>/invoke``    methods that may be called on one model
``GET  /api/sessions``              open model sessions
``GET  /api/sessions/<id>``         state of one session
``GET  /api/notebook/<model_id>``   generated ``.ipynb`` download (``?parameters=``)
``POST /api/playground``            ``{"model_id": ..., "parameters": {...}}`` -> run the model
``POST /api/playground/jobs``       one run or a list of runs -> ``202`` and a ``job_id``
``POST /api/models/<id>/invoke``    ``{"method": ..., "arguments": {...}}`` -> call a method
``POST /api/models/<id>/sessions``  build the model once and keep it alive
``POST /api/sessions/<id>/invoke``  call a method on the live instance
``DELETE /api/sessions/<id>``       close a session
``POST /api/notebook``              ``{"model_id": ..., "parameters": {...}}`` -> notebook JSON
``POST /api/search``                deprecated alias of ``GET /api/search``

Any other path is resolved inside ``models/`` so that the HTML assets shipped
with the project (``index.html``, ``preview.html``, ``favicon.svg``) are served
as static files.

Configuration
-------------
``LM_HOST``                interface to bind (default ``127.0.0.1``)
``LM_PORT``                TCP port (default ``8000``)
``LM_API_TOKEN``           require ``Authorization: Bearer <token>`` on ``/api``
``LM_CORS_ORIGINS``        comma separated allowed origins (``*`` allows any)
``LM_RATE_LIMIT``          requests per minute and client (``0`` disables it)
``LM_REQUEST_TIMEOUT``     seconds a client may take to send its request
``LM_CATALOG_TTL``         seconds between checks of the model sources (``0`` = never)
``LM_API_CACHE_CONTROL``   ``Cache-Control`` of the cacheable JSON endpoints
``LM_PLAYGROUND``          set it to ``0``/``false`` to disable ``POST /api/playground``
``LM_PLAYGROUND_TIMEOUT``  seconds a playground run may take (default ``60``)
``LM_PLAYGROUND_CPU``      CPU seconds the child process may consume
``LM_PLAYGROUND_MEMORY``   address space limit of the child process, in MiB
``LM_PLAYGROUND_FILES``    open file descriptors the child process may hold
``LM_PLAYGROUND_BLOCKED_MODULES``
                           comma separated modules the catalog code may not import
                           (the guard never applies to third party libraries; an empty
                           value disables it)
``LM_PLAYGROUND_MAX_BATCH``
                           maximum number of runs accepted by one job
``LM_JOBS_WORKERS``        worker threads used by the asynchronous jobs
``LM_JOBS_RETENTION``      seconds a finished job is kept in memory
``LM_MAX_JOBS``            maximum number of jobs kept in memory
``LM_INVOKE``              set it to ``0``/``false`` to disable the method invocation API
``LM_INVOKE_TIMEOUT``      seconds a method call may take (default: ``LM_PLAYGROUND_TIMEOUT``)
``LM_INVOKE_MAX_CONCURRENCY``
                           method calls that may run at the same time
``LM_INVOKE_METHODS``      explicit allowlist of method names (replaces the policy)
``LM_INVOKE_BLOCKED_METHODS``
                           extra method names to block
``LM_INVOKE_ALLOW_SIDE_EFFECTS``
                           allow ``save``/``load``/``plot``/... methods too
``LM_SESSIONS``            set it to ``0``/``false`` to disable the sessions
``LM_MAX_SESSIONS``        live model instances kept in memory
``LM_SESSION_TTL``         seconds an idle session stays alive
``LM_SESSION_TIMEOUT``     seconds a call on a session may take

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
import difflib
import hashlib
import hmac
import html
import json
import logging
import math
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import notebook_generator
from model_catalog import (
    DEFAULT_PACKAGE_DIR,
    REPO_ROOT,
    build_catalog,
    category_index,
    category_meta,
    facet_counts,
    filter_by_category,
    filter_by_tags,
    find_model,
    guess_widget,
    normalize_tags,
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
_SAFE_REF = re.compile(r"^[A-Za-z0-9._/@^~-]+$")


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean environment variable (``0``/``false``/``no`` are false)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


#: Largest request body accepted by the JSON endpoints (1 MiB).
MAX_BODY_BYTES = 1_048_576
#: Chunk size used while reading a request body, so a lying ``Content-Length``
#: cannot make the server allocate or block on a single unbounded read.
BODY_CHUNK_BYTES = 65_536
#: Largest page returned by ``GET /api/models``.
MAX_PAGE_SIZE = 200
#: ``Cache-Control`` used for the static assets under ``models/``.
STATIC_CACHE_CONTROL = "public, max-age=300, must-revalidate"
#: ``Cache-Control`` used for the JSON endpoints derived from the catalog.
API_CACHE_CONTROL = os.environ.get("LM_API_CACHE_CONTROL") or "public, max-age=60, must-revalidate"
#: Version exposed through ``X-API-Version`` and the OpenAPI document.
API_VERSION = "1.0"
#: Seconds a client may take to send a request before it is dropped.
REQUEST_TIMEOUT = float(os.environ.get("LM_REQUEST_TIMEOUT") or 30)
#: Seconds a ``POST /api/playground`` run may take.
PLAYGROUND_TIMEOUT = float(os.environ.get("LM_PLAYGROUND_TIMEOUT") or 60)
#: CPU seconds the playground child may consume before the kernel kills it.
PLAYGROUND_CPU_SECONDS = int(os.environ.get("LM_PLAYGROUND_CPU") or 30)
#: Address space of the playground child, in MiB (``0`` disables the limit).
PLAYGROUND_MEMORY_MB = int(os.environ.get("LM_PLAYGROUND_MEMORY") or 2048)
#: Open file descriptors the playground child may hold (``0`` disables the limit).
PLAYGROUND_MAX_FILES = int(os.environ.get("LM_PLAYGROUND_FILES") or 128)
#: Largest file the playground child may write, in MiB (``0`` disables the limit).
PLAYGROUND_MAX_FILE_MB = int(os.environ.get("LM_PLAYGROUND_FILE_SIZE") or 32)
#: Modules the catalog code may not import.  ``LM_PLAYGROUND_BLOCKED_MODULES``
#: replaces the list; an empty value disables the guard entirely.  The guard only
#: inspects the imports performed by the catalog code (the model module, its
#: package and the files inside the repository), never the ones performed inside
#: third party libraries, which need ``subprocess``/``urllib`` internally.
_DEFAULT_BLOCKED_MODULES = (
    "ftplib",
    "pty",
    "smtplib",
    "subprocess",
    "telnetlib",
    "webbrowser",
    "xmlrpc",
)
if "LM_PLAYGROUND_BLOCKED_MODULES" in os.environ:
    PLAYGROUND_BLOCKED_MODULES = frozenset(
        name.strip()
        for name in os.environ["LM_PLAYGROUND_BLOCKED_MODULES"].replace(",", " ").split()
        if name.strip()
    )
else:
    PLAYGROUND_BLOCKED_MODULES = frozenset(_DEFAULT_BLOCKED_MODULES)
#: Maximum number of runs accepted by a single playground job.
PLAYGROUND_MAX_BATCH = int(os.environ.get("LM_PLAYGROUND_MAX_BATCH") or 10)
#: Worker threads used to run playground jobs in the background.
JOBS_WORKERS = max(1, int(os.environ.get("LM_JOBS_WORKERS") or 2))
#: Seconds a finished job is kept in memory before it is pruned.
JOBS_RETENTION = float(os.environ.get("LM_JOBS_RETENTION") or 900)
#: Maximum number of jobs kept in memory.
JOBS_MAX = int(os.environ.get("LM_MAX_JOBS") or 200)
#: Seconds between checks of the model sources (``0`` disables the hot reload).
CATALOG_TTL = float(os.environ.get("LM_CATALOG_TTL") or 2)
#: Allowed CORS origins (``LM_CORS_ORIGINS``, comma separated; ``*`` allows any).
CORS_ORIGINS: Tuple[str, ...] = tuple(
    origin.strip().rstrip("/")
    for origin in (os.environ.get("LM_CORS_ORIGINS") or "").split(",")
    if origin.strip()
)
#: Optional bearer token that protects every ``/api`` endpoint (``LM_API_TOKEN``).
API_TOKEN = (os.environ.get("LM_API_TOKEN") or "").strip()
#: Requests per minute and client (``LM_RATE_LIMIT``, ``0`` disables the limiter).
RATE_LIMIT = max(0, int(os.environ.get("LM_RATE_LIMIT") or 0))
#: Date advertised through ``Sunset`` for the deprecated ``POST /api/search``.
SEARCH_ALIAS_SUNSET = (os.environ.get("LM_SEARCH_ALIAS_SUNSET") or "2026-12-31").strip()
#: Whether ``POST /api/models/<id>/invoke`` may call model methods (``LM_INVOKE``).
INVOKE_ENABLED = _env_flag("LM_INVOKE", True)
#: Seconds a single method call may take (``LM_INVOKE_TIMEOUT``).
INVOKE_TIMEOUT = float(os.environ.get("LM_INVOKE_TIMEOUT") or PLAYGROUND_TIMEOUT)
#: Number of method calls that may run at the same time (``LM_INVOKE_MAX_CONCURRENCY``).
INVOKE_MAX_CONCURRENCY = max(1, int(os.environ.get("LM_INVOKE_MAX_CONCURRENCY") or 4))
#: Method name fragments that are never invoked (filesystem/side effect methods).
INVOKE_SIDE_EFFECT_METHODS = (
    "save",
    "load",
    "plot",
    "write",
    "download",
    "delete",
    "reset",
    "remove",
    "close",
    "shutdown",
)
#: Explicit allowlist of method names (``LM_INVOKE_METHODS``); empty = automatic.
INVOKE_ALLOWED_METHODS = frozenset(
    name.strip()
    for name in (os.environ.get("LM_INVOKE_METHODS") or "").replace(",", " ").split()
    if name.strip()
)
#: Extra method names to block on top of :data:`INVOKE_SIDE_EFFECT_METHODS`.
INVOKE_BLOCKED_METHODS = frozenset(
    name.strip()
    for name in (os.environ.get("LM_INVOKE_BLOCKED_METHODS") or "").replace(",", " ").split()
    if name.strip()
)
#: Allow the side effect methods too (``LM_INVOKE_ALLOW_SIDE_EFFECTS``).
INVOKE_ALLOW_SIDE_EFFECTS = _env_flag("LM_INVOKE_ALLOW_SIDE_EFFECTS", False)
#: Whether persistent sessions are available (``LM_SESSIONS``).
SESSIONS_ENABLED = _env_flag("LM_SESSIONS", True)
#: Number of live model instances kept by the session store (``LM_MAX_SESSIONS``).
MAX_SESSIONS = max(1, int(os.environ.get("LM_MAX_SESSIONS") or 4))
#: Seconds an idle session is kept alive (``LM_SESSION_TTL``).
SESSION_TTL = float(os.environ.get("LM_SESSION_TTL") or 900)
#: Seconds a call on a session may take (``LM_SESSION_TIMEOUT``).
SESSION_TIMEOUT = float(os.environ.get("LM_SESSION_TIMEOUT") or INVOKE_TIMEOUT)
#: Marker used to separate the playground result from anything the model prints.
_PLAYGROUND_MARKER = "__LIKELIHOOD_PLAYGROUND__"


#: Whether ``POST /api/playground`` may construct models (``--no-playground``).
PLAYGROUND_ENABLED = _env_flag("LM_PLAYGROUND", True)


class CatalogStore:
    """Keep the introspected catalog in sync with the model sources.

    The catalog used to be built once at import time, so editing a model forced a
    restart.  The store remembers the ``mtime`` of every source file and rebuilds
    the catalog when one of them changes, at most once every ``ttl`` seconds.
    """

    def __init__(self, package_dir: Optional[Path] = None, ttl: float = CATALOG_TTL) -> None:
        self.package_dir = Path(package_dir) if package_dir else DEFAULT_PACKAGE_DIR
        self.ttl = ttl
        self._lock = threading.Lock()
        self._catalog: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._mtimes: Dict[str, int] = {}
        self._checked_at = 0.0
        self.reloads = 0

    def load(self) -> List[Dict[str, Any]]:
        """Rebuild the catalog unconditionally and remember the source mtimes."""
        with self._lock:
            self._catalog = build_catalog(self.package_dir)
            self._by_id = {model["id"]: model for model in self._catalog}
            self._mtimes = self.source_mtimes()
            self._checked_at = time.monotonic()
            self.reloads += 1
            return self._catalog

    @property
    def catalog(self) -> List[Dict[str, Any]]:
        """The catalog currently being served."""
        return self._catalog

    @property
    def by_id(self) -> Dict[str, Dict[str, Any]]:
        """The catalog indexed by model id."""
        return self._by_id

    @property
    def fingerprint(self) -> str:
        """Short digest of the served catalog, used for the ``ETag`` responses."""
        digest = hashlib.sha1()
        digest.update(str(sorted(self._mtimes.items())).encode("utf-8"))
        for model in self._catalog:
            digest.update(f"{model['id']}:{model['signature']}".encode("utf-8"))
        return digest.hexdigest()[:16]

    def source_mtimes(self) -> Dict[str, int]:
        """``{path: mtime_ns}`` of every model source of the package."""
        mtimes: Dict[str, int] = {}
        if not self.package_dir.is_dir():
            return mtimes
        for path in sorted(self.package_dir.rglob("*.py")):
            try:
                mtimes[str(path)] = path.stat().st_mtime_ns
            except OSError:  # pragma: no cover - the file vanished mid-scan
                continue
        return mtimes

    def changed(self) -> bool:
        """True when a source file was added, removed or modified."""
        return self.source_mtimes() != self._mtimes

    def refresh(self, force: bool = False) -> bool:
        """Rebuild the catalog when the sources changed; ``True`` when it reloaded."""
        if self.ttl <= 0 and not force:
            return False
        now = time.monotonic()
        with self._lock:
            if not force and now - self._checked_at < self.ttl:
                return False
            self._checked_at = now
            if not force and self.source_mtimes() == self._mtimes:
                return False
        self.load()
        return True


# The catalog is rebuilt on demand (see :class:`CatalogStore`); ``CATALOG`` and
# ``MODELS_BY_ID`` stay as module level aliases for backwards compatibility.
CATALOG_STORE = CatalogStore()
CATALOG: List[Dict[str, Any]] = CATALOG_STORE.load()
MODELS_BY_ID: Dict[str, Dict[str, Any]] = CATALOG_STORE.by_id


class Metrics:
    """Per endpoint counters and latency percentiles behind ``GET /api/health``."""

    def __init__(self, samples: int = 512, max_endpoints: int = 200) -> None:
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self._samples = samples
        self._max_endpoints = max_endpoints
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.total_requests = 0
        self.total_errors = 0
        self.statuses: Dict[str, int] = {}

    def record(self, endpoint: str, status: int, duration: float) -> None:
        """Account one request of ``endpoint`` that finished with ``status``.

        Unknown labels beyond ``max_endpoints`` are folded into ``other`` so an
        arbitrary amount of distinct URLs cannot grow the counters forever.
        """
        with self._lock:
            if endpoint not in self.endpoints and len(self.endpoints) >= self._max_endpoints:
                endpoint = "other"
            entry = self.endpoints.setdefault(
                endpoint,
                {"count": 0, "errors": 0, "latency": deque(maxlen=self._samples), "statuses": {}},
            )
            entry["count"] += 1
            entry["statuses"][str(status)] = entry["statuses"].get(str(status), 0) + 1
            entry["latency"].append(duration * 1000.0)
            self.total_requests += 1
            self.statuses[str(status)] = self.statuses.get(str(status), 0) + 1
            if status >= 400:
                entry["errors"] += 1
                self.total_errors += 1

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
        return round(ordered[index], 2)

    def snapshot(self) -> Dict[str, Any]:
        """Immutable view of the counters, ready to be serialised as JSON."""
        with self._lock:
            endpoints = {
                name: {
                    "count": entry["count"],
                    "errors": entry["errors"],
                    "statuses": dict(entry["statuses"]),
                    "latency_ms": {
                        "p50": self._percentile(entry["latency"], 0.50),
                        "p95": self._percentile(entry["latency"], 0.95),
                        "max": round(max(entry["latency"]), 2) if entry["latency"] else 0.0,
                    },
                }
                for name, entry in sorted(self.endpoints.items())
            }
            return {
                "uptime_seconds": round(time.monotonic() - self.started, 3),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "statuses": dict(self.statuses),
                "endpoints": endpoints,
            }


METRICS = Metrics()


class RateLimiter:
    """Sliding window limiter with one bucket per client address."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = {}

    def hit(self, key: str) -> Optional[float]:
        """Register a request; return the seconds to wait when over the limit."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return max(0.0, self.window - (now - bucket[0]))
            bucket.append(now)
            return None

    def reset(self) -> None:
        """Forget every bucket (used by the tests)."""
        with self._lock:
            self._hits.clear()


RATE_LIMITER = RateLimiter(RATE_LIMIT)


def endpoint_label(method: str, path: str) -> str:
    """Group requests into a small set of metric buckets.

    ``/api/models/Arima`` and ``/api/models/HMM`` are reported together as
    ``GET /api/models/<model_id>`` so the counters stay readable.
    """
    clean = path.rstrip("/") or "/"
    patterns = (
        (re.compile(r"^/api/models/[^/]+/source$"), "/api/models/<model_id>/source"),
        (re.compile(r"^/api/models/[^/]+/versions$"), "/api/models/<model_id>/versions"),
        (re.compile(r"^/api/models/[^/]+/invoke$"), "/api/models/<model_id>/invoke"),
        (re.compile(r"^/api/models/[^/]+/sample$"), "/api/models/<model_id>/sample"),
        (re.compile(r"^/api/models/[^/]+/sessions$"), "/api/models/<model_id>/sessions"),
        (re.compile(r"^/api/models/[^/]+$"), "/api/models/<model_id>"),
        (re.compile(r"^/api/playground/jobs/[^/]+$"), "/api/playground/jobs/<job_id>"),
        (re.compile(r"^/api/playground/[^/]+$"), "/api/playground/<model_id>"),
        (re.compile(r"^/api/sessions/[^/]+/invoke$"), "/api/sessions/<session_id>/invoke"),
        (re.compile(r"^/api/sessions/[^/]+$"), "/api/sessions/<session_id>"),
        (re.compile(r"^/api/notebook/[^/]+$"), "/api/notebook/<model_id>"),
        (re.compile(r"^/preview/.*$"), "/preview/<model_id>"),
    )
    for pattern, replacement in patterns:
        if pattern.match(clean):
            return f"{method} {replacement}"
    return f"{method} {clean}"


def model_brief(model: Dict[str, Any]) -> Dict[str, Any]:
    """Small projection used by listings and search results."""
    return {
        "id": model["id"],
        "title": model["title"],
        "icon": model["icon"],
        "category": model["category"],
        "category_label": model["category_label"],
        "short_description": model["short_description"],
        "tags": model["tags"],
        "module": model["module"],
        "signature": model["signature"],
        "parameter_count": len(model["parameters"]),
        "method_count": len(model["methods"]),
        "line_count": model["line_count"],
    }


def model_payload(model: Dict[str, Any], include_source: bool = True) -> Dict[str, Any]:
    """Full JSON representation of ``model``."""
    payload = dict(model)
    if not include_source:
        payload.pop("source", None)
    payload["preview_url"] = f"/preview/{model['id']}"
    payload["notebook_url"] = f"/api/notebook/{model['id']}"
    payload["source_url"] = f"/api/models/{model['id']}/source"
    payload["source_download_url"] = f"/api/models/{model['id']}/source?download=1"
    payload["source_versions_url"] = f"/api/models/{model['id']}/versions"
    payload["playground_url"] = f"/api/playground/{model['id']}"
    payload["playground"] = playground_spec(model)
    payload["invoke_url"] = f"/api/models/{model['id']}/invoke"
    payload["sessions_url"] = f"/api/models/{model['id']}/sessions"
    payload["invokable_methods"] = [method["name"] for method in invokable_methods(model)]
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


#: Argument names that usually carry a matrix (``samples × features``).
MATRIX_ARGUMENTS = frozenset(
    {
        "x",
        "xs",
        "data",
        "features",
        "inputs",
        "samples",
        "predictors",
        "covariates",
        "design",
        "matrix",
        "train_data",
        "test_data",
        "x_train",
        "x_test",
    }
)
#: Argument names that usually carry a categorical or one dimensional sequence.
SEQUENCE_ARGUMENTS = frozenset(
    {
        "y",
        "ys",
        "target",
        "targets",
        "labels",
        "classes",
        "sequence",
        "sequences",
        "observations",
        "obs",
        "states",
        "state_sequence",
        "true_states",
        "categories",
        "indices",
        "indexes",
        "values",
        "series",
        "returns",
        "prices",
        "signal",
        "signals",
    }
)
#: Constructor parameters that describe how many categories/states exist.
COUNT_PARAMETERS = (
    "n_observations",
    "n_states",
    "n_classes",
    "n_clusters",
    "n_categories",
    "num_observations",
    "num_states",
    "num_classes",
    "n_components",
)
#: Constructor parameters that describe the length of a sequence or a window.
LENGTH_PARAMETERS = (
    "window",
    "window_size",
    "lags",
    "lag",
    "seq_len",
    "sequence_length",
    "length",
    "size",
    "n_steps",
    "steps",
    "horizon",
    "periods",
    "lookback",
)
#: Constructor parameters that describe how many columns/features exist.
WIDTH_PARAMETERS = (
    "n_features",
    "n_inputs",
    "input_dim",
    "features",
    "dim",
    "n_columns",
    "n_variables",
    "embedding_dim",
    "hidden_dim",
)
#: Argument names that carry a shape tuple such as ``(None, 4)``.
SHAPE_ARGUMENTS = frozenset({"shape", "input_shape", "output_shape", "dims", "dimensions"})
#: Argument names that carry a list of sequences instead of a single one.
MULTI_SEQUENCE_ARGUMENTS = frozenset({"sequences", "batch", "batches"})
#: Unannotated arguments that are usually integers.
SCALAR_INT_ARGUMENTS = frozenset(
    {
        "epochs",
        "num_epochs",
        "n_epochs",
        "num_episodes",
        "num_steps",
        "n_steps",
        "iterations",
        "n_iterations",
        "nwalkers",
        "n_walkers",
        "batch_size",
        "top_k",
    }
)
#: Unannotated arguments that are usually fractions.
SCALAR_FLOAT_ARGUMENTS = frozenset(
    {
        "epsilon",
        "gamma",
        "alpha",
        "sigma",
        "noise",
        "tol",
        "frac",
        "threshold_factor",
        "validation_split",
        "learning_rate",
    }
)
#: Unannotated arguments that are usually flags.
SCALAR_BOOL_ARGUMENTS = frozenset({"verbose", "mode", "use_scaler", "normalize", "highlight"})
#: Sensible scalar samples for well known argument names (used when there is no
#: default): a class with one unit or one class is a poor starting point.
SCALAR_SAMPLE_VALUES: Dict[str, Any] = {
    "input_shape_parm": 4,
    "output_shape_parm": 1,
    "input_shape": 4,
    "output_shape": 1,
    "input_dim": 4,
    "n_features": 3,
    "n_inputs": 3,
    "units": 8,
    "num_neurons": 8,
    "hidden_dim": 8,
    "num_classes": 3,
    "n_classes": 3,
    "n_observations": 3,
    "n_states": 2,
    "n_clusters": 2,
    "n_models": 5,
    "num_layers": 2,
    "depth": 2,
    "n_estimators": 10,
    "maxlen": 100,
    "epochs": 1,
    "num_epochs": 1,
    "n_epochs": 1,
    "batch_size": 8,
    "top_k": 3,
    "patience": 2,
    "vae_units": 2,
    "lora_rank": 4,
    "window": 3,
    "lags": 2,
    "n_steps": 1,
    "num_steps": 1,
    "horizon": 1,
    "tol": 0.001,
    "dropout": 0.1,
    "l2_reg": 0.0,
    "noise": 0.0,
    "p": 1.0,
    "q": 0.0,
}
#: Type hints that mean "a collection", even without a known argument name.
CONTAINER_HINTS = ("ndarray", "array", "list", "tuple", "set", "series", "sequence")
#: Size of the generated payloads, so a sample never becomes a huge request.
DUMMY_ROWS = 3
DUMMY_COLUMNS = 2
DUMMY_ROWS_RANGE = (2, 10)
DUMMY_COLUMNS_RANGE = (1, 8)
DUMMY_CATEGORIES_RANGE = (2, 10)


def _int_parameter(parameters: Dict[str, Any], names: Sequence[str], default: int) -> int:
    """First integer recorded in ``parameters`` under one of ``names``."""
    for name, value in parameters.items():
        if name.lower() not in names or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _clamp(value: int, bounds: Tuple[int, int]) -> int:
    """Keep ``value`` inside the inclusive ``bounds``."""
    return max(bounds[0], min(bounds[1], value))


def dummy_json_value(name: str, type_hint: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """JSON placeholder for an argument called ``name`` typed ``type_hint``.

    The payload is derived from the constructor parameters (number of states,
    window size, features...) so ``fit``/``predict``/``forward`` can be tried
    from the invoke tab without writing JSON by hand.  It is only a sample: it
    never evaluates code and the caller is expected to edit it.
    """
    parameters = parameters or {}
    lowered = (name or "").lower()
    type_lower = (type_hint or "").lower()
    base = lowered
    for suffix in ("_train", "_test", "_valid", "_validation", "_val", "_data"):
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    categories = _clamp(
        _int_parameter(parameters, COUNT_PARAMETERS, DUMMY_COLUMNS), DUMMY_CATEGORIES_RANGE
    )
    rows = _clamp(_int_parameter(parameters, LENGTH_PARAMETERS, DUMMY_ROWS), DUMMY_ROWS_RANGE)
    columns = _clamp(
        _int_parameter(parameters, WIDTH_PARAMETERS, DUMMY_COLUMNS), DUMMY_COLUMNS_RANGE
    )

    if "dataframe" in type_lower:
        return [
            {f"col_{index + 1}": float(row + index + 1) for index in range(columns)}
            for row in range(rows)
        ]
    if "dict" in type_lower or "mapping" in type_lower:
        return {f"clave_{index + 1}": float(index + 1) for index in range(min(columns, 3))}
    if "str" in type_lower:
        if any(hint in type_lower for hint in ("list", "sequence", "array")):
            return [f"texto_{index + 1}" for index in range(min(rows, DUMMY_ROWS))]
        return "example"
    if base in SHAPE_ARGUMENTS or lowered in SHAPE_ARGUMENTS:
        return [int(index + 1) for index in range(min(rows, DUMMY_ROWS))]
    if base in MULTI_SEQUENCE_ARGUMENTS:
        return [[int(index % categories) for index in range(rows)] for _ in range(2)]
    if base in MATRIX_ARGUMENTS or lowered in MATRIX_ARGUMENTS:
        return [[float(row + column + 1) for column in range(columns)] for row in range(rows)]
    if base in SEQUENCE_ARGUMENTS or lowered in SEQUENCE_ARGUMENTS:
        return [int(index % categories) for index in range(rows)]
    if any(hint in type_lower for hint in CONTAINER_HINTS):
        return [float(index + 1) for index in range(rows)]
    if lowered.endswith(("_path", "_file", "_folder", "_dir")):
        return "model"
    if base in SCALAR_BOOL_ARGUMENTS:
        return False
    if base in SCALAR_FLOAT_ARGUMENTS:
        return 0.1
    if (
        base in SCALAR_INT_ARGUMENTS
        or base in LENGTH_PARAMETERS
        or base in WIDTH_PARAMETERS
        or base in COUNT_PARAMETERS
    ):
        return max(1, rows)
    return None


def _sample_value(parameter: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Any:
    """JSON friendly sample of ``parameter``, generated when it is needed.

    The catalog records Python source expressions (``np.array([])``, ``[]``),
    which are not JSON values, and the parameters of a method do not record a
    sample at all.  The value prefers the default (what the model really uses),
    then a name aware scalar, then the catalog sample, and finally a small
    payload shaped from the constructor ``parameters``.
    """
    default = _default_value(parameter)
    if default not in (None, [], {}, (), ""):
        return default
    name = parameter.get("name") or ""
    kind = parameter.get("widget") or guess_widget(parameter.get("type") or "")
    if kind in ("int", "float") and name in SCALAR_SAMPLE_VALUES:
        return SCALAR_SAMPLE_VALUES[name]
    literal = _literal_for_kind(parameter, parameter.get("sample"), fallback=False)
    if literal not in (None, [], {}, (), ""):
        return literal
    if kind in ("raw", "text"):
        return dummy_json_value(name, parameter.get("type") or "", parameters)
    if kind == "int":
        return 1
    if kind == "float":
        return 0.1
    if kind == "bool":
        return False
    if kind == "str":
        return dummy_json_value(name, "str", parameters) or "example"
    return None


def _dict_with_integer_keys(type_hint: str) -> bool:
    """True for ``Dict[int, ...]``: JSON object keys are always strings."""
    lowered = (type_hint or "").lower()
    return bool(re.search(r"(?:dict|mapping)\s*\[\s*(?:int|integer|np\.int\d*)", lowered))


def parameter_is_jsonable(parameter: Dict[str, Any]) -> bool:
    """Whether the constructor parameter can be sent as JSON to the child.

    ``env``/``model`` style parameters receive Python objects, so the model can
    only be built from Python (or from a session created in Python); the API
    says so instead of failing later with a confusing missing-field error.
    """
    if parameter.get("variadic"):
        return False
    type_hint = parameter.get("type") or ""
    kind = parameter.get("widget") or guess_widget(type_hint)
    if kind in ("int", "float", "bool", "str", "text"):
        return True
    if _dict_with_integer_keys(type_hint):
        return False
    if any(hint in type_hint.lower() for hint in CONTAINER_HINTS):
        return True
    return dummy_json_value(parameter.get("name") or "", type_hint) is not None


def unsupported_parameters(model: Dict[str, Any]) -> List[Dict[str, str]]:
    """Required constructor parameters that cannot be expressed as JSON."""
    blocked: List[Dict[str, str]] = []
    for parameter in model.get("parameters") or []:
        if parameter.get("variadic") or not parameter.get("required"):
            continue
        if parameter_is_jsonable(parameter):
            continue
        type_hint = parameter.get("type") or "Any"
        if _dict_with_integer_keys(type_hint):
            reason = "necesita un dict con claves enteras (JSON solo admite texto)"
        elif type_hint in ("Any", ""):
            reason = "necesita un objeto Python"
        else:
            reason = f"necesita una instancia de {type_hint}"
        blocked.append({"name": parameter.get("name") or "", "type": type_hint, "reason": reason})
    return blocked


def _missing_parameters_error(
    fields: Dict[str, Dict[str, Any]], missing: List[str]
) -> ValidationError:
    """Explain which required parameters are missing (and which need Python)."""
    blocked = [name for name in missing if not parameter_is_jsonable(fields[name])]
    if blocked:
        names = ", ".join(f"'{name}'" for name in blocked)
        return ValidationError(
            "No se puede construir el modelo desde la API: "
            f"{names} necesita(n) objetos Python que no se pueden enviar como JSON",
            missing=missing,
            unsupported=blocked,
        )
    return ValidationError(f"Faltan parámetros obligatorios: {', '.join(missing)}", missing=missing)


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
    blocked = unsupported_parameters(model)
    return {
        "model_id": model["id"],
        "model_name": model["name"],
        "import_statement": model["import_statement"],
        "endpoint": "/api/playground",
        "jobs_endpoint": "/api/playground/jobs",
        "buildable": not blocked,
        "unsupported": blocked,
        "fields": playground_fields(model),
    }


def constructor_samples(model: Dict[str, Any]) -> Dict[str, Any]:
    """Constructor values used as the context of the generated samples.

    Every parameter takes its default; required ones fall back to the generated
    sample (``Arima.datapoints`` becomes a small matrix, for instance).
    """
    values: Dict[str, Any] = {}
    for field in playground_fields(model):
        value = field["default"]
        if value is None and field["required"]:
            value = field["sample"]
        if value is not None:
            values[field["name"]] = value
    return values


_BOOL_TRUE = {"true", "1", "yes", "y", "si", "sí", "on"}
_BOOL_FALSE = {"false", "0", "no", "n", "off"}
#: Type hints that expect a JSON collection (list or object).
CONTAINER_ARGUMENT_HINTS = (
    "ndarray",
    "tensor",
    "array",
    "list",
    "tuple",
    "sequence",
    "series",
    "dataframe",
    "set",
    "dict",
    "mapping",
)
#: Hints that cannot be an empty list (a tensor needs at least one value).
NON_EMPTY_HINTS = ("ndarray", "tensor", "array", "series", "dataframe")


def _expects_container(type_hint: str) -> bool:
    """True when the annotation describes a JSON collection."""
    lowered = (type_hint or "").lower()
    scalar = any(hint in lowered for hint in ("str", "int", "float", "bool"))
    if scalar and ("union" in lowered or "|" in lowered):
        # ``Union[str, pd.Series]`` still accepts a plain string.
        return False
    return any(hint in lowered for hint in CONTAINER_ARGUMENT_HINTS)


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
    hints = str(field.get("type") or "").lower()
    if _expects_container(hints):
        if not isinstance(value, (list, dict)):
            raise ValueError(f"se esperaba una lista o matriz JSON para '{name}': {value!r}")
        if isinstance(value, list) and not value and any(hint in hints for hint in NON_EMPTY_HINTS):
            raise ValueError(f"'{name}' no puede ser una lista vacía")
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


#: Common body of every child interpreter: resource limits and the import guard.
#: Each bootstrap below appends its own entry point to this prelude.
#:
#: The guard is *frame aware*: it only rejects the imports performed by the
#: catalog code itself (the model module, its package and anything inside the
#: repository), so a library that needs ``subprocess`` or ``urllib`` internally
#: (matplotlib, torch, tensorflow, ...) keeps working.
_SANDBOX_PRELUDE = r"""
import builtins
import importlib
import json
import math
import os
import sys
import traceback

payload = json.loads(sys.stdin.readline())
marker = payload["marker"]


def _emit(data):
    sys.stdout.write("\n" + marker + json.dumps(data, default=str) + "\n")
    sys.stdout.flush()


def _limit(target, value):
    import resource

    if not value:
        return
    attribute = getattr(resource, target, None)
    if attribute is None:
        return
    try:
        soft, hard = resource.getrlimit(attribute)
        ceiling = value if hard in (-1, resource.RLIM_INFINITY) else min(value, hard)
        resource.setrlimit(attribute, (ceiling, hard))
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        pass


limits = payload.get("limits") or {}
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None
if resource is not None:
    _limit("RLIMIT_CPU", limits.get("cpu"))
    _limit("RLIMIT_AS", limits.get("memory"))
    _limit("RLIMIT_DATA", limits.get("memory"))
    _limit("RLIMIT_NOFILE", limits.get("files"))
    _limit("RLIMIT_FSIZE", limits.get("file_size"))

blocked = frozenset(payload.get("blocked") or ())
if blocked:
    original_import = builtins.__import__
    package = payload.get("package") or ""
    project_root = os.path.abspath(payload.get("project_root") or "")

    def _catalog_frame(frame):
        # True when the frame runs catalog code, not a library.
        name = frame.f_globals.get("__name__") or ""
        if name == "__main__" or name == package or (
            package and name.startswith(package + ".")
        ):
            return True
        filename = frame.f_globals.get("__file__") or ""
        if not filename or not project_root:
            return False
        try:
            return os.path.abspath(filename).startswith(project_root)
        except (OSError, ValueError):  # pragma: no cover - exotic paths
            return False

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level == 0 and root in blocked:
            try:
                caller = sys._getframe(1)
            except (AttributeError, ValueError):  # pragma: no cover - no frames
                caller = None
            if caller is None or _catalog_frame(caller):
                raise ImportError(
                    "El playground no permite importar %r desde el código del catálogo; "
                    "ajusta LM_PLAYGROUND_BLOCKED_MODULES si una librería lo necesita"
                    % name
                )
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import

types = payload.get("types") or {}
argument_types = payload.get("argument_types") or {}


def _convert(name, value, hints):
    # JSON has no arrays: turn lists into numpy/pandas objects when the catalog
    # says the argument is a container of that kind, and fall back otherwise.
    hint = (hints.get(name) or "").lower()
    if not isinstance(value, list) or not hint:
        return value
    if "ndarray" in hint or "tensor" in hint or hint.rstrip("]").endswith("array"):
        try:
            import numpy

            return numpy.asarray(value)
        except Exception:  # pragma: no cover - numpy missing
            return value
    if "dataframe" in hint or "series" in hint:
        try:
            import pandas

            return pandas.DataFrame(value) if "dataframe" in hint else pandas.Series(value)
        except Exception:  # pragma: no cover - pandas missing
            return value
    return value


def _convert_all(values, hints=None):
    hints = types if hints is None else hints
    return {key: _convert(key, value, hints) for key, value in (values or {}).items()}


def _safe(value, depth=0):
    # JSON friendly projection of a Python value returned by a model method.
    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 100000:
            return value[:100000]
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in list(value)[:1000]]
    if isinstance(value, dict):
        return {str(key): _safe(item, depth + 1) for key, item in list(value.items())[:1000]}
    for attribute in ("tolist", "to_dict", "item"):
        converter = getattr(value, attribute, None)
        if callable(converter):
            try:
                return _safe(converter(), depth + 1)
            except Exception:  # pragma: no cover - exotic objects
                break
    return None


def _describe(instance):
    try:
        attributes = sorted(name for name in vars(instance) if not name.startswith("_"))[:25]
    except TypeError:
        attributes = []
    return {
        "type": type(instance).__name__,
        "module": type(instance).__module__,
        "repr": repr(instance)[:2000],
        "attributes": attributes,
    }


def _build():
    module = importlib.import_module(payload["module"])
    target = getattr(module, payload["name"])
    return target(**_convert_all(payload.get("parameters")))


def _input_shape(value):
    # Shape hint ``(None, n_features)`` derived from the JSON value of an input.
    if not isinstance(value, (list, tuple)) or not value:
        return None
    first = value[0]
    if isinstance(first, (list, tuple)):
        return (None, len(first))
    return (None,)


def _prepare(instance, method_name, arguments):
    # Keras models may not set ``built`` from their own ``build`` override, so a
    # private marker keeps the instance prepared across session calls (rebuilding
    # it would throw away the trained weights of the classifier).
    if method_name == "build" or not hasattr(instance, "built"):
        return
    if getattr(instance, "built", False) or getattr(instance, "_lm_prepared", False):
        return
    build = getattr(instance, "build", None)
    if not callable(build):
        return
    shape = None
    for value in (arguments or {}).values():
        shape = _input_shape(value)
        if shape is not None:
            break
    if shape is None:
        shape = (None, 1)
    try:
        build(shape)
    except BaseException:
        # The real call reports the original error with a clearer message.
        return
    try:
        instance._lm_prepared = True
    except BaseException:
        pass
"""


#: Builds the model and describes it (``POST /api/playground``).  The payload
#: arrives through ``stdin`` and the result is written to ``stdout`` after a
#: marker line, which keeps any ``print`` of the model apart.
_PLAYGROUND_BOOTSTRAP = _SANDBOX_PRELUDE + r"""
_emit(_describe(_build()))
"""


#: Same isolation, but it also calls one method of the freshly built instance, so
#: every ``POST /api/models/<id>/invoke`` request starts from a clean object.
_INVOKE_BOOTSTRAP = _SANDBOX_PRELUDE + r"""
try:
    instance = _build()
    arguments = _convert_all(payload.get("arguments"), argument_types)
    _prepare(instance, payload["method"], arguments)
    method = getattr(instance, payload["method"])
    value = method(**arguments)
except BaseException as exc:
    import traceback as _traceback

    _emit(
        {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": _traceback.format_exc()[-4000:],
        }
    )
    sys.exit(3)

result = _describe(instance)
result["ok"] = True
result["result"] = _safe(value)
result["result_type"] = type(value).__name__
result["result_repr"] = repr(value)[:2000]
_emit(result)
"""


#: Long lived worker behind ``POST /api/models/<id>/sessions``: it builds the
#: model once and then answers one JSON line per method call, so ``fit`` followed
#: by ``predict`` keeps the state of the instance.
_SESSION_BOOTSTRAP = _SANDBOX_PRELUDE + r"""
try:
    instance = _build()
except BaseException as exc:
    import traceback as _traceback

    _emit(
        {
            "event": "error",
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": _traceback.format_exc()[-4000:],
        }
    )
    sys.exit(3)

ready = _describe(instance)
ready["event"] = "ready"
_emit(ready)

import time

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except ValueError as exc:
        _emit({"id": None, "ok": False, "error": "Petición inválida: %s" % exc})
        continue
    action = request.get("action")
    if action == "close":
        break
    if action == "ping":
        _emit({"id": request.get("id"), "ok": True, "result": "pong", "elapsed": 0.0})
        continue
    started = time.monotonic()
    try:
        hints = request.get("argument_types") or argument_types
        arguments = _convert_all(request.get("arguments"), hints)
        _prepare(instance, request["method"], arguments)
        method = getattr(instance, request["method"])
        value = method(**arguments)
    except BaseException as exc:
        import traceback as _traceback

        _emit(
            {
                "id": request.get("id"),
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": _traceback.format_exc()[-4000:],
                "elapsed": round(time.monotonic() - started, 3),
            }
        )
        continue
    _emit(
        {
            "id": request.get("id"),
            "ok": True,
            "result": _safe(value),
            "result_type": type(value).__name__,
            "result_repr": repr(value)[:2000],
            "elapsed": round(time.monotonic() - started, 3),
        }
    )
"""


class APIError(Exception):
    """Base class of every error that maps onto an HTTP status."""

    status = 500

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        self.headers = dict(headers or {})
        self.extra = extra


class ValidationError(APIError):
    """The request is well formed but its content is not usable (``400``)."""

    status = 400


class AuthenticationError(APIError):
    """The request lacks a valid API token (``401``)."""

    status = 401


class ModelNotFound(APIError):
    """The requested model id does not exist (``404``)."""

    status = 404


class RequestTimeout(APIError):
    """The client took too long to send the request (``408``)."""

    status = 408


class RequestBodyTooLarge(APIError, ValueError):
    """The request body exceeded :data:`MAX_BODY_BYTES` (``413``)."""

    status = 413


class UnsupportedMediaType(APIError):
    """The request body is not JSON (``415``)."""

    status = 415


class RateLimitExceeded(APIError):
    """Too many requests from the same client (``429``)."""

    status = 429


class PlaygroundError(APIError):
    """A playground run failed; ``status`` is the HTTP status to report."""

    def __init__(self, message: str, status: int = 422, detail: str = "") -> None:
        super().__init__(message, status=status, detail=detail)
        self.detail = detail


class InvocationError(PlaygroundError):
    """A model method call failed (``422``), timed out (``504``) or died (``503``)."""


def _last_error_line(stderr: str) -> str:
    """The final meaningful line of a traceback (usually the exception)."""
    for line in reversed([line.strip() for line in stderr.splitlines()]):
        if line:
            return line
    return ""


def playground_limits() -> Dict[str, int]:
    """``resource`` limits applied inside the playground child process.

    The values are absolute (bytes, seconds, file descriptors) and are only
    *lowered*, never raised: the child can never get more than the parent, so a
    leaked limit cannot weaken the server.
    """
    return {
        "cpu": PLAYGROUND_CPU_SECONDS,
        "memory": PLAYGROUND_MEMORY_MB * 1024 * 1024 if PLAYGROUND_MEMORY_MB else 0,
        "files": PLAYGROUND_MAX_FILES,
        "file_size": PLAYGROUND_MAX_FILE_MB * 1024 * 1024 if PLAYGROUND_MAX_FILE_MB else 0,
    }


def sandbox_payload(
    model: Dict[str, Any], types: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Configuration shared by every child interpreter (limits, guard, marker)."""
    return {
        "marker": _PLAYGROUND_MARKER,
        "limits": playground_limits(),
        "blocked": sorted(PLAYGROUND_BLOCKED_MODULES),
        # The import guard only applies to this package (and to anything that
        # lives inside the repository), never to third party libraries.
        "package": str(model["module"]).split(".")[0],
        "project_root": str(REPO_ROOT),
        "types": types or {},
    }


def child_environment() -> Dict[str, str]:
    """Environment of the child interpreters (same package, unbuffered pipes)."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), environment.get("PYTHONPATH", "")) if part
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _spawn(script: str) -> subprocess.Popen:
    """Start a child interpreter running ``script`` with piped stdin/stdout."""
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        env=child_environment(),
        text=True,
        bufsize=1,
    )


def _marker_payload(line: str) -> Optional[str]:
    """The JSON written after the marker in a child line, if any."""
    if line.startswith(_PLAYGROUND_MARKER):
        return line[len(_PLAYGROUND_MARKER) :]
    return None


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
            **sandbox_payload(model, constructor_types(model)),
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PLAYGROUND_BOOTSTRAP],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=child_environment(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PlaygroundError(
            f"La ejecución superó el límite de {timeout:g} segundos", status=504
        ) from exc
    elapsed = time.monotonic() - started

    stdout = completed.stdout or ""
    stderr = (completed.stderr or "").strip()
    result_line = ""
    for line in stdout.splitlines():
        payload_line = _marker_payload(line)
        if payload_line is not None:
            result_line = payload_line
    if not result_line:
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
    if result.get("ok") is False:
        raise PlaygroundError(
            friendly_invocation_error(result.get("error") or "No se pudo construir el modelo"),
            detail=(result.get("traceback") or stderr)[-4000:],
        )
    result.pop("ok", None)
    result["elapsed"] = round(elapsed, 3)
    result["stdout"] = stdout.split(_PLAYGROUND_MARKER)[0][-4000:].strip()
    result["stderr"] = stderr[-4000:]
    return result


def constructor_types(model: Dict[str, Any]) -> Dict[str, str]:
    """``{parameter: python type}`` of the constructor, for the child converter."""
    return {
        parameter["name"]: parameter.get("type") or ""
        for parameter in model.get("parameters") or []
        if not parameter.get("variadic")
    }


# ---------------------------------------------------------------------------
# Invocación de métodos (``POST /api/models/<id>/invoke`` y sesiones)
# ---------------------------------------------------------------------------


def method_fields(
    method: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Describe the arguments of ``method`` exactly like the playground does.

    ``parameters`` are the constructor values the model will be built with;
    they shape the generated samples (``n_states``, ``window``, ``n_features``).
    """
    fields: List[Dict[str, Any]] = []
    for parameter in method.get("parameters") or []:
        if parameter.get("variadic"):
            # ``*args`` / ``**kwargs`` cannot be expressed as a JSON object.
            continue
        fields.append(
            {
                "name": parameter["name"],
                "type": parameter.get("type") or "",
                "kind": parameter.get("widget") or guess_widget(parameter.get("type") or ""),
                "required": bool(parameter.get("required")),
                "default": _default_value(parameter),
                "sample": _sample_value(parameter, parameters),
                "description": parameter.get("description") or "",
            }
        )
    return fields


def method_allowed(name: str, method: Dict[str, Any]) -> bool:
    """Whether the invocation policy lets a client call ``method``.

    The rule is intentionally conservative: private helpers, documented-only
    entries (`` `fit ` `` coming from a docstring), properties and anything that
    looks like a filesystem side effect are rejected.  ``LM_INVOKE_METHODS``
    replaces the automatic decision with an explicit allowlist and
    ``LM_INVOKE_ALLOW_SIDE_EFFECTS=1`` re-enables the side effect methods.
    """
    if not name or not name.isidentifier() or name.startswith("_"):
        return False
    if name in INVOKE_BLOCKED_METHODS:
        return False
    if method.get("kind") not in ("method", "static"):
        return False
    if INVOKE_ALLOWED_METHODS:
        return name in INVOKE_ALLOWED_METHODS
    if not INVOKE_ALLOW_SIDE_EFFECTS and any(
        token in name.lower() for token in INVOKE_SIDE_EFFECT_METHODS
    ):
        return False
    if method.get("inherited") and not method.get("parameters"):
        return False
    return True


def invokable_methods(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Methods of ``model`` that the API is allowed to call."""
    return [
        method
        for method in model.get("methods") or []
        if method_allowed(method.get("name") or "", method)
    ]


def find_method(model: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """The catalog entry of ``name`` inside ``model``, if it exists."""
    for method in model.get("methods") or []:
        if method.get("name") == name:
            return method
    return None


def method_for_invocation(model: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The catalog entry of an invokable ``name`` (raises the API errors)."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError("El campo 'method' es obligatorio")
    method = find_method(model, cleaned)
    if method is None:
        raise ModelNotFound(f'El modelo "{model["id"]}" no declara el método "{cleaned}"')
    if not method_allowed(cleaned, method):
        raise APIError(
            f'El método "{cleaned}" no está permitido por la política de invocación',
            status=403,
            method=cleaned,
        )
    return method


def sample_arguments(
    model: Dict[str, Any], method: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """JSON placeholders for every argument of ``method``.

    ``parameters`` shape the payload exactly like the constructor that will be
    used when the method is invoked (states, window, features...).
    """
    context = constructor_samples(model) if parameters is None else parameters
    return {
        field["name"]: field["sample"]
        for field in method_fields(method, context)
        if field.get("sample") is not None
    }


def invoke_spec(model: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a client needs in order to call the methods of ``model``."""
    context = constructor_samples(model)
    blocked = unsupported_parameters(model)
    methods = [
        {
            "name": method["name"],
            "signature": method["signature"],
            "description": method.get("description") or "",
            "kind": method.get("kind") or "method",
            "fields": method_fields(method, context),
        }
        for method in invokable_methods(model)
    ]
    return {
        "model_id": model["id"],
        "model_name": model["name"],
        "import_statement": model["import_statement"],
        "endpoint": f"/api/models/{model['id']}/invoke",
        "sample_endpoint": f"/api/models/{model['id']}/sample",
        "sessions_url": f"/api/models/{model['id']}/sessions",
        "session_url_template": "/api/sessions/{session_id}",
        "session_invoke_url_template": "/api/sessions/{session_id}/invoke",
        "enabled": bool(INVOKE_ENABLED),
        "sessions_enabled": bool(SESSIONS_ENABLED),
        "buildable": not blocked,
        "unsupported": blocked,
        "timeout_seconds": INVOKE_TIMEOUT,
        "session_ttl_seconds": SESSION_TTL,
        "max_sessions": MAX_SESSIONS,
        "side_effects_blocked": not INVOKE_ALLOW_SIDE_EFFECTS,
        "methods": methods,
    }


def coerce_model_parameters(
    model: Dict[str, Any], raw: Dict[str, Any], require_all: bool = True
) -> Dict[str, Any]:
    """Validate the constructor arguments of ``model``."""
    fields = {field["name"]: field for field in playground_fields(model)}
    unknown = sorted(name for name in raw if name not in fields)
    if unknown:
        raise ValidationError(f"Parámetros desconocidos: {', '.join(unknown)}", unknown=unknown)
    values: Dict[str, Any] = {}
    problems: List[str] = []
    for name, value in raw.items():
        if value is None and not fields[name]["required"]:
            # ``None`` means "use the default" for optional parameters.
            continue
        try:
            values[name] = coerce_playground_value(fields[name], value)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValidationError("; ".join(problems), invalid=problems)
    if require_all:
        missing = [
            field["name"]
            for field in fields.values()
            if field["required"] and field["name"] not in values
        ]
        if missing:
            raise _missing_parameters_error(fields, missing)
    return values


def resolved_parameters(model: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Constructor values for a sample request: ``raw`` plus the catalog defaults."""
    provided = coerce_model_parameters(model, raw, require_all=False)
    values = {**constructor_samples(model), **provided}
    missing = [
        field["name"]
        for field in playground_fields(model)
        if field["required"] and field["name"] not in values
    ]
    if missing:
        raise _missing_parameters_error(
            {field["name"]: field for field in playground_fields(model)}, missing
        )
    return values


def coerce_arguments(fields: Sequence[Dict[str, Any]], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the arguments of a method call."""
    by_name = {field["name"]: field for field in fields}
    unknown = sorted(name for name in raw if name not in by_name)
    if unknown:
        raise ValidationError(f"Argumentos desconocidos: {', '.join(unknown)}", unknown=unknown)
    values: Dict[str, Any] = {}
    problems: List[str] = []
    for name, value in raw.items():
        if value is None and not by_name[name]["required"]:
            continue
        try:
            values[name] = coerce_playground_value(by_name[name], value)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValidationError("; ".join(problems), invalid=problems)
    missing = [
        field["name"] for field in fields if field["required"] and field["name"] not in values
    ]
    if missing:
        raise ValidationError(
            f"Faltan argumentos obligatorios: {', '.join(missing)}", missing=missing
        )
    return values


def prepare_invocation(
    model: Dict[str, Any], data: Dict[str, Any], with_parameters: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Validate an invoke payload and return ``(method, arguments, parameters)``."""
    method = method_for_invocation(model, str(data.get("method") or ""))
    raw_arguments = data.get("arguments", data.get("args", {}))
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ValidationError("El campo 'arguments' debe ser un objeto")
    arguments = coerce_arguments(method_fields(method), raw_arguments)
    parameters: Dict[str, Any] = {}
    if with_parameters:
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise ValidationError("El campo 'parameters' debe ser un objeto")
        parameters = coerce_model_parameters(model, raw_parameters)
    return method, arguments, parameters


_KERAS_POSITIONAL_ERROR = "Only input tensors may be passed as positional arguments"


def friendly_invocation_error(
    message: str,
    method: str = "",
    fields: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Append an actionable hint to the low level errors of the child process."""
    text = message or ""
    if _KERAS_POSITIONAL_ERROR in text:
        tensors = [
            f"'{field['name']}' ({field['type']})"
            for field in (fields or [])
            if _expects_container(str(field.get("type") or ""))
        ]
        hint = " Debe ser una matriz/lista JSON: " + ", ".join(tensors) + "." if tensors else ""
        return (
            f"{text} · Keras solo acepta tensores como argumento posicional: "
            "revisa que el valor de entrada sea una matriz/lista JSON (no un "
            f"número o un texto).{hint}"
        )
    if method == "call" and "NoneType' object is not callable" in text:
        return (
            f"{text} · El método call necesita que la instancia esté construida "
            "(sus capas internas siguen en None): llama primero a "
            "build_encoder_decoder o build sobre la misma instancia."
        )
    return text


def run_invoke(
    model: Dict[str, Any],
    method: Dict[str, Any],
    parameters: Dict[str, Any],
    arguments: Dict[str, Any],
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Build ``model`` and call ``method`` in a throw-away child interpreter.

    No state survives the call: two requests always start from a fresh object,
    which is what makes the endpoint safe to expose next to the catalog.
    """
    timeout = INVOKE_TIMEOUT if timeout is None else timeout
    payload = json.dumps(
        {
            "module": model["module"],
            "name": model["name"],
            "parameters": parameters,
            "method": method["name"],
            "arguments": arguments,
            **sandbox_payload(model, constructor_types(model)),
            "argument_types": {field["name"]: field["type"] for field in method_fields(method)},
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _INVOKE_BOOTSTRAP],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=child_environment(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise InvocationError(
            f"La llamada superó el límite de {timeout:g} segundos", status=504
        ) from exc
    elapsed = time.monotonic() - started

    stdout = completed.stdout or ""
    stderr = (completed.stderr or "").strip()
    result_line = ""
    for line in stdout.splitlines():
        payload_line = _marker_payload(line)
        if payload_line is not None:
            result_line = payload_line
    if not result_line:
        raise InvocationError(
            _last_error_line(stderr) or "No se pudo construir el modelo",
            detail=stderr[-4000:],
        )
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as exc:
        raise InvocationError(
            "El ejecutable devolvió una respuesta inválida", detail=stdout[-2000:]
        ) from exc
    if not result.get("ok"):
        raise InvocationError(
            friendly_invocation_error(
                result.get("error") or "La llamada falló",
                method["name"],
                method_fields(method),
            ),
            detail=(result.get("traceback") or stderr)[-4000:],
        )
    result.pop("ok", None)
    result["method"] = method["name"]
    result["elapsed"] = round(elapsed, 3)
    result["stdout"] = stdout.split(_PLAYGROUND_MARKER)[0][-4000:].strip()
    result["stderr"] = stderr[-4000:]
    return result


class ModelSession:
    """A live model instance kept inside its own child process.

    The worker builds the model once and then answers one JSON line per method
    call, so ``fit`` followed by ``predict`` sees the same object.  Every call is
    serialised by a lock (the pipe protocol is not re-entrant) and a call that
    exceeds the timeout kills the worker: a hung model cannot leak into the next
    request.
    """

    def __init__(
        self,
        session_id: str,
        model: Dict[str, Any],
        process: "subprocess.Popen[str]",
        timeout: float,
    ) -> None:
        self.id = session_id
        self.model = model
        self.process = process
        self.timeout = timeout
        self.created_at = time.time()
        self.last_used = self.created_at
        self.calls = 0
        self.errors = 0
        self.info: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._responses: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stdout: Deque[str] = deque(maxlen=200)
        self._stderr: Deque[str] = deque(maxlen=200)
        self._next_id = 0
        self._closed = False
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # -- pipes ------------------------------------------------------------

    def _read_stdout(self) -> None:
        """Forward the protocol lines to the queue and keep the model output."""
        stream = self.process.stdout
        if stream is None:  # pragma: no cover - defensive
            self._responses.put(None)
            return
        for line in stream:
            payload_line = _marker_payload(line.rstrip("\n"))
            if payload_line is not None:
                self._responses.put(payload_line)
            else:
                self._stdout.append(line.rstrip("\n"))
        self._responses.put(None)

    def _read_stderr(self) -> None:
        """Keep the last lines of stderr for the error reports."""
        stream = self.process.stderr
        if stream is None:  # pragma: no cover - defensive
            return
        for line in stream:
            self._stderr.append(line.rstrip("\n"))

    def stderr_tail(self) -> str:
        """Human readable tail of the worker stderr."""
        return "\n".join(list(self._stderr)[-20:])[-4000:]

    def stdout_tail(self) -> str:
        """Output the model printed during the session (truncated)."""
        return "\n".join(list(self._stdout)[-20:])[-4000:]

    # -- protocol ---------------------------------------------------------

    def _send(self, command: Dict[str, Any]) -> None:
        stream = self.process.stdin
        if stream is None:  # pragma: no cover - defensive
            raise InvocationError("La sesión no está disponible", status=503)
        try:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise InvocationError(
                "El proceso de la sesión terminó", status=503, detail=self.stderr_tail()
            ) from exc

    def _receive(self, timeout: float) -> Dict[str, Any]:
        try:
            raw = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise InvocationError(
                f"La llamada superó el límite de {timeout:g} segundos", status=504
            ) from exc
        if raw is None:
            raise InvocationError(
                "El proceso de la sesión terminó", status=503, detail=self.stderr_tail()
            )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise InvocationError(
                "El proceso de la sesión devolvió una respuesta inválida", status=503
            ) from exc

    def wait_ready(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for the constructor of the worker (``ready``/``error``)."""
        response = self._receive(self.timeout if timeout is None else timeout)
        if response.get("event") == "error":
            raise InvocationError(
                response.get("error") or "No se pudo construir el modelo",
                detail=(response.get("traceback") or self.stderr_tail())[-4000:],
            )
        self.info = {
            key: response[key]
            for key in ("type", "module", "repr", "attributes")
            if key in response
        }
        return self.info

    def invoke(self, method: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call ``method`` on the live instance and return its JSON result."""
        if self._closed:
            raise InvocationError("La sesión está cerrada", status=404)
        with self._lock:
            self._next_id += 1
            identifier = self._next_id
            found = find_method(self.model, method) or {}
            hints = {field["name"]: field["type"] for field in method_fields(found)}
            self._send(
                {
                    "id": identifier,
                    "method": method,
                    "arguments": arguments,
                    "argument_types": hints,
                }
            )
            try:
                response = self._receive(self.timeout)
            except InvocationError:
                # A timed out or dead worker cannot be reused.
                self.errors += 1
                self._closed = True
                self._terminate()
                raise
            self.calls += 1
            self.last_used = time.time()
            if response.get("id") not in (None, identifier):  # pragma: no cover - protocol
                LOGGER.warning("Respuesta fuera de orden en la sesión %s", self.id)
            if not response.get("ok"):
                self.errors += 1
                raise InvocationError(
                    friendly_invocation_error(
                        response.get("error") or "La llamada falló",
                        method,
                        method_fields(found),
                    ),
                    detail=(response.get("traceback") or "")[-4000:],
                )
            elapsed = response.get("elapsed")
            if not isinstance(elapsed, (int, float)):
                elapsed = 0.0
            return {
                "result": response.get("result"),
                "result_type": response.get("result_type") or "",
                "result_repr": response.get("result_repr") or "",
                "elapsed": elapsed,
            }

    # -- lifecycle --------------------------------------------------------

    def alive(self) -> bool:
        """True when the worker process is still running."""
        return not self._closed and self.process.poll() is None

    def _terminate(self) -> None:
        """Close the pipe and stop the worker (never takes the lock)."""
        try:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.write(json.dumps({"action": "close"}) + "\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                self.process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):  # pragma: no cover - defensive
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn worker
            self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable worker
                pass

    def close(self) -> None:
        """Stop the worker and release its pipes."""
        with self._lock:
            if self._closed and self.process.poll() is not None:
                return
            self._closed = True
            self._terminate()

    def to_dict(self, include_output: bool = False) -> Dict[str, Any]:
        """JSON projection of the session."""
        payload: Dict[str, Any] = {
            "session_id": self.id,
            "model_id": self.model["id"],
            "status": "open" if self.alive() else "closed",
            "type": self.info.get("type") or "",
            "module": self.info.get("module") or "",
            "repr": self.info.get("repr") or "",
            "attributes": self.info.get("attributes") or [],
            "created_at": self.created_at,
            "last_used": self.last_used,
            "idle_seconds": round(time.time() - self.last_used, 3),
            "calls": self.calls,
            "errors": self.errors,
            "timeout_seconds": self.timeout,
        }
        if include_output:
            payload["stdout"] = self.stdout_tail()
            payload["stderr"] = self.stderr_tail()
        return payload


def start_session(
    model: Dict[str, Any], parameters: Dict[str, Any], timeout: float
) -> ModelSession:
    """Spawn a worker, build the model inside it and wait until it is ready."""
    session_id = uuid.uuid4().hex[:12]
    try:
        process = _spawn(_SESSION_BOOTSTRAP)
    except OSError as exc:  # pragma: no cover - python disappeared
        raise InvocationError(f"No se pudo iniciar la sesión: {exc}", status=503) from exc
    session = ModelSession(session_id, model, process, timeout)
    payload = json.dumps(
        {
            "module": model["module"],
            "name": model["name"],
            "parameters": parameters,
            **sandbox_payload(model, constructor_types(model)),
        }
    )
    try:
        stream = process.stdin
        if stream is None:  # pragma: no cover - defensive
            raise InvocationError("No se pudo comunicar con la sesión", status=503)
        stream.write(payload + "\n")
        stream.flush()
        session.wait_ready(min(timeout, 60.0))
    except (InvocationError, BrokenPipeError, OSError):
        session.close()
        raise
    return session


class SessionStore:
    """Keeps a bounded number of live model instances (``/api/sessions``)."""

    def __init__(
        self,
        maximum: int = MAX_SESSIONS,
        ttl: float = SESSION_TTL,
        timeout: float = SESSION_TIMEOUT,
    ) -> None:
        self._maximum = max(1, maximum)
        self._ttl = max(1.0, ttl)
        self._timeout = timeout
        self._lock = threading.Lock()
        self._sessions: Dict[str, ModelSession] = {}
        self.created = 0

    def create(self, model: Dict[str, Any], parameters: Dict[str, Any]) -> ModelSession:
        """Start a new session, evicting the least recently used one if needed."""
        with self._lock:
            self._prune()
            evicted: Optional[ModelSession] = None
            if len(self._sessions) >= self._maximum:
                evicted = min(self._sessions.values(), key=lambda session: session.last_used)
                self._sessions.pop(evicted.id, None)
            self.created += 1
        if evicted is not None:
            LOGGER.info("Sesión %s desalojada (máximo %d)", evicted.id, self._maximum)
            evicted.close()
        session = start_session(model, parameters, self._timeout)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[ModelSession]:
        """The open session ``session_id``, or ``None``."""
        with self._lock:
            self._prune()
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        """Close and forget ``session_id``; ``True`` when it existed."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def recent(self) -> List[ModelSession]:
        """Every open session, most recently used first."""
        with self._lock:
            self._prune()
            return sorted(
                self._sessions.values(), key=lambda session: session.last_used, reverse=True
            )

    def _prune(self) -> None:
        """Close the sessions that are idle for too long or already dead."""
        now = time.time()
        expired = [
            session
            for session in self._sessions.values()
            if not session.alive() or now - session.last_used > self._ttl
        ]
        for session in expired:
            self._sessions.pop(session.id, None)
        for session in expired:
            session.close()

    def snapshot(self) -> Dict[str, Any]:
        """Counters exposed by ``GET /api/health``."""
        with self._lock:
            return {
                "max_sessions": self._maximum,
                "ttl_seconds": self._ttl,
                "timeout_seconds": self._timeout,
                "open": sum(1 for session in self._sessions.values() if session.alive()),
                "created": self.created,
            }

    def shutdown(self) -> None:
        """Close every session (used when the server stops)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


SESSIONS = SessionStore()
#: Cap of concurrent ``/invoke`` calls (each one spawns a child process).
INVOKE_SLOTS = threading.BoundedSemaphore(INVOKE_MAX_CONCURRENCY)


class PlaygroundJob:
    """One asynchronous playground request, possibly holding several runs.

    ``POST /api/playground/jobs`` returns its id immediately; the client polls
    ``GET /api/playground/jobs/<id>`` until ``status`` leaves ``pending``/
    ``running``.  A pool of worker threads keeps the cost of the background
    machinery bounded while every model still runs in its own child process.
    """

    def __init__(self, runs: List[Dict[str, Any]]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.runs = runs
        self.status = "pending"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.results: List[Dict[str, Any]] = []
        self.error = ""

    @property
    def batch(self) -> bool:
        """True when the job groups more than one run."""
        return len(self.runs) > 1

    @property
    def done(self) -> bool:
        """True once the job reached a final state."""
        return self.status in {"succeeded", "failed", "partial"}

    def elapsed(self) -> Optional[float]:
        """Seconds between the start and the end of the job, when known."""
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 3)

    def to_dict(self, include_results: bool = True) -> Dict[str, Any]:
        """JSON projection of the job."""
        payload: Dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "batch": self.batch,
            "runs": len(self.runs),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": self.elapsed(),
            "error": self.error,
        }
        if include_results:
            payload["results"] = self.results
        return payload


class JobQueue:
    """Background executor for playground runs (``/api/playground/jobs``)."""

    def __init__(
        self,
        workers: int = JOBS_WORKERS,
        retention: float = JOBS_RETENTION,
        maximum: int = JOBS_MAX,
    ) -> None:
        self._workers = max(1, workers)
        self._retention = max(1.0, retention)
        self._maximum = max(1, maximum)
        self._lock = threading.Lock()
        self._jobs: Dict[str, PlaygroundJob] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    def _pool(self) -> ThreadPoolExecutor:
        """Lazily created thread pool (never started while unused)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="lm-playground"
            )
        return self._executor

    def _prune(self) -> None:
        """Drop expired jobs and keep the store below ``maximum``."""
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.done and now - (job.finished_at or now) > self._retention
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
        if len(self._jobs) <= self._maximum:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.done),
            key=lambda job: job.finished_at or job.created_at,
        )
        for job in finished[: len(self._jobs) - self._maximum]:
            self._jobs.pop(job.id, None)

    def submit(self, runs: List[Dict[str, Any]]) -> PlaygroundJob:
        """Queue ``runs`` (``[{"model": ..., "parameters": ...}, ...]``)."""
        job = PlaygroundJob(runs)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        self._pool().submit(self._run, job)
        return job

    def _run(self, job: PlaygroundJob) -> None:
        """Execute every run of ``job`` and store the individual outcomes."""
        job.status = "running"
        job.started_at = time.time()
        successes = 0
        for run in job.runs:
            model = run["model"]
            parameters = run["parameters"]
            try:
                result = run_playground(model, parameters)
            except PlaygroundError as exc:
                job.results.append(
                    {
                        "success": False,
                        "model_id": model["id"],
                        "parameters": parameters,
                        "status": exc.status,
                        "error": str(exc),
                        "detail": exc.detail,
                    }
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Fallo inesperado del job %s en %s", job.id, model["id"])
                job.results.append(
                    {
                        "success": False,
                        "model_id": model["id"],
                        "parameters": parameters,
                        "status": 500,
                        "error": str(exc),
                        "detail": "",
                    }
                )
                continue
            successes += 1
            entry = {"success": True, "model_id": model["id"], "parameters": parameters}
            entry.update(result)
            job.results.append(entry)
        total = len(job.runs)
        if successes == total:
            job.status = "succeeded"
        elif successes:
            job.status = "partial"
        else:
            job.status = "failed"
            errors = [result.get("error", "") for result in job.results if result.get("error")]
            job.error = "; ".join(errors)[:500]
        job.finished_at = time.time()

    def get(self, job_id: str) -> Optional[PlaygroundJob]:
        """The job with ``job_id``, or ``None`` when unknown or expired."""
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> List[PlaygroundJob]:
        """The ``limit`` most recent jobs, newest first."""
        with self._lock:
            self._prune()
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return jobs[: max(1, limit)]

    def snapshot(self) -> Dict[str, Any]:
        """Counters exposed by ``GET /api/health``."""
        with self._lock:
            active = sum(1 for job in self._jobs.values() if not job.done)
            return {
                "workers": self._workers,
                "retention_seconds": self._retention,
                "max_jobs": self._maximum,
                "stored": len(self._jobs),
                "active": active,
            }

    def shutdown(self) -> None:
        """Stop accepting work and release the worker threads."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


JOBS = JobQueue()


def _git(*arguments: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Run ``git`` inside the repository and capture its output."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_versions(path: str, limit: int = 10) -> List[Dict[str, str]]:
    """Commits that touched ``path``, newest first.

    Raises :class:`APIError` with status ``503`` when git is unavailable or the
    path has no history, which is the honest answer for that endpoint.
    """
    completed = _git(
        "log",
        f"--max-count={max(1, limit)}",
        "--date=iso-strict",
        "--format=%H%x1f%ad%x1f%an%x1f%s",
        "--",
        path,
    )
    if completed.returncode != 0:
        raise APIError(
            "No se pudo leer el historial de git para este modelo",
            status=503,
            detail=(completed.stderr or "").strip()[-500:],
        )
    versions: List[Dict[str, str]] = []
    for line in (completed.stdout or "").splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        commit, date, author, subject = parts
        versions.append(
            {
                "commit": commit,
                "short_commit": commit[:8],
                "date": date,
                "author": author,
                "subject": subject,
            }
        )
    return versions


def git_blob(path: str, ref: str) -> str:
    """Content of ``path`` at ``ref``; raises :class:`APIError` when missing."""
    if not _SAFE_REF.match(ref) or ref.startswith("-"):
        # A leading ``-`` would be parsed by git as an option, never as a ref.
        raise ValidationError(f"Referencia de git inválida: {ref!r}")
    completed = _git("show", f"{ref}:{path}")
    if completed.returncode != 0:
        raise APIError(
            f'No se pudo leer "{path}" en la referencia "{ref}"',
            status=404,
            detail=(completed.stderr or "").strip()[-500:],
        )
    return completed.stdout


_JSON_SCHEMA_TYPES = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "text": "string",
}


def parameter_schema(parameter: Dict[str, Any]) -> Dict[str, Any]:
    """JSON Schema for one constructor argument of a model.

    The catalog already knows the annotation, the default and the documentation of
    every parameter, so the OpenAPI document can be derived without any manual
    bookkeeping.
    """
    kind = parameter.get("widget") or guess_widget(parameter["type"])
    schema: Dict[str, Any] = {
        "title": parameter["name"],
        "x-python-type": parameter["type"],
        "description": parameter["description"] or f"Parámetro `{parameter['name']}`.",
    }
    if kind in _JSON_SCHEMA_TYPES:
        schema["type"] = _JSON_SCHEMA_TYPES[kind]
    if parameter.get("sample") is not None:
        schema["x-python-default"] = parameter["sample"]
    default = _default_value(parameter)
    if default is not None:
        schema["default"] = default
    return schema


def model_request_schema(model: Dict[str, Any]) -> Dict[str, Any]:
    """Schema of the ``parameters`` object that constructs ``model``."""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for parameter in model["parameters"]:
        if parameter.get("variadic"):
            continue
        properties[parameter["name"]] = parameter_schema(parameter)
        if parameter["required"]:
            required.append(parameter["name"])
    schema: Dict[str, Any] = {
        "type": "object",
        "title": f"{model['id']} parameters",
        "description": model["short_description"] or model["title"],
        "properties": properties,
        "additionalProperties": False,
        "x-model-id": model["id"],
    }
    if required:
        schema["required"] = required
    return schema


def _error_response(description: str) -> Dict[str, Any]:
    """Standard ``success: false`` error body."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


def _json_response(description: str, schema: Any) -> Dict[str, Any]:
    """JSON response with an inline schema or a ``$ref``."""
    return {"description": description, "content": {"application/json": {"schema": schema}}}


def _query_parameter(name: str, description: str, **extra: Any) -> Dict[str, Any]:
    """Query string parameter definition."""
    parameter: Dict[str, Any] = {"name": name, "in": "query", "description": description}
    parameter.update(extra)
    return parameter


def _model_path_parameter() -> Dict[str, Any]:
    """The ``{model_id}`` path parameter shared by several operations."""
    return {
        "name": "model_id",
        "in": "path",
        "required": True,
        "description": "Identificador del modelo (`id` del catálogo).",
        "schema": {"type": "string"},
    }


def _session_path_parameter() -> Dict[str, Any]:
    """The ``{session_id}`` path parameter of the model sessions."""
    return {
        "name": "session_id",
        "in": "path",
        "required": True,
        "description": "Identificador devuelto por `POST /api/models/<id>/sessions`.",
        "schema": {"type": "string"},
    }


def source_diff(path: str, current: str, ref: str) -> Dict[str, Any]:
    """Unified diff between ``path`` at ``ref`` and its current content."""
    previous = git_blob(path, ref)
    diff_lines = list(
        difflib.unified_diff(
            previous.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"{path}@{ref}",
            tofile=path,
        )
    )
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return {
        "ref": ref,
        "path": path,
        "diff": "".join(diff_lines),
        "added": added,
        "removed": removed,
        "changes": added + removed,
    }


def _openapi_paths() -> Dict[str, Any]:
    """Path items of the OpenAPI document, grouped to keep the code readable."""
    model_path = _model_path_parameter()
    paths: Dict[str, Any] = {}
    paths.update(_openapi_catalog_paths(model_path))
    paths.update(_openapi_source_paths(model_path))
    paths.update(_openapi_playground_paths(model_path))
    paths.update(_openapi_notebook_paths(model_path))
    return paths


def _openapi_catalog_paths(model_path: Dict[str, Any]) -> Dict[str, Any]:
    """Health, categories, model listing, model detail and search."""
    return {
        "/api/health": {
            "get": {
                "tags": ["operación"],
                "summary": "Estado del servidor, contadores y métricas por endpoint",
                "responses": {"200": _json_response("Estado y métricas", {"type": "object"})},
            }
        },
        "/api/categories": {
            "get": {
                "tags": ["catálogo"],
                "summary": "Categorías con su número de modelos",
                "responses": {"200": _json_response("Categorías", {"type": "object"})},
            }
        },
        "/api/models": {
            "get": {
                "tags": ["catálogo"],
                "summary": "Lista paginada de modelos",
                "parameters": [
                    _query_parameter("category", "Filtra por categoría", schema={"type": "string"}),
                    _query_parameter("q", "Búsqueda de texto libre", schema={"type": "string"}),
                    _query_parameter(
                        "tags",
                        "Etiquetas separadas por comas; se exigen todas",
                        schema={"type": "string"},
                    ),
                    _query_parameter(
                        "limit",
                        "Tamaño de página",
                        schema={"type": "integer", "minimum": 1, "maximum": MAX_PAGE_SIZE},
                    ),
                    _query_parameter(
                        "offset",
                        "Desplazamiento dentro del resultado",
                        schema={"type": "integer", "minimum": 0},
                    ),
                ],
                "responses": {
                    "200": _json_response(
                        "Página de modelos con `has_more` y `next`",
                        {
                            "type": "object",
                            "properties": {
                                "models": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/ModelBrief"},
                                },
                                "total": {"type": "integer"},
                                "matched": {"type": "integer"},
                                "offset": {"type": "integer"},
                                "limit": {"type": ["integer", "null"]},
                                "has_more": {"type": "boolean"},
                                "next": {"type": ["string", "null"]},
                            },
                        },
                    ),
                    "400": _error_response("Parámetros de paginación inválidos"),
                },
            }
        },
        "/api/models/{model_id}": {
            "get": {
                "tags": ["catálogo"],
                "summary": "Todo lo conocido sobre un modelo",
                "parameters": [model_path],
                "responses": {
                    "200": _json_response("Modelo completo", {"type": "object"}),
                    "404": _error_response("Modelo no encontrado"),
                },
            }
        },
        "/api/search": {
            "get": {
                "tags": ["catálogo"],
                "summary": "Búsqueda con facetas por categoría y etiqueta",
                "parameters": [
                    _query_parameter(
                        "q", "Término de búsqueda", required=True, schema={"type": "string"}
                    ),
                    _query_parameter(
                        "tags", "Etiquetas separadas por comas", schema={"type": "string"}
                    ),
                    _query_parameter("limit", "Máximo de resultados", schema={"type": "integer"}),
                ],
                "responses": {
                    "200": _json_response("Resultados y facetas", {"type": "object"}),
                    "400": _error_response("Falta `q`"),
                },
            }
        },
        "/api/models/{model_id}/invoke": {
            "get": {
                "tags": ["invocación"],
                "summary": "Métodos que se pueden invocar en el modelo",
                "parameters": [model_path],
                "responses": {
                    "200": _json_response(
                        "Métodos permitidos y sus argumentos", {"type": "object"}
                    ),
                    "404": _error_response("Modelo no encontrado"),
                },
            },
            "post": {
                "tags": ["invocación"],
                "summary": "Construye el modelo y llama a un método (sin estado)",
                "parameters": [model_path],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/InvokeRequest"}
                        }
                    },
                },
                "responses": {
                    "200": _json_response(
                        "Resultado de la llamada",
                        {"$ref": "#/components/schemas/InvokeResult"},
                    ),
                    "400": _error_response("Argumentos o parámetros inválidos"),
                    "403": _error_response("Método bloqueado por la política o API deshabilitada"),
                    "404": _error_response("Modelo o método no encontrado"),
                    "422": _error_response("El modelo lanzó una excepción"),
                    "429": _error_response("Demasiadas llamadas simultáneas"),
                    "504": _error_response("Se agotó el tiempo de la llamada"),
                },
            },
        },
        "/api/models/{model_id}/sample": {
            "post": {
                "tags": ["invocación"],
                "summary": "Datos de ejemplo (JSON) para los argumentos de un método",
                "parameters": [model_path],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SampleRequest"}
                        }
                    },
                },
                "responses": {
                    "200": _json_response(
                        "Argumentos de ejemplo listos para `POST .../invoke`",
                        {"$ref": "#/components/schemas/SampleResponse"},
                    ),
                    "400": _error_response("Parámetros inválidos"),
                    "403": _error_response("Método bloqueado por la política o API deshabilitada"),
                    "404": _error_response("Modelo o método no encontrado"),
                },
            }
        },
        "/api/models/{model_id}/sessions": {
            "post": {
                "tags": ["invocación"],
                "summary": "Construye el modelo una vez y mantiene la instancia viva",
                "parameters": [model_path],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SessionRequest"}
                        }
                    },
                },
                "responses": {
                    "201": _json_response(
                        "Sesión creada", {"$ref": "#/components/schemas/Session"}
                    ),
                    "400": _error_response("Parámetros inválidos"),
                    "403": _error_response("Sesiones deshabilitadas"),
                    "404": _error_response("Modelo no encontrado"),
                    "422": _error_response("El constructor del modelo falló"),
                },
            }
        },
        "/api/sessions": {
            "get": {
                "tags": ["invocación"],
                "summary": "Sesiones abiertas",
                "responses": {"200": _json_response("Lista de sesiones", {"type": "object"})},
            }
        },
        "/api/sessions/{session_id}": {
            "get": {
                "tags": ["invocación"],
                "summary": "Estado de una sesión",
                "parameters": [_session_path_parameter()],
                "responses": {
                    "200": _json_response("Sesión", {"$ref": "#/components/schemas/Session"}),
                    "404": _error_response("Sesión desconocida o expirada"),
                },
            },
            "delete": {
                "tags": ["invocación"],
                "summary": "Cierra la sesión y su proceso",
                "parameters": [_session_path_parameter()],
                "responses": {
                    "200": _json_response("Sesión cerrada", {"type": "object"}),
                    "404": _error_response("Sesión desconocida o expirada"),
                },
            },
        },
        "/api/sessions/{session_id}/invoke": {
            "post": {
                "tags": ["invocación"],
                "summary": "Llama a un método sobre la instancia viva",
                "parameters": [_session_path_parameter()],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/InvokeRequest"}
                        }
                    },
                },
                "responses": {
                    "200": _json_response(
                        "Resultado de la llamada",
                        {"$ref": "#/components/schemas/InvokeResult"},
                    ),
                    "400": _error_response("Argumentos inválidos"),
                    "403": _error_response("Método bloqueado por la política"),
                    "404": _error_response("Sesión no encontrada"),
                    "422": _error_response("El modelo lanzó una excepción"),
                    "504": _error_response("Se agotó el tiempo de la llamada"),
                },
            }
        },
        "/api/openapi.json": {
            "get": {
                "tags": ["operación"],
                "summary": "Este documento",
                "responses": {"200": _json_response("Especificación OpenAPI", {"type": "object"})},
            }
        },
    }


def _openapi_source_paths(model_path: Dict[str, Any]) -> Dict[str, Any]:
    """Raw source, differential view and git history of a model."""
    return {
        "/api/models/{model_id}/source": {
            "get": {
                "tags": ["fuentes"],
                "summary": "Código fuente del modelo, o su diff contra una referencia de git",
                "parameters": [
                    model_path,
                    _query_parameter(
                        "download", "`1` envía el archivo como adjunto", schema={"type": "string"}
                    ),
                    _query_parameter(
                        "diff",
                        "Referencia de git (por ejemplo `HEAD~1`)",
                        schema={"type": "string"},
                    ),
                ],
                "responses": {
                    "200": {"description": "Código fuente o diff unificado"},
                    "400": _error_response("Referencia de git inválida"),
                    "404": _error_response("Modelo o referencia no encontrada"),
                    "503": _error_response("git no está disponible"),
                },
            }
        },
        "/api/models/{model_id}/versions": {
            "get": {
                "tags": ["fuentes"],
                "summary": "Commits que modificaron el código fuente del modelo",
                "parameters": [
                    model_path,
                    _query_parameter(
                        "limit",
                        "Número de commits",
                        schema={"type": "integer", "minimum": 1, "maximum": 100},
                    ),
                ],
                "responses": {
                    "200": _json_response("Historial del archivo", {"type": "object"}),
                    "404": _error_response("Modelo no encontrado"),
                    "503": _error_response("git no está disponible"),
                },
            }
        },
    }


def _openapi_playground_paths(model_path: Dict[str, Any]) -> Dict[str, Any]:
    """Playground specification, synchronous run and asynchronous jobs."""
    request_ref = {"$ref": "#/components/schemas/PlaygroundRequest"}
    return {
        "/api/playground/{model_id}": {
            "get": {
                "tags": ["playground"],
                "summary": "Especificación del playground de un modelo",
                "parameters": [model_path],
                "responses": {
                    "200": _json_response("Especificación", {"type": "object"}),
                    "404": _error_response("Modelo no encontrado"),
                },
            }
        },
        "/api/playground": {
            "post": {
                "tags": ["playground"],
                "summary": "Construye un modelo con los parámetros enviados",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": request_ref}},
                },
                "responses": {
                    "200": _json_response(
                        "Modelo construido", {"$ref": "#/components/schemas/PlaygroundResult"}
                    ),
                    "400": _error_response("Parámetros inválidos"),
                    "403": _error_response("Playground deshabilitado"),
                    "404": _error_response("Modelo no encontrado"),
                    "422": _error_response("El constructor del modelo falló"),
                    "504": _error_response("Se agotó el tiempo de ejecución"),
                },
            }
        },
        "/api/playground/jobs": {
            "get": {
                "tags": ["playground"],
                "summary": "Jobs recientes",
                "parameters": [
                    _query_parameter("limit", "Número de jobs", schema={"type": "integer"})
                ],
                "responses": {"200": _json_response("Lista de jobs", {"type": "object"})},
            },
            "post": {
                "tags": ["playground"],
                "summary": "Encola una ejecución o un lote de ejecuciones",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "oneOf": [
                                    request_ref,
                                    {"type": "array", "items": request_ref},
                                ]
                            }
                        }
                    },
                },
                "responses": {
                    "202": _json_response("Job aceptado", {"$ref": "#/components/schemas/Job"}),
                    "400": _error_response("Lote vacío o inválido"),
                    "403": _error_response("Playground deshabilitado"),
                    "404": _error_response("Modelo no encontrado"),
                },
            },
        },
        "/api/playground/jobs/{job_id}": {
            "get": {
                "tags": ["playground"],
                "summary": "Estado y resultados de un job",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": _json_response("Job", {"$ref": "#/components/schemas/Job"}),
                    "404": _error_response("Job desconocido o expirado"),
                },
            }
        },
    }


def _openapi_notebook_paths(model_path: Dict[str, Any]) -> Dict[str, Any]:
    """Notebook download and generation."""
    return {
        "/api/notebook/{model_id}": {
            "get": {
                "tags": ["notebooks"],
                "summary": "Descarga el notebook generado",
                "parameters": [
                    model_path,
                    _query_parameter(
                        "parameters",
                        "JSON con los valores del playground usados como defaults",
                        schema={"type": "string"},
                    ),
                ],
                "responses": {
                    "200": {"description": "Notebook (nbformat 4.5)"},
                    "400": _error_response("`parameters` no es un objeto JSON"),
                    "404": _error_response("Modelo no encontrado"),
                },
            }
        },
        "/api/notebook": {
            "post": {
                "tags": ["notebooks"],
                "summary": "Devuelve el notebook como JSON",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PlaygroundRequest"}
                        }
                    },
                },
                "responses": {
                    "200": _json_response("Notebook", {"type": "object"}),
                    "400": _error_response("Petición inválida"),
                    "404": _error_response("Modelo no encontrado"),
                },
            }
        },
    }


def openapi_spec(catalog: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """OpenAPI 3.1 document generated from the live catalog."""
    schemas: Dict[str, Any] = {
        "Error": {
            "type": "object",
            "required": ["success", "error"],
            "properties": {
                "success": {"type": "boolean", "const": False},
                "error": {"type": "string"},
            },
        },
        "ModelBrief": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "icon": {"type": "string"},
                "category": {"type": "string"},
                "category_label": {"type": "string"},
                "short_description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "module": {"type": "string"},
                "signature": {"type": "string"},
                "parameter_count": {"type": "integer"},
                "method_count": {"type": "integer"},
                "line_count": {"type": "integer"},
            },
        },
        "PlaygroundRequest": {
            "type": "object",
            "required": ["model_id"],
            "properties": {
                "model_id": {"type": "string", "enum": [model["id"] for model in catalog]},
                "parameters": {
                    "type": "object",
                    "description": "Argumentos del constructor; ver los esquemas `*Parameters`.",
                    "additionalProperties": True,
                },
            },
        },
        "PlaygroundResult": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "model_id": {"type": "string"},
                "parameters": {"type": "object"},
                "status": {"type": "integer"},
                "error": {"type": "string"},
                "type": {"type": "string"},
                "module": {"type": "string"},
                "repr": {"type": "string"},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "elapsed": {"type": "number"},
            },
        },
        "Job": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "succeeded", "partial", "failed"],
                },
                "batch": {"type": "boolean"},
                "runs": {"type": "integer"},
                "created_at": {"type": "number"},
                "started_at": {"type": ["number", "null"]},
                "finished_at": {"type": ["number", "null"]},
                "elapsed": {"type": ["number", "null"]},
                "error": {"type": "string"},
                "poll_url": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/PlaygroundResult"},
                },
            },
        },
        "InvokeRequest": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Método a llamar; ver `GET /api/models/<id>/invoke`.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Argumentos del método; las listas se convierten en "
                    "`ndarray`/`DataFrame` cuando la firma lo declara.",
                    "additionalProperties": True,
                },
                "parameters": {
                    "type": "object",
                    "description": "Parámetros del constructor (solo en la llamada sin estado).",
                    "additionalProperties": True,
                },
            },
        },
        "InvokeResult": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "model_id": {"type": "string"},
                "session_id": {"type": ["string", "null"]},
                "method": {"type": "string"},
                "arguments": {"type": "object"},
                "parameters": {"type": "object"},
                "calls": {"type": "integer"},
                "result": {
                    "description": "Valor devuelto, proyectado a JSON (listas/objetos).",
                },
                "result_type": {"type": "string"},
                "result_repr": {"type": "string"},
                "elapsed": {"type": "number"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "error": {"type": "string"},
                "detail": {"type": "string"},
            },
        },
        "SampleRequest": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Método cuyos argumentos se quieren ejemplificar.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Valores del constructor que ajustan la forma de los datos.",
                    "additionalProperties": True,
                },
            },
        },
        "SampleResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "model_id": {"type": "string"},
                "method": {"type": "string"},
                "parameters": {"type": "object"},
                "arguments": {
                    "type": "object",
                    "description": "Argumentos de ejemplo listos para `POST .../invoke`.",
                },
            },
        },
        "SessionRequest": {
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "description": "Parámetros del constructor del modelo.",
                    "additionalProperties": True,
                }
            },
        },
        "Session": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "model_id": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "closed"]},
                "type": {"type": "string"},
                "module": {"type": "string"},
                "repr": {"type": "string"},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "created_at": {"type": "number"},
                "last_used": {"type": "number"},
                "idle_seconds": {"type": "number"},
                "calls": {"type": "integer"},
                "errors": {"type": "integer"},
                "timeout_seconds": {"type": "number"},
                "invoke_url": {"type": "string"},
                "parameters": {"type": "object"},
            },
        },
    }
    for model in catalog:
        schemas[f"{model['id']}Parameters"] = model_request_schema(model)
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Likelihood model catalog",
            "version": API_VERSION,
            "description": (
                "Catálogo de los modelos del paquete `likelihood` generado a partir de las "
                "firmas reales de las clases. Todas las claves JSON usan `snake_case`."
            ),
        },
        "servers": [{"url": "/"}],
        "tags": [
            {"name": "catálogo", "description": "Modelos, categorías y búsqueda"},
            {"name": "fuentes", "description": "Código fuente y su historial"},
            {"name": "playground", "description": "Construcción de modelos en un proceso hijo"},
            {"name": "invocación", "description": "Llamadas a métodos y sesiones con estado"},
            {"name": "notebooks", "description": "Generación de notebooks"},
            {"name": "operación", "description": "Salud, métricas y documentación"},
        ],
        "paths": _openapi_paths(),
        "components": {"schemas": schemas},
    }


_DOCS_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API del catálogo · documentación</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; }
  header { padding: 1rem 1.5rem; background: #1f2937; color: #f9fafb; }
  header a { color: #8ab4f8; }
  main { padding: 1.5rem; }
  code { background: #e2e8f0; padding: .1rem .3rem; border-radius: .25rem; }
  .hint { color: #475569; }
</style>
</head>
<body>
<header>
  <strong>API del catálogo Likelihood</strong> ·
  <a href="/api/openapi.json">/api/openapi.json</a> ·
  <a href="/">Volver al catálogo</a>
</header>
<main>
  <p class="hint">Especificación OpenAPI {{VERSION}} generada desde el catálogo:
  <a href="/api/openapi.json">descargar JSON</a>.</p>
  <div id="swagger-ui">
    <p class="hint">No se pudo cargar el visor interactivo (sin acceso a la CDN).
    Descarga la especificación y ábrela en
    <a href="https://editor.swagger.io/">editor.swagger.io</a>.</p>
  </div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
  <script>
    (function () {
      if (!window.SwaggerUIBundle) return;
      document.getElementById('swagger-ui').innerHTML = '';
      window.SwaggerUIBundle({
        url: '/api/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true
      });
    })();
  </script>
</main>
</body>
</html>
"""


def render_docs_page() -> str:
    """The small viewer served by ``GET /docs``."""
    return _DOCS_PAGE.replace("{{VERSION}}", html.escape(API_VERSION))


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
        "MODEL_SHORT_DESCRIPTION": _escape(model["short_description"]),
        "MODEL_DESCRIPTION": _escape(model["full_description"]).replace("\n", "<br>"),
        "MODEL_TAGS": tags_html(model),
        "MODEL_PARAMETERS": parameters_html(model),
        "MODEL_ATTRIBUTES_SECTION": attributes_section_html(model),
        "MODEL_METHODS": methods_html(model),
        "MODEL_SOURCE": _escape(model["source"]),
        "MODEL_EXAMPLE": _escape(model["example"]),
        "MODEL_IMPORT": _escape(model["import_statement"]),
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
        "MODEL_INVOKE_JSON": _script_json(invoke_spec(model)),
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
    #: Seconds a client may take to send a request.  ``StreamRequestHandler.setup``
    #: applies it to the socket, so a stalled client cannot block a worker.
    timeout = REQUEST_TIMEOUT

    catalog: List[Dict[str, Any]] = CATALOG
    template: str = ""
    playground_enabled: bool = PLAYGROUND_ENABLED
    invoke_enabled: bool = INVOKE_ENABLED
    sessions_enabled: bool = SESSIONS_ENABLED
    #: Kept as class attributes so tests (or an embedder) can override them
    #: without touching the environment.
    cors_origins: Tuple[str, ...] = CORS_ORIGINS
    api_token: str = API_TOKEN
    rate_limiter: RateLimiter = RATE_LIMITER
    metrics: Metrics = METRICS
    jobs: JobQueue = JOBS
    sessions: SessionStore = SESSIONS
    invoke_slots: threading.BoundedSemaphore = INVOKE_SLOTS
    store: CatalogStore = CATALOG_STORE

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

    # -- lifecycle --------------------------------------------------------

    def _start_request(self) -> None:
        """Initialise the per request state used by the metrics and the logs."""
        self.request_id = uuid.uuid4().hex[:12]
        self.started_at = time.monotonic()
        self.status_code = 0
        self.body_read = False
        self.route = endpoint_label(self.command, unquote(urlparse(self.path).path))

    def _finish_request(self) -> None:
        """Record the outcome of the request in :class:`Metrics`."""
        started = getattr(self, "started_at", None)
        if started is None:
            return
        self.metrics.record(self.route, self.status_code or 500, time.monotonic() - started)

    def _reload_catalog(self) -> None:
        """Refresh the catalog when the model sources changed on disk."""
        if self.catalog is not self.store.catalog:
            return  # an explicitly injected catalog (tests, --package-dir) wins
        if not self.store.refresh():
            return
        global CATALOG, MODELS_BY_ID
        CATALOG = self.store.catalog
        MODELS_BY_ID = self.store.by_id
        ModelAPIHandler.catalog = CATALOG
        LOGGER.info("Catálogo recargado desde %s: %d modelos", self.store.package_dir, len(CATALOG))

    def _check_auth(self, path: str) -> None:
        """Reject ``/api`` requests that do not carry the configured token."""
        if not self.api_token or not path.startswith("/api/"):
            return
        if path == "/api/health":
            return  # liveness probes stay open, they expose no model data
        header = self.headers.get("Authorization") or ""
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
        else:
            token = (self.headers.get("X-API-Token") or "").strip()
        if token and hmac.compare_digest(token, self.api_token):
            return
        raise AuthenticationError(
            "Token de API inválido o ausente",
            headers={"WWW-Authenticate": 'Bearer realm="likelihood"'},
        )

    def _check_rate_limit(self) -> None:
        """Apply the sliding window limiter, when one is configured."""
        retry_after = self.rate_limiter.hit(self.client_address[0])
        if retry_after is None:
            return
        wait = max(1, math.ceil(retry_after))
        raise RateLimitExceeded(
            f"Demasiadas peticiones; reintenta en {wait} segundos",
            headers={"Retry-After": str(wait)},
        )

    def _dispatch(self, route: Callable[[str, Dict[str, List[str]]], None]) -> None:
        """Run ``route`` with authentication, rate limiting, metrics and logging.

        Every failure ends up here: the mapped exceptions keep their status,
        anything else is a ``500`` with an exception log that includes the
        endpoint, the latency and the request id.
        """
        self._start_request()
        path = unquote(urlparse(self.path).path)
        try:
            self._reload_catalog()
            self._check_auth(path)
            self._check_rate_limit()
            route(path, self._query())
        except APIError as exc:
            self._log_failure(exc)
            self._send_api_error(exc)
        except (socket.timeout, TimeoutError) as exc:
            error = RequestTimeout(f"La petición superó el límite de {REQUEST_TIMEOUT:g} segundos")
            self._log_failure(error, cause=exc)
            self._send_api_error(error)
        except BrokenPipeError:  # pragma: no cover - client closed the socket
            LOGGER.debug("[%s] Cliente desconectado en %s", self.request_id, self.route)
        except Exception:  # pragma: no cover - last resort, keep the server alive
            LOGGER.exception(
                "[%s] Error inesperado en %s (%.1f ms)",
                self.request_id,
                self.route,
                (time.monotonic() - self.started_at) * 1000.0,
            )
            self._send_error_json(500, "Error interno del servidor", request_id=self.request_id)
        finally:
            self._finish_request()
            LOGGER.info(
                "[%s] %s %s %d %.1f ms",
                self.request_id,
                self.address_string(),
                self.route,
                self.status_code,
                (time.monotonic() - self.started_at) * 1000.0,
            )

    def _log_failure(self, exc: APIError, cause: Optional[BaseException] = None) -> None:
        """Structured log line for the expected (mapped) errors."""
        LOGGER.warning(
            "[%s] %s %s -> %d (%.1f ms): %s",
            self.request_id,
            self.address_string(),
            self.route,
            exc.status,
            (time.monotonic() - self.started_at) * 1000.0,
            exc,
            exc_info=cause,
        )

    # -- responses --------------------------------------------------------

    # -- responses --------------------------------------------------------

    def _cors_headers(self) -> Dict[str, str]:
        """``Access-Control-*`` headers for the current ``Origin``, when allowed."""
        origin = (self.headers.get("Origin") or "").strip().rstrip("/")
        if not origin or not self.cors_origins:
            return {}
        wildcard = "*" in self.cors_origins
        if not wildcard and origin not in self.cors_origins:
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

    def send_response(self, code: int, message: Optional[str] = None) -> None:
        """Remember the status (the metrics need it) and keep the default headers."""
        self.status_code = code
        super().send_response(code, message)

    def end_headers(self) -> None:
        """Add the CORS and versioning headers to every single response."""
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        self.send_header("X-API-Version", API_VERSION)
        self.send_header("X-Request-Id", getattr(self, "request_id", ""))
        super().end_headers()

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

    def _send_cacheable_json(
        self,
        status: int,
        payload: Any,
        cache_key: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """JSON response with ``ETag``/``Cache-Control`` support.

        The catalog only changes when a source file changes (see
        :class:`CatalogStore`), so the catalog fingerprint plus the request path is
        a valid, cheap ``ETag``; a matching ``If-None-Match`` answers ``304``.
        """
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        digest = hashlib.sha1(
            f"{cache_key}:{self.store.fingerprint}:{API_VERSION}".encode("utf-8")
        ).hexdigest()[:20]
        etag = f'"{digest}"'
        headers: Dict[str, str] = {"Cache-Control": API_CACHE_CONTROL, "ETag": etag}
        headers.update(extra_headers or {})
        cached = (self.headers.get("If-None-Match") or "").strip()
        if cached in {etag, f"W/{etag}"}:
            self._send_bytes(304, b"", "application/json; charset=utf-8", headers)
            return
        self._send_bytes(status, body, "application/json; charset=utf-8", headers)

    def _send_error_json(self, status: int, message: str, **extra: Any) -> None:
        payload = {"success": False, "error": message}
        payload.update(extra)
        self._send_json(status, payload)

    def _send_api_error(self, exc: APIError) -> None:
        """Serialise a mapped exception (status, headers and extra fields)."""
        payload = {"success": False, "error": str(exc)}
        payload.update(exc.extra)
        if exc.status in (408, 413):
            # The pending bytes must not be parsed as the next request.
            self.close_connection = True
        elif exc.status in (401, 403, 429) and not getattr(self, "body_read", True):
            # The request was rejected before its body was consumed: keeping the
            # connection alive would make the server read that body as a request.
            declared = self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding")
            if declared:
                self.close_connection = True
        headers = {"Cache-Control": "no-store"}
        headers.update(exc.headers)
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(exc.status, body, "application/json; charset=utf-8", headers)

    def _send_redirect(self, location: str, status: int = 302) -> None:
        """Answer with a body-less redirect (used by ``/favicon.ico``)."""
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()

    def _read_chunk(self, size: int) -> bytes:
        """Read up to ``size`` bytes, mapping a socket timeout onto ``408``."""
        try:
            return self.rfile.read(size)
        except (socket.timeout, TimeoutError) as exc:
            raise RequestTimeout(
                f"La lectura del cuerpo superó {REQUEST_TIMEOUT:g} segundos"
            ) from exc

    def _read_exactly(self, length: int) -> bytes:
        """Read ``length`` bytes in bounded chunks, never trusting the client.

        Reading in chunks means a client that declares a body it never sends can
        only keep the server busy until the socket timeout, and a premature EOF is
        reported as a ``400`` instead of being parsed as a truncated payload.
        """
        chunks: List[bytes] = []
        remaining = length
        while remaining > 0:
            chunk = self._read_chunk(min(BODY_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValidationError("El cuerpo de la petición está incompleto")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_chunked_body(self) -> bytes:
        """Decode a ``Transfer-Encoding: chunked`` body, capping its total size."""
        chunks: List[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline(1024)
            if not line:
                raise ValidationError("El cuerpo de la petición está incompleto")
            try:
                size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            except ValueError as exc:
                raise ValidationError("Bloque 'chunked' inválido en el cuerpo") from exc
            if size == 0:
                self.rfile.readline(1024)  # final CRLF or trailers
                break
            total += size
            if total > MAX_BODY_BYTES:
                raise RequestBodyTooLarge(
                    f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
                )
            chunks.append(self._read_exactly(size))
            self.rfile.readline(1024)  # CRLF after every chunk
        return b"".join(chunks)

    def _read_body(self) -> bytes:
        """The raw body of the request (``b""`` when there is none).

        ``Content-Length`` is validated but never trusted on its own: the body is
        read in :data:`BODY_CHUNK_BYTES` chunks, capped at :data:`MAX_BODY_BYTES`
        and the socket timeout aborts a client that stops sending data.
        """
        transfer_encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in transfer_encoding:
            body = self._read_chunked_body()
            self.body_read = True
            return body
        raw_length = (self.headers.get("Content-Length") or "").strip()
        if not raw_length:
            self.body_read = True
            return b""
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValidationError("Cabecera Content-Length inválida") from exc
        if length < 0:
            raise ValidationError("Cabecera Content-Length inválida")
        if length > MAX_BODY_BYTES:
            raise RequestBodyTooLarge(
                f"El cuerpo supera el máximo permitido de {MAX_BODY_BYTES} bytes"
            )
        body = self._read_exactly(length)
        self.body_read = True
        return body

    def _read_json_body(self, allow_list: bool = False) -> Any:
        """Parse the request body as JSON.

        Returns an empty dict for an empty body and, when ``allow_list`` is set, a
        JSON array too.  Raises :class:`ValidationError` for malformed payloads,
        :class:`UnsupportedMediaType` for non JSON content and
        :class:`RequestBodyTooLarge` when :data:`MAX_BODY_BYTES` is exceeded.
        """
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type not in _JSON_TYPES:
            raise UnsupportedMediaType(
                f"Tipo de contenido no soportado: {content_type}; usa application/json"
            )
        raw = self._read_body()
        if not raw.strip():
            return [] if allow_list else {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Cuerpo JSON inválido: {exc}") from exc
        if isinstance(data, dict):
            return data
        if allow_list and isinstance(data, list):
            return data
        raise ValidationError("El cuerpo JSON debe ser un objeto")

    def _query(self) -> Dict[str, List[str]]:
        return parse_qs(urlparse(self.path).query, keep_blank_values=True)

    def _models(self, query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Apply the ``category``, ``q`` and ``tags`` filters to the catalog."""
        selected = self.catalog
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

    def do_HEAD(self) -> None:  # noqa: N802
        """HEAD reuses the GET routing; ``_send_bytes`` skips the body."""
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(self._route_post)

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch(self._route_delete)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """CORS preflight: answer ``204`` when an allowed origin asks for it."""
        self._start_request()
        try:
            self._handle_options()
        except APIError as exc:  # pragma: no cover - defensive
            self._send_api_error(exc)
        finally:
            self._finish_request()

    def _handle_options(self) -> None:
        """Reply to the preflight of the allowed origins, else ``405``."""
        origin = (self.headers.get("Origin") or "").strip().rstrip("/")
        allowed = bool(origin) and ("*" in self.cors_origins or origin in self.cors_origins)
        if not self.cors_origins or not allowed:
            self._send_json(
                405,
                {
                    "success": False,
                    "error": f"Método no soportado: {self.command}",
                    "allow": "GET, HEAD, POST",
                },
                {"Allow": "GET, HEAD, POST"},
            )
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-API-Token, If-None-Match",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _route_get(self, path: str, query: Dict[str, List[str]]) -> None:
        """Dispatch a ``GET``/``HEAD`` request."""
        if path in ("/", "/index.html"):
            self._serve_asset(INDEX_FILE)
        elif path == "/favicon.ico":
            self._send_redirect("/favicon.svg", 308)
        elif path == "/docs":
            self._send_bytes(
                200,
                render_docs_page().encode("utf-8"),
                "text/html; charset=utf-8",
                {"Cache-Control": "no-store"},
            )
        elif path == "/api/health":
            self._handle_health()
        elif path == "/api/openapi.json":
            self._handle_openapi()
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
                {"success": False, "error": "Método no soportado: GET", "allow": "POST"},
                {"Allow": "POST"},
            )
        elif path == "/api/playground/jobs":
            self._handle_job_list(query)
        elif path.startswith("/api/playground/jobs/"):
            self._handle_job_get(path[len("/api/playground/jobs/") :])
        elif path.startswith("/api/playground/"):
            self._handle_playground_spec(path[len("/api/playground/") :])
        elif path == "/api/sessions":
            self._handle_session_list(query)
        elif path.startswith("/api/sessions/"):
            self._handle_session_get(path[len("/api/sessions/") :])
        elif path.startswith("/api/notebook/"):
            self._handle_notebook_download(path[len("/api/notebook/") :], query)
        elif path.startswith("/api/"):
            self._send_error_json(404, f"Endpoint desconocido: {path}")
        elif path.startswith("/preview/"):
            self._handle_preview(path[len("/preview/") :])
        else:
            self._serve_static(path)

    def _route_post(self, path: str, _query: Dict[str, List[str]]) -> None:
        """Dispatch a ``POST`` request (every endpoint reads its own body)."""
        if path == "/api/notebook":
            self._handle_notebook(self._read_json_body())
        elif path == "/api/playground":
            self._handle_playground(self._read_json_body())
        elif path == "/api/playground/jobs":
            self._handle_job_create(self._read_json_body(allow_list=True))
        elif path == "/api/search":
            self._handle_search(self._read_json_body(), deprecated=True)
        elif path.startswith("/api/models/"):
            self._route_model_post(path[len("/api/models/") :])
        elif path.startswith("/api/sessions/") and path.endswith("/invoke"):
            session_id = path[len("/api/sessions/") : -len("/invoke")].strip("/")
            self._handle_session_invoke(session_id, self._read_json_body())
        else:
            self._send_error_json(404, f"Endpoint desconocido: {path}")

    def _route_delete(self, path: str, _query: Dict[str, List[str]]) -> None:
        """``DELETE`` only exists for the model sessions."""
        if path.startswith("/api/sessions/"):
            self._handle_session_delete(path[len("/api/sessions/") :])
        else:
            self._send_error_json(404, f"Endpoint desconocido: {path}")

    def _route_model_post(self, remainder: str) -> None:
        """``POST /api/models/<id>/{invoke,sessions,sample}``."""
        parts = [part for part in remainder.split("/") if part]
        if len(parts) == 2 and parts[1] == "invoke":
            self._handle_invoke(parts[0], self._read_json_body())
        elif len(parts) == 2 and parts[1] == "sessions":
            self._handle_session_create(parts[0], self._read_json_body())
        elif len(parts) == 2 and parts[1] == "sample":
            self._handle_sample(parts[0], self._read_json_body())
        else:
            self._send_error_json(404, f"Endpoint desconocido: /api/models/{remainder}")

    def _route_model(self, remainder: str, query: Dict[str, List[str]]) -> None:
        """Dispatch ``/api/models/<id>``, ``.../source``, ``.../versions`` and ``.../invoke``."""
        parts = [part for part in remainder.split("/") if part]
        if len(parts) == 1:
            self._handle_get_model(parts[0])
        elif len(parts) == 2 and parts[1] == "source":
            self._handle_model_source(parts[0], query)
        elif len(parts) == 2 and parts[1] == "versions":
            self._handle_model_versions(parts[0], query)
        elif len(parts) == 2 and parts[1] == "invoke":
            self._handle_invoke_spec(parts[0])
        else:
            self._send_error_json(404, f"Endpoint desconocido: /api/models/{remainder}")

    def _reject_method(self) -> None:
        """Answer ``405`` for the methods the catalog does not implement."""
        self._start_request()
        try:
            allow = "GET, HEAD, POST, DELETE, OPTIONS"
            self._send_json(
                405,
                {
                    "success": False,
                    "error": f"Método no soportado: {self.command}",
                    "allow": allow,
                },
                {"Allow": allow},
            )
        finally:
            self._finish_request()

    do_PUT = _reject_method
    do_PATCH = _reject_method

    def _handle_health(self) -> None:
        counts = summarize(self.catalog)
        self._send_json(
            200,
            {
                "success": True,
                "status": "ok",
                "version": package_version(),
                "api_version": API_VERSION,
                "models": counts["total"],
                "categories": len(category_index(self.catalog)),
                "summary": counts,
                "playground": {
                    "enabled": bool(self.playground_enabled),
                    "timeout_seconds": PLAYGROUND_TIMEOUT,
                    "blocked_modules": sorted(PLAYGROUND_BLOCKED_MODULES),
                },
                "jobs": self.jobs.snapshot(),
                "sessions": self.sessions.snapshot(),
                "invoke": {
                    "enabled": bool(self.invoke_enabled),
                    "sessions_enabled": bool(self.sessions_enabled),
                    "timeout_seconds": INVOKE_TIMEOUT,
                    "max_concurrency": INVOKE_MAX_CONCURRENCY,
                    "side_effects_blocked": not INVOKE_ALLOW_SIDE_EFFECTS,
                    "allowed_methods": sorted(INVOKE_ALLOWED_METHODS),
                    "blocked_methods": sorted(INVOKE_BLOCKED_METHODS),
                },
                "catalog": {"reloads": self.store.reloads, "ttl_seconds": self.store.ttl},
                "metrics": self.metrics.snapshot(),
            },
        )

    def _handle_openapi(self) -> None:
        """Serve the OpenAPI document generated from the catalog."""
        self._send_cacheable_json(200, openapi_spec(self.catalog), "/api/openapi.json")

    def _handle_categories(self) -> None:
        categories = category_index(self.catalog)
        self._send_cacheable_json(
            200,
            {"success": True, "count": len(categories), "categories": categories},
            "/api/categories",
        )

    def _handle_list_models(self, query: Dict[str, List[str]]) -> None:
        try:
            limit, offset = parse_pagination(query)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        models = self._models(query)
        if limit is not None:
            page = models[offset : offset + limit]
        else:
            page = models[offset:] if offset else models
        has_more = offset + len(page) < len(models)
        headers: Dict[str, str] = {}
        next_url: Optional[str] = None
        if has_more and page:
            next_url = self._page_url(query, offset + len(page), limit)
            headers["Link"] = f'<{next_url}>; rel="next"'
        self._send_cacheable_json(
            200,
            {
                "success": True,
                "count": len(page),
                "total": len(self.catalog),
                "matched": len(models),
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "next": next_url,
                "categories": category_index(self.catalog),
                "models": [model_brief(model) for model in page],
            },
            f"/api/models?{urlencode(query, doseq=True)}",
            headers,
        )

    @staticmethod
    def _page_url(query: Dict[str, List[str]], offset: int, limit: Optional[int]) -> str:
        """Absolute path of the next page, preserving every other filter."""
        params: List[Tuple[str, str]] = []
        for key, values in query.items():
            if key in ("offset", "limit"):
                continue
            params.extend((key, value) for value in values)
        params.append(("offset", str(offset)))
        if limit is not None:
            params.append(("limit", str(limit)))
        return f"/api/models?{urlencode(params)}"

    def _handle_get_model(self, model_id: str) -> None:
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        self._send_cacheable_json(
            200,
            {"success": True, "model": model_payload(model)},
            f"/api/models/{model['id']}",
        )

    def _handle_model_source(self, model_id: str, query: Dict[str, List[str]]) -> None:
        """Serve the raw source of one model, its diff against a git ref, or a download."""
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        ref = (query.get("diff") or [""])[0].strip()
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        }
        if ref:
            diff = source_diff(model["path"], model["source"], ref)
            headers["X-Diff-Changes"] = str(diff["changes"])
            headers["X-Diff-Ref"] = ref
            self._send_json(200, {"success": True, "model_id": model["id"], **diff}, headers)
            return
        download = (query.get("download") or [""])[0].strip().lower()
        if download in {"1", "true", "yes", "download"}:
            headers["Content-Disposition"] = f'attachment; filename="{model["id"]}.py"'
        self._send_bytes(
            200,
            model["source"].encode("utf-8"),
            "text/x-python; charset=utf-8",
            headers,
        )

    def _handle_model_versions(self, model_id: str, query: Dict[str, List[str]]) -> None:
        """List the commits that touched the source file of ``model_id``."""
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        try:
            limit, _ = parse_pagination({**query, "limit": query.get("limit") or ["10"]})
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        versions = git_versions(model["path"], limit or 10)
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "path": model["path"],
                "count": len(versions),
                "versions": versions,
            },
        )

    def _handle_playground_spec(self, model_id: str) -> None:
        """Return the playground specification of one model."""
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "enabled": bool(self.playground_enabled),
                "playground": playground_spec(model),
            },
        )

    def _find_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``model_id`` of a JSON payload (``400`` when missing)."""
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            raise ValidationError("El campo 'model_id' es obligatorio")
        model = find_model(self.catalog, model_id)
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        return model

    def _model_or_404(self, model_id: str) -> Dict[str, Any]:
        """Resolve ``model_id`` of a path (``404`` when missing)."""
        model = find_model(self.catalog, (model_id or "").strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        return model

    def _prepare_run(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate one playground payload and return ``(model, parameters)``.

        Shared by the synchronous endpoint and the asynchronous jobs so both
        enforce exactly the same rules: known model, known parameter names and
        JSON values coercible to the declared types.
        """
        model = self._find_model(data)
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise ValidationError("El campo 'parameters' debe ser un objeto")
        return model, coerce_model_parameters(model, raw_parameters)

    def _handle_playground(self, data: Dict[str, Any]) -> None:
        """Construct a model with the parameters sent by the browser."""
        if not self.playground_enabled:
            raise APIError("El playground está deshabilitado en este servidor", status=403)
        model, values = self._prepare_run(data)
        try:
            result = run_playground(model, values)
        except PlaygroundError as exc:
            exc.extra.setdefault("model_id", model["id"])
            raise
        payload = {"success": True, "model_id": model["id"], "parameters": values}
        payload.update(result)
        self._send_json(200, payload)

    def _handle_job_create(self, body: Any) -> None:
        """Queue one run (or a batch) and answer ``202`` with the job id."""
        if not self.playground_enabled:
            raise APIError("El playground está deshabilitado en este servidor", status=403)
        if isinstance(body, list):
            entries: Any = body
        elif isinstance(body, dict) and ("jobs" in body or "batch" in body):
            entries = body.get("jobs") or body.get("batch")
        elif isinstance(body, dict) and ("model_id" in body or "id" in body):
            entries = [body]
        else:
            entries = []
        if not isinstance(entries, list) or not entries:
            raise ValidationError("El campo 'jobs' debe ser una lista no vacía")
        if len(entries) > PLAYGROUND_MAX_BATCH:
            raise ValidationError(f"El lote supera el máximo de {PLAYGROUND_MAX_BATCH} ejecuciones")
        runs: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValidationError(f"La ejecución {index} debe ser un objeto")
            try:
                model, values = self._prepare_run(entry)
            except APIError as exc:
                exc.extra.setdefault("run_index", index)
                raise
            runs.append({"model": model, "parameters": values})
        job = self.jobs.submit(runs)
        poll_url = f"/api/playground/jobs/{job.id}"
        self._send_json(
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

    def _job_payload(self, job: PlaygroundJob) -> Dict[str, Any]:
        """The JSON body of one job."""
        payload: Dict[str, Any] = {"success": True}
        payload.update(job.to_dict())
        payload["poll_url"] = f"/api/playground/jobs/{job.id}"
        return payload

    def _handle_job_get(self, job_id: str) -> None:
        """State (and results) of one asynchronous job."""
        job = self.jobs.get(job_id.strip())
        if job is None:
            raise ModelNotFound(f"Job no encontrado o expirado: {job_id}")
        self._send_json(200, self._job_payload(job))

    def _handle_job_list(self, query: Dict[str, List[str]]) -> None:
        """The most recent jobs, without their results."""
        try:
            limit, _ = parse_pagination({**query, "limit": query.get("limit") or ["20"]})
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        jobs = self.jobs.recent(limit or 20)
        self._send_json(
            200,
            {
                "success": True,
                "count": len(jobs),
                "jobs": [job.to_dict(include_results=False) for job in jobs],
            },
        )

    # -- invocación de métodos ---------------------------------------------

    def _require_invoke(self) -> None:
        if not self.invoke_enabled:
            raise APIError(
                "La invocación de métodos está deshabilitada en este servidor", status=403
            )

    def _require_sessions(self) -> None:
        self._require_invoke()
        if not self.sessions_enabled:
            raise APIError("Las sesiones están deshabilitadas en este servidor", status=403)

    def _acquire_invoke_slot(self) -> None:
        """Bound the number of concurrent ``/invoke`` child processes."""
        if not self.invoke_slots.acquire(blocking=False):
            raise RateLimitExceeded(
                "Hay demasiadas llamadas en curso; reintenta en unos segundos",
                headers={"Retry-After": "5"},
            )

    def _handle_invoke_spec(self, model_id: str) -> None:
        """``GET /api/models/<id>/invoke``: the methods a client may call."""
        model = self._model_or_404(model_id)
        self._send_json(200, {"success": True, **invoke_spec(model)})

    def _handle_sample(self, model_id: str, data: Dict[str, Any]) -> None:
        """``POST /api/models/<id>/sample``: dummy arguments for one method."""
        self._require_invoke()
        model = self._model_or_404(model_id)
        method = method_for_invocation(model, str(data.get("method") or ""))
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise ValidationError("El campo 'parameters' debe ser un objeto")
        parameters = resolved_parameters(model, raw_parameters)
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "method": method["name"],
                "parameters": parameters,
                "arguments": sample_arguments(model, method, parameters),
            },
        )

    def _handle_invoke(self, model_id: str, data: Dict[str, Any]) -> None:
        """``POST /api/models/<id>/invoke``: build the model and call a method."""
        self._require_invoke()
        model = self._model_or_404(model_id)
        method, arguments, parameters = prepare_invocation(model, data)
        self._acquire_invoke_slot()
        try:
            result = run_invoke(model, method, parameters, arguments)
        except PlaygroundError as exc:
            exc.extra.setdefault("model_id", model["id"])
            raise
        finally:
            self.invoke_slots.release()
        payload = {
            "success": True,
            "model_id": model["id"],
            "method": method["name"],
            "arguments": arguments,
            "parameters": parameters,
            "session_id": None,
        }
        payload.update(result)
        self._send_json(200, payload)

    def _handle_session_create(self, model_id: str, data: Dict[str, Any]) -> None:
        """``POST /api/models/<id>/sessions``: build once, keep the instance."""
        self._require_sessions()
        model = self._model_or_404(model_id)
        raw_parameters = data.get("parameters", data.get("params", {}))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise ValidationError("El campo 'parameters' debe ser un objeto")
        parameters = coerce_model_parameters(model, raw_parameters)
        session = self.sessions.create(model, parameters)
        location = f"/api/sessions/{session.id}"
        payload: Dict[str, Any] = {
            "success": True,
            "parameters": parameters,
            "invoke_url": f"{location}/invoke",
        }
        payload.update(session.to_dict())
        self._send_json(201, payload, {"Location": location, "Cache-Control": "no-store"})

    def _handle_session_get(self, session_id: str) -> None:
        """``GET /api/sessions/<id>``: state of one live session."""
        self._require_sessions()
        session = self.sessions.get(session_id.strip())
        if session is None:
            raise ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        payload: Dict[str, Any] = {"success": True}
        payload.update(session.to_dict(include_output=True))
        self._send_json(200, payload)

    def _handle_session_invoke(self, session_id: str, data: Dict[str, Any]) -> None:
        """``POST /api/sessions/<id>/invoke``: call a method on the live instance."""
        self._require_sessions()
        session = self.sessions.get(session_id.strip())
        if session is None:
            raise ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        method, arguments, _ = prepare_invocation(session.model, data, with_parameters=False)
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
        self._send_json(200, payload)

    def _handle_session_delete(self, session_id: str) -> None:
        """``DELETE /api/sessions/<id>``: stop the worker and forget the session."""
        self._require_sessions()
        if not self.sessions.drop(session_id.strip()):
            raise ModelNotFound(f"Sesión no encontrada o expirada: {session_id}")
        self._send_json(
            200, {"success": True, "session_id": session_id.strip(), "status": "closed"}
        )

    def _handle_session_list(self, _query: Dict[str, List[str]]) -> None:
        """``GET /api/sessions``: every open session."""
        self._require_sessions()
        sessions = self.sessions.recent()
        summary = self.sessions.snapshot()
        self._send_json(
            200,
            {
                "success": True,
                "count": len(sessions),
                "max_sessions": summary["max_sessions"],
                "ttl_seconds": summary["ttl_seconds"],
                "sessions": [session.to_dict() for session in sessions],
            },
        )

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

    def _notebook_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validated parameter values that become the widget defaults.

        Only values that ``coerce_playground_value`` accepts are kept, so a
        notebook can never embed a Python expression coming from the browser.
        """
        raw = data.get("parameters", data.get("params")) if isinstance(data, dict) else None
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValidationError("El campo 'parameters' debe ser un objeto")
        model = self._find_model(data)
        return coerce_model_parameters(model, raw, require_all=False)

    def _handle_notebook(self, data: Dict[str, Any]) -> None:
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            raise ValidationError("El campo 'model_id' es obligatorio")
        model = find_model(self.catalog, model_id)
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        parameters = self._notebook_parameters(data)
        notebook = notebook_generator.build_notebook(model, parameters or None)
        self._send_json(
            200,
            {
                "success": True,
                "model_id": model["id"],
                "filename": f"{model['id']}.ipynb",
                "parameters": parameters,
                "notebook": notebook,
            },
        )

    def _handle_notebook_download(self, model_id: str, query: Dict[str, List[str]]) -> None:
        model = find_model(self.catalog, model_id.strip())
        if model is None:
            raise ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        parameters: Dict[str, Any] = {}
        raw = (query.get("parameters") or [""])[0].strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"El parámetro 'parameters' no es JSON válido: {exc}"
                ) from exc
            parameters = self._notebook_parameters({"model_id": model["id"], "parameters": parsed})
        notebook = notebook_generator.build_notebook(model, parameters or None)
        body = json.dumps(notebook, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(
            200,
            body,
            f"{_NOTEBOOK_TYPE}; charset=utf-8",
            {"Content-Disposition": f'attachment; filename="{model["id"]}.ipynb"'},
        )

    def _deprecation_headers(self) -> Dict[str, str]:
        """``Deprecation``/``Sunset``/``Link`` for the legacy ``POST /api/search``."""
        return {
            "Deprecation": "true",
            "Sunset": SEARCH_ALIAS_SUNSET,
            "Link": '</api/search?q=...>; rel="successor-version"',
        }

    def _search_payload(
        self, term: str, tags: Sequence[str], limit: int, deprecated: bool = False
    ) -> Dict[str, Any]:
        """Search results plus the category/tag facets of the same query."""
        matches = search_models(self.catalog, term, tags)
        # The facets are computed *before* the tag filter, so the counters tell
        # how many models each tag would keep.
        base = search_models(self.catalog, term)
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
            "results": [model_brief(model) for model in selected],
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

    def _handle_search(self, data: Dict[str, Any], deprecated: bool = False) -> None:
        query = str(data.get("query") or data.get("q") or "").strip()
        if not query:
            raise ValidationError("El campo 'query' es obligatorio")
        tags = data.get("tags") or []
        payload = self._search_payload(query, tags, limit=0, deprecated=deprecated)
        headers = self._deprecation_headers() if deprecated else {}
        self._send_json(200, payload, headers)

    def _handle_search_query(self, query: Dict[str, List[str]]) -> None:
        """Search endpoint used by the catalog autocomplete."""
        term = (query.get("q") or query.get("query") or [""])[0].strip()
        if not term:
            raise ValidationError("El parámetro 'q' es obligatorio")
        try:
            limit, _ = parse_pagination({**query, "limit": query.get("limit") or ["10"]})
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        tags = query.get("tags") or []
        self._send_json(200, self._search_payload(term, tags, limit=limit or 10))

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

    def server_close(self) -> None:
        """Release the sockets, the job workers and the live model sessions."""
        try:
            JOBS.shutdown()
        finally:
            try:
                SESSIONS.shutdown()
            finally:
                super().server_close()


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
        f"Catálogo Likelihood v{package_version()} (API v{API_VERSION})",
        f"Modelos descubiertos : {counts['total']} en {len(category_index(CATALOG))} categorías",
        f"Catálogo web         : http://{address}:{actual_port}/",
        f"API                  : http://{address}:{actual_port}/api/models",
        f"Documentación        : http://{address}:{actual_port}/docs",
        "Playground           : "
        + (
            f"http://{address}:{actual_port}/api/playground"
            if ModelAPIHandler.playground_enabled
            else "deshabilitado"
        ),
        "Seguridad            : "
        + ", ".join(
            [
                f"token={'sí' if ModelAPIHandler.api_token else 'no'}",
                f"rate_limit={RATE_LIMITER.limit or 'no'}",
                f"cors={','.join(ModelAPIHandler.cors_origins) or 'no'}",
            ]
        ),
        f"Recarga de catálogo  : {'cada %gs' % CATALOG_TTL if CATALOG_TTL > 0 else 'desactivada'}",
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

    global CATALOG, MODELS_BY_ID, RATE_LIMITER
    if arguments.package_dir is not None:
        CATALOG_STORE.package_dir = Path(arguments.package_dir)
        CATALOG = CATALOG_STORE.load()
        MODELS_BY_ID = CATALOG_STORE.by_id
        ModelAPIHandler.catalog = CATALOG
        ModelAPIHandler.store = CATALOG_STORE
    if arguments.no_playground:
        ModelAPIHandler.playground_enabled = False
    if arguments.no_invoke:
        ModelAPIHandler.invoke_enabled = False
        ModelAPIHandler.sessions_enabled = False
    if arguments.token is not None:
        ModelAPIHandler.api_token = arguments.token.strip()
    if arguments.cors_origin:
        ModelAPIHandler.cors_origins = tuple(
            origin.strip().rstrip("/") for origin in arguments.cors_origin if origin.strip()
        )
    if arguments.rate_limit is not None:
        RATE_LIMITER = RateLimiter(arguments.rate_limit)
        ModelAPIHandler.rate_limiter = RATE_LIMITER

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
