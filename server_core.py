"""Shared domain layer of the Likelihood catalogue server.

This module keeps the code that is independent of the HTTP framework: the
introspected catalog, the sandboxed playground, method invocation, live sessions,
asynchronous jobs, notebook/preview rendering and the helpers they need.  The
FastAPI application in :mod:`server_fastapi` builds its routes on top of it.

The standard-library HTTP server that used to live here was removed; run the
catalogue with :mod:`server_fastapi` instead::

    uvicorn server_fastapi:app
    python3 server_fastapi.py
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import html
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

from model_catalog import (
    DEFAULT_PACKAGE_DIR,
    REPO_ROOT,
    build_catalog,
    category_meta,
    guess_widget,
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
#: Smallest response that is compressed with gzip, in bytes
#: (``LM_GZIP_MIN_BYTES``; ``0`` disables the compression middleware).
_GZIP_MIN_RAW = (os.environ.get("LM_GZIP_MIN_BYTES") or "").strip()
GZIP_MIN_BYTES = max(0, int(_GZIP_MIN_RAW)) if _GZIP_MIN_RAW else 1024
#: Honour ``X-Forwarded-For`` when identifying the client (``LM_TRUST_PROXY``).
#: Enable it **only** behind a reverse proxy that overwrites the header,
#: otherwise a client could spoof its address to escape the rate limiter.
TRUST_PROXY = _env_flag("LM_TRUST_PROXY", False)
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

    The catalog used to be built once at import time, so editing a model forced a restart.  The
    store remembers the ``mtime`` of every source file and rebuilds the catalog when one of them
    changes, at most once every ``ttl`` seconds.
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


def _prom_label(value: Any) -> str:
    """Escape ``value`` so it can be used inside a Prometheus label."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


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

        Unknown labels beyond ``max_endpoints`` are folded into ``other`` so an arbitrary amount of
        distinct URLs cannot grow the counters forever.
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

    def prometheus(self, namespace: str = "likelihood") -> str:
        """The same counters in the Prometheus text exposition format.

        Every series carries the ``endpoint`` label (``GET /api/models``), which is the coarse
        bucket of :func:`endpoint_label`, so the cardinality stays bounded no matter which model ids
        are requested.
        """
        snapshot = self.snapshot()
        lines: List[str] = []

        def series(name: str, kind: str, help_text: str) -> None:
            lines.append(f"# HELP {namespace}_{name} {help_text}")
            lines.append(f"# TYPE {namespace}_{name} {kind}")

        series("uptime_seconds", "gauge", "Seconds since the server started.")
        lines.append(f"{namespace}_uptime_seconds {snapshot['uptime_seconds']}")

        series("requests_total", "counter", "Requests served, per endpoint.")
        for name, entry in snapshot["endpoints"].items():
            lines.append(
                f'{namespace}_requests_total{{endpoint="{_prom_label(name)}"}} {entry["count"]}'
            )

        series("request_errors_total", "counter", "Responses with a 4xx/5xx status.")
        for name, entry in snapshot["endpoints"].items():
            lines.append(
                f'{namespace}_request_errors_total{{endpoint="{_prom_label(name)}"}} '
                f'{entry["errors"]}'
            )

        series("request_duration_seconds", "summary", "Latency of the served requests.")
        for name, entry in snapshot["endpoints"].items():
            label = _prom_label(name)
            for quantile, key in ((0.5, "p50"), (0.95, "p95")):
                lines.append(
                    f'{namespace}_request_duration_seconds{{endpoint="{label}",'
                    f'quantile="{quantile}"}} {entry["latency_ms"][key] / 1000.0}'
                )
            lines.append(
                f'{namespace}_request_duration_seconds_max{{endpoint="{label}"}} '
                f'{entry["latency_ms"]["max"] / 1000.0}'
            )

        series("responses_total", "counter", "Responses grouped by HTTP status code.")
        for status, count in snapshot["statuses"].items():
            lines.append(f'{namespace}_responses_total{{status="{_prom_label(status)}"}} {count}')

        return "\n".join(lines) + "\n"


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

    ``/api/models/Arima`` and ``/api/models/HMM`` are reported together as ``GET
    /api/models/<model_id>`` so the counters stay readable.
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
        "import_statement": model.get("import_statement")
        or (f"from {model['module']} import {model['id']}"),
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

    The payload is derived from the constructor parameters (number of states, window size,
    features...) so ``fit``/``predict``/``forward`` can be tried from the invoke tab without writing
    JSON by hand.  It is only a sample: it never evaluates code and the caller is expected to edit
    it.
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

    The catalog records Python source expressions (``np.array([])``, ``[]``), which are not JSON
    values, and the parameters of a method do not record a sample at all.  The value prefers the
    default (what the model really uses), then a name aware scalar, then the catalog sample, and
    finally a small payload shaped from the constructor ``parameters``.
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

    ``env``/``model`` style parameters receive Python objects, so the model can only be built from
    Python (or from a session created in Python); the API says so instead of failing later with a
    confusing missing-field error.
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

    Every parameter takes its default; required ones fall back to the generated sample
    (``Arima.datapoints`` becomes a small matrix, for instance).
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

    ``limit`` is ``None`` when the caller did not ask for pagination, which keeps the endpoint
    backwards compatible.
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
constructor_parameters = payload.get("parameters") or {}


def _convert(name, value, hints):
    # JSON has no arrays: turn lists into numpy/pandas objects when the catalog
    # says the argument is a container of that kind, and fall back otherwise.
    hint = (hints.get(name) or "").lower()
    if not isinstance(value, list) or not hint:
        return value
    if "tuple" in hint:
        # ``input_shape`` and friends are annotated as tuples: Keras inspects the
        # type, so a JSON list must become a real tuple.
        return tuple(value)
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
    for attribute in ("tolist", "to_dict", "item", "numpy"):
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


#: Methods that build the instance themselves (or only inspect it): calling
#: ``build`` before them would create the Keras layers twice and make an explicit
#: ``build`` fail with "You cannot add new elements of state ... already built".
_NO_AUTO_BUILD = frozenset({"build", "build_encoder_decoder", "get_config", "from_config"})


def _default_input_shape():
    # Prefer the constructor hint (``input_shape_parm``/``n_features``) over the
    # generic one-feature shape, so ``build`` receives a coherent shape even when
    # the method has no array argument.
    for name in ("input_shape_parm", "input_shape", "n_features", "num_features", "n_columns"):
        value = constructor_parameters.get(name)
        if isinstance(value, int) and value > 0:
            return (None, value)
    return (None, 1)


def _prepare(instance, method_name, arguments):
    # Keras models may not set ``built`` from their own ``build`` override, so a
    # private marker keeps the instance prepared across session calls (rebuilding
    # it would throw away the trained weights of the classifier).
    if method_name in _NO_AUTO_BUILD or not hasattr(instance, "built"):
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
        shape = _default_input_shape()
    try:
        build(shape)
    except BaseException:
        # The real call reports the original error with a clearer message.
        return
    try:
        instance._lm_prepared = True
    except BaseException:
        pass


def _mark_prepared(instance, method_name):
    # An explicit ``build`` leaves the instance ready for the following calls.
    if method_name != "build":
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
    _mark_prepared(instance, payload["method"])
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
        if request["method"] == "build" and (
            getattr(instance, "_lm_prepared", False) or getattr(instance, "built", False)
        ):
            # An explicit ``build`` on an instance that is already prepared cannot
            # add its layers again.  Recreate it from the session constructor
            # parameters so ``build`` starts from a clean object.
            instance = _build()
        _prepare(instance, request["method"], arguments)
        method = getattr(instance, request["method"])
        value = method(**arguments)
        _mark_prepared(instance, request["method"])
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

    The values are absolute (bytes, seconds, file descriptors) and are only *lowered*, never raised:
    the child can never get more than the parent, so a leaked limit cannot weaken the server.
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

    ``parameters`` are the constructor values the model will be built with; they shape the generated
    samples (``n_states``, ``window``, ``n_features``).
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

    ``parameters`` shape the payload exactly like the constructor that will be used when the method
    is invoked (states, window, features...).
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


def parameters_signature(parameters: Dict[str, Any]) -> str:
    """Stable signature of the constructor values that built a session.

    Constructor values are JSON literals (the same objects the playground validates), so sorting the
    keys is enough to compare a payload with the values of a live instance without depending on the
    key order.
    """
    return json.dumps(parameters or {}, sort_keys=True, default=str, ensure_ascii=False)


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
        parameters: Dict[str, Any],
        process: "subprocess.Popen[str]",
        timeout: float,
    ) -> None:
        self.id = session_id
        self.model = model
        self.parameters = dict(parameters or {})
        self.parameters_signature = parameters_signature(self.parameters)
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
            "parameters": self.parameters,
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
    session = ModelSession(session_id, model, parameters, process, timeout)
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

    def replace(self, session: ModelSession, parameters: Dict[str, Any]) -> ModelSession:
        """Rebuild ``session`` with new constructor values.

        A live instance cannot change its constructor arguments, so the worker is replaced by one
        created with ``parameters`` and the old id stops working. The previous session is detached
        while the replacement is built and put back if the constructor fails, so a bad payload never
        kills a valid session.
        """
        with self._lock:
            previous = self._sessions.pop(session.id, None)
        try:
            replacement = self.create(session.model, parameters)
        except BaseException:
            if previous is not None:
                with self._lock:
                    self._sessions.setdefault(previous.id, previous)
            raise
        if previous is not None:
            previous.close()
        return replacement

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

    ``POST /api/playground/jobs`` returns its id immediately; the client polls ``GET
    /api/playground/jobs/<id>`` until ``status`` leaves ``pending``/ ``running``.  A pool of worker
    threads keeps the cost of the background machinery bounded while every model still runs in its
    own child process.
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
        self.cancel_requested = False

    @property
    def batch(self) -> bool:
        """True when the job groups more than one run."""
        return len(self.runs) > 1

    @property
    def done(self) -> bool:
        """True once the job reached a final state."""
        return self.status in {"succeeded", "failed", "partial", "cancelled"}

    @property
    def active(self) -> bool:
        """True while the job still has work to do."""
        return self.status in {"pending", "running"}

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
            "cancel_requested": self.cancel_requested,
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
            if job.cancel_requested:
                break
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
        if job.cancel_requested:
            job.status = "cancelled"
            job.error = job.error or f"Job cancelado tras {successes} de {total} ejecuciones"
        elif successes == total:
            job.status = "succeeded"
        elif successes:
            job.status = "partial"
        else:
            job.status = "failed"
            errors = [result.get("error", "") for result in job.results if result.get("error")]
            job.error = "; ".join(errors)[:500]
        job.finished_at = time.time()

    def cancel(self, job_id: str) -> Optional[PlaygroundJob]:
        """Ask ``job_id`` to stop; ``None`` when the job is unknown.

        A job that has not started yet is marked ``cancelled`` right away.  A running job keeps the
        outcome of the run in flight and stops before the next one, because a model executes in its
        own child process and is not interrupted half way.  Finished jobs are returned untouched
        (``DELETE`` is idempotent, and :meth:`forget` removes them from the store).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.active:
                job.cancel_requested = True
                if job.status == "pending":
                    job.status = "cancelled"
                    job.finished_at = time.time()
            return job

    def forget(self, job_id: str) -> Optional[PlaygroundJob]:
        """Remove a *finished* job from the store; the job object is returned."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.active:
                return job
            return self._jobs.pop(job_id, None)

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

    Raises :class:`APIError` with status ``503`` when git is unavailable or the path has no history,
    which is the honest answer for that endpoint.
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


def _escape(value: Any) -> str:
    """HTML-escape ``value`` for use inside markup."""
    return html.escape(str(value), quote=True)


def tag_url(tag: str, back_url: str = "/") -> str:
    """Catalogue URL that filters by ``tag`` (``/?tag=Deep%20Learning``)."""
    base = back_url.split("?")[0] or "/"
    return f"{base}?tag={quote(tag, safe='')}"


def tags_html(model: Dict[str, Any], back_url: str = "/") -> str:
    """Links to the catalogue filtered by each tag of the model."""
    return "".join(
        f'<a class="tag" href="{_escape(tag_url(tag, back_url))}"'
        f' title="Ver los modelos con la etiqueta {_escape(tag)}">{_escape(tag)}</a>'
        for tag in model["tags"]
    )


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

    Rendering the heading together with its content means the template never has to hide empty
    sections with JavaScript.
    """
    cards = attributes_html(model)
    if not cards:
        return ""
    return f'<h2>Atributos</h2>\n<div class="grid">{cards}</div>'


def related_html(entries: Sequence[Dict[str, Any]]) -> str:
    """Section with links to the models related to the current one.

    ``entries`` are the records of :func:`model_catalog.related_models`, so the shared tags (or the
    category, when there are none) tell *why* a model is being suggested.  An empty list renders
    nothing at all.
    """
    cards: List[str] = []
    for entry in entries:
        model = entry["model"]
        shared = list(entry.get("shared_tags") or [])
        hint = "Comparte: " + ", ".join(shared[:3]) if shared else model["category_label"]
        cards.append(
            f'<a class="related-card" href="/preview/{quote(model["id"], safe="")}">'
            f'<span class="related-icon" aria-hidden="true">{_escape(model["icon"])}</span>'
            '<span class="related-body">'
            f'<span class="related-title">{_escape(model["title"])}</span>'
            f'<span class="related-desc">{_escape(model["short_description"])}</span>'
            f'<span class="related-hint">{_escape(hint)}</span>'
            "</span></a>"
        )
    if not cards:
        return ""
    return (
        '<h2 id="relacionados">Modelos relacionados</h2>\n<div class="related-grid">'
        + ("".join(cards))
        + "</div>"
    )


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
    related: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Render the HTML preview of ``model`` from ``template``.

    Placeholders are plain ``{{NAME}}`` tokens replaced through ``str.replace``; this keeps the
    CSS/JS braces of the template untouched and avoids the ``str.format`` escaping problems of
    inline f-strings.
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
        "MODEL_TAGS": tags_html(model, back_url),
        "MODEL_PARAMETERS": parameters_html(model),
        "MODEL_ATTRIBUTES_SECTION": attributes_section_html(model),
        "MODEL_METHODS": methods_html(model),
        "MODEL_RELATED": related_html(list(related or [])),
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


_NOTEBOOK_TYPE = "application/x-ipynb+json"


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

    Returns ``None`` when the path escapes ``root``, points at a hidden file or does not resolve to
    a readable file.  This replaces the naive ``base / user_input`` joins that allowed directory
    traversal.
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
