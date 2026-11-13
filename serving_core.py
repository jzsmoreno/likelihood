"""Model serving layer of the Likelihood catalogue server.

The endpoint store keeps the deployment records, runs the inference calls through the
existing child-process sandbox of :mod:`server_core` and records one monitoring event
per request.  Nothing in this module duplicates the model loading or invocation logic:
a deployment is a persisted configuration that is validated with ``prepare_load`` /
``run_load`` and served with ``run_invoke``.

Design notes
------------
* A deployment is stateless at the process level: every prediction starts a fresh
  child interpreter, exactly like ``POST /api/models/<id>/invoke``.  That keeps the
  catalogue safe to expose and makes a restart of the server cheap; the deployment
  record (configuration, status and logs) is persisted in ``deployments.json``.
* The monitoring data comes from ``events.jsonl``: one JSON line per request with the
  real latency, status, endpoint, model and error.  No metric is invented.
* Saved artifacts are validated while the deployment is created (status
  ``deploying``); the request never blocks on a cold TensorFlow import because the
  validation runs in a background thread and the UI polls the status.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import server_core as core

LOGGER = logging.getLogger("likelihood.serving")

#: Directory that keeps ``deployments.json`` and ``events.jsonl``.  Tests inject a
#: temporary directory; production can point ``LM_SERVING_STORE`` to a volume.
STORE_DIR = Path(
    os.environ.get("LM_SERVING_STORE") or (core.REPO_ROOT / "output" / "serving")
).expanduser()

#: Maximum number of requests kept in memory and replayed on startup.
MAX_EVENTS = max(100, int(os.environ.get("LM_SERVING_MAX_EVENTS") or 20000))
#: Set ``LM_SERVING_LOG_INPUTS=0`` to stop storing the short input preview.
LOG_INPUTS = core._env_flag("LM_SERVING_LOG_INPUTS", True)
#: Seconds a deployment may spend loading the saved artifact.
DEPLOY_TIMEOUT = float(os.environ.get("LM_SERVING_DEPLOY_TIMEOUT") or 600)
#: Default per-request timeout of an endpoint (``LM_SERVING_TIMEOUT``).
DEFAULT_TIMEOUT = float(os.environ.get("LM_SERVING_TIMEOUT") or core.INVOKE_TIMEOUT)
#: Default number of concurrent predictions per endpoint.
DEFAULT_CONCURRENCY = max(1, int(os.environ.get("LM_SERVING_CONCURRENCY") or 1))
#: Upper bound accepted for ``max_concurrency``.
MAX_CONCURRENCY = 32
#: Background workers that validate/load deployments.
DEPLOY_WORKERS = max(1, int(os.environ.get("LM_SERVING_WORKERS") or 2))

_ENDPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,62}$")
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Keys of the prediction body that are part of the request envelope, not arguments.
_RESERVED_BODY_KEYS = frozenset(
    {"arguments", "args", "inputs", "input", "timeout", "timeout_seconds", "metadata"}
)

#: Methods that train, persist or prepare a model instead of serving predictions.
#: They are rejected by the deployment validator (the existing invocation policy
#: already blocks the filesystem side effects such as ``save``/``load``).
NON_INFERENCE_METHODS = frozenset(
    {
        "build",
        "build_encoder_decoder",
        "compile",
        "fit",
        "fit_predict",
        "fit_transform",
        "partial_fit",
        "train",
        "train_on_batch",
        "get_config",
        "get_params",
        "set_params",
        "apply_initialization",
        "play_one_step",
        "freeze_encoder_decoder",
        "unfreeze_encoder_decoder",
        "get_digraph",
    }
)

#: Prefixes that clearly describe setup, training or persistence rather than inference.
NON_INFERENCE_PREFIXES = (
    "fit",
    "train",
    "build",
    "compile",
    "save",
    "load",
    "plot",
    "render",
    "viz",
    "freeze",
    "unfreeze",
    "set_",
)

#: Name hints that identify an inference method in the UI and the documentation.
INFERENCE_HINTS = (
    "predict",
    "predict_proba",
    "infer",
    "forecast",
    "score",
    "decision_function",
    "transform",
    "forward",
    "call",
    "sample",
    "generate",
    "encode",
    "decode",
    "simulate",
    "step",
    "act",
    "probability",
    "anomaly",
)

#: Human labels and tone classes used by the UI and the JSON payloads.
STATUS_LABELS = {
    "deploying": ("Desplegando", "progress"),
    "running": ("En ejecución", "success"),
    "stopped": ("Detenido", "muted"),
    "failed": ("Fallido", "danger"),
}


def utc_now() -> datetime:
    """Return the current UTC time with a second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment: Optional[datetime] = None) -> str:
    """Format ``moment`` (now by default) as an ISO-8601 UTC string."""
    return (moment or utc_now()).strftime(_ISO_FORMAT)


def parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp, accepting the trailing ``Z``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile matching ``Metrics._percentile``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 2)


def slugify(value: str) -> str:
    """Turn free text into an endpoint id (``Modelo Arima`` -> ``modelo-arima``)."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return cleaned[:63].strip("-") or "endpoint"


def is_inference_method(name: str) -> bool:
    """Whether ``name`` can be exposed as an inference method.

    Every public, JSON-callable method can be served, except the ones that train,
    compile, build, render or persist a model.
    """
    cleaned = (name or "").strip().lower()
    if not cleaned or cleaned in NON_INFERENCE_METHODS:
        return False
    if cleaned.startswith(NON_INFERENCE_PREFIXES):
        return False
    return True


def has_inference_hint(name: str) -> bool:
    """Whether the name looks like a prediction/transform operation.

    The hint only drives presentation (the "inferencia" badge); it never hides a
    callable method.
    """
    cleaned = (name or "").strip().lower()
    return any(hint in cleaned for hint in INFERENCE_HINTS)


def _json_bytes(value: Any) -> int:
    """Size of the JSON representation of ``value`` (best effort)."""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _preview(value: Any, limit: int = 180) -> str:
    """Compact, truncated representation used by the monitoring event."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    return text[:limit]


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (readers never see a half file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class EventLog:
    """Bounded, append-only request log behind the monitoring dashboard."""

    def __init__(self, directory: Path, limit: int = MAX_EVENTS) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "events.jsonl"
        self.limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.limit)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        self._events.append(event)
        except OSError as exc:  # pragma: no cover - unreadable volume
            LOGGER.warning("No se pudo leer %s: %s", self.path, exc)

    def record(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append ``event`` to memory and to disk, returning the stored record."""
        stored = dict(event)
        stored.setdefault("id", uuid.uuid4().hex[:12])
        stored.setdefault("timestamp", time.time())
        stored.setdefault("at", iso())
        with self._lock:
            self._events.append(stored)
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(stored, ensure_ascii=False))
                    handle.write("\n")
                if self.path.stat().st_size > 16 * 1024 * 1024:
                    self._rotate_locked()
            except OSError as exc:  # pragma: no cover - disk full / read only
                LOGGER.warning("No se pudo persistir el evento de serving: %s", exc)
        return stored

    def _rotate_locked(self) -> None:
        """Rewrite the file with the in-memory tail (bounded growth)."""
        try:
            _atomic_write(
                self.path,
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self._events),
            )
        except OSError as exc:  # pragma: no cover - defensive
            LOGGER.warning("No se pudo rotar %s: %s", self.path, exc)

    def events(self) -> List[Dict[str, Any]]:
        """A snapshot of every stored event, oldest first."""
        with self._lock:
            return list(self._events)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """The newest ``limit`` events, newest first."""
        with self._lock:
            return list(reversed(list(self._events)[-max(1, limit) :]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class ServingStore:
    """Deployment registry, lifecycle and inference gateway."""

    def __init__(self, directory: Optional[Path] = None, max_events: Optional[int] = None) -> None:
        self.directory = Path(directory or STORE_DIR)
        self.deployments_path = self.directory / "deployments.json"
        self.events = EventLog(self.directory, max_events or MAX_EVENTS)
        self._lock = threading.RLock()
        self._deployments: Dict[str, Dict[str, Any]] = {}
        self._semaphores: Dict[str, threading.BoundedSemaphore] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=DEPLOY_WORKERS, thread_name_prefix="lm-serving"
        )
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.deployments_path.is_file():
            return
        try:
            with open(self.deployments_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupt store
            LOGGER.warning("No se pudo leer %s: %s", self.deployments_path, exc)
            return
        records = data.get("deployments") if isinstance(data, dict) else data
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict) or not record.get("id"):
                continue
            deployment = self._normalise_record(record)
            if deployment["status"] == "deploying":
                # No worker survived the restart: the deployment never finished.
                deployment["status"] = "failed"
                deployment["last_error"] = (
                    "El servidor se reinició mientras el endpoint se estaba desplegando"
                )
                deployment["updated_at"] = iso()
                deployment["log"].append(
                    {"at": iso(), "level": "error", "message": deployment["last_error"]}
                )
            self._deployments[deployment["id"]] = deployment
            self._semaphore(deployment)

    @staticmethod
    def _normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Fill the fields a deployment persisted by an older version may miss."""
        return {
            "id": str(record.get("id")),
            "name": str(record.get("name") or record.get("id")),
            "description": str(record.get("description") or ""),
            "model_id": str(record.get("model_id") or ""),
            "model_name": str(record.get("model_name") or record.get("model_id") or ""),
            "model_title": str(record.get("model_title") or record.get("model_name") or ""),
            "model_icon": str(record.get("model_icon") or "📦"),
            "model_category": str(record.get("model_category") or ""),
            "model_module": str(record.get("model_module") or ""),
            "model_signature": str(record.get("model_signature") or ""),
            "method": str(record.get("method") or ""),
            "method_signature": str(record.get("method_signature") or ""),
            "method_description": str(record.get("method_description") or ""),
            "method_fields": list(record.get("method_fields") or []),
            "parameters": dict(record.get("parameters") or {}),
            "load": dict(record.get("load") or {"enabled": False}),
            "config": {
                "timeout_seconds": float(
                    (record.get("config") or {}).get("timeout_seconds") or DEFAULT_TIMEOUT
                ),
                "max_concurrency": int(
                    (record.get("config") or {}).get("max_concurrency") or DEFAULT_CONCURRENCY
                ),
            },
            "status": str(record.get("status") or "failed"),
            "created_at": str(record.get("created_at") or iso()),
            "updated_at": str(record.get("updated_at") or record.get("created_at") or iso()),
            "deployed_at": record.get("deployed_at"),
            "stopped_at": record.get("stopped_at"),
            "revision": int(record.get("revision") or 1),
            "last_error": str(record.get("last_error") or ""),
            "loaded_artifact": record.get("loaded_artifact"),
            "log": list(record.get("log") or []),
        }

    def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "updated_at": iso(),
            "deployments": [self._deployments[key] for key in sorted(self._deployments)],
        }
        try:
            _atomic_write(
                self.deployments_path,
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            )
        except OSError as exc:  # pragma: no cover - read only volume
            LOGGER.warning("No se pudo persistir %s: %s", self.deployments_path, exc)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _public(deployment: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of ``deployment`` so callers cannot mutate the registry."""
        return json.loads(json.dumps(deployment, ensure_ascii=False, default=str))

    def _semaphore(self, deployment: Dict[str, Any]) -> threading.BoundedSemaphore:
        concurrency = max(1, int(deployment["config"]["max_concurrency"]))
        semaphore = self._semaphores.get(deployment["id"])
        if semaphore is None or getattr(semaphore, "_lm_limit", None) != concurrency:
            semaphore = threading.BoundedSemaphore(concurrency)
            setattr(semaphore, "_lm_limit", concurrency)
            self._semaphores[deployment["id"]] = semaphore
        return semaphore

    @staticmethod
    def _log(deployment: Dict[str, Any], message: str, level: str = "info") -> None:
        deployment["log"] = (
            list(deployment.get("log") or [])
            + [{"at": iso(), "level": level, "message": str(message)[:2000]}]
        )[-40:]
        deployment["updated_at"] = iso()

    def _find(self, endpoint_id: str) -> Dict[str, Any]:
        deployment = self._deployments.get(str(endpoint_id or "").strip())
        if deployment is None:
            raise core.ModelNotFound(f"Endpoint no encontrado: {endpoint_id}")
        return deployment

    def get(self, endpoint_id: str) -> Dict[str, Any]:
        """Return a copy of one deployment (``404`` when missing)."""
        with self._lock:
            return self._public(self._find(endpoint_id))

    def list(self) -> List[Dict[str, Any]]:
        """Every deployment, most recently created first."""
        with self._lock:
            records = [self._public(item) for item in self._deployments.values()]
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return records

    def snapshot(self) -> Dict[str, Any]:
        """Small status report embedded in ``GET /api/health``."""
        with self._lock:
            statuses: Dict[str, int] = {}
            for deployment in self._deployments.values():
                statuses[deployment["status"]] = statuses.get(deployment["status"], 0) + 1
            return {
                "endpoints": len(self._deployments),
                "statuses": statuses,
                "events": len(self.events),
                "store": str(self.directory),
            }

    # -- validation -------------------------------------------------------

    def validate(self, model: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one deployment payload and return its normalised form.

        Raises
        ------
        core.APIError
            With a ``400``/``404``/``422`` status when a field, the method, a
            constructor argument or the saved-model path is not usable.
        """
        if not isinstance(data, dict):
            raise core.ValidationError("La configuración del endpoint debe ser un objeto")

        requested_name = str(data.get("endpoint_name") or data.get("name") or "").strip()
        method_name = str(data.get("method") or "").strip()
        if not method_name:
            raise core.ValidationError("El campo 'method' es obligatorio")
        method = core.method_for_invocation(model, method_name)
        if not is_inference_method(method["name"]):
            raise core.ValidationError(
                f'El método "{method["name"]}" entrena o prepara el modelo; '
                "selecciona un método de inferencia (predict, forward, transform…)"
            )
        if not core.method_is_jsonable(method):
            raise core.ValidationError(
                f'El método "{method["name"]}" tiene argumentos que no se pueden enviar como JSON'
            )

        name = requested_name or f"{slugify(model['id'])}-{slugify(method['name'])}"
        if not _ENDPOINT_NAME.match(name):
            raise core.ValidationError(
                "El nombre del endpoint debe tener entre 2 y 63 caracteres y usar solo "
                "letras, números, puntos, guiones y guiones bajos"
            )

        description = str(data.get("description") or "").strip()[:500]
        raw_parameters = data.get("parameters", data.get("params", {})) or {}
        if not isinstance(raw_parameters, dict):
            raise core.ValidationError("El campo 'parameters' debe ser un objeto")

        load = _normalise_load(data.get("load"), raw_parameters, model)
        warnings: List[str] = []
        if load["enabled"]:
            prepared = core.prepare_load(
                model,
                {
                    "loader": load["loader"],
                    "path": load["path"],
                    "parameters": load["parameters"],
                },
            )
            load.update(
                {
                    "loader": prepared["name"],
                    "path": prepared["path"],
                    "path_parameter": prepared.get("path_parameter"),
                    "parameters": prepared["parameters"],
                    "requires_instance": bool(prepared["requires_instance"]),
                }
            )
            parameters = prepared["parameters"]
            artifact = _artifact_path(prepared["path"])
            if not artifact.exists():
                warnings.append(
                    f'La ruta "{prepared["path"]}" no existe todavía; el despliegue '
                    "fallará al intentar cargar el modelo."
                )
        else:
            parameters = core.coerce_model_parameters(model, raw_parameters, require_all=True)

        config = _normalise_config(data.get("config"))

        return {
            "name": name,
            "description": description,
            "model_id": model["id"],
            "model_name": model["name"],
            "model_title": model.get("title") or model["name"],
            "model_icon": model.get("icon") or "📦",
            "model_category": model.get("category") or "",
            "model_module": model.get("module") or "",
            "model_signature": model.get("signature") or "",
            "method": method["name"],
            "method_signature": method.get("signature") or method["name"],
            "method_description": method.get("description") or "",
            "method_fields": core.method_fields(method, parameters),
            "parameters": parameters,
            "load": load,
            "config": config,
            "_method": method,
            "warnings": warnings,
        }

    def preview(self, model: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Validation result for ``POST /api/serving/validate`` (never raises)."""
        try:
            form = self.validate(model, data)
        except core.APIError as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
                "error": str(exc),
                "extra": exc.extra,
            }
        fields = form["method_fields"]
        examples = {
            field["name"]: field.get("sample")
            for field in fields
            if field.get("sample") is not None
        }
        if len(fields) == 1 and fields[0].get("sample") is not None:
            payload_example: Dict[str, Any] = {"input": fields[0]["sample"]}
        else:
            payload_example = {"inputs": examples}
        return {
            "valid": True,
            "errors": [],
            "warnings": form["warnings"],
            "name": form["name"],
            "method": {
                "name": form["method"],
                "signature": form["method_signature"],
                "description": form["method_description"],
                "fields": fields,
            },
            "parameters": form["parameters"],
            "load": form["load"],
            "config": form["config"],
            "payload_example": payload_example,
        }

    # -- lifecycle --------------------------------------------------------

    def create(self, model: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a deployment and validate/load it in the background (``202``)."""
        form = self.validate(model, data)
        warnings = form.pop("warnings", [])
        form.pop("_method", None)
        with self._lock:
            for existing in self._deployments.values():
                if existing["name"].lower() == form["name"].lower():
                    raise core.ValidationError(f'Ya existe un endpoint llamado "{form["name"]}"')
            identifier = slugify(form["name"])
            base = identifier
            counter = 2
            while identifier in self._deployments:
                identifier = f"{base}-{counter}"
                counter += 1
            deployment = {
                **form,
                "id": identifier,
                "status": "deploying",
                "created_at": iso(),
                "updated_at": iso(),
                "deployed_at": None,
                "stopped_at": None,
                "revision": 1,
                "last_error": "",
                "loaded_artifact": None,
                "log": [],
            }
            self._log(deployment, "Configuración validada; iniciando el despliegue")
            for warning in warnings:
                self._log(deployment, warning, level="warn")
            self._deployments[identifier] = deployment
            self._semaphore(deployment)
            self._persist_locked()
            result = self._public(deployment)
            self._executor.submit(self._deploy, identifier, model)
        return result

    def _deploy(self, endpoint_id: str, model: Dict[str, Any]) -> None:
        """Validate the method and load the saved artifact (background worker)."""
        with self._lock:
            deployment = self._deployments.get(endpoint_id)
            if deployment is None:
                return
            deployment["status"] = "deploying"
            deployment["last_error"] = ""
            self._log(deployment, f'Validando el método "{deployment["method"]}"')
            self._persist_locked()
        try:
            method = core.find_method(model, deployment["method"])
            if method is None or not core.method_allowed(deployment["method"], method):
                raise core.ValidationError(
                    f'El método "{deployment["method"]}" ya no está disponible en ' f'{model["id"]}'
                )
            if not core.method_is_jsonable(method):
                raise core.ValidationError(
                    f'El método "{deployment["method"]}" no admite argumentos JSON'
                )
            with self._lock:
                deployment = self._deployments.get(endpoint_id)
                if deployment is None:
                    return
                deployment["method_signature"] = method.get("signature") or method["name"]
                deployment["method_description"] = method.get("description") or ""
                deployment["method_fields"] = core.method_fields(method, deployment["parameters"])
                self._persist_locked()

            artifact = None
            if deployment["load"].get("enabled"):
                with self._lock:
                    deployment = self._deployments.get(endpoint_id)
                    if deployment is None:
                        return
                    self._log(
                        deployment,
                        "Cargando el modelo guardado desde " f'"{deployment["load"].get("path")}"',
                    )
                    self._persist_locked()
                prepared = core.prepare_load(
                    model,
                    {
                        "loader": deployment["load"].get("loader"),
                        "path": deployment["load"].get("path"),
                        "parameters": deployment["parameters"],
                    },
                )
                result = core.run_load(model, prepared, timeout=DEPLOY_TIMEOUT)
                artifact = {
                    "loader": prepared["name"],
                    "path": prepared["path"],
                    "type": result.get("type"),
                    "module": result.get("module"),
                    "repr": result.get("repr"),
                    "elapsed": result.get("elapsed"),
                    "loaded_at": iso(),
                }

            with self._lock:
                deployment = self._deployments.get(endpoint_id)
                if deployment is None:
                    return
                deployment["status"] = "running"
                deployment["deployed_at"] = iso()
                deployment["stopped_at"] = None
                deployment["loaded_artifact"] = artifact
                deployment["last_error"] = ""
                if artifact:
                    self._log(
                        deployment,
                        f'Modelo guardado cargado con {artifact["loader"]}() '
                        f'en {artifact.get("elapsed") or 0:g} s',
                        level="success",
                    )
                self._log(deployment, "Endpoint listo para recibir peticiones", level="success")
                self._persist_locked()
        except core.APIError as exc:
            self._fail(endpoint_id, str(exc), getattr(exc, "detail", ""))
        except BaseException as exc:  # pragma: no cover - defensive
            LOGGER.exception("Fallo inesperado desplegando %s", endpoint_id)
            self._fail(endpoint_id, f"{type(exc).__name__}: {exc}", "")

    def _fail(self, endpoint_id: str, message: str, detail: str = "") -> None:
        with self._lock:
            deployment = self._deployments.get(endpoint_id)
            if deployment is None:
                return
            deployment["status"] = "failed"
            deployment["last_error"] = message
            self._log(deployment, message, level="error")
            if detail:
                self._log(deployment, detail[-1500:], level="detail")
            self._persist_locked()

    def stop(self, endpoint_id: str) -> Dict[str, Any]:
        """Stop an endpoint (rejected while it is deploying)."""
        with self._lock:
            deployment = self._find(endpoint_id)
            if deployment["status"] == "deploying":
                raise core.APIError(
                    "El endpoint se está desplegando; espera a que termine antes de detenerlo",
                    status=409,
                )
            if deployment["status"] != "stopped":
                deployment["status"] = "stopped"
                deployment["stopped_at"] = iso()
                self._log(deployment, "Endpoint detenido por el usuario", level="warn")
                self._persist_locked()
            return self._public(deployment)

    def restart(self, endpoint_id: str, model: Dict[str, Any]) -> Dict[str, Any]:
        """Restart/start an endpoint, revalidating its configuration and artifact."""
        with self._lock:
            deployment = self._find(endpoint_id)
            if deployment["status"] == "deploying":
                raise core.APIError("El endpoint ya se está desplegando", status=409)
            deployment["status"] = "deploying"
            deployment["last_error"] = ""
            deployment["revision"] = int(deployment.get("revision") or 1) + 1
            deployment["stopped_at"] = None
            self._log(deployment, "Reiniciando el endpoint", level="warn")
            self._persist_locked()
            result = self._public(deployment)
            self._executor.submit(self._deploy, endpoint_id, model)
        return result

    def delete(self, endpoint_id: str) -> Dict[str, Any]:
        """Remove an endpoint (rejected while it is deploying)."""
        with self._lock:
            deployment = self._find(endpoint_id)
            if deployment["status"] == "deploying":
                raise core.APIError(
                    "El endpoint se está desplegando; espera a que termine antes de eliminarlo",
                    status=409,
                )
            removed = self._deployments.pop(deployment["id"])
            self._semaphores.pop(deployment["id"], None)
            self._persist_locked()
        return self._public(removed)

    def shutdown(self) -> None:
        """Release the background workers (called from the app lifespan)."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- inference --------------------------------------------------------

    def predict(
        self,
        endpoint_id: str,
        model: Dict[str, Any],
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one inference through the endpoint and record its monitoring event.

        The request body accepts three equivalent forms::

            {"input": 42}                  # single-argument shorthand
            {"inputs": {"data": [...]}}    # named arguments
            {"arguments": {"data": [...]}} # invoke-compatible envelope
        """
        if not isinstance(body, dict):
            raise core.ValidationError("El cuerpo de la petición debe ser un objeto JSON")
        with self._lock:
            deployment = self._find(endpoint_id)
            snapshot = self._public(deployment)
        if snapshot["status"] != "running":
            message = (
                f'El endpoint "{snapshot["name"]}" no está en ejecución ' f'({snapshot["status"]})'
            )
            self._record_error(snapshot, None, message, status=409, error_type="EndpointNotRunning")
            exc = core.APIError(message, status=409)
            exc.extra["endpoint_status"] = snapshot["status"]
            raise exc

        try:
            method = core.find_method(model, snapshot["method"])
            if method is None or not core.method_allowed(snapshot["method"], method):
                raise core.ValidationError(
                    f'El método "{snapshot["method"]}" ya no está disponible en ' f'{model["id"]}'
                )
            fields = core.method_fields(method, snapshot["parameters"])
            raw_arguments = _arguments_from_body(body, fields)
            arguments = core.coerce_arguments(fields, raw_arguments)
            timeout = core.requested_timeout(body, snapshot["config"]["timeout_seconds"])
        except core.APIError as exc:
            self._record_error(
                snapshot,
                body,
                str(exc),
                status=exc.status,
                error_type=type(exc).__name__,
            )
            raise

        semaphore = self._semaphore(snapshot)
        if not semaphore.acquire(blocking=False):
            message = (
                f'El endpoint "{snapshot["name"]}" está atendiendo el máximo de '
                f'{snapshot["config"]["max_concurrency"]} peticiones simultáneas'
            )
            self._record_error(snapshot, arguments, message, status=429, error_type="Busy")
            raise core.RateLimitExceeded(message, headers={"Retry-After": "1"})

        started = time.monotonic()
        request_bytes = _json_bytes(body)
        invoke_load = _invoke_load(snapshot["load"])
        if invoke_load and not invoke_load.get("path_parameter"):
            # Records persisted by an older version may not carry the loader
            # parameter name; resolve it from the live catalog.
            try:
                loader = core.resolve_loader(model, invoke_load.get("name") or "")
                invoke_load["path_parameter"] = loader.get("path_parameter")
            except core.APIError:
                pass
        try:
            result = core.run_invoke(
                model,
                method,
                snapshot["parameters"],
                arguments,
                timeout=timeout,
                load=invoke_load,
            )
            elapsed_ms = round(
                float(result.get("elapsed") or (time.monotonic() - started)) * 1000.0, 2
            )
            event = self._record_success(
                snapshot,
                arguments,
                result,
                elapsed_ms,
                request_bytes=request_bytes,
            )
            return {
                "success": True,
                "endpoint_id": snapshot["id"],
                "endpoint_name": snapshot["name"],
                "model_id": snapshot["model_id"],
                "method": snapshot["method"],
                "prediction": result.get("result"),
                "prediction_type": result.get("result_type"),
                "prediction_repr": result.get("result_repr"),
                "latency_ms": elapsed_ms,
                "request_id": event["id"],
                "timestamp": event["at"],
                "source": ("loaded_model" if snapshot["load"].get("enabled") else "constructor"),
                "arguments": arguments,
            }
        except core.APIError as exc:
            self._record_error(
                snapshot,
                arguments,
                str(exc),
                status=exc.status,
                error_type=type(exc).__name__,
                detail=getattr(exc, "detail", ""),
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
            raise
        except BaseException as exc:  # pragma: no cover - defensive
            self._record_error(
                snapshot,
                arguments,
                f"{type(exc).__name__}: {exc}",
                status=500,
                error_type=type(exc).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000.0, 2),
            )
            raise core.InvocationError(f"La inferencia falló: {type(exc).__name__}: {exc}") from exc
        finally:
            semaphore.release()

    def _record_success(
        self,
        deployment: Dict[str, Any],
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        elapsed_ms: float,
        request_bytes: int = 0,
    ) -> Dict[str, Any]:
        event = self._event(deployment, "success", 200, elapsed_ms)
        event.update(
            {
                "request_bytes": request_bytes,
                "response_bytes": _json_bytes(result.get("result")),
                "input_preview": _preview(arguments) if LOG_INPUTS else "",
                "output_preview": _preview(result.get("result")) if LOG_INPUTS else "",
            }
        )
        return self.events.record(event)

    def _record_error(
        self,
        deployment: Dict[str, Any],
        arguments: Optional[Dict[str, Any]],
        message: str,
        status: int = 422,
        error_type: str = "",
        detail: str = "",
        elapsed_ms: float = 0.0,
    ) -> Dict[str, Any]:
        event = self._event(deployment, "error", status, elapsed_ms)
        event.update(
            {
                "request_bytes": _json_bytes(arguments or {}),
                "response_bytes": 0,
                "input_preview": _preview(arguments or {}) if LOG_INPUTS else "",
                "output_preview": "",
                "error_type": error_type or "Error",
                "error_message": str(message)[:1000],
                "error_detail": (detail or "")[-4000:],
            }
        )
        return self.events.record(event)

    @staticmethod
    def _event(
        deployment: Dict[str, Any], status: str, http_status: int, elapsed_ms: float
    ) -> Dict[str, Any]:
        return {
            "endpoint_id": deployment["id"],
            "endpoint_name": deployment["name"],
            "model_id": deployment["model_id"],
            "model_name": deployment["model_name"],
            "method": deployment["method"],
            "status": status,
            "http_status": int(http_status),
            "latency_ms": round(float(elapsed_ms), 2),
            "source": "serving",
        }

    def events_for(self, endpoint_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Newest events of one endpoint."""
        wanted = str(endpoint_id or "").strip()
        return [
            event
            for event in self.events.recent(max(limit * 8, limit))
            if str(event.get("endpoint_id") or "") == wanted
        ][:limit]


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _artifact_path(path: str) -> Path:
    """Resolve a saved-model path the same way the sandbox does."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else core.LOAD_ROOT / candidate


def _normalise_config(raw: Any) -> Dict[str, Any]:
    """Validate the runtime options of an endpoint."""
    config = raw if isinstance(raw, dict) else {}
    timeout = config.get("timeout_seconds", config.get("timeout", DEFAULT_TIMEOUT))
    if isinstance(timeout, bool):
        raise core.ValidationError("El campo 'timeout_seconds' debe ser un número")
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise core.ValidationError("El campo 'timeout_seconds' debe ser un número") from exc
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise core.ValidationError("El campo 'timeout_seconds' debe ser mayor que 0")
    if timeout_value > core.INVOKE_MAX_TIMEOUT:
        raise core.ValidationError(
            f"El campo 'timeout_seconds' no puede superar {core.INVOKE_MAX_TIMEOUT:g} segundos"
        )
    concurrency = config.get("max_concurrency", DEFAULT_CONCURRENCY)
    if isinstance(concurrency, bool):
        raise core.ValidationError("El campo 'max_concurrency' debe ser un entero")
    try:
        concurrency_value = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise core.ValidationError("El campo 'max_concurrency' debe ser un entero") from exc
    if not 1 <= concurrency_value <= MAX_CONCURRENCY:
        raise core.ValidationError(
            f"El campo 'max_concurrency' debe estar entre 1 y {MAX_CONCURRENCY}"
        )
    return {"timeout_seconds": timeout_value, "max_concurrency": concurrency_value}


def _normalise_load(raw: Any, parameters: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the optional saved-model loading configuration."""
    if raw is None or raw is False:
        return {"enabled": False}
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise core.ValidationError("El campo 'load' debe ser un objeto")
    path = str(raw.get("path") or "").strip()
    enabled = bool(raw.get("enabled", bool(path)))
    if not enabled:
        return {"enabled": False}
    if not path:
        raise core.ValidationError(
            "El campo 'load.path' es obligatorio cuando se carga un modelo guardado"
        )
    if not model.get("loaders"):
        raise core.ValidationError(
            f'El modelo "{model["id"]}" no declara un método load() compatible'
        )
    loader_name = str(raw.get("loader") or raw.get("name") or "").strip()
    loader = core.resolve_loader(model, loader_name)
    loader_parameters = raw.get("parameters")
    if loader_parameters is not None and not isinstance(loader_parameters, dict):
        raise core.ValidationError("El campo 'load.parameters' debe ser un objeto")
    return {
        "enabled": True,
        "loader": loader["name"],
        "path": path,
        "parameters": dict(loader_parameters if loader_parameters is not None else parameters),
        "requires_instance": bool(loader.get("requires_instance")),
    }


def _invoke_load(load: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate the persisted load block into the ``run_invoke`` payload."""
    if not load or not load.get("enabled"):
        return None
    return {
        "name": load.get("loader"),
        "path": load.get("path"),
        "path_parameter": load.get("path_parameter"),
        "requires_instance": bool(load.get("requires_instance")),
    }


def _arguments_from_body(body: Dict[str, Any], fields: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the method arguments from the supported prediction envelopes."""
    if "arguments" in body:
        raw = body["arguments"]
        label = "arguments"
    elif "inputs" in body:
        raw = body["inputs"]
        label = "inputs"
    elif "input" in body:
        if len(fields) != 1:
            raise core.ValidationError(
                "El campo 'input' solo se admite cuando el método declara un único "
                "argumento; usa 'inputs' con los nombres reales"
            )
        return {fields[0]["name"]: body["input"]}
    else:
        raw = {key: value for key, value in body.items() if key not in _RESERVED_BODY_KEYS}
        label = "body"
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise core.ValidationError(f"El campo '{label}' debe ser un objeto")
    return raw


def method_example(method_fields: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Example request body generated from the real method fields."""
    examples = {
        field["name"]: field.get("sample")
        for field in method_fields
        if field.get("sample") is not None
    }
    if len(method_fields) == 1:
        value = method_fields[0].get("sample")
        return {"input": value if value is not None else "YOUR_INPUT"}
    if examples:
        return {"inputs": examples}
    return {"arguments": {}}


def build_docs(
    deployment: Dict[str, Any],
    model: Dict[str, Any],
    method: Dict[str, Any],
    base_url: str = "",
    auth_required: bool = False,
) -> Dict[str, Any]:
    """Generate the interactive API documentation of one endpoint.

    Everything is derived from the deployed model, the deployed method and the
    constructor values of the deployment; there is no hardcoded example.
    """
    fields = core.method_fields(method, deployment.get("parameters") or {})
    payload = method_example(fields)
    path = f"/api/serving/endpoints/{deployment['id']}/predict"
    endpoint_url = (base_url.rstrip("/") + path) if base_url else path
    headers = {"Content-Type": "application/json"}
    if auth_required:
        headers["Authorization"] = "Bearer $LM_API_TOKEN"
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    headers_json = json.dumps(
        {"Authorization": "Bearer <TOKEN>"} if auth_required else {},
        ensure_ascii=False,
    )

    python_lines = [
        "import os",
        "import requests",
        "",
        f'url = "{endpoint_url}"',
        "",
        "payload = " + payload_json,
        "",
    ]
    if auth_required:
        python_lines.extend(
            [
                "# El servidor exige LM_API_TOKEN; no escribas el token en el código.",
                'headers = {"Authorization": "Bearer " + os.environ.get("LM_API_TOKEN", "")}',
                "",
                "response = requests.post(url, json=payload, headers=headers)",
            ]
        )
    else:
        python_lines.append("response = requests.post(url, json=payload)")
    python_lines.extend(["response.raise_for_status()", "print(response.json())"])
    python_code = "\n".join(python_lines)

    curl_lines = [
        f'curl -X POST "{endpoint_url}" \\',
        '  -H "Content-Type: application/json" \\',
    ]
    if auth_required:
        curl_lines.append('  -H "Authorization: Bearer $LM_API_TOKEN" \\')
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    shell_payload = compact.replace("'", "'\"'\"'")
    curl_lines.append(f"  -d '{shell_payload}'")
    curl_code = "\n".join(curl_lines)

    javascript_lines = [
        f'const url = "{endpoint_url}";',
        f"const payload = {json.dumps(payload, ensure_ascii=False, indent=2)};",
        "",
        "const response = await fetch(url, {",
        '  method: "POST",',
        "  headers: {",
        '    "Content-Type": "application/json",',
    ]
    if auth_required:
        javascript_lines.append('    "Authorization": "Bearer <TOKEN>",')
    javascript_lines.extend(
        [
            "  },",
            "  body: JSON.stringify(payload),",
            "});",
            "",
            "if (!response.ok) throw new Error(await response.text());",
            "console.log(await response.json());",
        ]
    )
    javascript_code = "\n".join(javascript_lines)

    http_lines = [
        f"POST {path} HTTP/1.1",
        "Host: <host>",
        "Content-Type: application/json",
    ]
    if auth_required:
        http_lines.append("Authorization: Bearer <TOKEN>")
    http_lines.extend(["", compact])
    http_code = "\n".join(http_lines)

    response_example = {
        "success": True,
        "endpoint_id": deployment["id"],
        "endpoint_name": deployment["name"],
        "model_id": deployment["model_id"],
        "method": deployment["method"],
        "prediction": "<salida JSON del método>",
        "prediction_type": "ndarray",
        "latency_ms": 42.1,
        "request_id": "5f2c1a9b3e4d",
        "timestamp": iso(),
        "source": ("loaded_model" if deployment.get("load", {}).get("enabled") else "constructor"),
        "arguments": {field["name"]: field.get("sample") for field in fields},
    }

    return {
        "endpoint_id": deployment["id"],
        "endpoint_name": deployment["name"],
        "endpoint_url": endpoint_url,
        "endpoint_path": path,
        "method": {
            "name": method["name"],
            "signature": method.get("signature") or "",
            "description": method.get("description") or "",
            "kind": method.get("kind") or "method",
            "inference_like": is_inference_method(method["name"]),
            "inference_hint": has_inference_hint(method["name"]),
        },
        "model": {
            "id": model["id"],
            "name": model["name"],
            "title": model.get("title") or model["name"],
            "module": model.get("module") or "",
            "signature": model.get("signature") or "",
            "category": model.get("category") or "",
            "category_label": model.get("category_label") or "",
        },
        "payload_example": payload,
        "headers_example": headers,
        "samples": [
            {"language": "python", "label": "Python", "code": python_code},
            {"language": "bash", "label": "cURL", "code": curl_code},
            {"language": "javascript", "label": "JavaScript", "code": javascript_code},
            {"language": "http", "label": "HTTP", "code": http_code},
        ],
        "request_schema": {
            "envelope": {
                "input": "Atajo para métodos con un único argumento.",
                "inputs": "Objeto con los argumentos por nombre.",
                "arguments": "Alias compatible con /api/models/<id>/invoke.",
            },
            "fields": [
                {
                    "name": field["name"],
                    "type": field.get("type") or "",
                    "required": bool(field.get("required")),
                    "default": field.get("default"),
                    "description": field.get("description") or "",
                    "example": field.get("sample"),
                    "jsonable": bool(field.get("jsonable", True)),
                }
                for field in fields
            ],
            "headers": headers_json,
        },
        "response_schema": {
            "success": "boolean",
            "prediction": "Salida JSON del método (objeto, lista o escalar).",
            "prediction_type": "string",
            "latency_ms": "number",
            "request_id": "string",
            "timestamp": "ISO-8601",
        },
        "response_example": response_example,
        "auth": {
            "required": bool(auth_required),
            "scheme": "Bearer" if auth_required else "",
            "header": "Authorization" if auth_required else "",
            "env_var": "LM_API_TOKEN" if auth_required else "",
            "note": (
                "Envía el token configurado en LM_API_TOKEN como "
                "`Authorization: Bearer <token>`. Las llamadas sin token reciben 401."
                if auth_required
                else "Este servidor no exige token para /api."
            ),
        },
        "source": {
            "type": (
                "loaded_model" if deployment.get("load", {}).get("enabled") else "constructor"
            ),
            "loader": (deployment.get("load") or {}).get("loader"),
            "path": (deployment.get("load") or {}).get("path"),
            "parameters": deployment.get("parameters") or {},
        },
        "generated_at": iso(),
    }
