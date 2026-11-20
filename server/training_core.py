"""Model training layer of the Likelihood catalogue server.

Training reuses the existing sandbox of :mod:`server_core` (``run_invoke``) instead of
spawning its own child interpreter: a job is a persisted record that, when started,
builds the constructor of a catalog model, calls a fitting method inside the sandbox
and asks the child to pickle the live instance into an artifact file
(``run_invoke(..., save_path=...)``).  A second kind, ``evaluate``, loads that
artifact back through the model's own ``load``/``load_model`` loader and runs a
scoring method on it.

Design notes
------------
* A training job is stateless at the process level: it runs in a background
  worker of a ``ThreadPoolExecutor`` that calls ``core.run_invoke`` in a fresh
  child interpreter.  If the server restarts, running jobs are marked ``failed``
  (a worker that dies with the process cannot resume a fitting loop).
* Artifacts live under ``STORE_DIR/artifacts`` with a deterministic name
  (``<artifact_id>.pkl``); they are plain pickle files written by the child,
  never by this module.  Deleting a fit job deletes its artifact; the job history
  of an evaluate job keeps the artifact reference of the fit job that owns it.
* Nothing here duplicates model loading: artifacts are only *paths* that a later
  deployment (``serving_core``) or an evaluation job resolves through the
  catalog model's declared loaders.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import server_core as core
from serving_core import iso

LOGGER = logging.getLogger("likelihood.training")

#: Directory that keeps ``jobs.json`` and the ``artifacts/`` folder.  Tests inject a
#: temporary directory; production can point ``LM_TRAINING_STORE`` to a volume.
STORE_DIR = Path(
    os.environ.get("LM_TRAINING_STORE") or (core.REPO_ROOT / "data" / "output" / "training")
).expanduser()

#: Seconds a fitting call may spend before the sandbox kills the child.
TRAIN_TIMEOUT = float(os.environ.get("LM_TRAINING_TIMEOUT") or core.INVOKE_TIMEOUT)
#: Background workers that execute training jobs.
TRAIN_WORKERS = max(1, int(os.environ.get("LM_TRAINING_WORKERS") or 1))
#: Maximum number of finished jobs persisted to ``jobs.json``.
MAX_PERSISTED = max(1, int(os.environ.get("LM_TRAINING_MAX_PERSISTED") or 500))

#: Method names that fit a model, in the order the catalogue prefers them.
FIT_METHODS = ("fit", "train", "fit_transform", "fit_predict", "fit_predict_transform")


def new_job_id() -> str:
    """Generate the unique identifier of one training job."""
    return f"trj-{uuid.uuid4().hex[:12]}"


def new_artifact_id() -> str:
    """Generate the unique identifier of one model artifact."""
    return f"art-{uuid.uuid4().hex[:12]}"


def artifact_path(artifact_id: str, directory: Optional[Path] = None) -> Path:
    """Absolute location of the ``.pkl`` file of ``artifact_id``.

    ``directory`` lets a :class:`TrainingStore` injected with its own directory
    (tests, a mounted volume) keep its artifacts next to ``jobs.json``.
    """
    return Path(directory or STORE_DIR) / "artifacts" / f"{artifact_id}.pkl"


def fit_method_name(model: Dict[str, Any]) -> str:
    """Name of the fitting method declared by ``model`` (``fit`` if none matches)."""
    for name in FIT_METHODS:
        if core.find_method(model, name) is not None:
            return name
    return "fit"


def resolve_fit_method_name(model: Dict[str, Any], requested: str = "") -> str:
    """Map the generic ``fit``/empty request onto the fitting method of ``model``."""
    cleaned = str(requested or "").strip()
    if cleaned and (cleaned.lower() != "fit" or core.find_method(model, cleaned) is not None):
        return cleaned
    return fit_method_name(model)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (readers never see a half file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def method_name_hint(model: Dict[str, Any]) -> bool:
    """Whether ``model`` declares a ``score``-like method (UI hint only)."""
    for method in model.get("methods") or []:
        if str(method.get("name") or "").lower() in ("score", "score_samples"):
            return True
    return False


class TrainingStore:
    """Training job registry, artifact store and job executor."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory or STORE_DIR)
        self.jobs_path = self.directory / "jobs.json"
        self.artifacts_dir = self.directory / "artifacts"
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=TRAIN_WORKERS, thread_name_prefix="lm-training"
        )
        self._load()

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        """Stop accepting new work (called from the app lifespan)."""
        self._executor.shutdown(wait=False)

    def _load(self) -> None:
        if not self.jobs_path.is_file():
            return
        try:
            with open(self.jobs_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("No se pudo leer %s: %s", self.jobs_path, "error de lectura")
            return
        records = data.get("jobs") if isinstance(data, dict) else data
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict) or not record.get("id") and not record.get("job_id"):
                continue
            job = self._normalise_record(record)
            if job["status"] in ("pending", "running"):
                # No worker survived the restart: the job never finished.
                job["status"] = "failed"
                job["error"] = (
                    "El servidor se reinició mientras el entrenamiento se estaba ejecutando"
                )
                job["updated_at"] = iso()
                job["log"].append({"at": iso(), "level": "error", "message": job["error"]})
            self._jobs[job["job_id"]] = job

    @staticmethod
    def _normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Fill the fields a job persisted by an older version may miss."""
        return {
            "job_id": str(record.get("id") or record.get("job_id") or new_job_id()),
            "kind": str(record.get("kind") or "fit"),
            "status": str(record.get("status") or "failed"),
            "name": str(record.get("name") or ""),
            "model_id": str(record.get("model_id") or ""),
            "model_name": str(record.get("model_name") or record.get("model_id") or ""),
            "method": str(record.get("method") or ""),
            "arguments": dict(record.get("arguments") or {}),
            "parameters": dict(record.get("parameters") or {}),
            "timeout_seconds": float(record.get("timeout_seconds") or TRAIN_TIMEOUT),
            "attempts": int(record.get("attempts") or 0),
            "created_at": str(record.get("created_at") or iso()),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "elapsed": record.get("elapsed"),
            "compute_seconds": record.get("compute_seconds"),
            "error": str(record.get("error") or ""),
            "result_type": record.get("result_type"),
            "result_repr": record.get("result_repr"),
            "instance": dict(record.get("instance") or {}),
            "artifact": record.get("artifact"),
            "log": list(record.get("log") or []),
            "updated_at": str(record.get("updated_at") or record.get("created_at") or iso()),
        }

    def _persist_locked(self) -> None:
        """Persist the job registry atomically (oldest finished jobs dropped)."""
        records = sorted(self._jobs.values(), key=lambda job: job["created_at"])
        if len(records) > MAX_PERSISTED:
            records = records[-MAX_PERSISTED:]
        try:
            _atomic_write(
                self.jobs_path,
                json.dumps({"version": 1, "jobs": records}, ensure_ascii=False, indent=2),
            )
        except OSError:
            LOGGER.warning("No se pudo persistir %s", self.jobs_path)

    # -- job registry --------------------------------------------------------

    def start(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ``spec`` and register a training job that runs in the background.

        ``kind`` is ``fit`` (default; builds the constructor and calls the fitting
        method, which defaults to the first ``fit``/``train`` method declared by
        the model) or ``evaluate`` (loads an artifact when one is given and runs a
        scoring method).  The returned job has ``status == "pending"``; poll
        ``GET /api/training/jobs/<job_id>`` for the outcome.
        """
        spec = dict(spec or {})
        kind = str(spec.get("kind") or "fit").strip().lower()
        if kind not in ("fit", "evaluate"):
            raise core.ValidationError("El campo 'kind' debe ser 'fit' o 'evaluate'")
        model_id = str(spec.get("model_id") or "").strip()
        if not model_id:
            raise core.ValidationError("El campo 'model_id' es obligatorio")
        model = self._find_model(model_id)
        if kind == "fit":
            method_name = resolve_fit_method_name(model, str(spec.get("method") or ""))
            method, arguments, parameters = self._prepare_invocation(model, spec, method_name)
            if method["name"].lower() in ("save", "load", "load_model", "save_model"):
                raise core.ValidationError(
                    f'El método "{method["name"]}" guarda o carga un modelo; '
                    "usa fit/train para entrenar."
                )
        else:
            artifact_id = str(spec.get("artifact_id") or "").strip()
            # The artifact is optional: without it the model is built from the
            # constructor and the scoring method runs on that fresh instance.
            artifact = self.artifact(artifact_id) if artifact_id else None
            if artifact_id and not artifact:
                raise core.ModelNotFound(f'Artefacto no encontrado: "{artifact_id}"')
            method_name = str(spec.get("method") or "").strip()
            method, arguments, parameters = self._prepare_invocation(model, spec, method_name)
        timeout = core.requested_timeout(spec, TRAIN_TIMEOUT)
        now = iso()
        with self._lock:
            job_id = new_job_id()
            job = self._normalise_record(
                {
                    "id": job_id,
                    "kind": kind,
                    "status": "pending",
                    "name": str(spec.get("name") or "").strip(),
                    "model_id": model["id"],
                    "model_name": model.get("name") or model["id"],
                    "method": method["name"],
                    "arguments": arguments,
                    "parameters": parameters,
                    "timeout_seconds": timeout,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            if kind == "fit":
                aid = new_artifact_id()
                job["artifact"] = {
                    "artifact_id": aid,
                    "path": str(artifact_path(aid, self.directory)),
                }
            else:
                job["artifact"] = dict(artifact) if artifact else None
            job["log"].append(
                {"at": now, "level": "info", "message": "Trabajo de entrenamiento en cola"}
            )
            self._jobs[job_id] = job
            self._persist_locked()
        self._executor.submit(self._run, job_id)
        return dict(job)

    def _prepare_invocation(
        self, model: Dict[str, Any], spec: Dict[str, Any], method_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Validate a job spec and resolve the constructor arguments.

        A training form usually starts from an empty ``parameters`` object, but
        many fit-capable models declare required constructor fields (``Arima``
        needs ``datapoints``).  Missing values fall back to the same catalog
        defaults/samples the Playground uses, so a job can be launched without
        writing JSON by hand.

        Training data is often posted inside ``arguments``; a key that names a
        constructor parameter and not a method argument (``datapoints`` for
        ``Arima.train``) is moved to ``parameters`` instead of being rejected.
        """
        method = core.method_for_invocation(model, method_name)
        raw_arguments = spec.get("arguments", spec.get("args"))
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            raise core.ValidationError("El campo 'arguments' debe ser un objeto")
        raw_parameters = spec.get("parameters", spec.get("params"))
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise core.ValidationError("El campo 'parameters' debe ser un objeto")
        arguments = dict(raw_arguments)
        parameters = dict(raw_parameters)
        constructor_names = {
            parameter["name"]
            for parameter in model.get("parameters") or []
            if not parameter.get("variadic")
        }
        method_names = {
            parameter["name"]
            for parameter in method.get("parameters") or []
            if not parameter.get("variadic")
        }
        for name in list(arguments):
            if name in constructor_names and name not in method_names:
                parameters.setdefault(name, arguments.pop(name))
        return (
            method,
            core.coerce_arguments(core.method_fields(method), arguments),
            core.resolved_parameters(model, parameters),
        )

    def _run(self, job_id: str) -> None:
        """Worker body: mark running, execute, persist the outcome."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "pending":
                return
            job["status"] = "running"
            job["started_at"] = iso()
            job["updated_at"] = iso()
            job["attempts"] += 1
            self._persist_locked()
        started = time.monotonic()
        try:
            with self._lock:
                job = self._jobs[job_id]
            if job["kind"] == "fit":
                outcome = self._run_fit(job)
            else:
                outcome = self._run_evaluate(job)
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "succeeded"
                job["finished_at"] = iso()
                job["elapsed"] = round(time.monotonic() - started, 3)
                job["compute_seconds"] = outcome.get("compute_seconds")
                job["result_type"] = outcome.get("result_type")
                job["result_repr"] = outcome.get("result_repr")
                job["instance"] = {
                    "type": outcome.get("type"),
                    "module": outcome.get("module"),
                    "repr": outcome.get("repr"),
                }
                job["updated_at"] = iso()
                job["log"].append(
                    {
                        "at": iso(),
                        "level": "info",
                        "message": f"Completado en {job['elapsed']} s (cómputo {job.get('compute_seconds')} s)",
                    }
                )
                if outcome.get("save_error"):
                    job["last_error"] = outcome["save_error"]
                    job["log"].append(
                        {"at": iso(), "level": "warning", "message": outcome["save_error"]}
                    )
                if job["kind"] == "fit" and outcome.get("save_bytes"):
                    job["artifact"]["size_bytes"] = int(outcome["save_bytes"])
                    job["artifact"]["model_id"] = job["model_id"]
                    job["artifact"]["model_name"] = job.get("model_name") or ""
                    job["artifact"]["method"] = job["method"]
                    job["artifact"]["created_at"] = iso()
                self._persist_locked()
        except core.APIError as exc:
            with self._lock:
                job = self._jobs.get(job_id) or {}
                job["status"] = "failed"
                job["finished_at"] = iso()
                job["elapsed"] = round(time.monotonic() - started, 3)
                job["error"] = str(exc)
                job["updated_at"] = iso()
                job["log"].append({"at": iso(), "level": "error", "message": str(exc)})
                self._persist_locked()
        except BaseException as exc:  # pragma: no cover - unexpected sandbox failure
            with self._lock:
                job = self._jobs.get(job_id) or {}
                job["status"] = "failed"
                job["finished_at"] = iso()
                job["elapsed"] = round(time.monotonic() - started, 3)
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["updated_at"] = iso()
                job["log"].append({"at": iso(), "level": "error", "message": job["error"]})
                self._persist_locked()

    def _run_fit(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run the fitting call through the sandbox and save the instance."""
        model = self._find_model(job["model_id"])
        method = core.find_method(model, job["method"])
        if method is None:
            raise core.ModelNotFound(
                f'El modelo "{model["id"]}" ya no declara el método "{job["method"]}"'
            )
        save_path = str((job.get("artifact") or {}).get("path") or "")
        if not save_path:
            raise core.ValidationError("El trabajo no tiene una ruta de artefacto")
        result = core.run_invoke(
            model,
            method,
            dict(job["parameters"]),
            dict(job["arguments"]),
            timeout=float(job["timeout_seconds"]),
            save_path=save_path,
        )
        if result.get("module"):
            job["log"].append(
                {"at": iso(), "level": "info", "message": f"Instancia: {result['module']}"}
            )
        return result

    def _run_evaluate(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Load an artifact (when given) and run a scoring method on the model."""
        model = self._find_model(job["model_id"])
        artifact = job["artifact"] or {}
        path = str(artifact.get("path") or "")
        load = None
        if path:
            loader = core.resolve_loader(model)
            load = {
                "name": loader["name"],
                "path": path,
                "path_parameter": loader.get("path_parameter") or "path",
                "requires_instance": bool(loader.get("requires_instance")),
                "module": model["module"],
            }
        method = core.find_method(model, job["method"])
        if method is None:
            raise core.ModelNotFound(
                f'El modelo "{model["id"]}" ya no declara el método "{job["method"]}"'
            )
        result = core.run_invoke(
            model,
            method,
            dict(job["parameters"]),
            dict(job["arguments"]),
            timeout=float(job["timeout_seconds"]),
            load=load,
        )
        return result

    # -- queries ---------------------------------------------------------------

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the job ``job_id`` or ``None``."""
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            return dict(job) if job else None

    def list(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """All jobs, newest first (optionally filtered by ``kind``)."""
        wanted = str(kind or "").strip().lower()
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]
        if wanted:
            jobs = [job for job in jobs if job["kind"] == wanted]
        jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
        return jobs

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.  Running jobs cannot be interrupted (the child
        process of ``run_invoke`` has no handle) and finished jobs are untouched."""
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            if not job:
                return False
            if job["status"] != "pending":
                return False
            job["status"] = "cancelled"
            job["error"] = "El trabajo fue cancelado antes de empezar"
            job["finished_at"] = iso()
            job["elapsed"] = 0.0
            job["updated_at"] = iso()
            job["log"].append({"at": iso(), "level": "info", "message": "Trabajo cancelado"})
            self._persist_locked()
        return True

    def delete(self, job_id: str) -> Dict[str, Any]:
        """Remove a job from the registry.  A fit job also removes its artifact
        file; the artifact reference of an evaluate job is kept in the history."""
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            if not job:
                raise core.ModelNotFound(f'Trabajo no encontrado: "{job_id}"')
            deleted_artifact = False
            if job["kind"] == "fit" and isinstance(job.get("artifact"), dict):
                path_value = str(job["artifact"].get("path") or "")
                if path_value:
                    try:
                        target = Path(path_value)
                        if target.is_file():
                            target.unlink()
                            deleted_artifact = True
                        job["artifact"].pop("size_bytes", None)
                    except OSError as exc:
                        job["log"].append(
                            {
                                "at": iso(),
                                "level": "warning",
                                "message": f"No se pudo borrar el artefacto: {exc}",
                            }
                        )
            del self._jobs[job["job_id"]]
            self._persist_locked()
        return {"id": job["job_id"], "deleted": True, "artifact_deleted": deleted_artifact}

    # -- artifacts -------------------------------------------------------------

    def artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return the artifact ``artifact_id`` (from a succeeded fit job)."""
        with self._lock:
            wanted = str(artifact_id or "").strip()
            for job in self._jobs.values():
                artifact = job.get("artifact")
                if not isinstance(artifact, dict):
                    continue
                if str(artifact.get("artifact_id") or "") == wanted:
                    return dict(artifact)
        return None

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """All artifacts of succeeded fit jobs, newest first."""
        found = []
        with self._lock:
            for job in self._jobs.values():
                if job["status"] != "succeeded":
                    continue
                artifact = job.get("artifact")
                if not isinstance(artifact, dict) or not artifact.get("artifact_id"):
                    continue
                record = dict(artifact)
                record["job_id"] = job["job_id"]
                record["status"] = (
                    "ok" if Path(str(record.get("path") or "")).is_file() else "missing"
                )
                found.append(record)
        found.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return found

    def delete_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Remove the artifact file and clear its reference in the fit job."""
        record = self.artifact(artifact_id)
        if not record:
            raise core.ModelNotFound(f'Artefacto no encontrado: "{artifact_id}"')
        deleted = False
        with self._lock:
            wanted = str(artifact_id or "").strip()
            for job in self._jobs.values():
                artifact = job.get("artifact")
                if (
                    not isinstance(artifact, dict)
                    or str(artifact.get("artifact_id") or "") != wanted
                ):
                    continue
                path = Path(str(artifact.get("path") or ""))
                if path.is_file():
                    try:
                        path.unlink()
                        deleted = True
                    except OSError as exc:
                        job["log"].append({"at": iso(), "level": "warning", "message": str(exc)})
                artifact.pop("size_bytes", None)
                break
        return {"id": wanted, "deleted": deleted}

    # -- helpers ---------------------------------------------------------------

    def _find_model(self, model_id: str) -> Dict[str, Any]:
        """Resolve ``model_id`` against the catalog (the single source of truth)."""
        model = core.MODELS_BY_ID.get(str(model_id or "").strip())
        if not model:
            raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        return model
