"""HTTP routes for Model Training.

The router lives in its own module to keep :mod:`server_fastapi` focused on the
catalogue.  It reuses the same exception hierarchy (``server_core.APIError``), so
the application error handlers, the authentication middleware, the rate limiter and
the Prometheus counters apply to the training endpoints without extra wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import server_core as core
import training_core
import training_datasets

#: Tags used by the OpenAPI document.
TRAINING_TAG = "training"


def get_settings(request: Request) -> Any:
    """Return the runtime settings of the running application."""
    return request.app.state.settings


def json_response(
    status: int, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
) -> JSONResponse:
    """JSON response with the same envelope used by the catalogue."""
    final_headers = {"Cache-Control": "no-store"}
    final_headers.update(headers or {})
    return JSONResponse(status_code=status, content=payload, headers=final_headers)


def _no_store(payload: Dict[str, Any], status: int = 200) -> JSONResponse:
    return json_response(status, payload)


def require_training(settings: Any) -> None:
    """Training reuses the invocation sandbox, so the invoke switch must be on."""
    if not settings.invoke_enabled:
        raise core.APIError(
            "La invocación de métodos está deshabilitada en este servidor", status=403
        )


class SeriesAnalysisRequest(BaseModel):
    """Body of ``POST /api/training/jobs/<id>/series``."""

    validation_ratio: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fracción final de la serie reservada para validación (0-1).",
    )
    n_steps: int = Field(default=12, ge=1, le=365, description="Pasos futuros a predecir.")
    scaling: Optional[str] = Field(
        default=None,
        description=(
            "detrend_minmax (DataScaler, por defecto), minmax_symmetric "
            "(solo [-1,1]), minmax, standard o none. Se aplica a la serie "
            "completa; si falta, scale=true equivale a detrend_minmax."
        ),
    )
    scale: bool = Field(
        default=True,
        description="Alias booleano obsoleto de scaling: true=detrend_minmax, false=none.",
    )
    validation_mode: Optional[str] = Field(
        default=None,
        description=(
            "holdout (por defecto): la validación entrena sin el tramo final y "
            "la predicción se reentrena con toda la serie. full: un único modelo "
            "entrenado con toda la serie (validación en muestra) usado también "
            "para la predicción futura."
        ),
    )
    sigma: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Umbral de denoising FFT (solo FourierRegression).",
    )
    mov: int = Field(default=200, ge=2, description="Iteraciones de la caminata ARIMA.")
    seed: int = Field(
        default=42,
        ge=0,
        description="Semilla de NumPy/Python para que el análisis sea reproducible.",
    )
    n_walkers: int = Field(
        default=1,
        ge=1,
        description=("Caminantes por ventana en FourierRegression o ``nwalkers`` en Arima.train."),
    )


class TrainingRequest(BaseModel):
    """Body of ``POST /api/training/jobs``."""

    model_config = ConfigDict(extra="allow")

    kind: Optional[str] = Field(default=None, description="'fit' (por defecto) o 'evaluate'.")
    model_id: str = Field(default="", description="Id del modelo del catálogo.")
    method: Optional[str] = Field(default=None, description="Método a ejecutar (fit por defecto).")
    arguments: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del método (datos de entrenamiento)."
    )
    args: Optional[Dict[str, Any]] = Field(default=None, description="Alias de ``arguments``.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )
    params: Optional[Dict[str, Any]] = Field(default=None, description="Alias de ``parameters``.")
    compile: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Opciones de compilación Keras para el adaptador fit "
            "(optimizer, loss, metrics...). Se combinan con las del servidor."
        ),
    )
    compile_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Alias de ``compile``."
    )
    control: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Control Keras por métrica objetivo: metric, mode, action "
            "('stop'/'reduce_lr'/'both'), value, patience, min_delta, factor, min_lr."
        ),
    )
    cross_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Validación cruzada k-fold para fit Keras con dataset 'xy': folds "
            "(2-10), metric (val_loss por defecto, o 'balanced' para combinar "
            "métricas), weights ({nombre: peso} para 'balanced'), mode "
            "(auto/min/max), shuffle y seed. El artefacto guardado corresponde "
            "al mejor fold."
        ),
    )
    normalization: Optional[str] = Field(
        default=None,
        description=(
            "Normalización de X/datapoints: none, standard, minmax (0..1) o "
            "minmax_symmetric (-1..1). No modifica y; usa target_normalization "
            "para regresión."
        ),
    )
    target_normalization: Optional[str] = Field(
        default=None,
        description=(
            "Normalización de y para regresión: auto (misma que X), none, "
            "standard, minmax o minmax_symmetric. Las etiquetas de "
            "clasificación nunca se escalan."
        ),
    )
    dataset: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Referencia de dataset: dataset_id, mode ('xy'/'datapoints'), "
            "features, target, task y sample_rows (muestra aleatoria "
            "reproducible; vacío = todas las filas)."
        ),
    )
    timeout: Optional[float] = Field(
        default=None, description="Segundos máximos (por defecto LM_TRAINING_TIMEOUT)."
    )
    timeout_seconds: Optional[float] = Field(default=None, description="Alias de ``timeout``.")
    artifact_id: Optional[str] = Field(
        default=None, description="Artefacto a cargar (solo kind='evaluate')."
    )
    name: Optional[str] = Field(default=None, description="Nombre visible del trabajo.")

    def payload(self) -> Dict[str, Any]:
        """Plain dict without the ``None`` placeholders (legacy aliases included)."""
        data = {key: value for key, value in self.model_dump().items() if value is not None}
        data.update(self.model_extra or {})
        return data


class DatasetURLRequest(BaseModel):
    """Body of ``POST /api/training/datasets/inspect``."""

    url: str = Field(default="", description="URL HTTP(S) de un CSV, XLSX o Parquet.")


def _job_brief(job: Dict[str, Any]) -> Dict[str, Any]:
    """Small job projection used by lists and filters."""
    artifact = job.get("artifact") or {}
    return {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "status": job["status"],
        "name": job.get("name") or "",
        "model_id": job.get("model_id") or "",
        "model_name": job.get("model_name") or job.get("model_id") or "",
        "method": job.get("method") or "",
        "adapter": job.get("adapter") or "",
        "dataset_name": (job.get("dataset") or {}).get("name") or "",
        "created_at": job.get("created_at") or "",
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed": job.get("elapsed"),
        "compute_seconds": job.get("compute_seconds"),
        "attempts": job.get("attempts") or 0,
        "error": job.get("error") or "",
        "artifact_id": artifact.get("artifact_id") if artifact else None,
        "artifact_path": artifact.get("path") if artifact else None,
        "artifact_size_bytes": artifact.get("size_bytes") if artifact else None,
        "result_type": job.get("result_type"),
        "result_repr": job.get("result_repr"),
    }


def _job_payload(settings: Any, job: Dict[str, Any]) -> Dict[str, Any]:
    """Full job payload with model metadata and artifact details."""
    payload = {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "status": job["status"],
        "name": job.get("name") or "",
        "model_id": job.get("model_id") or "",
        "model_name": job.get("model_name") or job.get("model_id") or "",
        "method": job.get("method") or "",
        "adapter": job.get("adapter") or "",
        "dataset": job.get("dataset"),
        "arguments": dict(job.get("arguments") or {}),
        "parameters": dict(job.get("parameters") or {}),
        "compile": dict(job.get("compile") or {}),
        "control": dict(job.get("control") or {}),
        "cross_validation": dict(job.get("cross_validation") or {}),
        "normalization": str(job.get("normalization") or "none"),
        "target_normalization": dict(job.get("target_normalization") or {}),
        "timeout_seconds": job.get("timeout_seconds"),
        "attempts": job.get("attempts") or 0,
        "created_at": job.get("created_at") or "",
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed": job.get("elapsed"),
        "compute_seconds": job.get("compute_seconds"),
        "error": job.get("error") or "",
        "error_detail": job.get("error_detail") or "",
        "result_type": job.get("result_type"),
        "result_repr": job.get("result_repr"),
        "result": job.get("result"),
        "instance": dict(job.get("instance") or {}),
        "log": list(job.get("log") or []),
        "updated_at": job.get("updated_at") or "",
    }
    artifact = job.get("artifact") or {}
    payload["artifact"] = dict(artifact) if artifact else None
    if artifact:
        path = str(artifact.get("path") or "")
        # Artifacts live under ``LM_TRAINING_STORE`` (outside ``web/``), so the
        # browser asset resolver cannot see them: check the path itself.
        payload["artifact"]["exists"] = Path(path).exists() if path else False
    catalog_model = None
    for item in settings.catalog:
        if item.get("id") == job.get("model_id"):
            catalog_model = item
            break
    if catalog_model is not None:
        payload["model"] = core.model_brief(catalog_model)
    payload.update(
        {
            "page_url": f"/training/{job['job_id']}",
            "auth_required": bool(settings.api_token),
        }
    )
    return payload


def build_router() -> APIRouter:
    """Build the training router."""
    router = APIRouter()

    @router.get("/api/training/overview", tags=[TRAINING_TAG])
    def training_overview(settings: Any = Depends(get_settings)) -> JSONResponse:
        """Training job counters and recent jobs."""
        require_training(settings)
        jobs = settings.training.list()
        statuses: Dict[str, int] = {}
        models = set()
        for job in jobs:
            statuses[job["status"]] = statuses.get(job["status"], 0) + 1
            models.add(job.get("model_id") or "")
        recent = [_job_brief(job) for job in jobs[:5]]
        return _no_store(
            {
                "success": True,
                "counts": {
                    "jobs": len(jobs),
                    "models_trained": len(models - {""}),
                    "statuses": statuses,
                },
                "recent": recent,
            }
        )

    @router.get("/api/training/models", tags=[TRAINING_TAG])
    def training_models(settings: Any = Depends(get_settings)) -> JSONResponse:
        """Fit-capable models with their fitting method and number of constructor fields."""
        require_training(settings)
        rows: List[Dict[str, Any]] = []
        for model in settings.catalog:
            # Models such as AutoQL require live Python objects in the
            # constructor (env/model); they cannot be built from JSON.
            if core.unsupported_parameters(model):
                continue
            fit = training_core.training_fit_method(model)
            if fit is None:
                continue
            example = training_core.training_example(model)
            dataset_mode = training_core.training_dataset_mode(model)
            dataset_task = training_core.training_dataset_task(model)
            series_mode = training_core.training_series_mode(model)
            model_parameters = model.get("parameters") or []
            constructor_fields = [p.get("name") for p in model_parameters if not p.get("variadic")]
            method_fields = [
                {"name": p.get("name"), "type": p.get("type"), "required": bool(p.get("required"))}
                for p in (fit.get("parameters") or [])
                if not p.get("variadic")
            ]
            rows.append(
                {
                    "id": model["id"],
                    "name": model.get("name") or model["id"],
                    "category": model.get("category"),
                    "icon": model.get("icon") or "🧪",
                    "fit_method": fit["name"],
                    "adapter": fit.get("framework") or "",
                    "method_fields": method_fields,
                    "constructor_fields": constructor_fields,
                    "example": example,
                    "dataset_mode": dataset_mode,
                    "dataset_task": dataset_task,
                    "series_mode": series_mode,
                }
            )
        rows.sort(key=lambda item: item["name"].lower())
        return _no_store({"success": True, "count": len(rows), "models": rows})

    @router.post("/api/training/datasets/inspect", tags=[TRAINING_TAG])
    def inspect_dataset(
        body: DatasetURLRequest, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Download and inspect a CSV/XLSX/Parquet URL."""
        require_training(settings)
        frame, name, source, fmt = training_datasets.load_url(body.url)
        dataset = settings.training.datasets.put(name, frame, source, fmt)
        return _no_store({"success": True, "dataset": dataset})

    @router.post("/api/training/datasets/upload", tags=[TRAINING_TAG])
    async def upload_dataset(
        file: UploadFile = File(...), settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Upload and inspect a CSV/XLSX/Parquet file."""
        require_training(settings)
        data = await file.read(training_datasets.MAX_BYTES + 1)
        if len(data) > training_datasets.MAX_BYTES:
            raise core.ValidationError(
                f"El dataset supera el máximo de "
                f"{training_datasets.MAX_BYTES // (1024 * 1024)} MiB"
            )
        frame, name, source, fmt = training_datasets.load_bytes(
            data, file.filename or "dataset.csv"
        )
        dataset = settings.training.datasets.put(name, frame, source, fmt)
        return _no_store({"success": True, "dataset": dataset})

    @router.get("/api/training/jobs", tags=[TRAINING_TAG])
    def list_jobs(
        settings: Any = Depends(get_settings),
        kind: str = Query("", description="fit, evaluate o vacío para todos."),
        status: str = Query("", description="pending, running, succeeded, failed o cancelled."),
    ) -> JSONResponse:
        """All training jobs, newest first (optionally filtered)."""
        require_training(settings)
        wanted_kind = (kind or "").strip().lower()
        wanted_status = (status or "").strip().lower()
        rows: List[Dict[str, Any]] = []
        for job in settings.training.list():
            if wanted_kind and job["kind"] != wanted_kind:
                continue
            if wanted_status and job["status"] != wanted_status:
                continue
            rows.append(_job_brief(job))
        return _no_store({"success": True, "count": len(rows), "jobs": rows})

    @router.get("/api/training/jobs/{job_id}", tags=[TRAINING_TAG])
    def get_job(job_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Full job payload with model metadata, artifact and log."""
        require_training(settings)
        job = settings.training.job(job_id)
        if not job:
            raise core.ModelNotFound(f'Trabajo no encontrado: "{job_id}"')
        payload = _job_payload(settings, job)
        return _no_store({"success": True, "job": payload, **payload})

    @router.post("/api/training/jobs/{job_id}/series", tags=[TRAINING_TAG])
    def analyse_series(
        job_id: str,
        body: SeriesAnalysisRequest,
        settings: Any = Depends(get_settings),
    ) -> JSONResponse:
        """Hold-out validation and future forecast for a series fit job."""
        require_training(settings)
        result = settings.training.analyse_series(job_id, body.model_dump())
        return _no_store({"success": True, **result})

    @router.post("/api/training/jobs/{job_id}/series/overwrite", tags=[TRAINING_TAG])
    def overwrite_series_model(
        job_id: str,
        body: SeriesAnalysisRequest,
        settings: Any = Depends(get_settings),
    ) -> JSONResponse:
        """Explicitly replace the fit artifact with the analysed configuration."""
        require_training(settings)
        result = settings.training.overwrite_series(job_id, body.model_dump())
        return _no_store({"success": True, **result})

    @router.post("/api/training/jobs", status_code=202, tags=[TRAINING_TAG])
    def create_job(
        body: TrainingRequest,
        settings: Any = Depends(get_settings),
    ) -> JSONResponse:
        """Start a training job that runs in the background."""
        require_training(settings)
        data = body.payload()
        job = settings.training.start(data)
        payload = _job_payload(settings, job)
        location = payload["page_url"]
        return json_response(
            202,
            {"success": True, "job": payload, **payload},
            {"Location": location, "Cache-Control": "no-store"},
        )

    @router.post("/api/training/jobs/{job_id}/cancel", tags=[TRAINING_TAG])
    def cancel_job(job_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Cancel a pending job."""
        require_training(settings)
        if not settings.training.cancel(job_id):
            raise core.ValidationError(f'No se puede cancelar: "{job_id}" no está pendiente.')
        job = settings.training.job(job_id)
        return _no_store(
            {"success": True, "cancelled": True, "job": _job_brief(job) if job else None}
        )

    @router.delete("/api/training/jobs/{job_id}", tags=[TRAINING_TAG])
    def delete_job(job_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Delete a training job (and its artifact, for fit jobs)."""
        require_training(settings)
        result = settings.training.delete(job_id)
        return _no_store({"success": True, "deleted": True, **result})

    @router.get("/api/training/artifacts", tags=[TRAINING_TAG])
    def list_artifacts(settings: Any = Depends(get_settings)) -> JSONResponse:
        """All artifacts of succeeded fit jobs, newest first."""
        require_training(settings)
        artifacts = settings.training.list_artifacts()
        return _no_store({"success": True, "count": len(artifacts), "artifacts": artifacts})

    @router.get("/api/training/artifacts/{artifact_id}", tags=[TRAINING_TAG])
    def get_artifact(artifact_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """One artifact produced by a fit job."""
        require_training(settings)
        artifact = settings.training.artifact(artifact_id)
        if not artifact:
            raise core.ModelNotFound(f'Artefacto no encontrado: "{artifact_id}"')
        return _no_store({"success": True, "artifact": artifact, **artifact})

    @router.delete("/api/training/artifacts/{artifact_id}", tags=[TRAINING_TAG])
    def delete_artifact(artifact_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Delete an artifact and clear its reference."""
        require_training(settings)
        result = settings.training.delete_artifact(artifact_id)
        return _no_store({"success": True, "deleted": result["deleted"], **result})

    return router
