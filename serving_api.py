"""HTTP routes for Model Serving and Model Monitoring.

The router lives in its own module to keep :mod:`server_fastapi` focused on the
catalogue.  It reuses the same exception hierarchy (``server_core.APIError``), so the
application error handlers, the authentication middleware, the rate limiter and the
Prometheus counters apply to the new endpoints without extra wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import monitoring_core
import server_core as core
import serving_core
from model_catalog import find_model

#: Tags used by the OpenAPI document.
SERVING_TAG = "serving"
MONITORING_TAG = "monitoring"


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


def _model_or_404(settings: Any, model_id: str) -> Dict[str, Any]:
    model = find_model(settings.catalog, (model_id or "").strip())
    if model is None:
        raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
    return model


def require_serving(settings: Any) -> None:
    """Serving reuses the invocation sandbox, so both switches must be on."""
    if not getattr(settings, "serving_enabled", True):
        raise core.APIError("Model Serving está deshabilitado en este servidor", status=403)
    if not settings.invoke_enabled:
        raise core.APIError(
            "La invocación de métodos está deshabilitada en este servidor", status=403
        )


class DeploymentRequest(BaseModel):
    """Body of ``POST /api/serving/endpoints`` and ``POST /api/serving/validate``."""

    model_config = ConfigDict(extra="allow")

    model_id: str = Field(default="", description="Modelo del catálogo que se despliega.")
    method: str = Field(default="", description="Método de inferencia que expone el endpoint.")
    endpoint_name: Optional[str] = Field(
        default=None, description="Nombre visible del endpoint (único)."
    )
    name: Optional[str] = Field(default=None, description="Alias de ``endpoint_name``.")
    description: Optional[str] = Field(default=None, description="Descripción libre.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Argumentos del constructor del modelo."
    )
    load: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Carga opcional de un modelo entrenado: ``{enabled, loader, path, " "parameters}``."
        ),
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="``{timeout_seconds, max_concurrency}`` del endpoint.",
    )

    def payload(self) -> Dict[str, Any]:
        """Plain dict without the ``None`` placeholders (legacy aliases included)."""
        data = {key: value for key, value in self.model_dump().items() if value is not None}
        data.update(self.model_extra or {})
        return data


def _brief(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Small endpoint projection used by lists and filters."""
    return {
        "id": deployment["id"],
        "name": deployment["name"],
        "model_id": deployment["model_id"],
        "model_name": deployment["model_name"],
        "model_title": deployment.get("model_title") or deployment["model_name"],
        "model_icon": deployment.get("model_icon") or "📦",
        "method": deployment["method"],
        "status": deployment["status"],
        "created_at": deployment["created_at"],
        "deployed_at": deployment.get("deployed_at"),
    }


def _stats_by_endpoint(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """All-time request counters grouped by endpoint (one pass)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        key = str(event.get("endpoint_id") or "")
        if key:
            groups.setdefault(key, []).append(event)
    return {
        key: {
            **monitoring_core.aggregate(group),
            "last_request_at": group[-1].get("at") if group else None,
        }
        for key, group in groups.items()
    }


def deployment_payload(
    settings: Any,
    deployment: Dict[str, Any],
    request: Request,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full endpoint payload with live model/method metadata and URLs."""
    payload = dict(deployment)
    payload.pop("_method", None)
    model = find_model(settings.catalog, deployment["model_id"])
    method = None
    if model is not None:
        method = core.find_method(model, deployment["method"])
    fields = (
        core.method_fields(method, deployment.get("parameters") or {})
        if method is not None
        else list(deployment.get("method_fields") or [])
    )
    base = str(request.base_url).rstrip("/")
    path = f"/api/serving/endpoints/{deployment['id']}/predict"
    label, tone = serving_core.STATUS_LABELS.get(
        deployment["status"], (deployment["status"], "muted")
    )
    payload.update(
        {
            "status_label": label,
            "status_tone": tone,
            "endpoint_url": base + path,
            "endpoint_path": path,
            "predict_url": path,
            "docs_url": f"/api/serving/endpoints/{deployment['id']}/docs",
            "events_url": f"/api/serving/endpoints/{deployment['id']}/events",
            "page_url": f"/serving/{deployment['id']}",
            "monitoring_url": f"/monitoring/{deployment['id']}",
            "auth_required": bool(settings.api_token),
            "model": (
                core.model_brief(model)
                if model is not None
                else {
                    "id": deployment["model_id"],
                    "title": deployment.get("model_title") or deployment["model_id"],
                    "icon": deployment.get("model_icon") or "📦",
                    "available": False,
                }
            ),
            "model_available": model is not None,
            "method_available": method is not None,
            "inference_like": serving_core.is_inference_method(deployment["method"]),
            "inference_hint": serving_core.has_inference_hint(deployment["method"]),
            "method_fields": fields,
            "example_payload": serving_core.method_example(fields),
            "stats": stats
            or {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
            },
            "deploy_timeout_seconds": serving_core.DEPLOY_TIMEOUT,
            "max_concurrency_limit": serving_core.MAX_CONCURRENCY,
        }
    )
    return payload


def build_router() -> APIRouter:
    """Build the serving + monitoring router."""
    router = APIRouter()

    # -- serving: catalogue ------------------------------------------------

    @router.get("/api/serving/overview", tags=[SERVING_TAG])
    def serving_overview(request: Request, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Deployment status counters and all-time inference totals."""
        deployments = settings.serving.list()
        events = settings.serving.events.events()
        totals = monitoring_core.aggregate(events)
        statuses: Dict[str, int] = {}
        models = set()
        for deployment in deployments:
            statuses[deployment["status"]] = statuses.get(deployment["status"], 0) + 1
            models.add(deployment["model_id"])
        recent = [
            _brief(deployment)
            for deployment in sorted(
                deployments, key=lambda item: item.get("created_at") or "", reverse=True
            )[:5]
        ]
        return _no_store(
            {
                "success": True,
                "counts": {
                    "endpoints": len(deployments),
                    "deployed_models": len(models),
                    "statuses": statuses,
                },
                "requests": {
                    "total": totals["requests"],
                    "successful": totals["successful"],
                    "failed": totals["failed"],
                    "error_rate": totals["error_rate"],
                    "avg_latency_ms": totals["avg_latency_ms"],
                },
                "recent": recent,
            }
        )

    @router.get("/api/serving/endpoints", tags=[SERVING_TAG])
    def list_endpoints(
        request: Request,
        settings: Any = Depends(get_settings),
        q: str = Query("", description="Busca por nombre, modelo o método."),
        status: str = Query("", description="running, stopped, failed o deploying."),
        model_id: str = Query("", description="Filtra por modelo del catálogo."),
    ) -> JSONResponse:
        """Every deployed endpoint with its all-time counters."""
        require_serving(settings)
        events = settings.serving.events.events()
        stats = _stats_by_endpoint(events)
        term = (q or "").strip().lower()
        wanted_status = (status or "").strip().lower()
        rows: List[Dict[str, Any]] = []
        for deployment in settings.serving.list():
            if model_id and deployment["model_id"] != model_id:
                continue
            if wanted_status and deployment["status"] != wanted_status:
                continue
            if term:
                haystack = " ".join(
                    [
                        deployment["name"],
                        deployment["model_id"],
                        deployment["model_name"],
                        deployment["method"],
                        deployment.get("description") or "",
                    ]
                ).lower()
                if term not in haystack:
                    continue
            rows.append(
                deployment_payload(
                    settings,
                    deployment,
                    request,
                    stats.get(deployment["id"]),
                )
            )
        statuses: Dict[str, int] = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        return _no_store(
            {
                "success": True,
                "count": len(rows),
                "statuses": statuses,
                "endpoints": rows,
            }
        )

    @router.post("/api/serving/endpoints", status_code=202, tags=[SERVING_TAG])
    def create_endpoint(
        request: Request,
        body: DeploymentRequest,
        settings: Any = Depends(get_settings),
    ) -> JSONResponse:
        """Deploy a model method as an HTTP inference endpoint.

        The configuration is validated synchronously; the saved artifact (when
        ``load.enabled``) is validated in the background and the endpoint reports
        ``deploying`` until it finishes.
        """
        require_serving(settings)
        data = body.payload()
        model = _model_or_404(settings, str(data.get("model_id") or ""))
        deployment = settings.serving.create(model, data)
        payload = deployment_payload(
            settings,
            deployment,
            request,
            {"requests": 0, "successful": 0, "failed": 0, "error_rate": 0.0, "avg_latency_ms": 0.0},
        )
        location = payload["page_url"]
        return json_response(
            202,
            {"success": True, "endpoint": payload, "deployment": payload, **payload},
            {"Location": location, "Cache-Control": "no-store"},
        )

    @router.post("/api/serving/validate", tags=[SERVING_TAG])
    def validate_endpoint(
        body: DeploymentRequest, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Validate a deployment payload without creating the endpoint."""
        require_serving(settings)
        data = body.payload()
        model = _model_or_404(settings, str(data.get("model_id") or ""))
        result = settings.serving.preview(model, data)
        return _no_store({"success": True, **result})

    @router.get("/api/serving/endpoints/{endpoint_id}", tags=[SERVING_TAG])
    def get_endpoint(
        endpoint_id: str, request: Request, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Configuration, deployment log and counters of one endpoint."""
        require_serving(settings)
        deployment = settings.serving.get(endpoint_id)
        stats = _stats_by_endpoint(settings.serving.events.events()).get(deployment["id"])
        payload = deployment_payload(settings, deployment, request, stats)
        return _no_store({"success": True, "endpoint": payload, **payload})

    @router.get("/api/serving/endpoints/{endpoint_id}/docs", tags=[SERVING_TAG])
    def endpoint_docs(
        endpoint_id: str, request: Request, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Code samples generated from the deployed model and method."""
        require_serving(settings)
        deployment = settings.serving.get(endpoint_id)
        model = _model_or_404(settings, deployment["model_id"])
        method = core.method_for_invocation(model, deployment["method"])
        docs = serving_core.build_docs(
            deployment,
            model,
            method,
            base_url=str(request.base_url),
            auth_required=bool(settings.api_token),
        )
        return _no_store(
            {
                "success": True,
                "docs": docs,
                "endpoint": deployment_payload(settings, deployment, request),
            }
        )

    @router.get("/api/serving/endpoints/{endpoint_id}/events", tags=[SERVING_TAG])
    def endpoint_events(
        endpoint_id: str,
        settings: Any = Depends(get_settings),
        limit: int = Query(100, ge=1, le=1000),
        status: str = Query("", description="success o error"),
    ) -> JSONResponse:
        """Recent request history of one endpoint (monitoring source)."""
        require_serving(settings)
        deployment = settings.serving.get(endpoint_id)
        events = settings.serving.events_for(deployment["id"], limit=limit)
        wanted = (status or "").strip().lower()
        if wanted:
            events = [event for event in events if event.get("status") == wanted]
        return _no_store(
            {
                "success": True,
                "endpoint_id": deployment["id"],
                "count": len(events),
                "events": events,
            }
        )

    @router.post("/api/serving/endpoints/{endpoint_id}/predict", tags=[SERVING_TAG])
    def predict(
        endpoint_id: str,
        settings: Any = Depends(get_settings),
        body: Dict[str, Any] = Body(
            default_factory=dict,
            description=(
                'Argumentos del método: ``{"input": ...}`` para un único '
                'argumento, ``{"inputs": {...}}`` o ``{"arguments": {...}}``.'
            ),
        ),
    ) -> JSONResponse:
        """Run one inference through the deployed endpoint.

        The call reuses the sandboxed ``run_invoke`` path, records a monitoring
        event and returns the JSON projection of the method result.
        """
        require_serving(settings)
        deployment = settings.serving.get(endpoint_id)
        model = _model_or_404(settings, deployment["model_id"])
        result = settings.serving.predict(deployment["id"], model, body)
        return _no_store(result, 200)

    @router.post("/api/serving/endpoints/{endpoint_id}/stop", tags=[SERVING_TAG])
    def stop_endpoint(
        endpoint_id: str, request: Request, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Stop an endpoint; predictions answer ``409`` until it is restarted."""
        require_serving(settings)
        deployment = settings.serving.stop(endpoint_id)
        return _no_store(
            {"success": True, "endpoint": deployment_payload(settings, deployment, request)}
        )

    @router.post("/api/serving/endpoints/{endpoint_id}/restart", tags=[SERVING_TAG])
    @router.post("/api/serving/endpoints/{endpoint_id}/start", tags=[SERVING_TAG])
    def restart_endpoint(
        endpoint_id: str, request: Request, settings: Any = Depends(get_settings)
    ) -> JSONResponse:
        """Restart a stopped endpoint (revalidates the method and the artifact)."""
        require_serving(settings)
        deployment = settings.serving.get(endpoint_id)
        model = _model_or_404(settings, deployment["model_id"])
        deployment = settings.serving.restart(deployment["id"], model)
        return json_response(
            202,
            {
                "success": True,
                "endpoint": deployment_payload(settings, deployment, request),
            },
            {"Cache-Control": "no-store"},
        )

    @router.delete("/api/serving/endpoints/{endpoint_id}", tags=[SERVING_TAG])
    def delete_endpoint(endpoint_id: str, settings: Any = Depends(get_settings)) -> JSONResponse:
        """Delete an endpoint. Its past events remain in the monitoring history."""
        require_serving(settings)
        deployment = settings.serving.delete(endpoint_id)
        return _no_store({"success": True, "endpoint_id": deployment["id"], "deleted": True})

    # -- monitoring --------------------------------------------------------

    def _window(range_key: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
        try:
            return monitoring_core.parse_range(range_key, start, end)
        except core.APIError:
            raise
        except ValueError as exc:
            raise core.ValidationError(str(exc)) from exc

    @router.get("/api/monitoring/overview", tags=[MONITORING_TAG])
    def monitoring_overview(
        settings: Any = Depends(get_settings),
        range: str = Query("24h", description="1h, 24h, 7d, 30d o custom."),
        start: Optional[str] = Query(None, description="Inicio ISO-8601 (custom)."),
        end: Optional[str] = Query(None, description="Fin ISO-8601 (custom)."),
        model_id: str = Query("", description="Filtra por modelo."),
        endpoint_id: str = Query("", description="Filtra por endpoint."),
    ) -> JSONResponse:
        """Global KPIs, time series, breakdowns, errors and resource sample."""
        window = _window(range, start, end)
        deployments = settings.serving.list()
        events = settings.serving.events.events()
        payload = monitoring_core.overview(events, deployments, window, model_id, endpoint_id)
        options = monitoring_core.endpoint_index(deployments, events)
        models: Dict[str, Dict[str, Any]] = {}
        for option in options:
            key = option.get("model_id") or ""
            if key:
                models.setdefault(
                    key,
                    {"id": key, "name": option.get("model_name") or key},
                )
        payload.update(
            {
                "success": True,
                "options": {
                    "endpoints": options,
                    "models": sorted(models.values(), key=lambda item: item["name"].lower()),
                },
                "deployments": [_brief(deployment) for deployment in deployments],
            }
        )
        return _no_store(payload)

    @router.get("/api/monitoring/endpoints", tags=[MONITORING_TAG])
    def monitoring_endpoints(
        settings: Any = Depends(get_settings),
        range: str = Query("24h"),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        model_id: str = Query(""),
    ) -> JSONResponse:
        """Per-endpoint aggregates for the selected window."""
        window = _window(range, start, end)
        deployments = {item["id"]: item for item in settings.serving.list()}
        events = monitoring_core.filter_events(
            settings.serving.events.events(),
            window["start"],
            window["end"],
            model_id=model_id,
        )
        rows = monitoring_core.endpoint_breakdown(events)
        for row in rows:
            deployment = deployments.get(row["endpoint_id"])
            row["status"] = deployment["status"] if deployment else "deleted"
            row["exists"] = deployment is not None
        totals = monitoring_core.aggregate(events, window["minutes"])
        return _no_store(
            {
                "success": True,
                "range": window,
                "count": len(rows),
                "totals": totals,
                "endpoints": rows,
            }
        )

    @router.get("/api/monitoring/models", tags=[MONITORING_TAG])
    def monitoring_models(
        settings: Any = Depends(get_settings),
        range: str = Query("24h"),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        endpoint_id: str = Query(""),
    ) -> JSONResponse:
        """Per-model aggregates for the selected window."""
        window = _window(range, start, end)
        events = monitoring_core.filter_events(
            settings.serving.events.events(),
            window["start"],
            window["end"],
            endpoint_id=endpoint_id,
        )
        rows = monitoring_core.model_breakdown(events)
        return _no_store(
            {
                "success": True,
                "range": window,
                "count": len(rows),
                "totals": monitoring_core.aggregate(events, window["minutes"]),
                "models": rows,
            }
        )

    @router.get("/api/monitoring/events", tags=[MONITORING_TAG])
    def monitoring_events(
        settings: Any = Depends(get_settings),
        range: str = Query("24h"),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        model_id: str = Query(""),
        endpoint_id: str = Query(""),
        status: str = Query("", description="success o error"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> JSONResponse:
        """Request history with the same filters as the dashboard."""
        window = _window(range, start, end)
        events = monitoring_core.filter_events(
            settings.serving.events.events(),
            window["start"],
            window["end"],
            endpoint_id,
            model_id,
            status,
        )
        newest = list(reversed(events))
        page = newest[offset : offset + limit]
        return _no_store(
            {
                "success": True,
                "range": window,
                "count": len(page),
                "total": len(events),
                "offset": offset,
                "limit": limit,
                "totals": monitoring_core.aggregate(events, window["minutes"]),
                "events": page,
            }
        )

    @router.get("/api/monitoring/endpoints/{endpoint_id}", tags=[MONITORING_TAG])
    def monitoring_endpoint(
        endpoint_id: str,
        request: Request,
        settings: Any = Depends(get_settings),
        range: str = Query("24h"),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> JSONResponse:
        """KPIs, trends, errors and request history of one endpoint."""
        window = _window(range, start, end)
        deployment = None
        try:
            deployment = settings.serving.get(endpoint_id)
        except core.ModelNotFound:
            deployment = None
        payload = monitoring_core.endpoint_detail(
            settings.serving.events.events(),
            endpoint_id,
            window,
            deployment=deployment,
            limit=limit,
        )
        payload["success"] = True
        payload["deployment"] = (
            deployment_payload(settings, deployment, request) if deployment else None
        )
        payload["resources"] = monitoring_core.resources_snapshot()
        return _no_store(payload)

    @router.get("/api/monitoring/resources", tags=[MONITORING_TAG])
    def monitoring_resources(settings: Any = Depends(get_settings)) -> JSONResponse:
        """Host/process resource sample and the per-endpoint limitation."""
        return _no_store({"success": True, "resources": monitoring_core.resources_snapshot()})

    return router
