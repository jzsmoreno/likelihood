"""Analytics over the real serving event log.

Every number returned by this module is computed from the events recorded by
:class:`serving_core.ServingStore` and from the deployment registry.  There is no
synthetic traffic: an empty installation returns zeroes and the dashboard shows its
empty state.

The module is intentionally independent from FastAPI; :mod:`serving_api` only
validates the query parameters and serialises the results.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import server_core as core
from serving_core import iso, parse_iso, percentile

#: Human label, duration in seconds and bucket size of every preset.
RANGE_PRESETS: Dict[str, Tuple[str, int, int]] = {
    "1h": ("Última hora", 3600, 60),
    "24h": ("Últimas 24 horas", 86400, 900),
    "7d": ("Últimos 7 días", 604800, 7200),
    "30d": ("Últimos 30 días", 2592000, 43200),
}

#: Candidate bucket sizes for a custom range (seconds).
_BUCKETS = (60, 300, 900, 1800, 3600, 10800, 21600, 43200, 86400, 604800)

#: Maximum custom range accepted (366 days).
MAX_RANGE_SECONDS = 366 * 86400

_BUCKET_LABELS = {
    60: "1 minuto",
    300: "5 minutos",
    900: "15 minutos",
    1800: "30 minutos",
    3600: "1 hora",
    10800: "3 horas",
    21600: "6 horas",
    43200: "12 horas",
    86400: "1 día",
    604800: "1 semana",
}


def _timestamp(value: Any, label: str) -> float:
    """Accept an epoch number or an ISO-8601 date."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        raise core.ValidationError(f"El parámetro '{label}' es obligatorio")
    try:
        return float(text)
    except ValueError:
        pass
    parsed = parse_iso(text)
    if parsed is None:
        raise core.ValidationError(
            f"El parámetro '{label}' debe usar el formato ISO-8601 (2026-01-31T12:00:00Z)"
        )
    return parsed.timestamp()


def parse_range(
    range_key: str = "24h",
    start: Any = None,
    end: Any = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Translate the dashboard controls into a concrete time window.

    ``range`` is one of ``1h``, ``24h``, ``7d``, ``30d`` or ``custom``; a custom
    window requires ``start`` and ``end`` (ISO-8601 or epoch seconds).
    """
    current = time.time() if now is None else float(now)
    key = (range_key or "24h").strip().lower()
    if key in RANGE_PRESETS:
        label, span, bucket = RANGE_PRESETS[key]
        end_ts = current
        start_ts = current - span
        if bucket < 1:
            bucket = 60
    elif key == "custom":
        start_ts = _timestamp(start, "start")
        end_ts = _timestamp(end, "end") if end not in (None, "") else current
        span = end_ts - start_ts
        if span <= 0:
            raise core.ValidationError("El rango personalizado necesita 'start' < 'end'")
        if span > MAX_RANGE_SECONDS:
            raise core.ValidationError("El rango personalizado no puede superar 366 días")
        bucket = next(
            (candidate for candidate in _BUCKETS if span / candidate <= 80),
            _BUCKETS[-1],
        )
        label = "Rango personalizado"
    else:
        allowed = ", ".join([*RANGE_PRESETS, "custom"])
        raise core.ValidationError(f"El parámetro 'range' debe ser uno de: {allowed}")

    span = end_ts - start_ts
    previous_end = start_ts
    previous_start = start_ts - span
    return {
        "key": key,
        "label": label,
        "start": start_ts,
        "end": end_ts,
        "span_seconds": span,
        "bucket_seconds": bucket,
        "bucket_label": _BUCKET_LABELS.get(bucket, f"{bucket} s"),
        "start_at": iso(datetime.fromtimestamp(start_ts, timezone.utc)),
        "end_at": iso(datetime.fromtimestamp(end_ts, timezone.utc)),
        "previous_start": previous_start,
        "previous_end": previous_end,
        "previous_start_at": iso(datetime.fromtimestamp(previous_start, timezone.utc)),
        "previous_end_at": iso(datetime.fromtimestamp(previous_end, timezone.utc)),
        "minutes": max(span / 60.0, 1.0 / 60.0),
        "generated_at": iso(datetime.fromtimestamp(current, timezone.utc)),
    }


def filter_events(
    events: Sequence[Dict[str, Any]],
    start: float,
    end: float,
    endpoint_id: str = "",
    model_id: str = "",
    status: str = "",
) -> List[Dict[str, Any]]:
    """Events inside ``[start, end)`` matching the optional filters."""
    wanted_endpoint = (endpoint_id or "").strip()
    wanted_model = (model_id or "").strip()
    wanted_status = (status or "").strip().lower()
    selected: List[Dict[str, Any]] = []
    for event in events:
        try:
            stamp = float(event.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if stamp < start or stamp >= end:
            continue
        if wanted_endpoint and str(event.get("endpoint_id") or "") != wanted_endpoint:
            continue
        if wanted_model and str(event.get("model_id") or "") != wanted_model:
            continue
        if wanted_status and str(event.get("status") or "").lower() != wanted_status:
            continue
        selected.append(event)
    selected.sort(key=lambda item: float(item.get("timestamp") or 0))
    return selected


def aggregate(events: Sequence[Dict[str, Any]], minutes: Optional[float] = None) -> Dict[str, Any]:
    """Totals, percentiles and byte counters of a set of events."""
    requests = len(events)
    successes = sum(1 for event in events if event.get("status") == "success")
    failures = sum(1 for event in events if event.get("status") == "error")
    latencies: List[float] = []
    statuses: Dict[str, int] = {}
    error_types: Dict[str, int] = {}
    endpoints = set()
    models = set()
    request_bytes = 0
    response_bytes = 0
    for event in events:
        try:
            latencies.append(float(event.get("latency_ms") or 0.0))
        except (TypeError, ValueError):
            pass
        status = str(event.get("http_status") or 0)
        statuses[status] = statuses.get(status, 0) + 1
        if event.get("status") == "error":
            label = str(event.get("error_type") or "Error")
            error_types[label] = error_types.get(label, 0) + 1
        if event.get("endpoint_id"):
            endpoints.add(str(event["endpoint_id"]))
        if event.get("model_id"):
            models.add(str(event["model_id"]))
        request_bytes += int(event.get("request_bytes") or 0)
        response_bytes += int(event.get("response_bytes") or 0)
    total_latency = sum(latencies)
    window_minutes = minutes if minutes and minutes > 0 else None
    return {
        "requests": requests,
        "successful": successes,
        "failed": failures,
        "success_rate": round(successes / requests * 100.0, 2) if requests else 0.0,
        "error_rate": round(failures / requests * 100.0, 2) if requests else 0.0,
        "avg_latency_ms": round(total_latency / requests, 2) if requests else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p90_latency_ms": percentile(latencies, 0.90),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "requests_per_minute": (round(requests / window_minutes, 3) if window_minutes else 0.0),
        "statuses": dict(sorted(statuses.items(), key=lambda item: item[0])),
        "error_types": dict(sorted(error_types.items(), key=lambda item: (-item[1], item[0]))),
        "endpoints": len(endpoints),
        "models": len(models),
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
    }


def bucketize(
    events: Sequence[Dict[str, Any]],
    start: float,
    end: float,
    bucket_seconds: int,
) -> List[Dict[str, Any]]:
    """Aggregate ``events`` into aligned time buckets for the charts."""
    bucket = max(1, int(bucket_seconds))
    first = math.floor(start / bucket) * bucket
    count = max(1, int(math.ceil((end - first) / bucket)))
    points: List[Dict[str, Any]] = []
    samples: List[List[float]] = [[] for _ in range(count)]
    for index in range(count):
        epoch = first + index * bucket
        points.append(
            {
                "t": epoch,
                "at": iso(datetime.fromtimestamp(epoch, timezone.utc)),
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "error_rate": 0.0,
                "throughput": 0.0,
                "avg_latency_ms": None,
                "p50_latency_ms": None,
                "p90_latency_ms": None,
                "p95_latency_ms": None,
                "p99_latency_ms": None,
            }
        )
    for event in events:
        try:
            stamp = float(event.get("timestamp"))
            latency = float(event.get("latency_ms") or 0.0)
        except (TypeError, ValueError):
            continue
        index = int((stamp - first) // bucket)
        if index < 0 or index >= count:
            continue
        point = points[index]
        point["requests"] += 1
        if event.get("status") == "success":
            point["successful"] += 1
        elif event.get("status") == "error":
            point["failed"] += 1
        samples[index].append(latency)
    bucket_minutes = bucket / 60.0
    for point, values in zip(points, samples):
        if point["requests"]:
            point["error_rate"] = round(point["failed"] / point["requests"] * 100.0, 2)
            point["throughput"] = round(point["requests"] / bucket_minutes, 3)
            point["avg_latency_ms"] = round(sum(values) / len(values), 2)
            point["p50_latency_ms"] = percentile(values, 0.50)
            point["p90_latency_ms"] = percentile(values, 0.90)
            point["p95_latency_ms"] = percentile(values, 0.95)
            point["p99_latency_ms"] = percentile(values, 0.99)
    return points


def endpoint_breakdown(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-endpoint totals, ordered by request volume."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        key = str(event.get("endpoint_id") or "unknown")
        groups.setdefault(key, []).append(event)
    rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        totals = aggregate(group)
        latest = group[-1] if group else {}
        rows.append(
            {
                "endpoint_id": key,
                "endpoint_name": latest.get("endpoint_name") or key,
                "model_id": latest.get("model_id") or "",
                "model_name": latest.get("model_name") or "",
                "method": latest.get("method") or "",
                "last_request_at": latest.get("at"),
                **totals,
            }
        )
    rows.sort(key=lambda row: (-row["requests"], row["endpoint_name"].lower()))
    return rows


def model_breakdown(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-model totals, ordered by request volume."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        key = str(event.get("model_id") or "unknown")
        groups.setdefault(key, []).append(event)
    rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        totals = aggregate(group)
        endpoints = {str(event.get("endpoint_id") or "") for event in group}
        rows.append(
            {
                "model_id": key,
                "model_name": group[-1].get("model_name") or key,
                "endpoints": len(endpoints),
                **totals,
            }
        )
    rows.sort(key=lambda row: (-row["requests"], row["model_name"].lower()))
    return rows


def latency_comparison(endpoints: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-endpoint latency percentiles for the comparison chart."""
    rows = [
        {
            "endpoint_id": endpoint["endpoint_id"],
            "endpoint_name": endpoint["endpoint_name"],
            "requests": endpoint["requests"],
            "avg_latency_ms": endpoint["avg_latency_ms"],
            "p50_latency_ms": endpoint["p50_latency_ms"],
            "p95_latency_ms": endpoint["p95_latency_ms"],
            "p99_latency_ms": endpoint["p99_latency_ms"],
            "error_rate": endpoint["error_rate"],
        }
        for endpoint in endpoints
        if endpoint["requests"]
    ]
    return sorted(rows, key=lambda row: -row["avg_latency_ms"])


def endpoint_index(
    deployments: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Every endpoint known by the store or by the event history."""
    index: Dict[str, Dict[str, Any]] = {}
    for deployment in deployments:
        index[deployment["id"]] = {
            "id": deployment["id"],
            "name": deployment["name"],
            "model_id": deployment.get("model_id") or "",
            "model_name": deployment.get("model_name") or "",
            "method": deployment.get("method") or "",
            "status": deployment.get("status") or "",
            "exists": True,
        }
    for event in events:
        key = str(event.get("endpoint_id") or "")
        if not key:
            continue
        entry = index.setdefault(
            key,
            {
                "id": key,
                "name": event.get("endpoint_name") or key,
                "model_id": event.get("model_id") or "",
                "model_name": event.get("model_name") or "",
                "method": event.get("method") or "",
                "status": "deleted",
                "exists": False,
            },
        )
        if not entry.get("model_id") and event.get("model_id"):
            entry["model_id"] = event["model_id"]
    return sorted(index.values(), key=lambda item: item["name"].lower())


def error_breakdown(events: Sequence[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
    """Error rate sources: types, HTTP statuses, endpoints and messages."""
    failures = [event for event in events if event.get("status") == "error"]
    by_type: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    by_endpoint: Dict[str, Dict[str, Any]] = {}
    messages: Dict[str, Dict[str, Any]] = {}
    for event in failures:
        error_type = str(event.get("error_type") or "Error")
        by_type[error_type] = by_type.get(error_type, 0) + 1
        status = str(event.get("http_status") or 0)
        by_status[status] = by_status.get(status, 0) + 1
        endpoint_id = str(event.get("endpoint_id") or "unknown")
        bucket = by_endpoint.setdefault(
            endpoint_id,
            {
                "endpoint_id": endpoint_id,
                "endpoint_name": event.get("endpoint_name") or endpoint_id,
                "model_id": event.get("model_id") or "",
                "count": 0,
                "last_at": event.get("at"),
            },
        )
        bucket["count"] += 1
        bucket["last_at"] = event.get("at") or bucket["last_at"]
        message = str(event.get("error_message") or error_type).strip()
        key = message.lower()[:180]
        entry = messages.setdefault(
            key,
            {
                "message": message[:300],
                "error_type": error_type,
                "count": 0,
                "endpoints": set(),
                "http_status": event.get("http_status"),
                "last_at": event.get("at"),
                "detail": (event.get("error_detail") or "")[-1200:],
            },
        )
        entry["count"] += 1
        entry["last_at"] = event.get("at") or entry["last_at"]
        if event.get("endpoint_id"):
            entry["endpoints"].add(str(event["endpoint_id"]))
    top_messages = sorted(messages.values(), key=lambda item: (-item["count"], item["message"]))
    for entry in top_messages:
        entry["endpoints"] = sorted(entry["endpoints"])
    return {
        "total": len(failures),
        "by_type": [
            {"label": key, "value": value}
            for key, value in sorted(by_type.items(), key=lambda item: (-item[1], item[0]))
        ],
        "by_status": [
            {"label": f"HTTP {key}", "value": value}
            for key, value in sorted(by_status.items(), key=lambda item: item[0])
        ],
        "by_endpoint": sorted(
            by_endpoint.values(), key=lambda item: (-item["count"], item["endpoint_name"])
        ),
        "top_messages": top_messages[: max(1, limit)],
    }


def resources_snapshot() -> Dict[str, Any]:
    """Host/process resource sample, when the optional ``psutil`` is installed.

    Inference runs in short-lived child processes (``server_core.run_invoke``), so
    there is no persistent per-endpoint process to sample.  Reporting a made-up
    per-endpoint CPU/GPU figure would be worse than reporting the limitation.
    """
    base = {
        "scope": "Proceso del servidor",
        "per_endpoint": False,
        "gpu": {
            "available": False,
            "reason": (
                "La utilización de GPU no se muestrea: las inferencias corren en procesos "
                "hijos efímeros y el servidor no abre el dispositivo."
            ),
        },
        "note": (
            "CPU y memoria corresponden al proceso del servidor. Para métricas por "
            "endpoint y GPU, instrumenta los procesos hijos o exporta "
            "/api/metrics a Prometheus."
        ),
    }
    try:
        import psutil  # type: ignore
    except ImportError:
        base.update(
            {
                "available": False,
                "reason": (
                    "Instala 'psutil' para muestrear CPU y memoria del servidor "
                    "(pip install psutil)."
                ),
            }
        )
        return base
    try:
        process = psutil.Process()
        memory = process.memory_info()
        virtual = psutil.virtual_memory()
        cpu = process.cpu_percent(interval=None)
        base.update(
            {
                "available": True,
                "cpu_percent": round(float(cpu), 2),
                "memory_mb": round(memory.rss / (1024 * 1024), 2),
                "memory_percent": (
                    round(memory.rss / virtual.total * 100.0, 2) if virtual.total else 0.0
                ),
                "system_cpu_percent": round(float(psutil.cpu_percent(interval=None)), 2),
                "system_memory_percent": round(float(virtual.percent), 2),
                "children": len(process.children(recursive=True)),
                "at": iso(),
            }
        )
    except (OSError, AttributeError) as exc:  # pragma: no cover - platform dependent
        base.update({"available": False, "reason": f"No se pudo muestrear el proceso: {exc}"})
    return base


def usage_patterns(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Request volume by UTC hour and weekday (usage patterns panel)."""
    hours = [0] * 24
    weekdays = [0] * 7
    for event in events:
        parsed = parse_iso(event.get("at"))
        if parsed is None:
            try:
                parsed = datetime.fromtimestamp(float(event.get("timestamp")), timezone.utc)
            except (TypeError, ValueError):
                continue
        hours[parsed.hour] += 1
        weekdays[parsed.weekday()] += 1
    names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    return {
        "hours": [{"label": f"{hour:02d}:00", "value": value} for hour, value in enumerate(hours)],
        "weekdays": [
            {"label": names[index], "value": value} for index, value in enumerate(weekdays)
        ],
    }


def _kpi(current: float, previous: Optional[float], unit: str = "") -> Dict[str, Any]:
    """KPI value plus its comparison with the previous window."""
    delta: Optional[float] = None
    if previous not in (None, 0):
        delta = round((float(current) - float(previous)) / float(previous) * 100.0, 1)
    return {
        "value": current,
        "previous": previous,
        "delta_pct": delta,
        "unit": unit,
    }


def _deployment_stats(deployments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    models = set()
    for deployment in deployments:
        statuses[deployment.get("status") or "unknown"] = (
            statuses.get(deployment.get("status") or "unknown", 0) + 1
        )
        if deployment.get("model_id"):
            models.add(str(deployment["model_id"]))
    return {
        "total": len(deployments),
        "deployed_models": len(models),
        "active_endpoints": statuses.get("running", 0),
        "statuses": statuses,
    }


def overview(
    events: Sequence[Dict[str, Any]],
    deployments: Sequence[Dict[str, Any]],
    window: Dict[str, Any],
    model_id: str = "",
    endpoint_id: str = "",
) -> Dict[str, Any]:
    """Global dashboard payload (KPIs, series, breakdowns and resources)."""
    current = filter_events(events, window["start"], window["end"], endpoint_id, model_id)
    previous = filter_events(
        events,
        window["previous_start"],
        window["previous_end"],
        endpoint_id,
        model_id,
    )
    deployments = [
        deployment
        for deployment in deployments
        if (not endpoint_id or deployment["id"] == endpoint_id)
        and (not model_id or deployment.get("model_id") == model_id)
    ]
    totals = aggregate(current, window["minutes"])
    previous_totals = aggregate(previous, window["minutes"])
    stats = _deployment_stats(deployments)
    kpis = {
        "deployed_models": _kpi(stats["deployed_models"], None, "modelos"),
        "active_endpoints": _kpi(stats["active_endpoints"], None, "endpoints"),
        "total_requests": _kpi(totals["requests"], previous_totals["requests"], "peticiones"),
        "successful_requests": _kpi(
            totals["successful"], previous_totals["successful"], "peticiones"
        ),
        "failed_requests": _kpi(totals["failed"], previous_totals["failed"], "peticiones"),
        "avg_latency_ms": _kpi(totals["avg_latency_ms"], previous_totals["avg_latency_ms"], "ms"),
        "error_rate": _kpi(totals["error_rate"], previous_totals["error_rate"], "%"),
        "requests_per_minute": _kpi(
            totals["requests_per_minute"],
            previous_totals["requests_per_minute"],
            "req/min",
        ),
    }
    endpoints = endpoint_breakdown(current)
    models = model_breakdown(current)
    errors = error_breakdown(current)
    return {
        "range": window,
        "filters": {"model_id": model_id, "endpoint_id": endpoint_id},
        "kpis": kpis,
        "totals": totals,
        "previous_totals": previous_totals,
        "deployments": stats,
        "series": {
            "activity": bucketize(current, window["start"], window["end"], window["bucket_seconds"])
        },
        "breakdown": {
            "endpoints": endpoints,
            "models": models,
            "latency": latency_comparison(endpoints),
            "errors": errors,
            "usage": usage_patterns(current),
        },
        "resources": resources_snapshot(),
        "meta": {
            "events": len(current),
            "previous_events": len(previous),
            "data_source": "Registro de inferencias de Model Serving (events.jsonl)",
            "note": (
                "Solo se contabilizan las peticiones recibidas por los endpoints de "
                "Model Serving; las llamadas al playground no forman parte de estas "
                "estadísticas."
            ),
            "generated_at": window["generated_at"],
        },
    }


def endpoint_detail(
    events: Sequence[Dict[str, Any]],
    endpoint_id: str,
    window: Dict[str, Any],
    deployment: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Statistics of one endpoint: KPIs, trends, errors and request history."""
    model_id = (deployment or {}).get("model_id") or ""
    current = filter_events(events, window["start"], window["end"], endpoint_id=endpoint_id)
    previous = filter_events(
        events,
        window["previous_start"],
        window["previous_end"],
        endpoint_id=endpoint_id,
    )
    totals = aggregate(current, window["minutes"])
    previous_totals = aggregate(previous, window["minutes"])
    kpis = {
        "total_requests": _kpi(totals["requests"], previous_totals["requests"], "peticiones"),
        "successful_requests": _kpi(
            totals["successful"], previous_totals["successful"], "peticiones"
        ),
        "failed_requests": _kpi(totals["failed"], previous_totals["failed"], "peticiones"),
        "error_rate": _kpi(totals["error_rate"], previous_totals["error_rate"], "%"),
        "avg_latency_ms": _kpi(totals["avg_latency_ms"], previous_totals["avg_latency_ms"], "ms"),
        "p95_latency_ms": _kpi(totals["p95_latency_ms"], previous_totals["p95_latency_ms"], "ms"),
        "requests_per_minute": _kpi(
            totals["requests_per_minute"],
            previous_totals["requests_per_minute"],
            "req/min",
        ),
    }
    latest = current[-1] if current else {}
    recent = [dict(item) for item in reversed(current[-max(1, limit) :])]
    for event in recent:
        event.setdefault("at", iso())
    return {
        "endpoint": (
            {
                "id": deployment["id"],
                "name": deployment["name"],
                "model_id": deployment["model_id"],
                "model_name": deployment["model_name"],
                "method": deployment["method"],
                "status": deployment["status"],
                "exists": True,
            }
            if deployment
            else {
                "id": endpoint_id,
                "name": latest.get("endpoint_name") or endpoint_id,
                "model_id": latest.get("model_id") or model_id,
                "model_name": latest.get("model_name") or "",
                "method": latest.get("method") or "",
                "status": "deleted",
                "exists": False,
            }
        ),
        "range": window,
        "kpis": kpis,
        "totals": totals,
        "previous_totals": previous_totals,
        "series": {
            "activity": bucketize(current, window["start"], window["end"], window["bucket_seconds"])
        },
        "errors": error_breakdown(current),
        "usage": usage_patterns(current),
        "recent": recent,
        "meta": {
            "events": len(current),
            "previous_events": len(previous),
            "history_limit": limit,
            "data_source": "Registro de inferencias de Model Serving (events.jsonl)",
            "generated_at": window["generated_at"],
        },
    }
