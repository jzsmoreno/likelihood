"""Tabular dataset loading for Model Training.

The training form can use a CSV/XLSX/Parquet file or a remote URL.  The parsed
``DataFrame`` stays in the server process; a job stores only a compact dataset
reference plus the selected feature/target columns.  This keeps ``jobs.json``
small even for the 20k-row housing example.
"""

from __future__ import annotations

import io
import ipaddress
import math
import os
import socket
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

import server_core as core

#: Maximum size of one uploaded or downloaded dataset.
MAX_BYTES = int(os.environ.get("LM_TRAINING_DATASET_BYTES") or 25 * 1024 * 1024)
#: Maximum rows accepted by the inspector (the housing example has ~20k).
MAX_ROWS = int(os.environ.get("LM_TRAINING_DATASET_ROWS") or 50_000)
#: Maximum columns accepted by the inspector.
MAX_COLUMNS = int(os.environ.get("LM_TRAINING_DATASET_COLUMNS") or 200)
#: Number of parsed datasets kept in memory.
MAX_DATASETS = int(os.environ.get("LM_TRAINING_DATASETS") or 20)
#: Rows shown in the preview table.
PREVIEW_ROWS = 8
#: Extensions the loader understands.
SUPPORTED_FORMATS = frozenset({"csv", "tsv", "xlsx", "parquet"})


def _now() -> str:
    """UTC timestamp used by the dataset registry."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_from_name(name: str) -> str:
    """Infer the tabular format from a filename or URL path."""
    suffix = Path(urlparse(name).path).suffix.lower().lstrip(".")
    if suffix == "xls":
        raise core.ValidationError("El formato .xls no está soportado; guarda el libro como .xlsx")
    if suffix in SUPPORTED_FORMATS:
        return suffix
    return "csv"


def _json_value(value: Any) -> Any:
    """Convert a pandas/numpy scalar into a JSON friendly value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate limits, normalise column names and drop completely empty rows."""
    if frame is None or frame.empty:
        raise core.ValidationError("El dataset está vacío")
    if len(frame.columns) > MAX_COLUMNS:
        raise core.ValidationError(
            f"El dataset tiene {len(frame.columns)} columnas; el máximo es {MAX_COLUMNS}"
        )
    if len(frame) > MAX_ROWS:
        raise core.ValidationError(f"El dataset tiene {len(frame)} filas; el máximo es {MAX_ROWS}")
    frame = frame.dropna(how="all").reset_index(drop=True).copy()
    names: List[str] = []
    seen: Dict[str, int] = {}
    for index, column in enumerate(frame.columns):
        name = str(column).strip() or f"column_{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    frame.columns = names
    if frame.empty:
        raise core.ValidationError("El dataset no contiene filas con datos")
    return frame


def _read_frame(data: bytes, fmt: str) -> pd.DataFrame:
    """Parse ``data`` according to ``fmt``."""
    if len(data) > MAX_BYTES:
        raise core.ValidationError(
            f"El dataset supera el máximo de {MAX_BYTES // (1024 * 1024)} MiB"
        )
    buffer = io.BytesIO(data)
    try:
        if fmt in ("csv", "tsv"):
            return pd.read_csv(buffer, sep="\t" if fmt == "tsv" else ",")
        if fmt == "xlsx":
            return pd.read_excel(buffer, sheet_name=0)
        if fmt == "parquet":
            return pd.read_parquet(buffer)
    except ImportError as exc:
        raise core.ValidationError(
            f"Falta una dependencia para leer {fmt}: instala openpyxl/pyarrow"
        ) from exc
    except Exception as exc:
        raise core.ValidationError(f"No se pudo leer el dataset {fmt}: {exc}") from exc
    raise core.ValidationError(f"Formato de dataset no soportado: {fmt}")


def _check_host(host: str) -> None:
    """Reject loopback/private hosts to avoid server-side request forgery."""
    if not host:
        raise core.ValidationError("La URL no tiene host")
    try:
        addresses = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise core.ValidationError(f"No se pudo resolver el host de la URL: {host}") from exc
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise core.ValidationError(
                "La URL apunta a una dirección privada o local y no está permitida"
            )


def load_url(url: str) -> Tuple[pd.DataFrame, str, str, str]:
    """Download and parse a tabular URL."""
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https"):
        raise core.ValidationError("La URL debe usar http o https")
    _check_host(parsed.hostname or "")
    request = Request(text, headers={"User-Agent": "likelihood-training/1.0"})
    try:
        with urlopen(request, timeout=25) as response:
            final = urlparse(response.geturl())
            _check_host(final.hostname or "")
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                raise core.ValidationError(
                    f"El dataset supera el máximo de {MAX_BYTES // (1024 * 1024)} MiB"
                )
            data = response.read(MAX_BYTES + 1)
            content_type = (response.headers.get("Content-Type") or "").split(";")[0]
    except core.APIError:
        raise
    except Exception as exc:
        raise core.ValidationError(f"No se pudo descargar la URL: {exc}") from exc
    if len(data) > MAX_BYTES:
        raise core.ValidationError(
            f"El dataset supera el máximo de {MAX_BYTES // (1024 * 1024)} MiB"
        )
    name = Path(parsed.path).name or "dataset.csv"
    fmt = _format_from_name(name)
    suffix = Path(name).suffix.lower()
    if suffix not in (".csv", ".tsv", ".xlsx", ".parquet"):
        if "parquet" in content_type:
            fmt = "parquet"
        elif "spreadsheet" in content_type or "excel" in content_type:
            fmt = "xlsx"
        else:
            fmt = "csv"
    return _normalise_frame(_read_frame(data, fmt)), name, text, fmt


def load_bytes(data: bytes, filename: str) -> Tuple[pd.DataFrame, str, str, str]:
    """Parse an uploaded file."""
    name = Path(filename or "dataset.csv").name
    fmt = _format_from_name(name)
    return _normalise_frame(_read_frame(data, fmt)), name, name, fmt


def _column_meta(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Describe every column for the browser selectors."""
    columns: List[Dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        numeric = bool(pd.api.types.is_numeric_dtype(series)) and not bool(
            pd.api.types.is_bool_dtype(series)
        )
        sample = None
        for value in series.dropna().head(1):
            sample = _json_value(value)
            break
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "numeric": numeric,
                "sample": sample,
            }
        )
    return columns


def _preview(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """First rows of the dataset, JSON safe."""
    rows = frame.head(PREVIEW_ROWS).to_dict(orient="records")
    return [{str(key): _json_value(value) for key, value in row.items()} for row in rows]


class DatasetStore:
    """Small in-memory registry of parsed datasets."""

    def __init__(self, max_items: int = MAX_DATASETS) -> None:
        self._lock = threading.RLock()
        self._items: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._max_items = max(1, int(max_items))

    def put(self, name: str, frame: pd.DataFrame, source: str, fmt: str) -> Dict[str, Any]:
        """Register ``frame`` and return its public metadata."""
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        record = {
            "dataset_id": dataset_id,
            "name": name,
            "source": source,
            "format": fmt,
            "frame": frame,
            "created_at": _now(),
        }
        with self._lock:
            self._items[dataset_id] = record
            self._order.append(dataset_id)
            while len(self._order) > self._max_items:
                oldest = self._order.pop(0)
                self._items.pop(oldest, None)
        return self.public(dataset_id) or {}

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return the internal record (including the DataFrame)."""
        with self._lock:
            return self._items.get(str(dataset_id or "").strip())

    def public(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return the JSON-friendly description of one dataset."""
        record = self.get(dataset_id)
        if record is None:
            return None
        frame = record["frame"]
        return {
            "dataset_id": record["dataset_id"],
            "name": record["name"],
            "source": record["source"],
            "format": record["format"],
            "created_at": record["created_at"],
            "rows": int(len(frame)),
            "columns": _column_meta(frame),
            "preview": _preview(frame),
        }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    names = [str(name) for name in columns]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise core.ValidationError(f"Columnas no encontradas: {', '.join(missing)}")
    return names


def build_xy(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    task: str = "auto",
) -> Dict[str, Any]:
    """Build ``x``/``y`` arrays for a classification or regression target."""
    feature_names = _require_columns(frame, features)
    target_name = str(target or "").strip()
    if not feature_names:
        raise core.ValidationError("Selecciona al menos una columna X")
    if not target_name or target_name not in frame.columns:
        raise core.ValidationError("Selecciona una columna objetivo válida")
    if target_name in feature_names:
        raise core.ValidationError("La columna objetivo no puede formar parte de X")

    x_frame = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = [name for name in feature_names if (frame[name].notna() & x_frame[name].isna()).any()]
    if invalid:
        raise core.ValidationError(f"Las columnas X deben ser numéricas: {', '.join(invalid)}")

    data = x_frame.copy()
    data["__target__"] = frame[target_name].values
    data = data.dropna()
    if data.empty:
        raise core.ValidationError("No quedan filas con X y objetivo válidos")

    task_name = str(task or "auto").strip().lower()
    if task_name not in ("auto", "classification", "regression"):
        raise core.ValidationError("La tarea debe ser classification, regression o auto")
    if task_name == "auto":
        task_name = (
            "classification"
            if not pd.api.types.is_numeric_dtype(frame[target_name])
            else "regression"
        )

    classes: Optional[Dict[str, int]] = None
    if task_name == "classification":
        codes, uniques = pd.factorize(data["__target__"], sort=True)
        y = [int(code) for code in codes]
        classes = {str(value): index for index, value in enumerate(uniques)}
    else:
        y_series = pd.to_numeric(data["__target__"], errors="coerce")
        invalid_target = data["__target__"].notna() & y_series.isna()
        if invalid_target.any():
            raise core.ValidationError("La columna Y debe ser numérica para regresión")
        valid = ~y_series.isna()
        data = data.loc[valid]
        y_series = y_series.loc[valid]
        y = [float(value) for value in y_series]

    x = data[feature_names].astype(float).values.tolist()
    return {
        "mode": "xy",
        "task": task_name,
        "features": feature_names,
        "target": target_name,
        "rows": len(y),
        "x": x,
        "y": y,
        "classes": classes,
    }


def build_datapoints(frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Build a flat numeric ``datapoints`` sequence from one column."""
    target_name = str(target or "").strip()
    if not target_name or target_name not in frame.columns:
        raise core.ValidationError("Selecciona una columna de datapoints válida")
    series = pd.to_numeric(frame[target_name], errors="coerce")
    if (frame[target_name].notna() & series.isna()).any():
        raise core.ValidationError("La columna de datapoints debe ser numérica")
    values = [float(value) for value in series.dropna()]
    if not values:
        raise core.ValidationError("La columna de datapoints no contiene valores")
    return {
        "mode": "datapoints",
        "task": "",
        "features": [],
        "target": target_name,
        "rows": len(values),
        "datapoints": values,
    }


def parameter_overrides(
    model: Dict[str, Any],
    features: Sequence[str],
    task: str,
    classes: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Constructor values implied by the selected dataset."""
    names = {parameter["name"] for parameter in model.get("parameters") or []}
    overrides: Dict[str, Any] = {}
    for key in (
        "input_shape_parm",
        "n_features",
        "n_inputs",
        "input_dim",
        "n_columns",
        "state_dim",
    ):
        if key in names:
            overrides[key] = len(features)
    if task == "classification" and "num_classes" in names and classes:
        overrides["num_classes"] = len(classes)
    if task == "regression":
        for key in ("output_shape_parm", "output_dim"):
            if key in names:
                overrides[key] = 1
    return overrides
