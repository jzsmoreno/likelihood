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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
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
#: Seed of the reproducible random sample used by ``sample_rows`` options.
SAMPLE_SEED = 42
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
                "unique": int(series.nunique(dropna=True)),
                "numeric": numeric,
                "sample": sample,
            }
        )
    return columns


def _preview(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """First rows of the dataset, JSON safe."""
    rows = frame.head(PREVIEW_ROWS).to_dict(orient="records")
    return [{str(key): _json_value(value) for key, value in row.items()} for row in rows]


def _sample_rows(data: Any, rows: Optional[int]) -> Any:
    """Keep every row, or a reproducible random sample of ``rows``.

    A random sample (instead of ``head``) keeps every class/region represented
    even when the file is sorted (the Iris CSV is grouped by species).
    """
    if rows is None or rows <= 0 or rows >= len(data):
        return data
    return data.sample(n=int(rows), random_state=SAMPLE_SEED)


def _limit_series(data: Any, rows: Optional[int]) -> Any:
    """Keep the most recent ``rows`` values (time order is never shuffled)."""
    if rows is None or rows <= 0 or rows >= len(data):
        return data
    return data.tail(int(rows))


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
    rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Build ``x``/``y`` arrays for a classification or regression target.

    ``rows`` keeps a reproducible random sample of that size instead of every
    valid row; ``None`` uses the whole dataset.
    """
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
        data = _sample_rows(data, rows)
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
        data = _sample_rows(data, rows)
        y_series = y_series.loc[data.index]
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


def build_datapoints(
    frame: pd.DataFrame, target: str, rows: Optional[int] = None
) -> Dict[str, Any]:
    """Build a flat numeric ``datapoints`` sequence from one column.

    ``rows`` keeps a reproducible random sample of that size instead of every
    value; ``None`` uses the whole column.
    """
    target_name = str(target or "").strip()
    if not target_name or target_name not in frame.columns:
        raise core.ValidationError("Selecciona una columna de datapoints válida")
    series = pd.to_numeric(frame[target_name], errors="coerce")
    if (frame[target_name].notna() & series.isna()).any():
        raise core.ValidationError("La columna de datapoints debe ser numérica")
    series = series.dropna()
    if not len(series):
        raise core.ValidationError("La columna de datapoints no contiene valores")
    series = _limit_series(series, rows)
    values = [float(value) for value in series]
    return {
        "mode": "datapoints",
        "task": "",
        "features": [],
        "target": target_name,
        "series": [target_name],
        "rows": len(values),
        "datapoints": values,
    }


def build_series(
    frame: pd.DataFrame,
    columns: Sequence[str],
    rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a ``(series, points)`` matrix from one or more numeric columns.

    ``FourierRegression`` denoises and forecasts every row of the matrix, so a
    single selected column still becomes a ``1 × n`` matrix.  ``rows`` keeps the
    most recent values without reordering the series.
    """
    names = _require_columns(frame, columns)
    if not names:
        raise core.ValidationError("Selecciona al menos una columna de la serie")
    selected = frame[names].apply(pd.to_numeric, errors="coerce")
    invalid = [name for name in names if (frame[name].notna() & selected[name].isna()).any()]
    if invalid:
        raise core.ValidationError(
            f"Las columnas de la serie deben ser numéricas: {', '.join(invalid)}"
        )
    data = selected.dropna()
    if data.empty:
        raise core.ValidationError("No quedan puntos con valores válidos")
    data = _limit_series(data, rows)
    values = data.astype(float).values.T.tolist()
    return {
        "mode": "datapoints",
        "task": "",
        "features": [],
        "target": names[0],
        "series": names,
        "rows": len(data),
        "indices": data.index.tolist(),
        "datapoints": values,
    }


#: Column names that usually hold the time axis of a series dataset.
DATE_COLUMN_HINTS = ("date", "fecha", "month", "mes", "time", "timestamp", "period", "ds")

#: Scaling methods accepted by the series analysis / DataScaler-compatible scaler.
SCALING_METHODS = ("detrend_minmax", "minmax_symmetric", "minmax", "standard", "none")


def scaling_method(raw: Any) -> str:
    """Validate the scaling selector of a series analysis."""
    value = str(raw or "detrend_minmax").strip().lower()
    aliases = {
        "detrend": "detrend_minmax",
        "data_scaler": "detrend_minmax",
        "datascaler": "detrend_minmax",
        "[-1,1]": "minmax_symmetric",
        "symmetric": "minmax_symmetric",
        "0-1": "minmax",
        "zscore": "standard",
        "std": "standard",
        "raw": "none",
    }
    value = aliases.get(value, value)
    if value not in SCALING_METHODS:
        raise core.ValidationError(
            "El escalado debe ser detrend_minmax, minmax_symmetric, minmax, standard o none"
        )
    return value


def fit_series_scaler(
    matrix: Sequence[Sequence[float]], method: str = "detrend_minmax"
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Scale every series with the selected method and return its statistics.

    * ``detrend_minmax`` — DataScaler: linear detrend + min/max to [-1, 1].
    * ``minmax_symmetric`` — min/max to [-1, 1] without detrending.
    * ``minmax`` — min/max to [0, 1] without detrending.
    * ``standard`` — z-score without detrending.
    * ``none`` — original values.
    """
    method = scaling_method(method)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise core.ValidationError("El escalado de series espera una matriz (series, puntos)")
    if method == "none":
        return array.tolist(), [{"mode": "none"} for _ in range(array.shape[0])]
    scaled = np.zeros_like(array)
    stats: List[Dict[str, Any]] = []
    for index, row in enumerate(array):
        x = np.arange(row.shape[0], dtype=float)
        if method == "detrend_minmax" and row.shape[0] > 1:
            coefficients = np.polyfit(x, row, 1)
        else:
            coefficients = np.array([0.0, 0.0])
        detrended = row - np.polyval(coefficients, x)
        if method == "standard":
            mean = float(np.mean(detrended))
            std = float(np.std(detrended)) or 1.0
            scaled[index, :] = (detrended - mean) / std
            stats.append({"mode": method, "mean": mean, "std": std})
            continue
        low = float(np.min(detrended))
        high = float(np.max(detrended))
        span = high - low or 1.0
        if method == "minmax":
            scaled[index, :] = (detrended - low) / span
        else:
            scaled[index, :] = 2.0 * (detrended - low) / span - 1.0
        stats.append(
            {
                "mode": method,
                "coef": [float(coefficients[0]), float(coefficients[1])],
                "low": low,
                "span": span,
            }
        )
    return scaled.tolist(), stats


def apply_series_scaler(
    values: Sequence[Sequence[float]],
    stats: Sequence[Dict[str, Any]],
    start: int = 0,
) -> List[List[float]]:
    """Inverse of :func:`fit_series_scaler` for values starting at ``start``."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    restored = np.zeros_like(array)
    for index, row in enumerate(array):
        stat = (
            stats[index]
            if index < len(stats)
            else {"mode": "detrend_minmax", "coef": [0.0, 0.0], "low": 0.0, "span": 1.0}
        )
        mode = str(stat.get("mode") or "detrend_minmax")
        if mode == "none":
            restored[index, :] = row
            continue
        if mode == "standard":
            restored[index, :] = row * stat["std"] + stat["mean"]
            continue
        if mode == "minmax":
            restored[index, :] = row * stat["span"] + stat["low"]
            continue
        x = np.arange(int(start), int(start) + row.shape[0], dtype=float)
        coefficients = stat["coef"]
        restored[index, :] = (row + 1.0) / 2.0 * stat["span"] + stat["low"]
        if mode == "detrend_minmax":
            restored[index, :] += coefficients[0] * x + coefficients[1]
    return restored.tolist()


def series_label_column(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Best column for x labels: a date-like column outside the series."""
    selected = {str(column) for column in columns}
    for name in frame.columns:
        if str(name) in selected:
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(
            series
        ):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return str(name)
    for name in frame.columns:
        if str(name).lower() in DATE_COLUMN_HINTS and str(name) not in selected:
            return str(name)
    return ""


def series_labels(frame: pd.DataFrame, column: str, indices: Sequence[Any]) -> List[str]:
    """Labels for the rows of a built series (dates when available)."""
    if not column or column not in frame.columns:
        return [str(index) for index in range(len(indices))]
    values = frame.loc[list(indices), column]
    labels: List[str] = []
    for value in values:
        text = str(value)
        # ``Timestamp`` strings carry a redundant midnight suffix.
        if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
            text = text[:10]
        labels.append(text)
    return labels


def future_series_labels(
    frame: pd.DataFrame, column: str, indices: Sequence[Any], steps: int
) -> List[str]:
    """Extrapolated labels for a forecast (dates when the axis is periodic)."""
    sample = ""
    if column and column in frame.columns and len(indices):
        sample = str(frame.loc[list(indices), column].iloc[-1])
    if column and column in frame.columns and len(indices) > 2:
        dates = pd.to_datetime(frame.loc[list(indices), column], errors="coerce")
        if dates.notna().all():
            with warnings.catch_warnings():
                # pandas 2.x announces a future ``infer_freq`` return type.
                warnings.simplefilter("ignore")
                frequency = pd.infer_freq(dates)
            if frequency is not None and not isinstance(frequency, str):
                frequency = getattr(frequency, "freqstr", str(frequency))
            if frequency:
                future = pd.date_range(dates.iloc[-1], periods=int(steps) + 1, freq=frequency)[1:]
                fmt = "%Y-%m" if len(sample) == 7 else "%Y-%m-%d"
                return [value.strftime(fmt) for value in future]
    return ["t+" + str(offset) for offset in range(1, int(steps) + 1)]


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
