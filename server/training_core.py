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
* Artifacts live under ``STORE_DIR/artifacts/<artifact_id>/``; the child writes
  the model using its native saver when available (or a Keras/pickle fallback)
  and this module adds ``artifact.json`` and the serving ``load_path``.  Deleting
  a fit job deletes its folder; an evaluate job keeps the artifact reference of
  the fit job that owns it.
* Nothing here duplicates model loading: artifacts are only *paths* that a later
  deployment (``serving_core``) or an evaluation job resolves through the
  catalog model's declared loaders.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import server_core as core
import training_datasets
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
#: CPU seconds a training child may consume before the kernel sends ``SIGXCPU``
#: (``LM_TRAINING_CPU``).  ``0`` (the default) disables the limit: the
#: wall-clock :data:`TRAIN_TIMEOUT` already bounds the job, while the playground
#: cap (30 s by default) kills multi-threaded TensorFlow long before its
#: wall-clock budget because the kernel counts CPU time across every thread.
TRAIN_CPU_SECONDS = max(0, int(os.environ.get("LM_TRAINING_CPU") or 0))

#: Method names that fit a model, in the order the catalogue prefers them.
FIT_METHODS = ("fit", "train", "fit_transform", "fit_predict", "fit_predict_transform")

#: Metrics offered by the training UI for target-based stopping / LR control.
TARGET_METRICS = (
    "loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "mae",
    "val_mae",
    "mse",
    "val_mse",
    "mape",
    "val_mape",
    "mean_absolute_percentage_error",
    "val_mean_absolute_percentage_error",
    "precision",
    "val_precision",
    "recall",
    "val_recall",
    "f1",
    "val_f1",
    "f1_score",
    "val_f1_score",
    "auc",
    "val_auc",
)

#: Metrics compiled into a Keras classifier by default.
CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1_score", "auc")

#: Actions available for a target metric control.
TARGET_ACTIONS = ("stop", "reduce_lr", "both")

#: Supported input-data normalizations.  Targets (``y``) are never changed.
NORMALIZATIONS = ("none", "standard", "minmax", "minmax_symmetric")

#: Method arguments that contain training features (not targets).
NORMALIZABLE_ARGUMENTS = ("x", "X", "datapoints")

#: Framework base classes whose inherited ``fit`` is not part of the AST catalogue.
FRAMEWORK_MODEL_BASES = frozenset({"Model"})

#: Arguments exposed by the server-side Keras ``fit`` adapter.
FRAMEWORK_FIT_PARAMETERS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "x",
        "type": "np.ndarray",
        "default": None,
        "required": True,
        "kind": "positional",
        "positional_only": False,
        "variadic": False,
        "description": "Datos de entrada (matriz JSON; el sandbox la convierte en ndarray).",
        "widget": "raw",
    },
    {
        "name": "y",
        "type": "np.ndarray",
        "default": None,
        "required": True,
        "kind": "positional",
        "positional_only": False,
        "variadic": False,
        "description": "Etiquetas o valores objetivo (lista/matriz JSON).",
        "widget": "raw",
    },
    {
        "name": "epochs",
        "type": "int",
        "default": "1",
        "required": False,
        "kind": "keyword_only",
        "positional_only": False,
        "variadic": False,
        "description": "Épocas de entrenamiento.",
        "widget": "int",
    },
    {
        "name": "batch_size",
        "type": "int",
        "default": "32",
        "required": False,
        "kind": "keyword_only",
        "positional_only": False,
        "variadic": False,
        "description": "Tamaño de lote.",
        "widget": "int",
    },
    {
        "name": "validation_split",
        "type": "float",
        "default": "0.0",
        "required": False,
        "kind": "keyword_only",
        "positional_only": False,
        "variadic": False,
        "description": "Fracción de datos reservada para validación.",
        "widget": "float",
    },
    {
        "name": "verbose",
        "type": "int",
        "default": "0",
        "required": False,
        "kind": "keyword_only",
        "positional_only": False,
        "variadic": False,
        "description": "Nivel de salida de Keras (0 silencioso).",
        "widget": "int",
    },
    {
        "name": "shuffle",
        "type": "bool",
        "default": "True",
        "required": False,
        "kind": "keyword_only",
        "positional_only": False,
        "variadic": False,
        "description": "Mezclar los datos en cada época.",
        "widget": "bool",
    },
)


def new_job_id() -> str:
    """Generate the unique identifier of one training job."""
    return f"trj-{uuid.uuid4().hex[:12]}"


def new_artifact_id() -> str:
    """Generate the unique identifier of one model artifact."""
    return f"art-{uuid.uuid4().hex[:12]}"


def artifact_path(artifact_id: str, directory: Optional[Path] = None) -> Path:
    """Directory that contains every file of ``artifact_id``.

    Each artifact gets its own folder so a model saved with its native
    ``save``/``save_model`` layout (a SavedModel directory, a ``.keras`` archive
    next to metadata, a pickle...) can be consumed later by Serving.
    ``directory`` lets a :class:`TrainingStore` injected with its own directory
    keep artifacts next to ``jobs.json``.
    """
    return Path(directory or STORE_DIR) / "artifacts" / artifact_id


def save_method_for(model: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(method_name, path_parameter)`` of the model's native save method."""
    for name in ("save", "save_model"):
        method = core.find_method(model, name)
        if method is None or method.get("kind") not in ("method", "static"):
            continue
        parameters = [
            parameter
            for parameter in method.get("parameters") or []
            if not parameter.get("variadic")
        ]
        if parameters:
            return name, parameters[0]["name"]
    return "", ""


def save_layout_for(model: Dict[str, Any]) -> str:
    """Return a special folder layout required by the model's Serving loader."""
    if save_method_for(model)[0]:
        return ""
    for loader in model.get("loaders") or []:
        if loader.get("name") == "load" and "autoencoders" in str(loader.get("module") or ""):
            # legacy AutoClassifier.load() expects config.json + component files.
            return "components"
    return ""


def remove_artifact(path: Path) -> bool:
    """Delete an artifact file or directory, returning whether it existed."""
    target = Path(path)
    try:
        if target.is_dir():
            shutil.rmtree(target)
            return True
        if target.is_file():
            target.unlink()
            return True
    except OSError:
        return False
    return False


def artifact_stats(path: Path) -> Tuple[int, List[str]]:
    """Size and relative filenames of a file or artifact directory."""
    target = Path(path)
    if target.is_file():
        try:
            return target.stat().st_size, [target.name]
        except OSError:
            return 0, []
    total = 0
    files: List[str] = []
    if not target.is_dir():
        return 0, []
    for item in sorted(target.rglob("*")):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
        files.append(str(item.relative_to(target)))
    return total, files


def artifact_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Small constructor values worth persisting for serving inference.

    Training datasets and large arrays are intentionally omitted; the artifact
    metadata only needs scalar shape/architecture values so Serving can build a
    payload example without loading the model in the API process.
    """
    kept: Dict[str, Any] = {}
    for name, value in (parameters or {}).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            kept[name] = value
    return kept


def normalization_method(raw: Any) -> str:
    """Normalize the friendly names accepted by the training UI/API."""
    value = str(raw or "none").strip().lower()
    aliases = {
        "std": "standard",
        "zscore": "standard",
        "z-score": "standard",
        "0-1": "minmax",
        "01": "minmax",
        "min-max": "minmax",
        "zero-one": "minmax",
        "-1-1": "minmax_symmetric",
        "minus-one-one": "minmax_symmetric",
        "symmetric": "minmax_symmetric",
    }
    value = aliases.get(value, value)
    if value not in NORMALIZATIONS:
        raise core.ValidationError(
            "El campo 'normalization' debe ser none, standard, minmax o minmax_symmetric"
        )
    return value


def _numeric_column(values: List[Any]) -> List[float]:
    """Validate one numeric column and return its finite float values."""
    numbers: List[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            raise core.ValidationError(
                "La normalización requiere datos numéricos en x/y/datapoints"
            )
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError as exc:
                raise core.ValidationError(
                    "La normalización requiere datos numéricos en x/y/datapoints"
                ) from exc
        if not isinstance(value, (int, float)):
            raise core.ValidationError(
                "La normalización requiere datos numéricos en x/y/datapoints"
            )
        number = float(value)
        if not math.isfinite(number):
            raise core.ValidationError(
                "La normalización requiere valores finitos en x/y/datapoints"
            )
        numbers.append(number)
    return numbers


def _scale_column(values: List[Any], method: str) -> Tuple[List[float], Dict[str, Any]]:
    """Scale one numeric column and return the metadata needed to invert it."""
    numbers = _numeric_column(values)
    if method == "none" or not numbers:
        return numbers, {"method": "none"}
    minimum = min(numbers)
    maximum = max(numbers)
    if method == "standard":
        mean = sum(numbers) / len(numbers)
        variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
        std = math.sqrt(variance)
        if std == 0:
            std = 1.0
        return [(value - mean) / std for value in numbers], {
            "method": "standard",
            "mean": mean,
            "std": std,
        }
    span = maximum - minimum
    stats: Dict[str, Any] = {"method": method, "min": minimum, "max": maximum}
    if span == 0:
        return [0.0 for _ in numbers], stats
    if method == "minmax":
        return [(value - minimum) / span for value in numbers], stats
    return [2.0 * (value - minimum) / span - 1.0 for value in numbers], stats


def _normalise_column(values: List[Any], method: str) -> List[float]:
    """Scale one numeric column with the selected normalization."""
    return _scale_column(values, method)[0]


def _scale_value(value: Any, method: str) -> Tuple[Any, Dict[str, Any]]:
    """Scale a flat sequence or a rectangular matrix, returning scaler metadata."""
    if method == "none" or not isinstance(value, list) or not value:
        return value, {"method": "none"}
    if all(not isinstance(item, (list, tuple)) for item in value):
        return _scale_column(value, method)
    rows: List[List[Any]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise core.ValidationError(
                "La normalización espera una matriz rectangular en x/y/datapoints"
            )
        rows.append(list(row))
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise core.ValidationError(
            "La normalización espera una matriz rectangular en x/y/datapoints"
        )
    columns: List[List[float]] = []
    stats: List[Dict[str, Any]] = []
    for index in range(width):
        scaled, column_stats = _scale_column([row[index] for row in rows], method)
        columns.append(scaled)
        stats.append(column_stats)
    scaled_rows = [[columns[index][row] for index in range(width)] for row in range(len(rows))]
    return scaled_rows, {"method": method, "columns": stats}


def _normalise_value(value: Any, method: str) -> Any:
    """Normalize a flat sequence or a rectangular matrix of features."""
    return _scale_value(value, method)[0]


def apply_normalization(
    arguments: Dict[str, Any], parameters: Dict[str, Any], method: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return copies of the feature arguments/parameters after scaling.

    ``y`` is deliberately excluded here: :func:`apply_target_normalization`
    scales a regression target only after the caller has resolved the task, so
    classification labels stay as integer codes.
    """
    if method == "none":
        return dict(arguments), dict(parameters)
    normalized_args = dict(arguments)
    normalized_params = dict(parameters)
    for name in NORMALIZABLE_ARGUMENTS:
        if name in normalized_args:
            normalized_args[name] = _normalise_value(normalized_args[name], method)
        if name in normalized_params:
            normalized_params[name] = _normalise_value(normalized_params[name], method)
    return normalized_args, normalized_params


#: Values of ``target_normalization`` that reuse the feature method on regression.
TARGET_NORMALIZATION_AUTO = ("", "auto", "same", "feature")


def target_normalization_method(raw: Any, task: str, feature_method: str) -> str:
    """Resolve the optional target scaling; only regression targets are scaled.

    ``auto`` (the default) reuses the feature normalization when the resolved
    task is a regression, and does nothing for classification labels (which are
    integer codes, not a continuous quantity).
    """
    task_name = str(task or "").strip().lower()
    value = str(raw or "").strip().lower()
    if value in TARGET_NORMALIZATION_AUTO:
        method = feature_method if task_name == "regression" else "none"
    else:
        method = normalization_method(value)
    if task_name == "classification" and method != "none":
        raise core.ValidationError(
            "La normalización de y solo está disponible para problemas de regresión"
        )
    return method


def apply_target_normalization(
    arguments: Dict[str, Any], method: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Scale the regression target ``y`` and return the scaler metadata.

    Classification labels are integer codes, so callers must resolve ``method``
    to ``none`` for a classification task before calling this helper.  The
    metadata is persisted in ``artifact.json`` so predictions can be mapped back
    to the original target units.
    """
    if method == "none" or "y" not in arguments:
        return dict(arguments), {}
    normalized = dict(arguments)
    normalized["y"], stats = _scale_value(normalized["y"], method)
    return normalized, stats


def supports_framework_fit(model: Dict[str, Any]) -> bool:
    """Whether ``model`` inherits a framework ``fit`` missing from the AST catalogue.

    ``model_catalog`` only merges methods inherited from classes inside the
    ``likelihood`` package, so ``keras.Model.fit`` never appears in
    ``model["methods"]``.  The training layer adds a small adapter for those
    models instead of modifying the package.
    """
    bases = {str(base or "").split(".")[-1] for base in model.get("bases") or []}
    return bool(bases & FRAMEWORK_MODEL_BASES) and str(model.get("module") or "").startswith(
        "likelihood.models"
    )


def framework_fit_method(model: Dict[str, Any]) -> Dict[str, Any]:
    """Synthetic catalogue entry for the inherited Keras ``fit`` method."""
    return {
        "name": "fit",
        "signature": "fit(x, y, epochs=1, batch_size=32, validation_split=0.0, verbose=0, shuffle=True)",
        "description": (
            "Entrena el modelo Keras con el compilador por defecto del servidor "
            "(optimizer/loss/metrics) o con el objeto 'compile' del job."
        ),
        "parameters": [dict(parameter) for parameter in FRAMEWORK_FIT_PARAMETERS],
        "kind": "method",
        "decorators": [],
        "inherited": False,
        "framework": "keras",
    }


def training_fit_method(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the declared fitting method or the Keras adapter, if any."""
    for name in FIT_METHODS:
        method = core.find_method(model, name)
        if method is not None:
            return method
    if supports_framework_fit(model):
        return framework_fit_method(model)
    return None


def fit_method_name(model: Dict[str, Any]) -> str:
    """Name of the fitting method declared by ``model`` (``fit`` if none matches)."""
    method = training_fit_method(model)
    return method["name"] if method else "fit"


def resolve_fit_method_name(model: Dict[str, Any], requested: str = "") -> str:
    """Map the generic ``fit``/empty request onto the fitting method of ``model``."""
    cleaned = str(requested or "").strip()
    if cleaned and (cleaned.lower() != "fit" or core.find_method(model, cleaned) is not None):
        return cleaned
    return fit_method_name(model)


def default_compile_options(model: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative Keras compile options for a framework ``fit``.

    Regression stays on ``mse`` by default.  ``mean_absolute_percentage_error``
    (alias ``mape``) is available as an explicit ``compile.loss`` and/or
    ``compile.metrics`` entry, and the target-metric control offers both names.
    """
    activation = str(parameters.get("classifier_activation") or "").strip().lower()
    classes = parameters.get("num_classes")
    if activation == "sigmoid" and classes == 1:
        return {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": list(CLASSIFICATION_METRICS),
        }
    if activation in ("softmax", "sigmoid") and isinstance(classes, int) and classes > 1:
        return {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": list(CLASSIFICATION_METRICS),
        }
    return {"optimizer": "adam", "loss": "mse"}


def resolve_compile_options(
    model: Dict[str, Any], parameters: Dict[str, Any], raw: Any
) -> Dict[str, Any]:
    """Merge the request's ``compile`` object with the server defaults.

    ``learning_rate`` is accepted as a convenience and validated here; the
    sandbox turns it into an optimizer instance because Keras' ``compile`` does
    not take a ``learning_rate`` keyword.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise core.ValidationError("El campo 'compile' debe ser un objeto")
    options = default_compile_options(model, parameters)
    options.update(raw)
    optimizer = options.get("optimizer")
    if optimizer is not None and not isinstance(optimizer, (str, dict)):
        raise core.ValidationError("El campo 'compile.optimizer' debe ser un nombre o un objeto")
    if isinstance(optimizer, str) and not optimizer.strip():
        options.pop("optimizer", None)
    learning_rate = options.get("learning_rate")
    if learning_rate is not None:
        if isinstance(learning_rate, bool):
            raise core.ValidationError("El campo 'compile.learning_rate' debe ser un número")
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as exc:
            raise core.ValidationError(
                "El campo 'compile.learning_rate' debe ser un número"
            ) from exc
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise core.ValidationError("El campo 'compile.learning_rate' debe ser mayor que 0")
        options["learning_rate"] = learning_rate
    return options


def normalize_training_control(raw: Any) -> Dict[str, Any]:
    """Validate the optional target-metric control of a Keras fit job.

    ``action`` may be ``stop`` (EarlyStopping at ``value``), ``reduce_lr``
    (ReduceLROnPlateau) or ``both``.  An empty control is allowed and means no
    callback is added.
    """
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise core.ValidationError("El campo 'control' debe ser un objeto")
    metric = str(raw.get("metric") or raw.get("monitor") or "").strip()
    if not metric:
        return {}
    action = str(raw.get("action") or "stop").strip().lower()
    action = {"early_stopping": "stop", "reduce": "reduce_lr"}.get(action, action)
    if action not in TARGET_ACTIONS:
        raise core.ValidationError(
            "El campo 'control.action' debe ser 'stop', 'reduce_lr' o 'both'"
        )
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in ("auto", "min", "max"):
        raise core.ValidationError("El campo 'control.mode' debe ser auto, min o max")

    def _optional_float(name: str, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise core.ValidationError(f"El campo 'control.{name}' debe ser un número")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise core.ValidationError(f"El campo 'control.{name}' debe ser un número") from exc

    def _float_value(
        name: str, value: Any, default: float, *, minimum: Optional[float] = None
    ) -> float:
        parsed = _optional_float(name, value)
        result = default if parsed is None else parsed
        if minimum is not None and result < minimum:
            raise core.ValidationError(f"El campo 'control.{name}' debe ser >= {minimum:g}")
        return result

    def _int_value(name: str, value: Any, default: int, *, minimum: int = 0) -> int:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            raise core.ValidationError(f"El campo 'control.{name}' debe ser un entero")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise core.ValidationError(f"El campo 'control.{name}' debe ser un entero") from exc
        if result < minimum:
            raise core.ValidationError(f"El campo 'control.{name}' debe ser >= {minimum}")
        return result

    value = _optional_float("value", raw.get("value", raw.get("target")))
    if action in ("stop", "both") and value is None:
        raise core.ValidationError(
            "El campo 'control.value' es obligatorio cuando la acción detiene el entrenamiento"
        )
    factor = _float_value("factor", raw.get("factor"), 0.5, minimum=0.000001)
    if factor >= 1:
        raise core.ValidationError("El campo 'control.factor' debe ser menor que 1")
    patience = _int_value("patience", raw.get("patience"), 3)
    min_delta = _float_value("min_delta", raw.get("min_delta"), 0.0, minimum=0.0)
    min_lr = _float_value("min_lr", raw.get("min_lr"), 0.000001, minimum=0.0)
    max_epochs = _int_value("max_epochs", raw.get("max_epochs", raw.get("epochs")), 50, minimum=1)
    return {
        "metric": metric,
        "mode": mode,
        "action": action,
        "value": value,
        "patience": patience,
        "min_delta": min_delta,
        "factor": factor,
        "min_lr": min_lr,
        "max_epochs": max_epochs,
        "restore_best_weights": bool(raw.get("restore_best_weights", False)),
    }


#: Curated examples for the framework models whose training objective is obvious.
EXAMPLE_JOBS: Dict[str, Dict[str, Any]] = {
    "AutoClassifier": {
        "method": "fit",
        "name": "AutoClassifier · clasificación",
        "label": "Clasificación",
        "note": "x: 12 filas × 4 variables; y: 3 clases (0, 1, 2).",
        "parameters": {
            "input_shape_parm": 4,
            "num_classes": 3,
            "units": 8,
            "activation": "relu",
            "classifier_activation": "softmax",
            "num_layers": 1,
        },
        "arguments": {
            "x": [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, 0.4, 0.3],
                [0.3, 0.4, 0.1, 0.2],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.2, 0.1, 0.3],
                [0.2, 0.5, 0.3, 0.1],
                [0.1, 0.3, 0.5, 0.2],
                [0.3, 0.1, 0.2, 0.5],
                [0.4, 0.2, 0.5, 0.1],
                [0.2, 0.4, 0.1, 0.5],
                [0.5, 0.1, 0.4, 0.2],
                [0.1, 0.5, 0.2, 0.4],
            ],
            "y": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "epochs": 2,
            "batch_size": 4,
        },
        "compile": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": list(CLASSIFICATION_METRICS),
        },
    },
    "GANRegressor": {
        "method": "fit",
        "name": "GANRegressor · regresión",
        "label": "Regresión",
        "note": "x: 8 filas × 3 variables; y: 8 valores continuos.",
        "parameters": {
            "input_shape_parm": 3,
            "output_shape_parm": 1,
            "num_neurons": 8,
            "depth": 2,
            "dropout": 0.0,
        },
        "arguments": {
            "x": [
                [0.1, 0.2, 0.3],
                [0.2, 0.4, 0.1],
                [0.3, 0.1, 0.5],
                [0.4, 0.3, 0.2],
                [0.5, 0.2, 0.4],
                [0.2, 0.5, 0.3],
                [0.3, 0.4, 0.1],
                [0.1, 0.3, 0.5],
            ],
            "y": [[0.5], [0.7], [0.6], [0.8], [0.4], [0.9], [0.55], [0.65]],
            "epochs": 2,
            "batch_size": 4,
        },
        "compile": {"optimizer": "adam", "loss": "mse", "metrics": ["mae"]},
    },
}


def training_example(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a ready-to-run example payload for ``model``, when available."""
    if core.unsupported_parameters(model):
        return None
    fit = training_fit_method(model)
    if fit is None:
        return None
    curated = EXAMPLE_JOBS.get(model["id"])
    if curated:
        example = copy.deepcopy(curated)
        example["method"] = fit["name"]
        return example
    try:
        parameters = core.constructor_samples(model)
        arguments = core.sample_arguments(model, fit, parameters)
    except (core.APIError, TypeError, ValueError):
        return None
    return {
        "method": fit["name"],
        "name": f"{model['id']} · ejemplo",
        "label": "",
        "note": "",
        "parameters": parameters,
        "arguments": arguments,
        "compile": {},
    }


def training_dataset_mode(model: Dict[str, Any]) -> str:
    """How a dataset maps onto ``model``: ``xy``, ``datapoints`` or ``generic``."""
    fit = training_fit_method(model)
    if fit is None:
        return "generic"
    names = {parameter["name"] for parameter in fit.get("parameters") or []}
    if {"x", "y"} <= names or {"X", "y"} <= names:
        return "xy"
    if any(parameter.get("name") == "datapoints" for parameter in model.get("parameters") or []):
        return "datapoints"
    return "generic"


def training_dataset_task(model: Dict[str, Any]) -> str:
    """The task implied by a curated example, if the model has one."""
    example = training_example(model) or {}
    label = str(example.get("label") or "").lower()
    if label.startswith("clas"):
        return "classification"
    if label.startswith("regres"):
        return "regression"
    return "auto"


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
        self.datasets = training_datasets.DatasetStore()
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
            "compile": dict(record.get("compile") or {}),
            "control": dict(record.get("control") or {}),
            "normalization": str(record.get("normalization") or "none"),
            "target_normalization": dict(record.get("target_normalization") or {}),
            "adapter": str(record.get("adapter") or ""),
            "dataset": dict(record.get("dataset") or {}) or None,
            "timeout_seconds": float(record.get("timeout_seconds") or TRAIN_TIMEOUT),
            "attempts": int(record.get("attempts") or 0),
            "created_at": str(record.get("created_at") or iso()),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "elapsed": record.get("elapsed"),
            "compute_seconds": record.get("compute_seconds"),
            "error": str(record.get("error") or ""),
            "error_detail": str(record.get("error_detail") or ""),
            "result_type": record.get("result_type"),
            "result_repr": record.get("result_repr"),
            "result": record.get("result"),
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
        compile_options: Dict[str, Any] = {}
        control: Dict[str, Any] = {}
        normalization = "none"
        target_normalization: Dict[str, Any] = {}
        dataset_meta: Optional[Dict[str, Any]] = None
        dataset_args: Dict[str, Any] = {}
        dataset_params: Dict[str, Any] = {}
        if kind == "fit":
            method_name = resolve_fit_method_name(model, str(spec.get("method") or ""))
            dataset_spec = spec.get("dataset")
            if dataset_spec:
                dataset_args, dataset_params, dataset_meta = self._dataset_values(
                    model, dataset_spec
                )
                raw_arguments = spec.get("arguments", spec.get("args")) or {}
                raw_parameters = spec.get("parameters", spec.get("params")) or {}
                if not isinstance(raw_arguments, dict):
                    raise core.ValidationError("El campo 'arguments' debe ser un objeto")
                if not isinstance(raw_parameters, dict):
                    raise core.ValidationError("El campo 'parameters' debe ser un objeto")
                validation_spec = dict(spec)
                validation_spec["arguments"] = {**raw_arguments, **dataset_args}
                validation_spec["parameters"] = {**raw_parameters, **dataset_params}
            else:
                validation_spec = spec
            method, arguments, parameters = self._prepare_invocation(
                model, validation_spec, method_name
            )
            normalization = normalization_method(spec.get("normalization"))
            arguments, parameters = apply_normalization(arguments, parameters, normalization)
            task = (
                str((dataset_meta or {}).get("task") or "")
                if dataset_meta
                else training_dataset_task(model)
            )
            target_method = target_normalization_method(
                spec.get("target_normalization"), task, normalization
            )
            if target_method != "none":
                arguments, target_stats = apply_target_normalization(arguments, target_method)
                if target_stats:
                    target_normalization = {"method": target_method, **target_stats}
            if method["name"].lower() in ("save", "load", "load_model", "save_model"):
                raise core.ValidationError(
                    f'El método "{method["name"]}" guarda o carga un modelo; '
                    "usa fit/train para entrenar."
                )
            if method.get("framework") == "keras":
                compile_options = resolve_compile_options(
                    model,
                    parameters,
                    spec.get("compile", spec.get("compile_options")),
                )
                control = normalize_training_control(spec.get("control"))
            elif spec.get("control"):
                raise core.ValidationError(
                    "El control por métrica objetivo solo está disponible para "
                    "entrenamientos Keras con fit"
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
        stored_arguments = dict(arguments)
        for key in dataset_args:
            stored_arguments.pop(key, None)
        if control.get("max_epochs"):
            # The target control implies a larger training budget than the
            # Keras adapter default (epochs=1), otherwise the callback would
            # never have a chance to stop later.
            stored_arguments["epochs"] = int(control["max_epochs"])
        stored_parameters = dict(parameters)
        # ``datapoints`` can be long (a whole CSV column); keep it in the
        # dataset reference and rebuild it in the worker.
        if "datapoints" in dataset_params:
            stored_parameters.pop("datapoints", None)
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
                    "arguments": stored_arguments,
                    "parameters": stored_parameters,
                    "compile": compile_options,
                    "control": control,
                    "normalization": normalization,
                    "target_normalization": target_normalization,
                    "adapter": method.get("framework") or "",
                    "dataset": dataset_meta,
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

    @staticmethod
    def _training_method(model: Dict[str, Any], method_name: str) -> Dict[str, Any]:
        """Resolve a catalog method, or the synthetic Keras ``fit`` adapter."""
        try:
            return core.method_for_invocation(model, method_name)
        except core.ModelNotFound:
            if str(method_name or "").strip().lower() == "fit" and supports_framework_fit(model):
                return framework_fit_method(model)
            raise

    def _dataset_values(
        self, model: Dict[str, Any], dataset_spec: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Expand a dataset reference into method arguments and constructor overrides."""
        if not isinstance(dataset_spec, dict):
            raise core.ValidationError("El campo 'dataset' debe ser un objeto")
        dataset_id = str(dataset_spec.get("dataset_id") or "").strip()
        record = self.datasets.get(dataset_id)
        if record is None:
            raise core.ModelNotFound(f'Dataset no encontrado: "{dataset_id}"')
        mode = str(dataset_spec.get("mode") or "").strip().lower()
        task = str(dataset_spec.get("task") or "auto").strip().lower()
        target = str(dataset_spec.get("target") or "").strip()
        features = dataset_spec.get("features") or []
        if not isinstance(features, list):
            raise core.ValidationError("El campo 'features' del dataset debe ser una lista")
        frame = record["frame"]
        args: Dict[str, Any] = {}
        params: Dict[str, Any] = {}
        if mode == "datapoints":
            built = training_datasets.build_datapoints(frame, target)
            params = {"datapoints": built["datapoints"]}
            summary = (
                f"{record['name']} · {built['rows']} valores → "
                f"datapoints desde '{built['target']}'"
            )
        elif mode == "xy":
            built = training_datasets.build_xy(frame, features, target, task)
            args = {"x": built["x"], "y": built["y"]}
            params = training_datasets.parameter_overrides(
                model, built["features"], built["task"], built.get("classes")
            )
            summary = (
                f"{record['name']} · {built['rows']} filas · "
                f"X: {len(built['features'])} columnas → {built['target']}"
            )
        else:
            raise core.ValidationError("El modo del dataset debe ser 'xy' o 'datapoints'")
        meta = {
            "dataset_id": dataset_id,
            "name": record.get("name") or "",
            "source": record.get("source") or "",
            "format": record.get("format") or "",
            "mode": mode,
            "task": built.get("task") or task,
            "features": built.get("features") or [],
            "target": built.get("target") or target,
            "rows": built.get("rows") or 0,
            "classes": built.get("classes"),
            "summary": summary,
        }
        return args, params, meta

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
        method = self._training_method(model, method_name)
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
                result_value = outcome.get("result")
                if isinstance(result_value, dict) and isinstance(result_value.get("history"), dict):
                    job["result"] = {"history": result_value["history"]}
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
                if outcome.get("save_warning"):
                    job["log"].append(
                        {"at": iso(), "level": "warning", "message": outcome["save_warning"]}
                    )
                if outcome.get("save_method"):
                    job["log"].append(
                        {
                            "at": iso(),
                            "level": "info",
                            "message": f"Artefacto guardado con {outcome['save_method']}",
                        }
                    )
                if job["kind"] == "fit":
                    artifact_dir = str((job.get("artifact") or {}).get("path") or "")
                    if artifact_dir:
                        folder = Path(artifact_dir)
                        size, files = artifact_stats(folder)
                        load_path = str(outcome.get("load_path") or "")
                        if not load_path and files:
                            load_path = artifact_dir
                        try:
                            folder.mkdir(parents=True, exist_ok=True)
                            loader_name = ""
                            try:
                                model_entry = self._find_model(job["model_id"])
                                loader_name = core.resolve_loader(model_entry).get("name") or ""
                            except core.APIError:
                                pass
                            metadata = {
                                "artifact_id": job["artifact"].get("artifact_id"),
                                "job_id": job["job_id"],
                                "model_id": job["model_id"],
                                "model_name": job.get("model_name") or "",
                                "method": job["method"],
                                "parameters": artifact_parameters(job.get("parameters") or {}),
                                "normalization": job.get("normalization") or "none",
                                "target_normalization": dict(job.get("target_normalization") or {}),
                                "created_at": iso(),
                                "load_path": load_path,
                                "loader": loader_name,
                                "files": files,
                                "size_bytes": size,
                            }
                            _atomic_write(
                                folder / "artifact.json",
                                json.dumps(metadata, ensure_ascii=False, indent=2),
                            )
                            size, files = artifact_stats(folder)
                        except OSError as exc:
                            job["log"].append(
                                {
                                    "at": iso(),
                                    "level": "warning",
                                    "message": f"No se pudo escribir artifact.json: {exc}",
                                }
                            )
                        job["artifact"]["size_bytes"] = size
                        job["artifact"]["files"] = files
                        job["artifact"]["load_path"] = load_path or artifact_dir
                        job["artifact"]["save_method"] = outcome.get("save_method") or ""
                        job["artifact"]["metadata_path"] = str(folder / "artifact.json")
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
                detail = str(getattr(exc, "detail", "") or "")
                if detail:
                    job["error_detail"] = detail[-8000:]
                job["updated_at"] = iso()
                job["log"].append({"at": iso(), "level": "error", "message": str(exc)})
                if detail:
                    # Full tracebacks are available in ``error_detail`` (shown
                    # in the error card); keeping them out of the timeline
                    # avoids mixing Keras progress with real log events.
                    job["log"].append(
                        {
                            "at": iso(),
                            "level": "error",
                            "message": "Detalle técnico disponible en la tarjeta de error",
                        }
                    )
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
        method = self._training_method(model, job["method"])
        if method is None:
            raise core.ModelNotFound(
                f'El modelo "{model["id"]}" ya no declara el método "{job["method"]}"'
            )
        save_path = str((job.get("artifact") or {}).get("path") or "")
        if not save_path:
            raise core.ValidationError("El trabajo no tiene una ruta de artefacto")
        parameters = dict(job["parameters"])
        arguments = dict(job["arguments"])
        control = dict(job.get("control") or {})
        if control.get("max_epochs"):
            arguments["epochs"] = int(control["max_epochs"])
        normalization = str(job.get("normalization") or "none")
        if job.get("dataset"):
            data_args, data_params, _ = self._dataset_values(model, job["dataset"])
            arguments.update(data_args)
            parameters.update(data_params)
            if normalization != "none":
                arguments, parameters = apply_normalization(arguments, parameters, normalization)
            target_method = str((job.get("target_normalization") or {}).get("method") or "none")
            if target_method != "none":
                arguments, target_stats = apply_target_normalization(arguments, target_method)
                if target_stats:
                    job["target_normalization"] = {"method": target_method, **target_stats}
        save_method, save_parameter = save_method_for(model)
        result = core.run_invoke(
            model,
            method,
            parameters,
            arguments,
            timeout=float(job["timeout_seconds"]),
            save_path=save_path,
            save_options={
                "method": save_method,
                "path_parameter": save_parameter,
                "layout": save_layout_for(model),
                "filename": "model",
                "pickle_name": "model.pkl",
            },
            compile_options=dict(job.get("compile") or {}) or None,
            training_controls=dict(job.get("control") or {}) or None,
            limits={"cpu": TRAIN_CPU_SECONDS},
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
        path = str(artifact.get("load_path") or artifact.get("path") or "")
        load = None
        if path:
            loader = core.resolve_loader(model)
            load = {
                "name": loader["name"],
                "path": path,
                "path_parameter": loader.get("path_parameter") or "path",
                "requires_instance": bool(loader.get("requires_instance")),
                "module": loader.get("module") or model["module"],
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
            limits={"cpu": TRAIN_CPU_SECONDS},
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
                    deleted_artifact = remove_artifact(Path(path_value))
                    for key in ("size_bytes", "files", "load_path", "save_method", "metadata_path"):
                        job["artifact"].pop(key, None)
                    if not deleted_artifact:
                        job["log"].append(
                            {
                                "at": iso(),
                                "level": "warning",
                                "message": "No se pudo borrar el artefacto",
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
                    "ok" if Path(str(record.get("path") or "")).exists() else "missing"
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
                deleted = remove_artifact(Path(str(artifact.get("path") or "")))
                for key in ("size_bytes", "files", "load_path", "save_method", "metadata_path"):
                    artifact.pop(key, None)
                if not deleted:
                    job["log"].append(
                        {
                            "at": iso(),
                            "level": "warning",
                            "message": "No se pudo borrar el artefacto",
                        }
                    )
                break
        return {"id": wanted, "deleted": deleted}

    # -- helpers ---------------------------------------------------------------

    def _find_model(self, model_id: str) -> Dict[str, Any]:
        """Resolve ``model_id`` against the catalog (the single source of truth)."""
        model = core.MODELS_BY_ID.get(str(model_id or "").strip())
        if not model:
            raise core.ModelNotFound(f'Modelo no encontrado: "{model_id}"')
        return model
