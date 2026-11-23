"""K-fold cross-validation helpers for the training layer.

The sandbox fits a single model per child process, so the training store
orchestrates cross-validation itself: it splits the dataset, calls the sandbox
once per fold with a ``validation_data`` pair and keeps the artifact of the best
fold.  This module contains the pure functions behind that orchestration so
they can be unit-tested without TensorFlow.

Run with::

    python3 -m pytest tests/test_cross_validation.py -v
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import server_core as core

#: Folds used when the request only enables cross-validation.
DEFAULT_FOLDS = 5

#: Bounds accepted by the wizard and the API.
MIN_FOLDS = 2
MAX_FOLDS = 10

#: Deterministic default so two runs of the same job are comparable.
DEFAULT_SEED = 42

#: Metric used to pick the best fold when the request does not choose one.
DEFAULT_METRIC = "val_loss"

#: Metric-name hints that should be minimised when ``mode`` is ``auto``.
MINIMISE_TOKENS = ("loss", "error", "mae", "mse", "rmse", "mape")


def infer_mode(metric: str) -> str:
    """Return ``min`` for loss-like metrics and ``max`` for the rest."""
    name = str(metric or "").lower()
    return "min" if any(token in name for token in MINIMISE_TOKENS) else "max"


def normalise_options(raw: Any) -> Dict[str, Any]:
    """Validate the ``cross_validation`` object of a training request.

    Parameters
    ----------
    raw : Any
        Request value.  ``True`` enables the defaults; a mapping may set
        ``folds`` (or ``k``), ``metric``, ``weights``, ``mode``, ``shuffle`` and
        ``seed``.  ``metric`` accepts a history key (``val_loss``,
        ``val_accuracy``…) or ``balanced`` to combine several metrics.

    Returns
    -------
    Dict[str, Any]
        ``{"folds", "metric", "weights", "mode", "shuffle", "seed"}`` ready to
        store on the job record.

    Raises
    ------
    server_core.ValidationError
        When a field has an invalid value.
    """
    if raw is None or raw is False:
        return {}
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise core.ValidationError("El campo 'cross_validation' debe ser un objeto")
    folds = raw.get("folds", raw.get("k", DEFAULT_FOLDS))
    if isinstance(folds, bool):
        raise core.ValidationError(
            "El campo 'cross_validation.folds' debe ser un entero entre %d y %d"
            % (MIN_FOLDS, MAX_FOLDS)
        )
    try:
        folds = int(folds)
    except (TypeError, ValueError) as exc:
        raise core.ValidationError(
            "El campo 'cross_validation.folds' debe ser un entero entre %d y %d"
            % (MIN_FOLDS, MAX_FOLDS)
        ) from exc
    if folds < MIN_FOLDS or folds > MAX_FOLDS:
        raise core.ValidationError(
            "El campo 'cross_validation.folds' debe estar entre %d y %d" % (MIN_FOLDS, MAX_FOLDS)
        )
    metric = str(raw.get("metric") or DEFAULT_METRIC).strip() or DEFAULT_METRIC
    if metric.lower() in ("balanced", "balance", "composite", "auto"):
        metric = "balanced"
    weights_raw = raw.get("weights")
    weights: Dict[str, float] = {}
    if weights_raw is not None:
        if not isinstance(weights_raw, dict):
            raise core.ValidationError("El campo 'cross_validation.weights' debe ser un objeto")
        for name, value in weights_raw.items():
            label = str(name or "").strip()
            if not label:
                raise core.ValidationError(
                    "Las claves de 'cross_validation.weights' deben ser nombres de métrica"
                )
            if isinstance(value, bool):
                raise core.ValidationError(
                    "El peso de la métrica %r debe ser un número mayor que 0" % label
                )
            try:
                weight = float(value)
            except (TypeError, ValueError) as exc:
                raise core.ValidationError(
                    "El peso de la métrica %r debe ser un número mayor que 0" % label
                ) from exc
            if not math.isfinite(weight) or weight <= 0:
                raise core.ValidationError(
                    "El peso de la métrica %r debe ser un número mayor que 0" % label
                )
            weights[label] = weight
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in ("auto", "min", "max"):
        raise core.ValidationError("El campo 'cross_validation.mode' debe ser auto, min o max")
    if metric == "balanced":
        mode = "max"
    elif mode == "auto":
        mode = infer_mode(metric)
    shuffle = raw.get("shuffle", True)
    seed = raw.get("seed", DEFAULT_SEED if shuffle else None)
    if seed is not None and seed != "":
        if isinstance(seed, bool):
            raise core.ValidationError("El campo 'cross_validation.seed' debe ser un entero")
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise core.ValidationError(
                "El campo 'cross_validation.seed' debe ser un entero"
            ) from exc
    else:
        seed = None
    if bool(shuffle) and seed is None:
        seed = DEFAULT_SEED
    return {
        "folds": folds,
        "metric": metric,
        "weights": weights,
        "mode": mode,
        "shuffle": bool(shuffle),
        "seed": seed,
    }


def make_folds(
    rows: int,
    folds: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Split ``rows`` samples into ``folds`` train/validation index pairs.

    The validation blocks are contiguous after the optional shuffle (the usual
    ``KFold`` behaviour).  Sizes differ by at most one sample and every sample
    appears exactly once in the validation sets.

    Parameters
    ----------
    rows : int
        Number of samples.
    folds : int
        Number of folds.
    shuffle : bool, optional
        Shuffle the sample order before splitting (default ``True``).
    seed : int, optional
        Seed of the shuffle.  ``None`` uses :data:`DEFAULT_SEED`.

    Returns
    -------
    List[Tuple[List[int], List[int]]]
        ``[(train_indices, validation_indices), ...]``.

    Raises
    ------
    server_core.ValidationError
        When there are fewer samples than folds or the fold count is invalid.
    """
    try:
        rows = int(rows)
        folds = int(folds)
    except (TypeError, ValueError) as exc:
        raise core.ValidationError("La validación cruzada necesita un dataset con filas") from exc
    if folds < MIN_FOLDS or folds > MAX_FOLDS:
        raise core.ValidationError(
            "El campo 'cross_validation.folds' debe estar entre %d y %d" % (MIN_FOLDS, MAX_FOLDS)
        )
    if rows < folds:
        raise core.ValidationError(
            "El dataset tiene %d filas y se piden %d folds; reduce los folds" % (rows, folds)
        )
    order = list(range(rows))
    if shuffle:
        random.Random(DEFAULT_SEED if seed is None else seed).shuffle(order)
    base, remainder = divmod(rows, folds)
    splits: List[Tuple[List[int], List[int]]] = []
    start = 0
    for index in range(folds):
        size = base + (1 if index < remainder else 0)
        validation = order[start : start + size]
        training = order[:start] + order[start + size :]
        splits.append((training, validation))
        start += size
    return splits


def select_rows(values: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    """Return ``values[i]`` for every index in ``indices`` (row order kept)."""
    return [values[index] for index in indices]


def fold_score(history: Dict[str, Any], metric: str) -> Optional[float]:
    """Last finite value of ``metric`` in a Keras history, or ``None``."""
    values = (history or {}).get(metric)
    if not isinstance(values, (list, tuple)):
        return None
    for value in reversed(values):
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _finite(value: Any) -> bool:
    """Whether ``value`` is a real, finite number (bools excluded)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def composite_scores(
    fold_metrics: Sequence[Dict[str, Any]],
    weights: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Optional[float]], List[Dict[str, Any]]]:
    """Combine several per-fold metrics into one 0..1 quality score per fold.

    Each metric is normalised across the folds so that higher is always better
    (``loss``/``mae``/``mse``/``rmse``/``mape`` are inverted) and then averaged
    with the requested weights.  Without explicit weights every validation
    metric present in all folds is combined with weight 1.

    Parameters
    ----------
    fold_metrics : Sequence[Dict[str, Any]]
        Last-epoch metrics of every fold (``{"val_accuracy": 0.9, ...}``).
    weights : Dict[str, Any], optional
        Metric name → positive weight.  When given it also acts as an allowlist:
        only those metrics are combined.

    Returns
    -------
    Tuple[List[Optional[float]], List[Dict[str, Any]]]
        The composite score per fold (``None`` when nothing can be combined)
        and the components actually used, each one shaped as
        ``{"metric", "weight", "direction"}``.
    """
    total_folds = len(fold_metrics)
    if not total_folds:
        return [], []
    requested: Dict[str, float] = {}
    for key, value in (weights or {}).items():
        label = str(key or "").strip()
        if label:
            requested[label] = float(value)
    if requested:
        candidates = list(requested)
    else:
        common: Optional[set] = None
        for metrics in fold_metrics:
            keys = {key for key, value in (metrics or {}).items() if _finite(value)}
            common = keys if common is None else (common & keys)
        common = common or set()
        validation = {key for key in common if str(key).lower().startswith("val_")}
        candidates = sorted(validation or common)
    components: List[Dict[str, Any]] = []
    columns: List[List[float]] = []
    for key in candidates:
        values: List[float] = []
        for metrics in fold_metrics:
            value = (metrics or {}).get(key)
            if not _finite(value):
                values = []
                break
            values.append(float(value))
        if len(values) != total_folds:
            continue
        weight = requested.get(key, 1.0)
        if weight <= 0:
            continue
        low, high = min(values), max(values)
        direction = infer_mode(key)
        if high == low:
            column = [0.5] * total_folds
        elif direction == "min":
            column = [(high - value) / (high - low) for value in values]
        else:
            column = [(value - low) / (high - low) for value in values]
        columns.append(column)
        components.append({"metric": key, "weight": weight, "direction": direction})
    if not components:
        return [None] * total_folds, []
    total_weight = sum(component["weight"] for component in components)
    scores: List[Optional[float]] = []
    for index in range(total_folds):
        combined = sum(
            component["weight"] * columns[position][index]
            for position, component in enumerate(components)
        )
        scores.append(round(combined / total_weight, 6))
    return scores, components


def is_better(score: float, best: float, mode: str) -> bool:
    """Whether ``score`` improves on ``best`` for the requested direction."""
    return score > best if str(mode).lower() == "max" else score < best


def summarise(scores: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean and population standard deviation of the fold scores."""
    values = [float(score) for score in scores if score is not None and math.isfinite(score)]
    if not values:
        return {"mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": round(mean, 6), "std": round(math.sqrt(variance), 6)}
