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
        ``folds`` (or ``k``), ``metric``, ``mode``, ``shuffle`` and ``seed``.

    Returns
    -------
    Dict[str, Any]
        ``{"folds", "metric", "mode", "shuffle", "seed"}`` ready to store on
        the job record.

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
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in ("auto", "min", "max"):
        raise core.ValidationError("El campo 'cross_validation.mode' debe ser auto, min o max")
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
        "mode": infer_mode(metric) if mode == "auto" else mode,
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
