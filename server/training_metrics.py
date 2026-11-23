"""Classification metrics computed inside the server process.

The published ``likelihood`` package is intentionally left untouched: the
training sandbox needs a reference implementation of Cohen's kappa that can be
unit-tested without TensorFlow, while the trainable Keras metric lives in the
sandbox bootstrap (:data:`server_core._INVOKE_BOOTSTRAP`).

Run with::

    python3 -m pytest tests/test_training_metrics.py -v
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


def _as_labels(values: Iterable) -> List:
    """Materialise ``values`` as plain Python labels (unwrapping numpy scalars)."""
    return [value.item() if hasattr(value, "item") else value for value in values]


def confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[Sequence] = None,
) -> Tuple[List, List[List[int]]]:
    """Build a confusion matrix with rows indexed by the true label.

    Parameters
    ----------
    y_true : Sequence
        True class labels.
    y_pred : Sequence
        Predicted class labels.
    labels : Sequence, optional
        Labels to include, in order.  Defaults to the sorted union of the labels
        found in ``y_true`` and ``y_pred``.

    Returns
    -------
    Tuple[List, List[List[int]]]
        The labels and the matrix, where ``matrix[i][j]`` counts the samples of
        true class ``labels[i]`` predicted as ``labels[j]``.  Pairs containing a
        label outside ``labels`` are ignored.
    """
    truth = _as_labels(y_true)
    predicted = _as_labels(y_pred)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(truth) | set(predicted))
    label_list = _as_labels(labels)
    positions = {label: position for position, label in enumerate(label_list)}
    matrix = [[0] * len(label_list) for _ in label_list]
    for actual, guess in zip(truth, predicted):
        if actual in positions and guess in positions:
            matrix[positions[actual]][positions[guess]] += 1
    return label_list, matrix


def cohens_kappa(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[Sequence] = None,
) -> float:
    """Return the unweighted Cohen's kappa coefficient of two label sequences.

    The coefficient compares the observed agreement with the agreement expected
    by chance, ``(p_o - p_e) / (1 - p_e)``: it is ``1`` for perfect agreement,
    ``0`` for chance-level agreement and negative when the agreement is worse
    than chance.  Degenerate inputs (no samples, or a single class with perfect
    agreement) return ``0.0`` instead of dividing by zero.

    Parameters
    ----------
    y_true : Sequence
        True class labels.
    y_pred : Sequence
        Predicted class labels.
    labels : Sequence, optional
        Labels to include, in order.  Defaults to the sorted union of the labels
        found in ``y_true`` and ``y_pred``.

    Returns
    -------
    float
        The unweighted Cohen's kappa coefficient.
    """
    _, matrix = confusion_matrix(y_true, y_pred, labels)
    total = sum(sum(row) for row in matrix)
    if not total:
        return 0.0
    size = len(matrix)
    observed = sum(matrix[index][index] for index in range(size)) / total
    row = [sum(values) for values in matrix]
    column = [sum(matrix[index][position] for index in range(size)) for position in range(size)]
    expected = sum(row[index] * column[index] for index in range(size)) / (total * total)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)
