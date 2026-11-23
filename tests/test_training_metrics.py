"""Tests for the server-side classification metric helpers.

Run with::

    python3 -m pytest tests/test_training_metrics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import training_metrics  # noqa: E402


def test_confusion_matrix_rows_are_true_labels():
    """``matrix[i][j]`` counts true class ``i`` predicted as class ``j``."""
    labels, matrix = training_metrics.confusion_matrix([0, 0, 0, 1], [0, 0, 1, 1], [0, 1])
    assert labels == [0, 1]
    assert matrix == [[2, 1], [0, 1]]


def test_kappa_is_one_for_perfect_agreement():
    assert training_metrics.cohens_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_agreement():
    assert training_metrics.cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_kappa_matches_a_known_binary_example():
    """2 true negatives, 1 false positive and 1 true positive -> kappa 0.5."""
    assert training_metrics.cohens_kappa([0, 0, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_kappa_handles_degenerate_inputs():
    assert training_metrics.cohens_kappa([], []) == 0.0
    # Every sample is ignored because the label is outside ``labels``.
    assert training_metrics.cohens_kappa([0, 1], [2, 3], [0, 1]) == 0.0
    # A single class agreed on perfectly has no chance-corrected variance.
    assert training_metrics.cohens_kappa([1, 1, 1], [1, 1, 1]) == 0.0


def test_kappa_is_zero_for_a_constant_predictor():
    """A model that always predicts one class agrees only by chance.

    This is the case behind the "kappa is always zero" report: on small or
    under-trained jobs the network predicts a single class, so ``p_o == p_e``
    and the coefficient is exactly ``0`` even when the accuracy looks familiar.
    """
    assert training_metrics.cohens_kappa([0, 1, 2, 0], [0, 0, 0, 0]) == pytest.approx(0.0)
    assert training_metrics.cohens_kappa([0, 0, 0, 1], [0, 0, 0, 0]) == pytest.approx(0.0)
    assert training_metrics.cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(0.0)


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        training_metrics.cohens_kappa([0], [0, 1])
