"""Tests for the k-fold helpers used by the training layer.

Run with::

    python3 -m pytest tests/test_cross_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import cross_validation  # noqa: E402
import server_core as core  # noqa: E402


def test_normalise_options_defaults():
    options = cross_validation.normalise_options(True)
    assert options == {
        "folds": 5,
        "metric": "val_loss",
        "weights": {},
        "mode": "min",
        "shuffle": True,
        "seed": 42,
    }


def test_normalise_options_accepts_k_and_infers_mode():
    options = cross_validation.normalise_options(
        {"k": 3, "metric": "val_accuracy", "shuffle": False}
    )
    assert options["folds"] == 3
    assert options["mode"] == "max"
    assert options["seed"] is None


@pytest.mark.parametrize(
    "raw",
    [{"folds": 1}, {"folds": 11}, {"folds": "x"}, {"mode": "up"}, {"seed": "x"}],
)
def test_normalise_options_rejects_invalid_values(raw):
    with pytest.raises(core.ValidationError):
        cross_validation.normalise_options(raw)


def test_make_folds_covers_every_sample_once():
    splits = cross_validation.make_folds(10, 3, shuffle=False)
    assert [len(validation) for _, validation in splits] == [4, 3, 3]
    assert [len(train) for train, _ in splits] == [6, 7, 7]
    assert all(set(train).isdisjoint(validation) for train, validation in splits)
    covered = sorted(index for _, validation in splits for index in validation)
    assert covered == list(range(10))


def test_make_folds_is_deterministic_with_seed():
    first = cross_validation.make_folds(12, 4, True, 7)
    second = cross_validation.make_folds(12, 4, True, 7)
    other = cross_validation.make_folds(12, 4, True, 8)
    assert first == second
    assert first != other


def test_make_folds_requires_enough_rows():
    with pytest.raises(core.ValidationError):
        cross_validation.make_folds(2, 3, shuffle=False)


def test_select_rows_keeps_the_requested_order():
    assert cross_validation.select_rows(["a", "b", "c"], [2, 0]) == ["c", "a"]


def test_fold_score_uses_the_last_finite_value():
    assert cross_validation.fold_score({"val_loss": [1.0, 0.8, None]}, "val_loss") == 0.8
    assert cross_validation.fold_score({"val_loss": []}, "val_loss") is None
    assert cross_validation.fold_score({}, "val_loss") is None


def test_is_better_respects_the_mode():
    assert cross_validation.is_better(0.5, 0.7, "min")
    assert not cross_validation.is_better(0.8, 0.7, "min")
    assert cross_validation.is_better(0.8, 0.7, "max")
    assert not cross_validation.is_better(0.5, 0.7, "max")


def test_summarise_returns_mean_and_population_std():
    summary = cross_validation.summarise([0.4, 0.2, 0.3])
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["std"] == pytest.approx(0.08165, abs=1e-6)
    assert cross_validation.summarise([None]) == {"mean": None, "std": None}


def test_normalise_options_accepts_balanced_and_weights():
    options = cross_validation.normalise_options(
        {"metric": "balanced", "weights": {"val_accuracy": 2, "val_loss": 1}, "mode": "min"}
    )
    assert options["metric"] == "balanced"
    assert options["mode"] == "max"  # composite scores are always maximised
    assert options["weights"] == {"val_accuracy": 2.0, "val_loss": 1.0}


@pytest.mark.parametrize(
    "weights",
    [{"val_loss": 0}, {"val_loss": -1}, {"val_loss": "x"}, [], {"": 1}],
)
def test_normalise_options_rejects_invalid_weights(weights):
    with pytest.raises(core.ValidationError):
        cross_validation.normalise_options({"metric": "balanced", "weights": weights})


def test_composite_scores_normalise_directions_with_equal_weights():
    scores, components = cross_validation.composite_scores(
        [
            {"val_loss": 0.2, "val_accuracy": 0.9},
            {"val_loss": 0.1, "val_accuracy": 0.7},
            {"val_loss": 0.3, "val_accuracy": 0.8},
        ]
    )
    assert scores == pytest.approx([0.75, 0.5, 0.25])
    assert [component["metric"] for component in components] == ["val_accuracy", "val_loss"]
    assert components[1]["direction"] == "min"


def test_composite_scores_honour_explicit_weights():
    scores, _ = cross_validation.composite_scores(
        [
            {"val_loss": 0.2, "val_accuracy": 0.9},
            {"val_loss": 0.1, "val_accuracy": 0.7},
            {"val_loss": 0.3, "val_accuracy": 0.8},
        ],
        {"val_loss": 3, "val_accuracy": 1},
    )
    assert scores == pytest.approx([0.625, 0.75, 0.125])


def test_composite_scores_skip_metrics_missing_from_some_folds():
    scores, components = cross_validation.composite_scores(
        [
            {"val_accuracy": 0.9, "val_loss": 0.2},
            {"val_accuracy": 0.7},
        ]
    )
    assert scores == pytest.approx([1.0, 0.0])
    assert [component["metric"] for component in components] == ["val_accuracy"]


def test_composite_scores_handle_empty_and_constant_metrics():
    empty_scores, empty_components = cross_validation.composite_scores([{}, {}])
    assert empty_scores == [None, None]
    assert empty_components == []
    constant, _ = cross_validation.composite_scores([{"val_accuracy": 0.8}, {"val_accuracy": 0.8}])
    assert constant == pytest.approx([0.5, 0.5])
