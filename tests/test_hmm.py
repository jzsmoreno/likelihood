"""Pytest tests for HMM Baum-Welch training."""

# python -m pytest tests/test_hmm.py tests/test_hmm_temp.py -v --tb=short 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from likelihood.models.hmm import HMM


@pytest.fixture
def true_params():
    """Return true HMM parameters for synthetic data generation."""
    return {
        "pi": np.array([0.4, 0.3, 0.3]),
        "A": np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]),
        "B": np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]),
    }


def _generate_sequence(pI, A, B, length):
    """Generate a single observation/state sequence from an HMM."""
    states = []
    obs = []
    state = np.random.choice(3, p=pI)
    for _ in range(length):
        states.append(state)
        obs.append(np.random.choice(2, p=B[state]))
        state = np.random.choice(3, p=A[state])
    return obs, states


@pytest.fixture
def synthetic_data(true_params):
    """Generate 20 sequences of length 20 from the true HMM."""
    np.random.seed(42)
    sequences = []
    true_states = []
    for _ in range(20):
        o, s = _generate_sequence(true_params["pi"], true_params["A"], true_params["B"], 20)
        sequences.append(o)
        true_states.append(s)
    return sequences, true_states


class TestHMMBaumWelch:
    """Tests for HMM Baum-Welch training with synthetic data."""

    def test_baum_welch_trains_without_error(self, synthetic_data):
        """Baum-Welch should run to completion without raising."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100, verbose=False)

        # Learned parameters should be valid probability distributions
        assert np.allclose(np.sum(hmm.pi), 1.0)
        assert np.allclose(np.sum(hmm.A, axis=1), 1.0)
        assert np.allclose(np.sum(hmm.B, axis=1), 1.0)

    def test_pi_is_reasonably_distributed(self, synthetic_data):
        """Initial state distribution should not collapse to a single state."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100, verbose=False)

        max_pi = np.max(hmm.pi)
        assert max_pi < 0.95, f"Pi appears collapsed: {hmm.pi}"

    def test_decoding_accuracy_above_chance(self, synthetic_data):
        """Viterbi decoding accuracy should exceed random baseline."""
        sequences, true_states = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100, verbose=False)

        acc = hmm.decoding_accuracy(sequences, true_states)
        # With 3 states, random baseline is ~33%. Trained model should do better.
        assert acc > 30.0, f"Decoding accuracy too low: {acc:.2f}%"

    def test_learned_A_is_row_stochastic(self, synthetic_data):
        """Transition matrix rows must sum to 1."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100, verbose=False)

        assert np.allclose(np.sum(hmm.A, axis=1), 1.0)
        assert np.all(hmm.A >= 0)

    def test_learned_B_is_row_stochastic(self, synthetic_data):
        """Emission matrix rows must sum to 1."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100, verbose=False)

        assert np.allclose(np.sum(hmm.B, axis=1), 1.0)
        assert np.all(hmm.B >= 0)
