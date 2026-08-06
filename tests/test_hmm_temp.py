"""Pytest verification tests for HMM core functionality."""

# python -m pytest tests/test_hmm.py tests/test_hmm_temp.py -v --tb=short 2>&1
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from likelihood.models.hmm import HMM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hmm_3states():
    """Return a fresh 3-state, 2-observation HMM."""
    return HMM(3, 2)


@pytest.fixture
def short_seq():
    """A short observation sequence used across multiple tests."""
    return [0, 1, 0, 0, 1]


@pytest.fixture
def synthetic_data():
    """Generate 20 sequences of length 20 from a known HMM."""
    np.random.seed(42)
    true_pi = np.array([0.4, 0.3, 0.3])
    true_A = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    true_B = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])

    sequences = []
    true_states = []
    for _ in range(20):
        state = np.random.choice(3, p=true_pi)
        obs_seq = []
        st_seq = []
        for _ in range(20):
            st_seq.append(state)
            obs_seq.append(np.random.choice(2, p=true_B[state]))
            state = np.random.choice(3, p=true_A[state])
        sequences.append(obs_seq)
        true_states.append(st_seq)
    return sequences, true_states


# ---------------------------------------------------------------------------
# Test 1: Forward / Backward consistency
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def test_alpha_rows_sum_to_one(self, hmm_3states, short_seq):
        """Each row of alpha (forward probabilities) should sum to 1."""
        alpha = hmm_3states.forward(short_seq)
        assert np.allclose(np.sum(alpha, axis=1), 1.0)

    def test_beta_is_valid(self, hmm_3states, short_seq):
        """Backward probabilities should be non-negative and finite."""
        beta = hmm_3states.backward(short_seq)

        assert np.all(beta >= 0)
        assert np.all(np.isfinite(beta))

    def test_alpha_no_nan(self, hmm_3states, short_seq):
        """Forward probabilities should contain no NaN values."""
        alpha = hmm_3states.forward(short_seq)
        assert not np.any(np.isnan(alpha))

    def test_beta_no_nan(self, hmm_3states, short_seq):
        """Backward probabilities should contain no NaN values."""
        beta = hmm_3states.backward(short_seq)
        assert not np.any(np.isnan(beta))


# ---------------------------------------------------------------------------
# Test 2: Viterbi
# ---------------------------------------------------------------------------


class TestViterbi:
    def test_output_length_matches_input(self, hmm_3states, short_seq):
        """Viterbi should return a state sequence of the same length."""
        states = hmm_3states.viterbi(short_seq)
        assert len(states) == len(short_seq)

    def test_states_in_valid_range(self, hmm_3states, short_seq):
        """All returned states must be valid indices (0..N-1)."""
        states = hmm_3states.viterbi(short_seq)
        assert all(0 <= s < 3 for s in states)


# ---------------------------------------------------------------------------
# Test 3: State probabilities
# ---------------------------------------------------------------------------


class TestStateProbabilities:
    def test_rows_sum_to_one(self, hmm_3states, short_seq):
        """Each time-step's state probability distribution must sum to 1."""
        probs = hmm_3states.state_probabilities(short_seq)
        assert np.allclose(np.sum(probs, axis=1), 1.0)


# ---------------------------------------------------------------------------
# Test 4: Sequence probability
# ---------------------------------------------------------------------------


class TestSequenceProbability:
    def test_probability_in_valid_range(self, hmm_3states, short_seq):
        """Sequence probability must be between 0 and 1."""
        prob = hmm_3states.sequence_probability(short_seq)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Baum-Welch with adequate data
# ---------------------------------------------------------------------------


class TestBaumWelchAdequateData:
    def test_pi_not_collapsed(self, synthetic_data):
        """Initial state distribution should not collapse to a single state."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100)
        assert np.max(hmm.pi) < 0.95, f"Pi collapsed: {hmm.pi}"

    def test_pi_sums_to_one(self, synthetic_data):
        """Learned pi must be a valid probability distribution."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100)
        assert np.allclose(np.sum(hmm.pi), 1.0)

    def test_A_rows_sum_to_one(self, synthetic_data):
        """Learned transition matrix rows must sum to 1."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100)
        assert np.allclose(np.sum(hmm.A, axis=1), 1.0)

    def test_B_rows_sum_to_one(self, synthetic_data):
        """Learned emission matrix rows must sum to 1."""
        sequences, _ = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100)
        assert np.allclose(np.sum(hmm.B, axis=1), 1.0)

    def test_decoding_accuracy_reasonable(self, synthetic_data):
        """Decoding accuracy should be above random baseline (~33%)."""
        sequences, true_states = synthetic_data
        hmm = HMM(3, 2)
        hmm.baum_welch(sequences, n_iterations=100)
        acc = hmm.decoding_accuracy(sequences, true_states)
        assert acc > 30.0, f"Accuracy too low: {acc:.2f}%"


# ---------------------------------------------------------------------------
# Test 6: Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_viterbi_preserved_after_roundtrip(self, hmm_3states, short_seq):
        """Viterbi output should be identical after save + load."""
        orig_vit = hmm_3states.viterbi(short_seq).copy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hmm")
            hmm_3states.save_model(path)
            loaded = HMM.load_model(path)
        assert np.array_equal(orig_vit, loaded.viterbi(short_seq))

    def test_pi_preserved_after_roundtrip(self, hmm_3states):
        """Initial distribution pi should be identical after save + load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hmm")
            hmm_3states.save_model(path)
            loaded = HMM.load_model(path)
        assert np.allclose(hmm_3states.pi, loaded.pi)


# ---------------------------------------------------------------------------
# Test 7: Long sequence no underflow
# ---------------------------------------------------------------------------


class TestLongSequenceNoUnderflow:
    def test_forward_no_nan_on_long_sequence(self):
        """Forward pass on 200-length sequence should produce no NaN."""
        hmm = HMM(5, 10)
        long_seq = list(np.random.randint(0, 10, size=200))
        alpha = hmm.forward(long_seq)
        assert not np.any(np.isnan(alpha))

    def test_backward_no_nan_on_long_sequence(self):
        """Backward pass on 200-length sequence should produce no NaN."""
        hmm = HMM(5, 10)
        long_seq = list(np.random.randint(0, 10, size=200))
        beta = hmm.backward(long_seq)
        assert not np.any(np.isnan(beta))

    def test_alpha_rows_positive_on_long_sequence(self):
        """Forward probabilities should all be strictly positive."""
        hmm = HMM(5, 10)
        long_seq = list(np.random.randint(0, 10, size=200))
        alpha = hmm.forward(long_seq)
        assert np.all(np.sum(alpha, axis=1) > 0)
