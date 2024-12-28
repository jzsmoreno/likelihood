import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
from IPython.display import clear_output


class HMM:
    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize parameters with random values
        self.pi = np.random.dirichlet(np.ones(n_states), size=1)[0]
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B = np.random.dirichlet(np.ones(n_observations), size=n_states)

    def save_model(self, filename: str = "./hmm") -> None:
        filename = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename: str = "./hmm") -> "HMM":
        filename = filename + ".pkl" if not filename.endswith(".pkl") else filename
        with open(filename, "rb") as f:
            return pickle.load(f)

    def forward(self, sequence: List[int]) -> np.ndarray:
        T = len(sequence)
        alpha = np.zeros((T, self.n_states))

        # Add a small constant (smoothing) to avoid log(0)
        epsilon = 1e-10  # Small value to avoid taking log(0)

        # Initialization (log-space)
        alpha[0] = np.log(self.pi + epsilon) + np.log(self.B[:, sequence[0]] + epsilon)
        alpha[0] -= np.log(np.sum(np.exp(alpha[0])))  # Normalization (log-space)

        # Recursion (log-space)
        for t in range(1, T):
            for i in range(self.n_states):
                alpha[t, i] = np.log(
                    np.sum(np.exp(alpha[t - 1] + np.log(self.A[:, i] + epsilon)))
                ) + np.log(self.B[i, sequence[t]] + epsilon)
            alpha[t] -= np.log(np.sum(np.exp(alpha[t])))  # Normalization

        return alpha

    def backward(self, sequence: List[int]) -> np.ndarray:
        T = len(sequence)
        beta = np.ones((T, self.n_states))

        # Backward recursion
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, sequence[t + 1]] * beta[t + 1])

        return beta

    def viterbi(self, sequence: List[int]) -> np.ndarray:
        T = len(sequence)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0] = self.pi * self.B[:, sequence[0]]

        # Recursion
        for t in range(1, T):
            for i in range(self.n_states):
                delta[t, i] = np.max(delta[t - 1] * self.A[:, i]) * self.B[i, sequence[t]]
                psi[t, i] = np.argmax(delta[t - 1] * self.A[:, i])

        # Reconstruct the most probable path
        state_sequence = np.zeros(T, dtype=int)
        state_sequence[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        return state_sequence

    def baum_welch(
        self, sequences: List[List[int]], n_iterations: int, verbose: bool = False
    ) -> None:
        for iteration in range(n_iterations):
            # Initialize accumulators
            A_num = np.zeros((self.n_states, self.n_states))
            B_num = np.zeros((self.n_states, self.n_observations))
            pi_num = np.zeros(self.n_states)

            for sequence in sequences:
                T = len(sequence)
                alpha = self.forward(sequence)
                beta = self.backward(sequence)

                # Update pi
                gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)
                pi_num += gamma[0]

                # Update A and B
                for t in range(T - 1):
                    xi = np.zeros((self.n_states, self.n_states))
                    denom = np.sum(alpha[t] * self.A * self.B[:, sequence[t + 1]] * beta[t + 1])

                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[i, j] = (
                                alpha[t, i]
                                * self.A[i, j]
                                * self.B[j, sequence[t + 1]]
                                * beta[t + 1, j]
                            ) / denom
                        A_num[i] += xi[i]

                    B_num[:, sequence[t]] += gamma[t]

                # For the last step of the sequence
                B_num[:, sequence[-1]] += gamma[-1]

            # Normalize and update parameters
            self.pi = pi_num / len(sequences)
            self.A = A_num / np.sum(A_num, axis=1, keepdims=True)
            self.B = B_num / np.sum(B_num, axis=1, keepdims=True)

            # Logging parameters every 10 iterations
            if iteration % 10 == 0 and verbose:
                os.system("cls" if os.name == "nt" else "clear")
                clear_output(wait=True)
                logging.info(f"Iteration {iteration}:")
                logging.info("Pi: %s", self.pi)
                logging.info("A:\n%s", self.A)
                logging.info("B:\n%s", self.B)

    def decoding_accuracy(self, sequences: List[List[int]], true_states: List[List[int]]) -> float:
        correct_predictions = 0
        total_predictions = 0

        for sequence, true_state in zip(sequences, true_states):
            predicted_states = self.viterbi(sequence)
            correct_predictions += np.sum(predicted_states == true_state)
            total_predictions += len(sequence)

        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

    def state_probabilities(self, sequence: List[int]) -> np.ndarray:
        """
        Returns the smoothed probabilities of the hidden states at each time step.
        This is done by using both forward and backward probabilities.
        """
        alpha = self.forward(sequence)
        beta = self.backward(sequence)

        # Compute smoothed probabilities (gamma)
        smoothed_probs = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

        return smoothed_probs

    def sequence_probability(self, sequence: List[int]) -> np.ndarray:
        return self.state_probabilities(sequence)[-1]
