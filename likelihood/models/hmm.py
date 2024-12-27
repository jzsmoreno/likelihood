import os

import numpy as np


class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize parameters with random values
        self.pi = np.random.dirichlet(np.ones(n_states), size=1)[0]
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B = np.random.dirichlet(np.ones(n_observations), size=n_states)

    def forward(self, sequence):
        T = len(sequence)
        alpha = np.zeros((T, self.n_states))

        # Initialization
        alpha[0] = self.pi * self.B[:, sequence[0]]
        alpha[0] /= np.sum(alpha[0])  # Normalization

        # Recursion
        for t in range(1, T):
            for i in range(self.n_states):
                alpha[t, i] = np.sum(alpha[t - 1] * self.A[:, i]) * self.B[i, sequence[t]]
            alpha[t] /= np.sum(alpha[t])  # Normalization

        return alpha

    def backward(self, sequence):
        T = len(sequence)
        beta = np.ones((T, self.n_states))

        # Backward recursion
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, sequence[t + 1]] * beta[t + 1])

        return beta

    def viterbi(self, sequence):
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

    def baum_welch(self, sequences, n_iterations):
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
                pi_num += gamma[0]  # Accumulate the first step of each sequence

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

            # Display the parameters every 10 iterations
            if iteration % 10 == 0:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"Iteration {iteration}:")
                print("Pi:", self.pi)
                print("A:\n", self.A)
                print("B:\n", self.B)

    def decoding_accuracy(self, sequences, true_states):
        correct_predictions = 0
        total_predictions = 0

        for sequence, true_state in zip(sequences, true_states):
            predicted_states = self.viterbi(sequence)
            correct_predictions += np.sum(predicted_states == true_state)
            total_predictions += len(sequence)

        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy


# Example usage
if __name__ == "__main__":
    # Define the HMM class (the one we just created)
    # If you haven't already, make sure to define the HMM class with the methods we discussed.

    # Define the parameters of the model
    n_states = 2  # Sunny (0), Rainy (1)
    n_observations = 2  # Walk (0), Shop (1)

    # Create an HMM instance
    hmm = HMM(n_states, n_observations)

    # Generate some synthetic observation sequences (e.g., 3 sequences of 5 days)
    # Each number represents an observation: 0 -> Walk, 1 -> Shop
    sequences = [
        [0, 1, 0, 0, 1],  # Sequence 1: Walk, Shop, Walk, Walk, Shop
        [1, 1, 0, 1, 0],  # Sequence 2: Shop, Shop, Walk, Shop, Walk
        [0, 0, 0, 1, 1],  # Sequence 3: Walk, Walk, Walk, Shop, Shop
    ]

    # Define the true hidden states for the sequences (ground truth for training/testing)
    true_states = [
        [0, 0, 0, 1, 1],  # Sequence 1: Sunny, Sunny, Sunny, Rainy, Rainy
        [1, 1, 0, 1, 1],  # Sequence 2: Rainy, Rainy, Sunny, Rainy, Rainy
        [0, 0, 0, 1, 1],  # Sequence 3: Sunny, Sunny, Sunny, Rainy, Rainy
    ]

    # Train the HMM using the Baum-Welch algorithm
    print("Training the HMM with Baum-Welch...")
    hmm.baum_welch(sequences, n_iterations=50)

    # After training, evaluate the model's accuracy
    accuracy = hmm.decoding_accuracy(sequences, true_states)
    print(f"Decoding Accuracy: {accuracy:.2f}%")

    # Now let's use the trained HMM to predict the hidden states for one of the sequences
    test_sequence = [0, 1, 0, 1, 0]  # Test sequence: Walk, Shop, Walk, Shop, Walk
    predicted_states = hmm.viterbi(test_sequence)

    print(f"Test Sequence: {test_sequence}")
    print(f"Predicted States: {predicted_states}")
