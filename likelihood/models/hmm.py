from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from tqdm.auto import tqdm


class HMM:
    """Discrete Hidden Markov Model with multinomial emissions.

    The model stores an initial-state distribution ``pi``, a state-transition
    matrix ``A``, and an observation-emission matrix ``B``. Forward/backward
    inference is performed in log space with per-time-step scaling for
    numerical stability.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_observations : int
        Number of distinct discrete observation symbols. Valid observation
        values are integers in ``[0, n_observations)``.

    Attributes
    ----------
    n_states : int
        Number of hidden states.
    n_observations : int
        Number of observation symbols.
    pi : NDArray[np.float64]
        Initial-state probabilities with shape ``(n_states,)``.
    A : NDArray[np.float64]
        Transition-probability matrix with shape
        ``(n_states, n_states)``. ``A[i, j]`` is the probability of moving
        from state ``i`` to state ``j``.
    B : NDArray[np.float64]
        Emission-probability matrix with shape
        ``(n_states, n_observations)``. ``B[i, k]`` is the probability of
        observing symbol ``k`` in state ``i``.
    """

    _EPSILON = 1e-10

    def __init__(self, n_states: int, n_observations: int) -> None:
        if n_states <= 0:
            raise ValueError("n_states must be greater than zero.")
        if n_observations <= 0:
            raise ValueError("n_observations must be greater than zero.")

        self.pi = np.random.dirichlet(np.ones(n_states))
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B = np.random.dirichlet(
            np.ones(n_observations),
            size=n_states,
        )

    def save_model(self, filename: str = "./hmm") -> None:
        filename = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def _model_path(filename: str | Path) -> Path:
        """Return ``filename`` with a ``.pkl`` suffix appended if necessary."""
        path = Path(filename)
        return path if str(path).endswith(".pkl") else Path(f"{path}.pkl")

    def _forward_with_scaling(self, sequence: List[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the scaled forward probabilities using log-space recursion.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        alpha_hat : np.ndarray
            Scaled forward probabilities. Each row sums to one.

        log_scale : np.ndarray
            Logarithm of the scaling factor at each time step.
            The log-likelihood of the observation sequence is

                log P(O | λ) = sum(log_scale).
        """
        T = len(sequence)
        epsilon = 1e-10
        log_alpha = np.zeros((T, self.n_states))
        log_scale = np.zeros(T)

        # Initialization
        log_alpha[0] = np.log(self.pi + epsilon) + np.log(self.B[:, sequence[0]] + epsilon)
        log_scale[0] = logsumexp(log_alpha[0])
        log_alpha[0] -= log_scale[0]

        # Forward recursion
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + np.log(self.A[:, j] + epsilon)
                ) + np.log(self.B[j, sequence[t]] + epsilon)

            log_scale[t] = logsumexp(log_alpha[t])
            log_alpha[t] -= log_scale[t]

        alpha_hat = np.exp(log_alpha)

        return alpha_hat, log_scale

    def _backward_with_scaling(
        self,
        sequence: List[int],
        log_scale: np.ndarray,
    ) -> np.ndarray:

        T = len(sequence)
        epsilon = 1e-10

        log_beta = np.zeros((T, self.n_states))

        # beta_hat[T-1] = 1
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = (
                    logsumexp(
                        np.log(self.A[i] + epsilon)
                        + np.log(self.B[:, sequence[t + 1]] + epsilon)
                        + log_beta[t + 1]
                    )
                    - log_scale[t + 1]
                )

        return np.exp(log_beta)

    def forward(self, sequence: List[int]) -> np.ndarray:
        """
        Computes the scaled forward probabilities.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        np.ndarray
            Scaled forward probabilities (alpha_hat).
            Each row is normalized to sum to one.

        Notes
        -----
        These are the scaled forward probabilities used for numerical
        stability. They are proportional to the true forward probabilities,
        but they are not equal to P(o1, ..., ot, qt=i).
        """
        alpha_hat, _ = self._forward_with_scaling(sequence)
        return alpha_hat

    def backward(self, sequence: List[int]) -> np.ndarray:
        """
        Computes the scaled backward probabilities.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        np.ndarray
            Scaled backward probabilities (beta_hat).

        Notes
        -----
        The backward probabilities are computed using the same scaling
        factors as the forward pass, ensuring consistency for smoothing
        and Baum-Welch re-estimation.
        """
        _, log_scale = self._forward_with_scaling(sequence)
        return self._backward_with_scaling(sequence, log_scale)

    def viterbi(self, sequence: List[int]) -> np.ndarray:
        """
        Computes the most likely hidden state sequence using the Viterbi
        algorithm in log-space.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        np.ndarray
            Most probable sequence of hidden states.
        """
        T = len(sequence)
        epsilon = 1e-10
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        log_delta[0] = np.log(self.pi + epsilon) + np.log(self.B[:, sequence[0]] + epsilon)

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                previous_scores = log_delta[t - 1] + np.log(self.A[:, j] + epsilon)
                psi[t, j] = np.argmax(previous_scores)
                log_delta[t, j] = np.max(previous_scores) + np.log(self.B[j, sequence[t]] + epsilon)

        # Backtracking
        state_sequence = np.zeros(T, dtype=int)
        state_sequence[-1] = np.argmax(log_delta[-1])

        for t in range(T - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        return state_sequence

    def baum_welch(
        self,
        sequences: List[List[int]],
        n_iterations: int,
        verbose: bool = False,
    ) -> None:
        """
        Baum-Welch algorithm (Expectation-Maximization) for estimating HMM
        parameters.

        Uses the scaled forward-backward algorithm with shared scaling
        factors between alpha and beta.

        Parameters
        ----------
        sequences : List[List[int]]
            Collection of observation sequences.

        n_iterations : int
            Number of EM iterations.

        verbose : bool
            Print parameters every 10 iterations.
        """
        epsilon = 1e-10

        for iteration in range(n_iterations):
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_observations))
            B_den = np.zeros(self.n_states)
            pi_num = np.zeros(self.n_states)

            for sequence in sequences:
                T = len(sequence)
                # Forward-backward with consistent scaling
                alpha_hat, log_scale = self._forward_with_scaling(sequence)
                beta_hat = self._backward_with_scaling(
                    sequence,
                    log_scale,
                )

                # Gamma:
                # probability of being in state i at time t
                gamma = alpha_hat * beta_hat
                gamma /= (
                    np.sum(
                        gamma,
                        axis=1,
                        keepdims=True,
                    )
                    + epsilon
                )

                # Initial state probabilities
                pi_num += gamma[0]
                gamma_sum = gamma.sum(axis=0)
                B_den += gamma_sum

                # Transition and emission updates
                for t in range(T - 1):
                    xi = (
                        alpha_hat[t, :, None]
                        * self.A
                        * self.B[:, sequence[t + 1]][None, :]
                        * beta_hat[t + 1, None, :]
                    )
                    xi /= np.sum(xi) + epsilon
                    A_num += xi
                    A_den += gamma[t]
                    B_num[:, sequence[t]] += gamma[t]

                # Last observation emission update
                B_num[:, sequence[-1]] += gamma[-1]
                B_den += np.sum(gamma, axis=0)

            # Maximization step
            self.pi = pi_num / (np.sum(pi_num) + epsilon)
            self.A = A_num / (A_den[:, None] + epsilon)
            self.B = B_num / (B_den[:, None] + epsilon)

            # Logging parameters every 10 iterations
            if iteration % 10 == 0 and verbose:
                subprocess.run("cls" if os.name == "nt" else "clear", shell=True, check=False)
                clear_output(wait=True)
                print(f"Iteration {iteration}:")
                print("Pi:")
                print(self.pi)
                print("\nA:")
                print(self.A)
                print("\nB:")
                print(self.B)

                print("\nA:")
                print(self.A)

                print("\nB:")
                print(self.B)

    def decoding_accuracy(
        self,
        sequences: List[List[int]],
        true_states: List[List[int]],
    ) -> float:
        """
        Computes Viterbi decoding accuracy after optimally aligning
        predicted hidden-state labels with the ground-truth labels.

        Parameters
        ----------
        sequences : List[List[int]]
            Observation sequences.
        true_states : List[List[int]]
            Ground-truth hidden-state sequences.

        Returns
        -------
        float
            Label-aligned decoding accuracy percentage.
        """
        predicted_states = np.concatenate([self.viterbi(sequence) for sequence in sequences])
        true_states_flat = np.concatenate([np.asarray(states) for states in true_states])
        predicted_labels = np.unique(predicted_states)
        true_labels = np.unique(true_states_flat)
        confusion = np.zeros(
            (len(predicted_labels), len(true_labels)),
            dtype=int,
        )
        for i, predicted_label in enumerate(predicted_labels):
            for j, true_label in enumerate(true_labels):
                confusion[i, j] = np.sum(
                    (predicted_states == predicted_label) & (true_states_flat == true_label)
                )
        row_ind, col_ind = linear_sum_assignment(-confusion)
        mapping = {predicted_labels[row]: true_labels[col] for row, col in zip(row_ind, col_ind)}
        aligned_predictions = np.array([mapping.get(state, -1) for state in predicted_states])
        accuracy = np.mean(aligned_predictions == true_states_flat) * 100

        return accuracy

    def state_probabilities(
        self,
        sequence: List[int],
    ) -> np.ndarray:
        """
        Computes smoothed hidden-state probabilities.

        Calculates:

            gamma_t(i) = P(q_t=i | O)

        using the scaled forward-backward algorithm.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        np.ndarray
            Matrix of shape (T, n_states), where each row
            contains the posterior probability of each hidden
            state at time t.
        """
        epsilon = 1e-10

        alpha_hat, log_scale = self._forward_with_scaling(sequence)

        beta_hat = self._backward_with_scaling(
            sequence,
            log_scale,
        )

        gamma = alpha_hat * beta_hat

        gamma /= (
            np.sum(
                gamma,
                axis=1,
                keepdims=True,
            )
            + epsilon
        )

        return gamma

    def sequence_probability(
        self,
        sequence: List[int],
    ) -> float:
        """
        Computes the likelihood of an observation sequence.

        Calculates:

            P(O | model)

        using the scaling factors from the forward algorithm.

        Parameters
        ----------
        sequence : List[int]
            Observation sequence.

        Returns
        -------
        float
            Probability of the observation sequence.
        """
        _, log_scale = self._forward_with_scaling(sequence)
        log_likelihood = np.sum(log_scale)

        return float(np.exp(log_likelihood))

    def sequence_log_probability(self, sequence: List[int]) -> float:
        """
        Computes log P(O | model).
        """
        _, log_scale = self._forward_with_scaling(sequence)
        return float(np.sum(log_scale))


def aligned_decoding_accuracy(model, sequences, true_states):
    """
    Computes Viterbi decoding accuracy after optimally aligning
    predicted hidden-state labels with the ground-truth labels.
    """

    y_pred = np.concatenate([model.viterbi(sequence) for sequence in sequences])

    y_true = np.concatenate([np.asarray(states) for states in true_states])

    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)

    confusion = np.zeros(
        (len(pred_labels), len(true_labels)),
        dtype=int,
    )

    for i, pred_label in enumerate(pred_labels):
        for j, true_label in enumerate(true_labels):
            confusion[i, j] = np.sum((y_pred == pred_label) & (y_true == true_label))

    row_ind, col_ind = linear_sum_assignment(-confusion)

    mapping = {pred_labels[row]: true_labels[col] for row, col in zip(row_ind, col_ind)}

    aligned_predictions = np.array([mapping.get(pred, -1) for pred in y_pred])

    accuracy = np.mean(aligned_predictions == y_true) * 100

    return accuracy


def _display_hmm_training_results(
    history_df,
    best_result,
):

    print("=" * 60)
    print("BEST HMM MODEL")
    print("=" * 60)

    best_summary = pd.DataFrame(
        {
            "Metric": [
                "Hidden states",
                "Restart",
                "Iterations",
                "Validation accuracy",
                "Avg. log-likelihood",
            ],
            "Value": [
                best_result["states"],
                best_result["restart"],
                best_result["iterations"],
                f'{best_result["accuracy"]:.2f}%',
                f'{best_result["avg_log_likelihood"]:.4f}',
            ],
        }
    )

    display(best_summary.style.hide(axis="index").set_caption("Best model"))

    print("\nAll experiments")

    styled_history = (
        history_df.sort_values(
            "accuracy",
            ascending=False,
        )
        .style.format(
            {
                "accuracy": "{:.2f}%",
                "log_likelihood": "{:.2f}",
                "avg_log_likelihood": "{:.4f}",
            }
        )
        .background_gradient(subset=["accuracy"])
        .hide(axis="index")
    )

    display(styled_history)

    summary = (
        history_df.groupby("states")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            best_accuracy=("accuracy", "max"),
            mean_iterations=("iterations", "mean"),
            mean_log_likelihood=(
                "avg_log_likelihood",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\nConfiguration summary")

    display(
        summary.style.format(
            {
                "mean_accuracy": "{:.2f}%",
                "std_accuracy": "{:.2f}",
                "best_accuracy": "{:.2f}%",
                "mean_iterations": "{:.1f}",
                "mean_log_likelihood": "{:.4f}",
            }
        )
        .background_gradient(subset=["best_accuracy"])
        .hide(axis="index")
    )

    # Accuracy by number of states
    plt.figure(figsize=(9, 5))

    for n_states in sorted(history_df["states"].unique()):
        subset = history_df[history_df["states"] == n_states]

        plt.scatter(
            [n_states] * len(subset),
            subset["accuracy"],
            alpha=0.7,
        )

    mean_accuracy = history_df.groupby("states")["accuracy"].mean()

    plt.plot(
        mean_accuracy.index,
        mean_accuracy.values,
        marker="o",
        linewidth=2,
        label="Mean accuracy",
    )

    plt.xlabel("Number of hidden states")
    plt.ylabel("Validation accuracy (%)")
    plt.title("HMM validation accuracy by configuration")

    plt.grid(alpha=0.25)

    plt.legend()
    plt.show()

    # Average log-likelihood
    mean_likelihood = history_df.groupby("states")["avg_log_likelihood"].mean()

    plt.figure(figsize=(9, 5))

    plt.plot(
        mean_likelihood.index,
        mean_likelihood.values,
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of hidden states")
    plt.ylabel("Average log-likelihood per observation")

    plt.title("HMM training likelihood by configuration")

    plt.grid(alpha=0.25)

    plt.show()


def train_best_hmm(
    train_sequences,
    val_sequences,
    val_states,
    state_configs,
    n_observations,
    max_iterations=200,
    n_restarts=5,
    tolerance=1e-4,
    patience=5,
    random_state=42,
    show_results=True,
):
    """
    Train multiple HMM configurations and return the best model.

    Parameters
    ----------
    train_sequences : list[list[int]]
        Training observation sequences.
    val_sequences : list[list[int]]
        Validation observation sequences.
    val_states : list[list[int]]
        Ground-truth hidden states for validation.
    state_configs : list[int]
        Numbers of hidden states to test.
    n_observations : int
        Number of possible observation symbols.
    max_iterations : int, default=200
        Maximum Baum-Welch iterations per restart.
    n_restarts : int, default=5
        Number of random initializations per configuration.
    tolerance : float, default=1e-4
        Minimum log-likelihood improvement considered meaningful.
    patience : int, default=5
        Number of consecutive iterations below tolerance before stopping.
    random_state : int, default=42
        Base random seed for reproducibility.
    show_results : bool, default=True
        Display notebook-friendly tables and plots.

    Returns
    -------
    best_model : HMM
        Best-performing model according to validation accuracy.
    best_result : dict
        Metrics corresponding to the best model.
    history_df : pandas.DataFrame
        Results from every configuration and restart.
    """

    best_model = None
    best_result = None
    best_accuracy = -np.inf

    history = []
    total_runs = len(state_configs) * n_restarts
    progress_bar = tqdm(
        total=total_runs,
        desc="Training HMM configurations",
        unit="model",
    )

    for n_states in state_configs:
        for restart in range(n_restarts):
            # Different but reproducible seed for every run
            seed = random_state + n_states * 1000 + restart
            np.random.seed(seed)
            hmm = HMM(
                n_states=n_states,
                n_observations=n_observations,
            )
            previous_log_likelihood = -np.inf
            no_improvement_count = 0
            likelihood_history = []

            for iteration in range(1, max_iterations + 1):

                # One EM iteration
                hmm.baum_welch(
                    train_sequences,
                    n_iterations=1,
                )

                # Work directly in log-space
                log_likelihood = sum(
                    hmm.sequence_log_probability(sequence) for sequence in train_sequences
                )

                likelihood_history.append(log_likelihood)

                # Skip convergence check on first iteration
                if np.isfinite(previous_log_likelihood):
                    improvement = log_likelihood - previous_log_likelihood
                    if improvement < tolerance:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                    if no_improvement_count >= patience:
                        break
                previous_log_likelihood = log_likelihood

            accuracy = aligned_decoding_accuracy(
                hmm,
                val_sequences,
                val_states,
            )

            n_train_observations = sum(len(sequence) for sequence in train_sequences)
            avg_log_likelihood = log_likelihood / n_train_observations

            result = {
                "states": n_states,
                "restart": restart + 1,
                "seed": seed,
                "iterations": iteration,
                "accuracy": accuracy,
                "log_likelihood": log_likelihood,
                "avg_log_likelihood": avg_log_likelihood,
                "converged": iteration < max_iterations,
            }

            history.append(result)
            # Select best validation model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = copy.deepcopy(hmm)
                best_result = copy.deepcopy(result)
            progress_bar.set_postfix(
                states=n_states,
                restart=restart + 1,
                accuracy=f"{accuracy:.2f}%",
                best=f"{best_accuracy:.2f}%",
            )
            progress_bar.update(1)
    progress_bar.close()
    history_df = pd.DataFrame(history)
    if show_results:
        _display_hmm_training_results(
            history_df,
            best_result,
        )

    return (
        best_model,
        best_result,
        history_df,
    )
