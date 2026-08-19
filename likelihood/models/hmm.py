from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from tqdm.auto import tqdm

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ObservationSequence: TypeAlias = Sequence[int] | IntArray


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

        self.n_states: int = n_states
        self.n_observations: int = n_observations
        self.pi: FloatArray = np.random.dirichlet(np.ones(n_states))
        self.A: FloatArray = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B: FloatArray = np.random.dirichlet(np.ones(n_observations), size=n_states)

    @staticmethod
    def _model_path(filename: str | Path) -> Path:
        """Return ``filename`` with a ``.pkl`` suffix appended if necessary."""
        path = Path(filename)
        return path if str(path).endswith(".pkl") else Path(f"{path}.pkl")

    def _as_observation_array(self, sequence: ObservationSequence) -> IntArray:
        """Convert and validate an observation sequence.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            One-dimensional sequence of discrete observation symbols.

        Returns
        -------
        NDArray[np.int64]
            Validated one-dimensional observation array.

        Raises
        ------
        ValueError
            If the sequence is empty, is not one-dimensional, or contains an
            observation outside ``[0, n_observations)``.
        """
        observations = np.asarray(sequence, dtype=np.int64)
        if observations.ndim != 1:
            raise ValueError("sequence must be one-dimensional.")
        if observations.size == 0:
            raise ValueError("sequence must contain at least one observation.")
        if observations.min() < 0 or observations.max() >= self.n_observations:
            raise ValueError(f"observations must be integers in [0, {self.n_observations}).")
        return observations

    def save_model(self, filename: str | Path = "./hmm") -> None:
        """Serialize the model to disk using pickle.

        Parameters
        ----------
        filename : str or pathlib.Path, default="./hmm"
            Destination path. ``.pkl`` is appended when it is not already
            present.
        """
        path = self._model_path(filename)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename: str | Path = "./hmm") -> HMM:
        """Load a pickled :class:`HMM` from disk.

        Parameters
        ----------
        filename : str or pathlib.Path, default="./hmm"
            Model path. ``.pkl`` is appended when it is not already present.

        Returns
        -------
        HMM
            Deserialized model.

        Raises
        ------
        TypeError
            If the pickle does not contain an ``HMM`` instance.
        """
        path = HMM._model_path(filename)
        with path.open("rb") as file:
            model = pickle.load(file)

        if not isinstance(model, HMM):
            raise TypeError(f"Expected a pickled HMM, got {type(model).__name__}.")
        return model

    def _forward_with_scaling(self, sequence: ObservationSequence) -> tuple[FloatArray, FloatArray]:
        """Compute scaled forward probabilities using log-space recursion.

        The recursion is sequential over time, but the transition from all
        source states to all destination states is vectorized at each step.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.

        Returns
        -------
        alpha_hat : NDArray[np.float64]
            Scaled forward probabilities with shape ``(T, n_states)``. Each
            row sums approximately to one.
        log_scale : NDArray[np.float64]
            Log scaling factor for each time step. The sequence log-likelihood
            is ``log_scale.sum()``.
        """
        observations = self._as_observation_array(sequence)
        n_steps = observations.size

        A = self.A + self._EPSILON
        emissions = self.B[:, observations].T + self._EPSILON

        alpha_hat = np.empty((n_steps, self.n_states), dtype=np.float64)
        scale = np.empty(n_steps, dtype=np.float64)

        alpha_hat[0] = (self.pi + self._EPSILON) * emissions[0]
        scale[0] = alpha_hat[0].sum()
        alpha_hat[0] /= scale[0]

        for t in range(1, n_steps):
            alpha_hat[t] = (alpha_hat[t - 1] @ A) * emissions[t]
            scale[t] = alpha_hat[t].sum()
            alpha_hat[t] /= scale[t]

        return alpha_hat, np.log(scale)

    def _backward_with_scaling(
        self,
        sequence: ObservationSequence,
        log_scale: FloatArray,
    ) -> FloatArray:
        """Compute scaled backward probabilities in log space.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.
        log_scale : NDArray[np.float64]
            Forward-pass log scaling factors with shape ``(T,)``.

        Returns
        -------
        NDArray[np.float64]
            Scaled backward probabilities with shape ``(T, n_states)``.

        Raises
        ------
        ValueError
            If ``log_scale`` does not have shape ``(T,)``.
        """
        observations = self._as_observation_array(sequence)
        n_steps = observations.size
        scale = np.asarray(log_scale, dtype=np.float64)
        if scale.shape != (n_steps,):
            raise ValueError(f"log_scale must have shape ({n_steps},).")

        A = self.A + self._EPSILON
        emissions = self.B[:, observations].T + self._EPSILON
        beta_hat = np.ones((n_steps, self.n_states), dtype=np.float64)
        scale = np.exp(scale)

        # beta_hat[T - 1] = 1, so log(beta_hat[T - 1]) = 0.
        for t in range(n_steps - 2, -1, -1):
            beta_hat[t] = A @ (emissions[t + 1] * beta_hat[t + 1])
            beta_hat[t] /= scale[t + 1]

        return beta_hat

    def forward(self, sequence: ObservationSequence) -> FloatArray:
        """Compute scaled forward probabilities for an observation sequence.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.

        Returns
        -------
        NDArray[np.float64]
            Matrix with shape ``(T, n_states)`` containing scaled forward
            probabilities. Each row is normalized approximately to one.

        Notes
        -----
        These values are proportional to the unscaled forward probabilities;
        they are not themselves ``P(o_1, ..., o_t, q_t=i)``.
        """
        alpha_hat, _ = self._forward_with_scaling(sequence)
        return alpha_hat

    def backward(self, sequence: ObservationSequence) -> FloatArray:
        """Compute scaled backward probabilities for an observation sequence.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.

        Returns
        -------
        NDArray[np.float64]
            Matrix with shape ``(T, n_states)`` containing backward
            probabilities scaled consistently with the forward pass.
        """
        _, log_scale = self._forward_with_scaling(sequence)
        return self._backward_with_scaling(sequence, log_scale)

    def viterbi(self, sequence: ObservationSequence) -> IntArray:
        """Decode the most likely hidden-state path with the Viterbi algorithm.

        The dynamic program remains sequential over time, while the score
        calculation for all source/destination state pairs is vectorized.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.

        Returns
        -------
        NDArray[np.int64]
            Most probable hidden-state sequence with shape ``(T,)``.
        """
        observations = self._as_observation_array(sequence)
        n_steps = observations.size

        log_A = np.log(self.A + self._EPSILON)
        log_emissions = np.log(self.B[:, observations].T + self._EPSILON)

        psi = np.empty((n_steps, self.n_states), dtype=np.int64)
        psi[0] = 0
        state_indices = np.arange(self.n_states)
        log_delta = np.log(self.pi + self._EPSILON) + log_emissions[0]

        for t in range(1, n_steps):
            transition_scores = log_delta[:, None] + log_A
            psi[t] = np.argmax(transition_scores, axis=0)
            log_delta = transition_scores[psi[t], state_indices] + log_emissions[t]

        state_sequence = np.empty(n_steps, dtype=np.int64)
        state_sequence[-1] = np.argmax(log_delta)

        for t in range(n_steps - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        return state_sequence

    def baum_welch(
        self,
        sequences: Sequence[ObservationSequence],
        n_iterations: int,
        verbose: bool = False,
    ) -> None:
        """Estimate model parameters with the Baum-Welch EM algorithm.

        Forward/backward inference is performed per sequence. Within each
        sequence, posterior transition probabilities (``xi``), state
        occupancies (``gamma``), and emission sufficient statistics are
        accumulated with NumPy operations instead of Python loops over time
        and states.

        Parameters
        ----------
        sequences : Sequence[Sequence[int] | NDArray[np.int64]]
            Collection of non-empty observation sequences.
        n_iterations : int
            Number of expectation-maximization iterations.
        verbose : bool, default=False
            If ``True``, print model parameters every 10 iterations, starting
            at iteration 0.

        Raises
        ------
        ValueError
            If ``n_iterations`` is negative or ``sequences`` is empty.
        """
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if len(sequences) == 0:
            raise ValueError("sequences must contain at least one sequence.")

        observations_list = [self._as_observation_array(seq) for seq in sequences]
        state_offsets = np.arange(self.n_states, dtype=np.int64)

        for iteration in range(n_iterations):
            A_num = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            A_den = np.zeros(self.n_states, dtype=np.float64)
            B_num = np.zeros((self.n_states, self.n_observations), dtype=np.float64)
            B_den = np.zeros(self.n_states, dtype=np.float64)
            pi_num = np.zeros(self.n_states, dtype=np.float64)

            A_current = self.A
            B_current = self.B
            A_stable = A_current + self._EPSILON
            B_stable = B_current + self._EPSILON

            for observations in observations_list:
                n_steps = observations.size
                emissions = B_stable[:, observations].T

                alpha_hat = np.empty((n_steps, self.n_states), dtype=np.float64)
                scale = np.empty(n_steps, dtype=np.float64)
                alpha_hat[0] = (self.pi + self._EPSILON) * emissions[0]
                scale[0] = alpha_hat[0].sum()
                alpha_hat[0] /= scale[0]

                for t in range(1, n_steps):
                    alpha_hat[t] = (alpha_hat[t - 1] @ A_stable) * emissions[t]
                    scale[t] = alpha_hat[t].sum()
                    alpha_hat[t] /= scale[t]

                beta_hat = np.ones((n_steps, self.n_states), dtype=np.float64)
                for t in range(n_steps - 2, -1, -1):
                    beta_hat[t] = A_stable @ (emissions[t + 1] * beta_hat[t + 1])
                    beta_hat[t] /= scale[t + 1]

                gamma = alpha_hat * beta_hat
                gamma /= gamma.sum(axis=1, keepdims=True) + self._EPSILON

                pi_num += gamma[0]
                gamma_sum = gamma.sum(axis=0)
                B_den += gamma_sum

                emission_indices = (observations[:, None] * self.n_states + state_offsets).ravel()
                B_num += (
                    np.bincount(
                        emission_indices,
                        weights=gamma.ravel(),
                        minlength=self.n_observations * self.n_states,
                    )
                    .reshape(self.n_observations, self.n_states)
                    .T
                )

                if n_steps > 1:
                    next_factor = B_current[:, observations[1:]].T * beta_hat[1:]
                    xi_den = np.sum((alpha_hat[:-1] @ A_current) * next_factor, axis=1)
                    xi_den += self._EPSILON

                    A_num += A_current * ((alpha_hat[:-1] / xi_den[:, None]).T @ next_factor)
                    A_den += gamma_sum - gamma[-1]

            self.pi = pi_num / (pi_num.sum() + self._EPSILON)
            self.A = A_num / (A_den[:, None] + self._EPSILON)
            self.B = B_num / (B_den[:, None] + self._EPSILON)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}")
                print("\nPi:")
                print(self.pi)
                print("\nA:")
                print(self.A)
                print("\nB:")
                print(self.B)

    def decoding_accuracy(
        self,
        sequences: Sequence[ObservationSequence],
        true_states: Sequence[ObservationSequence],
    ) -> float:
        """Compute label-aligned Viterbi decoding accuracy.

        Hidden-state labels are permutation-invariant. This method builds a
        confusion matrix between predicted and true labels, uses the Hungarian
        algorithm to find the optimal label assignment, then reports the
        percentage of correctly aligned states.

        Parameters
        ----------
        sequences : Sequence[Sequence[int] | NDArray[np.int64]]
            Observation sequences to decode.
        true_states : Sequence[Sequence[int] | NDArray[np.int64]]
            Ground-truth state sequences corresponding to ``sequences``.

        Returns
        -------
        float
            Label-aligned decoding accuracy as a percentage in ``[0, 100]``.

        Raises
        ------
        ValueError
            If the collections are empty, contain different numbers of
            sequences, or corresponding sequence lengths differ.
        """
        if len(sequences) == 0:
            raise ValueError("sequences must contain at least one sequence.")
        if len(sequences) != len(true_states):
            raise ValueError("sequences and true_states must contain the same number of sequences.")

        predicted_parts: list[IntArray] = []
        true_parts: list[IntArray] = []

        for sequence, states in zip(sequences, true_states):
            predicted = self.viterbi(sequence)
            truth = np.asarray(states, dtype=np.int64)
            if truth.ndim != 1:
                raise ValueError("Each true-state sequence must be one-dimensional.")
            if predicted.size != truth.size:
                raise ValueError(
                    "Each observation sequence and true-state sequence must have "
                    "the same length."
                )
            predicted_parts.append(predicted)
            true_parts.append(truth)

        predicted_states = np.concatenate(predicted_parts)
        true_states_flat = np.concatenate(true_parts)

        predicted_labels, predicted_inverse = np.unique(predicted_states, return_inverse=True)
        true_labels, true_inverse = np.unique(true_states_flat, return_inverse=True)

        confusion = np.bincount(
            predicted_inverse * true_labels.size + true_inverse,
            minlength=predicted_labels.size * true_labels.size,
        ).reshape(predicted_labels.size, true_labels.size)

        row_ind, col_ind = linear_sum_assignment(-confusion)
        aligned_label_by_index = np.full(predicted_labels.size, -1, dtype=np.int64)
        aligned_label_by_index[row_ind] = true_labels[col_ind]
        aligned_predictions = aligned_label_by_index[predicted_inverse]

        return float(np.mean(aligned_predictions == true_states_flat) * 100.0)

    def state_probabilities(self, sequence: ObservationSequence) -> FloatArray:
        """Compute smoothed posterior hidden-state probabilities.

        This returns ``gamma[t, i] = P(q_t=i | O)`` using the scaled
        forward-backward algorithm.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence of length ``T``.

        Returns
        -------
        NDArray[np.float64]
            Posterior state-probability matrix with shape ``(T, n_states)``.
            Each row sums approximately to one.
        """
        alpha_hat, log_scale = self._forward_with_scaling(sequence)
        beta_hat = self._backward_with_scaling(sequence, log_scale)

        gamma = alpha_hat * beta_hat
        gamma /= gamma.sum(axis=1, keepdims=True) + self._EPSILON
        return gamma

    def sequence_probability(self, sequence: ObservationSequence) -> float:
        """Return the likelihood ``P(O | model)`` of an observation sequence.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence.

        Returns
        -------
        float
            Sequence likelihood. For long sequences this value can underflow
            to zero; use :meth:`sequence_log_probability` when possible.
        """
        return float(np.exp(self.sequence_log_probability(sequence)))

    def sequence_log_probability(self, sequence: ObservationSequence) -> float:
        """Return the log-likelihood ``log P(O | model)`` of a sequence.

        Parameters
        ----------
        sequence : Sequence[int] or NDArray[np.int64]
            Observation sequence.

        Returns
        -------
        float
            Sequence log-likelihood computed from the forward scaling factors.
        """
        _, log_scale = self._forward_with_scaling(sequence)
        return float(log_scale.sum())


def aligned_decoding_accuracy(
    model: HMM,
    sequences: list[list[int]],
    true_states: list[list[int]],
) -> float:
    """
    Compute Viterbi decoding accuracy after optimal hidden-state label alignment.

    Hidden-state labels learned by an HMM are arbitrary. For example,
    predicted state 0 may correspond to ground-truth state 2. This function
    finds the optimal one-to-one mapping between predicted and true state
    labels using the Hungarian algorithm and computes the resulting decoding
    accuracy.

    Parameters
    ----------
    model : HMM
        Trained HMM used to decode the observation sequences.
    sequences : list[list[int]]
        Collection of observation sequences.
    true_states : list[list[int]]
        Ground-truth hidden-state sequences corresponding to ``sequences``.

    Returns
    -------
    float
        Decoding accuracy after optimal state-label alignment, expressed as
        a percentage between 0 and 100.

    Raises
    ------
    ValueError
        If ``sequences`` and ``true_states`` contain different numbers of
        sequences, if they are empty, or if the total number of predicted
        and true states differs.
    """
    if len(sequences) != len(true_states):
        raise ValueError("sequences and true_states must contain the same number " "of sequences.")
    if not sequences:
        raise ValueError("sequences and true_states must not be empty.")
    y_pred = np.concatenate([model.viterbi(sequence) for sequence in sequences])
    y_true = np.concatenate([np.asarray(states, dtype=np.int64) for states in true_states])
    if y_pred.size != y_true.size:
        raise ValueError(
            "The total number of predicted states must equal the total "
            "number of ground-truth states."
        )

    pred_labels, pred_indices = np.unique(
        y_pred,
        return_inverse=True,
    )

    true_labels, true_indices = np.unique(
        y_true,
        return_inverse=True,
    )

    confusion = np.zeros(
        (pred_labels.size, true_labels.size),
        dtype=np.int64,
    )

    np.add.at(
        confusion,
        (pred_indices, true_indices),
        1,
    )

    row_ind, col_ind = linear_sum_assignment(
        confusion,
        maximize=True,
    )

    aligned_label_lookup = np.full(
        pred_labels.size,
        -1,
        dtype=np.int64,
    )

    aligned_label_lookup[row_ind] = true_labels[col_ind]
    aligned_predictions = aligned_label_lookup[pred_indices]
    accuracy = np.mean(aligned_predictions == y_true) * 100.0
    return float(accuracy)


def _display_hmm_training_results(
    history_df: pd.DataFrame,
    best_result: Mapping[str, Any] | pd.Series,
) -> None:
    """Display HMM experiment rankings, configuration summaries, and diagnostics."""

    def _section(title: str, subtitle: str | None = None) -> None:
        """Display a section heading with an optional subtitle."""
        subtitle_html = (
            f"""
            <div style="
                color: #64748b;
                font-size: 12px;
                margin-top: 2px;
            ">
                {subtitle}
            </div>
            """
            if subtitle
            else ""
        )

        display(HTML(f"""
                <div style="margin: 22px 0 10px 0;">
                    <div style="
                        font-size: 16px;
                        font-weight: 650;
                        color: #0f172a;
                    ">
                        {title}
                    </div>
                    {subtitle_html}
                </div>
                """))

    def _metric(
        label: str,
        value: object,
        accent: bool = False,
    ) -> str:
        """Build an HTML metric card."""
        bg = "#eff6ff" if accent else "#f8fafc"
        border = "#bfdbfe" if accent else "#e2e8f0"
        value_color = "#1d4ed8" if accent else "#0f172a"

        return f"""
        <div style="
            flex: 1;
            min-width: 125px;
            padding: 12px 14px;
            background: {bg};
            border: 1px solid {border};
            border-radius: 8px;
        ">
            <div style="
                color: #64748b;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: .04em;
                margin-bottom: 4px;
            ">
                {label}
            </div>

            <div style="
                color: {value_color};
                font-size: 20px;
                font-weight: 700;
                line-height: 1.2;
            ">
                {value}
            </div>
        </div>
        """

    experiments: pd.DataFrame = history_df.sort_values(
        ["accuracy", "avg_log_likelihood"],
        ascending=[False, False],
    ).reset_index(drop=True)

    summary: pd.DataFrame = (
        history_df.groupby("states")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            best_accuracy=("accuracy", "max"),
            mean_iterations=("iterations", "mean"),
            mean_log_likelihood=("avg_log_likelihood", "mean"),
        )
        .reset_index()
        .sort_values("states")
    )

    states: list[int] = sorted(int(state) for state in history_df["states"].unique())

    plot_summary: pd.DataFrame = (
        history_df.groupby("states")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_likelihood=("avg_log_likelihood", "mean"),
            std_likelihood=("avg_log_likelihood", "std"),
        )
        .reindex(states)
    )

    selected_mask: pd.Series = experiments["states"].eq(best_result["states"]) & experiments[
        "restart"
    ].eq(best_result["restart"])

    selected_index: int | None = (
        int(experiments.index[selected_mask][0]) if selected_mask.any() else None
    )

    display(HTML(f"""
            <div style="
                padding: 16px 18px;
                margin-bottom: 14px;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: white;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 16px;
                ">
                    <div>
                        <div style="
                            font-size: 19px;
                            font-weight: 700;
                            color: #0f172a;
                        ">
                            HMM Training Results
                        </div>

                        <div style="
                            margin-top: 3px;
                            color: #64748b;
                            font-size: 12px;
                        ">
                            Model selection across
                            {len(experiments)} experiments
                        </div>
                    </div>

                    <div style="
                        padding: 5px 9px;
                        border-radius: 6px;
                        background: #ecfdf5;
                        color: #047857;
                        font-size: 11px;
                        font-weight: 650;
                        white-space: nowrap;
                    ">
                        Best configuration selected
                    </div>
                </div>
            </div>
            """))

    metrics_html: str = "".join(
        [
            _metric(
                "Hidden states",
                best_result["states"],
            ),
            _metric(
                "Restart",
                best_result["restart"],
            ),
            _metric(
                "Iterations",
                best_result["iterations"],
            ),
            _metric(
                "Validation accuracy",
                f'{best_result["accuracy"]:.2f}%',
                accent=True,
            ),
            _metric(
                "Avg. log-likelihood",
                f'{best_result["avg_log_likelihood"]:.4f}',
            ),
        ]
    )

    display(HTML(f"""
            <div style="
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 4px;
            ">
                {metrics_html}
            </div>
            """))

    _section(
        "Experiments",
        "Ranked by validation accuracy and average log-likelihood.",
    )

    configurations: pd.Series = (
        experiments["states"].astype(int).astype(str)
        + " states · R"
        + experiments["restart"].astype(int).astype(str)
    )

    if selected_index is not None:
        configurations.iloc[selected_index] = f"★ {configurations.iloc[selected_index]}"

    experiment_view: pd.DataFrame = pd.DataFrame(
        {
            "Config": configurations,
            "Accuracy": experiments["accuracy"],
            "Avg. LL": experiments["avg_log_likelihood"],
            "Iter.": experiments["iterations"],
        }
    )

    def _highlight_selected(row: pd.Series) -> list[str]:
        """Return CSS styles for the selected experiment row."""
        if selected_index is not None and row.name == selected_index:
            return ["background-color: #eff6ff; font-weight: 650;" for _ in row]

        return ["" for _ in row]

    styled_history = (
        experiment_view.style.format(
            {
                "Accuracy": "{:.2f}%",
                "Avg. LL": "{:.4f}",
                "Iter.": "{:.0f}",
            }
        )
        .apply(
            _highlight_selected,
            axis=1,
        )
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("border-collapse", "separate"),
                        ("border-spacing", "0"),
                        ("font-size", "11px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("position", "sticky"),
                        ("top", "0"),
                        ("z-index", "2"),
                        ("background-color", "#f8fafc"),
                        ("color", "#475569"),
                        ("font-weight", "650"),
                        ("text-align", "right"),
                        ("padding", "5px 8px"),
                        ("border-bottom", "1px solid #e2e8f0"),
                        ("white-space", "nowrap"),
                    ],
                },
                {
                    "selector": "th:first-child",
                    "props": [
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "right"),
                        ("padding", "4px 8px"),
                        ("border-bottom", "1px solid #f1f5f9"),
                        ("white-space", "nowrap"),
                    ],
                },
                {
                    "selector": "td:first-child",
                    "props": [
                        ("text-align", "left"),
                        ("color", "#334155"),
                    ],
                },
                {
                    "selector": "tr:hover td",
                    "props": [
                        ("background-color", "#f8fafc"),
                    ],
                },
            ]
        )
    )

    display(HTML(f"""
            <div style="
                max-height: 280px;
                overflow-y: auto;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: white;
            ">
                {styled_history.to_html()}
            </div>
            """))

    _section(
        "Configuration summary",
        "Aggregate performance by number of hidden states.",
    )

    styled_summary = (
        summary.style.format(
            {
                "mean_accuracy": "{:.2f}%",
                "std_accuracy": "{:.2f}",
                "best_accuracy": "{:.2f}%",
                "mean_iterations": "{:.1f}",
                "mean_log_likelihood": "{:.4f}",
            }
        )
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("font-size", "12px"),
                        ("border-collapse", "collapse"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f8fafc"),
                        ("color", "#475569"),
                        ("font-weight", "650"),
                        ("text-align", "center"),
                        ("padding", "7px 10px"),
                        ("border-bottom", "1px solid #e2e8f0"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("padding", "7px 10px"),
                        ("border-bottom", "1px solid #f1f5f9"),
                    ],
                },
            ]
        )
    )

    display(styled_summary)

    _section(
        "Performance diagnostics",
        "Individual restarts, mean performance, variability, and selected model.",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.2),
    )

    fig.patch.set_facecolor("white")

    def _plot_metric(
        ax: Axes,
        metric: str,
        mean_column: str,
        std_column: str,
        title: str,
        subtitle: str,
        ylabel: str,
        line_color: str,
        error_color: str,
        annotation_format: str,
    ) -> None:
        """Plot experiment values, aggregate variability, and the selected model."""
        for n_states in states:
            subset: pd.DataFrame = history_df[history_df["states"] == n_states]

            jitter = np.linspace(-0.10, 0.10, len(subset)) if len(subset) > 1 else np.array([0.0])

            ax.scatter(
                n_states + jitter,
                subset[metric],
                s=30,
                alpha=0.38,
                color="#64748b",
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

        ax.errorbar(
            states,
            plot_summary[mean_column],
            yerr=plot_summary[std_column].fillna(0),
            fmt="o-",
            color=line_color,
            ecolor=error_color,
            linewidth=2,
            markersize=5.5,
            capsize=3,
            capthick=1,
            label="Mean ± std.",
            zorder=3,
        )

        best_value: float = float(best_result[metric])

        ax.scatter(
            best_result["states"],
            best_value,
            marker="*",
            s=180,
            color="#f59e0b",
            edgecolors="#92400e",
            linewidth=0.7,
            label="Selected",
            zorder=5,
        )

        ax.annotate(
            annotation_format.format(best_value),
            xy=(
                best_result["states"],
                best_value,
            ),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#92400e",
        )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=15,
            color="#0f172a",
        )

        ax.text(
            0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color="#64748b",
        )

        ax.set_xlabel(
            "Hidden states",
            fontsize=9,
            color="#475569",
        )

        ax.set_ylabel(
            ylabel,
            fontsize=9,
            color="#475569",
        )

        ax.set_xticks(states)

        ax.tick_params(
            axis="both",
            labelsize=8.5,
            colors="#475569",
        )

        ax.grid(
            axis="y",
            alpha=0.12,
            linestyle="--",
        )

        ax.grid(
            axis="x",
            visible=False,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e2e8f0")
        ax.spines["bottom"].set_color("#e2e8f0")

        ax.legend(
            frameon=False,
            fontsize=8,
            loc="best",
        )

    _plot_metric(
        axes[0],
        metric="accuracy",
        mean_column="mean_accuracy",
        std_column="std_accuracy",
        title="Validation accuracy",
        subtitle="Performance across random restarts",
        ylabel="Accuracy (%)",
        line_color="#2563eb",
        error_color="#bfdbfe",
        annotation_format="{:.2f}%",
    )

    _plot_metric(
        axes[1],
        metric="avg_log_likelihood",
        mean_column="mean_likelihood",
        std_column="std_likelihood",
        title="Average log-likelihood",
        subtitle="Model fit across random restarts",
        ylabel="Log-likelihood / observation",
        line_color="#7c3aed",
        error_color="#ddd6fe",
        annotation_format="{:.4f}",
    )

    plt.tight_layout(w_pad=3)
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
