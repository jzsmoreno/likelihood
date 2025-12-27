import logging
import os

import networkx as nx
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import sys
import warnings
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

from .figures import *


class suppress_prints:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout


def suppress_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


class TransformRange:
    """
    Generates a new DataFrame with ranges represented as strings.

    Transforms numerical columns into categorical range bins with descriptive labels.
    """

    def __init__(self, columns_bin_sizes: Dict[str, int]) -> None:
        """Initializes the class with the original DataFrame.

        Parameters
        ----------
        columns_bin_sizes : `dict`
            A dictionary where the keys are column names and the values are the bin sizes.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        """
        self.info = {}
        self.columns_bin_sizes = columns_bin_sizes

    def _create_bins_and_labels(
        self, min_val: Union[int, float], max_val: Union[int, float], bin_size: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Creates the bin edges and their labels.

        Parameters
        ----------
        min_val : `int` or `float`
            The minimum value for the range.
        max_val : `int` or `float`
            The maximum value for the range.
        bin_size : `int`
            The size of each bin.

        Returns
        -------
        bins : `np.ndarray`
            The bin edges.
        labels : `list`
            The labels for the bins.

        Raises
        ------
        ValueError
            If bin_size is not positive or if min_val >= max_val.
        """
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        start = int(min_val)
        end = int(max_val) + bin_size

        bins = np.arange(start, end + 1, bin_size)

        if bins[-1] <= max_val:
            bins = np.append(bins, max_val + 1)

        lower_bin_edge = -np.inf
        upper_bin_edge = np.inf

        labels = [f"{int(bins[i])}-{int(bins[i+1] - 1)}" for i in range(len(bins) - 1)]
        end = int(bins[-1] - 1)
        bins = bins.tolist()
        bins.insert(0, lower_bin_edge)
        bins.append(upper_bin_edge)
        labels.insert(0, f"< {start}")
        labels.append(f"> {end}")
        return bins, labels

    def _transform_column_to_ranges(
        self, df: pd.DataFrame, column: str, bin_size: int, fit: bool = True
    ) -> pd.Series:
        """
        Transforms a column in the DataFrame into range bins.

        Parameters
        ----------
        df : `pd.DataFrame`
            The original DataFrame to transform.
        column : `str`
            The name of the column to transform.
        bin_size : `int`
            The size of each bin.

        Returns
        -------
        `pd.Series`
            A Series with the range labels.

        Raises
        ------
        KeyError
            If column is not found in the DataFrame.
        ValueError
            If bin_size is not positive or if column contains non-numeric data.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        df_ = df.copy()  # Create a copy to avoid modifying the original
        numeric_series = pd.to_numeric(df_[column], errors="coerce")
        if fit:
            self.df = df_.copy()
            if column not in df_.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")

            if bin_size <= 0:
                raise ValueError("bin_size must be positive")

            if numeric_series.isna().all():
                raise ValueError(f"Column '{column}' contains no valid numeric data")

            min_val = numeric_series.min()
            max_val = numeric_series.max()

            if min_val == max_val:
                return pd.Series(
                    [f"{int(min_val)}-{int(max_val)}"] * len(df_), name=f"{column}_range"
                )
            self.info[column] = {"min_value": min_val, "max_value": max_val, "range": bin_size}
        else:
            min_val = self.info[column]["min_value"]
            max_val = self.info[column]["max_value"]
            bin_size = self.info[column]["range"]

        bins, labels = self._create_bins_and_labels(min_val, max_val, bin_size)
        return pd.cut(numeric_series, bins=bins, labels=labels, right=False, include_lowest=True)

    def transform(
        self, df: pd.DataFrame, drop_original: bool = False, fit: bool = True
    ) -> pd.DataFrame:
        """
        Creates a new DataFrame with range columns.

        Parameters
        ----------
        df : `pd.DataFrame`
            The original DataFrame to transform.
        drop_original : `bool`, optional
            If True, drops original columns from the result, by default False
        fit : `bool`, default=True
            Whether to compute bin edges based on the data (True) or use predefined binning (False).

        Returns
        -------
        `pd.DataFrame`
            A DataFrame with the transformed range columns.

        Raises
        ------
        TypeError
            If columns_bin_sizes is not a dictionary.
        """
        if not isinstance(self.columns_bin_sizes, dict):
            raise TypeError("columns_bin_sizes must be a dictionary")

        if not self.columns_bin_sizes:
            return pd.DataFrame()

        range_columns = {}
        for column, bin_size in self.columns_bin_sizes.items():
            range_columns[f"{column}_range"] = self._transform_column_to_ranges(
                df, column, bin_size, fit
            )

        result_df = pd.DataFrame(range_columns)

        if not drop_original:
            original_cols = [col for col in df.columns if col not in self.columns_bin_sizes]
            if original_cols:
                result_df = pd.concat([df[original_cols], result_df], axis=1)

        return result_df

    def get_range_info(self, column: str) -> Dict[str, Union[int, float, List[str]]]:
        """
        Get information about the range transformation for a specific column.

        Parameters
        ----------
        column : `str`
            The name of the column to analyze.

        Returns
        -------
        `dict`
            Dictionary containing min_val, max_val, bin_size, and labels.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        numeric_series = pd.to_numeric(self.df[column], errors="coerce")
        min_val = numeric_series.min()
        max_val = numeric_series.max()

        return {
            "min_value": min_val,
            "max_value": max_val,
            "range": max_val - min_val,
            "column": column,
        }


def remove_collinearity(df: pd.DataFrame, threshold: float = 0.9):
    """
    Removes highly collinear features from the DataFrame based on a correlation threshold.

    This function calculates the correlation matrix of the DataFrame and removes columns
    that are highly correlated with any other column in the DataFrame. It uses an absolute
    correlation value greater than the specified threshold to identify which columns to drop.

    Parameters
    ----------
    df : `pd.DataFrame`
        The input DataFrame containing numerical data.
    threshold : `float`
        The correlation threshold above which features will be removed. Default is `0.9`.

    Returns
    -------
    df_reduced : `pd.DataFrame`
        A DataFrame with highly collinear features removed.
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]
    df_reduced = df.drop(columns=to_drop)

    return df_reduced


def train_and_insights(
    x_data: np.ndarray,
    y_act: np.ndarray,
    model: tf.keras.Model,
    patience: int = 3,
    reg: bool = False,
    frac: float = 1.0,
    **kwargs: Optional[Dict],
) -> tf.keras.Model:
    """
    Train a Keras model and provide insights on the training and validation metrics.

    Parameters
    ----------
    x_data : `np.ndarray`
        Input data for training the model.
    y_act : `np.ndarray`
        Actual labels corresponding to x_data.
    model : `tf.keras.Model`
        The Keras model to train.
    patience : `int`
        The patience parameter for early stopping callback (default is 3).
    reg : `bool`
        Flag to determine if residual analysis should be performed (default is `False`).
    frac : `float`
        Fraction of data to use (default is 1.0).

    Keyword Arguments
    -----------------
    Additional keyword arguments passed to the `model.fit` function, such as validation split and callbacks.

    Returns
    -------
    `tf.keras.Model`
        The trained model after fitting.
    """
    validation_split = kwargs.get("validation_split", 0.2)
    callback = kwargs.get(
        "callback", [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)]
    )

    for key in ["validation_split", "callback"]:
        if key in kwargs:
            del kwargs[key]

    history = model.fit(
        x_data,
        y_act,
        validation_split=validation_split,
        verbose=False,
        callbacks=callback,
        **kwargs,
    )

    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    columns = hist.columns
    train_err, train_metric = columns[0], columns[1]
    val_err, val_metric = columns[2], columns[3]
    train_err, val_err = hist[train_err].values, hist[val_err].values

    with suppress_prints():
        n = int(len(x_data) * frac)
        y_pred = model.predict(x_data[:n])
        y_act = y_act[:n]

    if reg:
        residual(y_act, y_pred)
        residual_hist(y_act, y_pred)
        act_pred(y_act, y_pred)

    loss_curve(hist["epoch"].values, train_err, val_err)

    return model


@tf.keras.utils.register_keras_serializable(package="Custom", name="LoRALayer")
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, units, rank=4, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.units = units
        self.rank = rank

    def build(self, input_shape):
        input_dim = input_shape[-1]
        print(f"Input shape: {input_shape}")

        if self.rank > input_dim:
            raise ValueError(
                f"Rank ({self.rank}) cannot be greater than input dimension ({input_dim})."
            )
        if self.rank > self.units:
            raise ValueError(
                f"Rank ({self.rank}) cannot be greater than number of units ({self.units})."
            )

        self.A = self.add_weight(
            shape=(input_dim, self.rank), initializer="random_normal", trainable=True, name="A"
        )
        self.B = self.add_weight(
            shape=(self.rank, self.units), initializer="random_normal", trainable=True, name="B"
        )
        print(f"Dense weights shape: {input_dim}x{self.units}")
        print(f"LoRA weights shape: A{self.A.shape}, B{self.B.shape}")

    def call(self, inputs):
        lora_output = tf.matmul(tf.matmul(inputs, self.A), self.B)
        return lora_output


def apply_lora(model, rank=4):
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Applying LoRA to layer {layer.name}")
            x = LoRALayer(units=layer.units, rank=rank)(x)
        else:
            x = layer(x)
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    return new_model


def graph_metrics(adj_matrix: np.ndarray, eigenvector_threshold: float = 1e-6) -> pd.DataFrame:
    """
    Calculate various graph metrics based on the given adjacency matrix and return them in a single DataFrame.

    Parameters
    ----------
    adj_matrix : `np.ndarray`
        The adjacency matrix representing the graph, where each element denotes the presence/weight of an edge between nodes.
    eigenvector_threshold : `float`
        A threshold for the eigenvector centrality calculation, used to determine the cutoff for small eigenvalues. Default is `1e-6`.

    Returns
    -------
    metrics_df : pd.DataFrame
        A DataFrame containing the following graph metrics as columns.
        - `Degree`: The degree of each node, representing the number of edges connected to each node.
        - `DegreeCentrality`: Degree centrality values for each node, indicating the number of direct connections each node has.
        - `ClusteringCoefficient`: Clustering coefficient values for each node, representing the degree to which nodes cluster together.
        - `EigenvectorCentrality`: Eigenvector centrality values, indicating the influence of a node in the graph based on the eigenvectors of the adjacency matrix.
        - `BetweennessCentrality`: Betweenness centrality values, representing the extent to which a node lies on the shortest paths between other nodes.
        - `ClosenessCentrality`: Closeness centrality values, indicating the inverse of the average shortest path distance from a node to all other nodes in the graph.
        - `Assortativity`: The assortativity coefficient of the graph, measuring the tendency of nodes to connect to similar nodes.

    Notes
    -----
    The returned DataFrame will have one row for each node and one column for each of the computed metrics.
    """
    adj_matrix = adj_matrix.astype(int)
    G = nx.from_numpy_array(adj_matrix)
    degree_centrality = nx.degree_centrality(G)
    clustering_coeff = nx.clustering(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        print("Power iteration failed to converge. Returning NaN for eigenvector centrality.")
        eigenvector_centrality = {node: float("nan") for node in G.nodes()}

    for node, centrality in eigenvector_centrality.items():
        if centrality < eigenvector_threshold:
            eigenvector_centrality[node] = 0.0
    degree = dict(G.degree())
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    metrics_df = pd.DataFrame(
        {
            "Degree": degree,
            "DegreeCentrality": degree_centrality,
            "ClusteringCoefficient": clustering_coeff,
            "EigenvectorCentrality": eigenvector_centrality,
            "BetweennessCentrality": betweenness_centrality,
            "ClosenessCentrality": closeness_centrality,
        }
    )
    metrics_df["Assortativity"] = assortativity

    return metrics_df


def print_trajectory_info(state, selected_option, action, reward, next_state, terminate, done):
    print("=" * 50)
    print("TRAJECTORY INFO".center(50, "="))
    print("=" * 50)
    print(f"State: {state}")
    print(f"Selected Option: {selected_option}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Next State: {next_state}")
    print(f"Terminate: {terminate}")
    print(f"Done: {done}")
    print("=" * 50)


def collect_experience(
    env: Any,
    model: torch.nn.Module,
    gamma: float = 0.99,
    lambda_parameter: float = 0.95,
    penalty_for_done_state: float = -1.0,
    tolerance: int = float("inf"),
    verbose: bool = False,
) -> tuple[List[tuple], List[float], List[float], List[float]]:
    """Gathers experience samples from an environment using a reinforcement learning model.

    Parameters
    ----------
    env : `Any`
        The environment to collect experience from.
    model : `torch.nn.Module`
        The reinforcement learning model (e.g., a torch neural network).
    gamma : float, optional
        Discount factor for future rewards, default=0.99.
    lambda_parameter : float, optional
        TD error correction parameter, default=0.95.
    penalty_for_done_state : float, optional
        Penalty applied to the state when the environment reaches a terminal state, default=-1.0.

    Returns
    -------
    trajectory : `list[tuple]`
        The return trajectory (state, selected_option, action, reward, next_state, terminate, done).
    returns : `list[float]`
        The list of cumulative returns.
    advantages : `list[float]`
        The list of advantage terms for each step.
    old_probs : `list[float]`
        The list of old policy probabilities for each step.
    """
    state = env.reset()
    done = False
    trajectory = []
    old_probs = []
    tolerance_count = 0

    while not done and tolerance_count < tolerance:
        state = state[0] if isinstance(state, tuple) else state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        option_probs, action_probs, termination_probs, selected_option, action = model(state_tensor)

        action = torch.multinomial(action_probs, 1).item()
        option = torch.multinomial(option_probs, 1).item()
        old_probs.append(action_probs[0, action].item())

        terminate = torch.bernoulli(termination_probs).item() > 0.5
        signature = env.step.__code__
        if signature.co_argcount > 2:
            next_state, reward, done, truncated, info = env.step(action, option)
        else:
            next_state, reward, done, truncated, info = env.step(action)

        if done:
            reward = penalty_for_done_state
        tolerance_count += 1
        trajectory.append((state, selected_option, action, reward, next_state, terminate, done))
        state = next_state
        if verbose:
            print_trajectory_info(
                state, selected_option, action, reward, next_state, terminate, done
            )

    returns = []
    advantages = []
    G = 0
    delta = 0

    for t in reversed(range(len(trajectory))):
        state, selected_option, action, reward, next_state, terminate, done = trajectory[t]

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        _, action_probs, _, _, _ = model(state_tensor)

        if t == len(trajectory) - 1:
            G = reward
            advantages.insert(0, G - action_probs[0, action].item())
        else:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            _, next_action_probs, _, _, _ = model(next_state_tensor)

            delta = (
                reward
                + gamma * next_action_probs[0, action].item()
                - action_probs[0, action].item()
            )
            G = reward + gamma * G
            advantages.insert(
                0, delta + gamma * lambda_parameter * advantages[0] if advantages else delta
            )

        returns.insert(0, G)

    return trajectory, returns, advantages, old_probs


def ppo_loss(
    advantages: torch.Tensor,
    old_action_probs: torch.Tensor,
    action_probs: torch.Tensor,
    epsilon: float = 0.2,
):
    """Computes the Proximal Policy Optimization (PPO) loss using the clipped objective.

    Parameters
    ----------
    advantages : `torch.Tensor`
        The advantages (delta) for each action taken, calculated as the difference between returns and value predictions.
    old_action_probs : `torch.Tensor`
        The action probabilities from the previous policy (before the current update).
    action_probs : `torch.Tensor`
        The action probabilities from the current policy (after the update).
    epsilon : `float`, optional, default=0.2
        The clipping parameter that limits how much the policy can change between updates.

    Returns
    -------
    loss : `torch.Tensor`
        The PPO loss, averaged across the batch of samples. The loss is computed using the clipped objective to penalize large policy updates.
    """

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    log_ratio = torch.log(action_probs + 1e-8) - torch.log(old_action_probs + 1e-8)
    ratio = torch.exp(log_ratio)  # π(a|s) / π_old(a|s)

    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss.mean()


def train_option_critic(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env: Any,
    num_epochs: int = 1_000,
    batch_size: int = 32,
    device: str = "cpu",
    beta: float = 1e-2,
    epsilon: float = 0.2,
    patience: int = 15,
    verbose: bool = False,
    **kwargs,
) -> tuple[torch.nn.Module, float]:
    """Trains an option critic model with the provided environment and hyperparameters.

    Parameters
    ----------
    model : `nn.Module`
        The neural network model to train.
    optimizer : `torch.optim.Optimizer`
        The optimizer for model updates.
    env : `Any`
        The environment for training.
    num_epochs : `int`
        Number of training epochs.
    batch_size : `int`
        Batch size per training step.
    device : `str`
        Target device (e.g., "cpu" or "cuda").
    beta : `float`
        Critic learning rate hyperparameter.
    epsilon : `float`, optional, default=0.2
        The clipping parameter that limits how much the policy can change between updates.
    patience : `int`
        Early stopping patience in epochs.

    Returns
    -------
    model : `nn.Module`
        Trained model.
    avg_epoch_loss : `float`
        Average loss per epoch over training.
    """
    losses = []
    best_loss_so_far = float("inf")
    best_advantage_so_far = 0.0
    patience_counter = 0
    patience_counter_advantage = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    advantages_per_epoch = []

    for epoch in range(num_epochs):
        trajectory, returns, advantages, old_probs = collect_experience(env, model, **kwargs)
        avg_advantage = sum(advantages) / len(advantages)
        advantages_per_epoch.append(avg_advantage)

        states = torch.tensor(np.array([t[0] for t in trajectory]), dtype=torch.float32).to(device)
        actions = torch.tensor([t[2] for t in trajectory], dtype=torch.long).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).view(-1, 1).to(device)

        dataset = TensorDataset(
            states, actions, returns_tensor, advantages_tensor, old_probs_tensor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_loss = 0
        num_batches = 0

        for (
            batch_states,
            batch_actions,
            batch_returns,
            batch_advantages,
            batch_old_probs,
        ) in dataloader:
            optimizer.zero_grad()

            option_probs, action_probs, termination_probs, selected_option, action = model(
                batch_states
            )

            batch_current_probs = action_probs.gather(1, batch_actions.unsqueeze(1))
            ppo_loss_value = ppo_loss(
                batch_advantages, batch_old_probs, batch_current_probs, epsilon=epsilon
            )

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)

            loss = ppo_loss_value + beta * entropy.mean()
            avg_advantages = batch_advantages.mean().item()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if avg_advantages > best_advantage_so_far:
                best_advantage_so_far = avg_advantages
                patience_counter_advantage = 0
            else:
                patience_counter_advantage += 1
            if patience_counter_advantage >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} after {patience} epochs without advantage improvement."
                    )
                break

            epoch_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            losses.append(avg_epoch_loss)

            if avg_epoch_loss < best_loss_so_far:
                best_loss_so_far = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} after {patience} epochs without improvement."
                    )
                break

            if verbose:
                if epoch % (num_epochs // 10) == 0:
                    print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")

    return model, avg_epoch_loss, advantages_per_epoch


def train_model_with_episodes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env: Any,
    num_episodes: int,
    episode_patience: int = 5,
    **kwargs,
):
    """Trains a model via reinforcement learning episodes.

    Parameters
    ----------
    model : `torch.nn.Module`
        The model to be trained.
    optimizer : `torch.optim.Optimizer`
        The optimizer for the model.
    env : `Any`
        The environment used for the episodes.
    num_episodes : `int`
        The number of episodes to train.

    Keyword Arguments
    -----------------
    Additional keyword arguments passed to the `train_option_critic` function.

    num_epochs : `int`
        Number of training epochs.
    batch_size : `int`
        Batch size per training step.
    gamma : `float`
        Discount factor for future rewards.
    device : `str`
        Target device (e.g., "cpu" or "cuda").
    beta : `float`
        Critic learning rate hyperparameter.
    patience : `int`
        Early stopping patience in epochs.

    Returns
    -------
    model : `torch.nn.Module`
        The trained model.
    best_loss_so_far : `float`
        The best loss value observed during training.
    """
    previous_weights = model.state_dict()
    best_loss_so_far = float("inf")
    loss_window = []
    average_loss = 0.0
    no_improvement_count = 0

    print(f"{'Episode':<12} {'Loss':<8} {'Best Loss':<17} {'Status':<15} {'Avg Loss':<4}")
    print("=" * 70)

    NEW_BEST_COLOR = "\033[92m"
    REVERT_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"
    advantages_per_episode = []

    for episode in range(num_episodes):
        model, loss, advantages = train_option_critic(model, optimizer, env, **kwargs)
        advantages_per_episode.extend(advantages)

        loss_window.append(loss)
        average_loss = sum(loss_window) / len(loss_window)

        if loss < best_loss_so_far:
            best_loss_so_far = loss
            previous_weights = model.state_dict()
            no_improvement_count = 0
            status = f"{NEW_BEST_COLOR}Updated{RESET_COLOR}"
        else:
            model.load_state_dict(previous_weights)
            no_improvement_count += 1
            status = f"{REVERT_COLOR}No Improvement{RESET_COLOR}"
        print(
            f"{episode + 1:<8} {loss:<12.4f} {best_loss_so_far:<15.4f} {status:<25} {average_loss:<12.4f}"
        )
        print("=" * 70)

        if no_improvement_count >= episode_patience:
            print(f"\nNo improvement for {episode_patience} episodes. Stopping early.")
            break

    print(f"\nTraining complete. Final best loss: {best_loss_so_far:.4f}")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 3))
    plt.plot(
        range(len(advantages_per_episode)),
        advantages_per_episode,
        marker=None,
        markersize=6,
        color=sns.color_palette("deep")[0],
        linestyle="-",
        linewidth=2,
    )
    plt.xscale("log")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Advantages", fontsize=12)
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

    return model, best_loss_so_far


if __name__ == "__main__":
    pass
