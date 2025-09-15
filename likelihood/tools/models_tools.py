import logging
import os

import networkx as nx
import pandas as pd
from pandas.core.frame import DataFrame

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import sys
import warnings
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from pandas.core.frame import DataFrame

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

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializes the class with the original DataFrame.

        Parameters
        ----------
        df : `pd.DataFrame`
            The original DataFrame to transform.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()  # Create a copy to avoid modifying the original

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

        labels = [f"{int(bins[i])}-{int(bins[i+1] - 1)}" for i in range(len(bins) - 1)]
        return bins, labels

    def _transform_column_to_ranges(self, column: str, bin_size: int) -> pd.Series:
        """
        Transforms a column in the DataFrame into range bins.

        Parameters
        ----------
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
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        if bin_size <= 0:
            raise ValueError("bin_size must be positive")

        numeric_series = pd.to_numeric(self.df[column], errors="coerce")
        if numeric_series.isna().all():
            raise ValueError(f"Column '{column}' contains no valid numeric data")

        min_val = numeric_series.min()
        max_val = numeric_series.max()

        if min_val == max_val:
            return pd.Series(
                [f"{int(min_val)}-{int(max_val)}"] * len(self.df), name=f"{column}_range"
            )

        bins, labels = self._create_bins_and_labels(min_val, max_val, bin_size)

        return pd.cut(numeric_series, bins=bins, labels=labels, right=False, include_lowest=True)

    def transform_dataframe(
        self, columns_bin_sizes: Dict[str, int], drop_original: bool = False
    ) -> pd.DataFrame:
        """
        Creates a new DataFrame with range columns.

        Parameters
        ----------
        columns_bin_sizes : `dict`
            A dictionary where the keys are column names and the values are the bin sizes.
        drop_original : `bool`, optional
            If True, drops original columns from the result, by default False

        Returns
        -------
        `pd.DataFrame`
            A DataFrame with the transformed range columns.

        Raises
        ------
        TypeError
            If columns_bin_sizes is not a dictionary.
        """
        if not isinstance(columns_bin_sizes, dict):
            raise TypeError("columns_bin_sizes must be a dictionary")

        if not columns_bin_sizes:
            return pd.DataFrame()

        range_columns = {}
        for column, bin_size in columns_bin_sizes.items():
            range_columns[f"{column}_range"] = self._transform_column_to_ranges(column, bin_size)

        result_df = pd.DataFrame(range_columns)

        if not drop_original:
            original_cols = [col for col in self.df.columns if col not in columns_bin_sizes]
            if original_cols:
                result_df = pd.concat([self.df[original_cols], result_df], axis=1)

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


def remove_collinearity(df: DataFrame, threshold: float = 0.9):
    """
    Removes highly collinear features from the DataFrame based on a correlation threshold.

    This function calculates the correlation matrix of the DataFrame and removes columns
    that are highly correlated with any other column in the DataFrame. It uses an absolute
    correlation value greater than the specified threshold to identify which columns to drop.

    Parameters
    ----------
    df : `DataFrame`
        The input DataFrame containing numerical data.
    threshold : `float`
        The correlation threshold above which features will be removed. Default is `0.9`.

    Returns
    -------
        DataFrame : A DataFrame with highly collinear features removed.
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

    Keyword Arguments:
    ------------------
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


def graph_metrics(adj_matrix: np.ndarray, eigenvector_threshold: float = 1e-6) -> DataFrame:
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
    DataFrame : A DataFrame containing the following graph metrics as columns.
        - `Degree Centrality`: Degree centrality values for each node, indicating the number of direct connections each node has.
        - `Clustering Coefficient`: Clustering coefficient values for each node, representing the degree to which nodes cluster together.
        - `Eigenvector Centrality`: Eigenvector centrality values, indicating the influence of a node in the graph based on the eigenvectors of the adjacency matrix.
        - `Degree`: The degree of each node, representing the number of edges connected to each node.
        - `Betweenness Centrality`: Betweenness centrality values, representing the extent to which a node lies on the shortest paths between other nodes.
        - `Closeness Centrality`: Closeness centrality values, indicating the inverse of the average shortest path distance from a node to all other nodes in the graph.
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
            "Degree Centrality": degree_centrality,
            "Clustering Coefficient": clustering_coeff,
            "Eigenvector Centrality": eigenvector_centrality,
            "Betweenness Centrality": betweenness_centrality,
            "Closeness Centrality": closeness_centrality,
        }
    )
    metrics_df["Assortativity"] = assortativity

    return metrics_df


if __name__ == "__main__":
    pass
