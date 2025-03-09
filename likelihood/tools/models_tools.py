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

import tensorflow as tf
from numpy import ndarray

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


def remove_collinearity(df: DataFrame, threshold: float = 0.9):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]
    df_reduced = df.drop(columns=to_drop)

    return df_reduced


def train_and_insights(
    x_data: ndarray,
    y_act: ndarray,
    model: tf.keras.Model,
    patience: int = 3,
    reg: bool = False,
    **kwargs,
):
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
        y_pred = model.predict(x_data)
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


def graph_metrics(adj_matrix, eigenvector_threshold=1e-6):
    """
    This function calculates the following graph metrics using the adjacency matrix:
    1. Degree Centrality
    2. Clustering Coefficient
    3. Eigenvector Centrality
    4. Degree
    5. Betweenness Centrality
    6. Closeness Centrality
    7. Assortativity
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
