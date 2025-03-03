import logging
import os

import networkx as nx
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf


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
