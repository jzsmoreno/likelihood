import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from pandas.core.frame import DataFrame
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel("ERROR")

from likelihood.tools import LoRALayer


def compare_similarity(arr1: List[Any], arr2: List[Any], threshold: float = 0.05) -> int:
    """Calculate the similarity between two arrays considering numeric values near to 1 in ratio."""

    def is_similar(a: Any, b: Any) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a == 0 and b == 0:
                return True
            if a == 0 or b == 0:
                return False
            # For numeric values, check if their ratio is within the threshold range
            ratio = max(a, b) / min(a, b)
            return 1 - threshold <= ratio <= 1 + threshold
        else:
            return a == b

    return sum(is_similar(a, b) for a, b in zip(arr1, arr2))


def cal_adjacency_matrix(
    df: DataFrame, exclude_subset: List[str] = [], sparse: bool = True, **kwargs
) -> Tuple[dict, np.ndarray]:
    """Calculates the adjacency matrix for a given DataFrame.
    The adjacency matrix is a matrix that represents the similarity between each pair of features.
    The similarity is calculated using the `compare_similarity` function.
    The resulting matrix is a square matrix with the same number of rows and columns as the rows of the input DataFrame.

    Parameters
    ----------
    df : `DataFrame`
        The input DataFrame containing the features.
    exclude_subset : `List[str]`, optional
        A list of features to exclude from the calculation of the adjacency matrix.
    sparse : `bool`, optional
        Whether to return a sparse matrix or a dense matrix.
    **kwargs : `dict`
        Additional keyword arguments to pass to the `compare_similarity` function.

    Keyword Arguments:
    ----------
    similarity: `int`
        The minimum number of features that must be the same in both arrays to be considered similar.
    threshold : `float`
        The threshold value used in the `compare_similarity` function. Default is 0.05.

    Returns
    -------
    adj_dict : `dict`
        A dictionary containing the features.
    adjacency_matrix : `ndarray`
        The adjacency matrix.
    """

    if len(exclude_subset) > 0:
        columns = [col for col in df.columns if col not in exclude_subset]
        df_ = df[columns].copy()
    else:
        df_ = df.copy()

    assert len(df_) > 0

    similarity = kwargs.get("similarity", len(df_.columns) - 1)
    threshold = kwargs.get("threshold", 0.05)
    assert similarity <= df_.shape[1]

    adj_dict = {index: row.tolist() for index, row in df_.iterrows()}

    adjacency_matrix = np.zeros((len(df_), len(df_)))

    for i in range(len(df_)):
        for j in range(len(df_)):
            if compare_similarity(adj_dict[i], adj_dict[j], threshold=threshold) >= similarity:
                adjacency_matrix[i][j] = 1

    if sparse:
        num_nodes = adjacency_matrix.shape[0]

        indices = np.argwhere(adjacency_matrix != 0.0)
        indices = tf.constant(indices, dtype=tf.int64)
        values = tf.constant(adjacency_matrix[indices[:, 0], indices[:, 1]], dtype=tf.float32)
        adjacency_matrix = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=(num_nodes, num_nodes)
        )

        return adj_dict, adjacency_matrix
    else:
        return adj_dict, adjacency_matrix


class Data:
    def __init__(
        self,
        df: DataFrame,
        target: str | None = None,
        exclude_subset: List[str] = [],
        **kwargs,
    ):
        sparse = kwargs.get("sparse", True)
        threshold = kwargs.get("threshold", 0.05)
        _, adjacency = cal_adjacency_matrix(
            df, exclude_subset=exclude_subset, sparse=sparse, threshold=threshold
        )
        if target is not None:
            X = df.drop(columns=[target] + exclude_subset)
        else:
            X = df.drop(columns=exclude_subset)
        self.columns = X.columns
        X = X.to_numpy()
        self.x = np.asarray(X).astype(np.float32)
        self.adjacency = adjacency
        if target is not None:
            self.y = np.asarray(df[target].values).astype(np.int32)


@tf.keras.utils.register_keras_serializable(package="Custom", name="VanillaGNNLayer")
class VanillaGNNLayer(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, rank=None, kernel_initializer="glorot_uniform", **kwargs):
        super(VanillaGNNLayer, self).__init__(**kwargs)
        self.dim_out = dim_out
        self.rank = rank
        self.kernel_initializer = kernel_initializer
        self.linear = None

    def build(self, input_shape):
        if self.rank:
            self.linear = LoRALayer(self.dim_out, rank=self.rank)
        else:
            self.linear = tf.keras.layers.Dense(
                self.dim_out, use_bias=False, kernel_initializer=self.kernel_initializer
            )
        super(VanillaGNNLayer, self).build(input_shape)

    def call(self, x, adjacency):
        x = self.linear(x)
        x = tf.sparse.sparse_dense_matmul(adjacency, x)
        return x

    def get_config(self):
        config = super(VanillaGNNLayer, self).get_config()
        config.update(
            {
                "dim_out": self.dim_out,
                "rank": self.rank,
                "kernel_initializer": (
                    None
                    if self.rank
                    else tf.keras.initializers.serialize(self.linear.kernel_initializer)
                ),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="VanillaGNN")
class VanillaGNN(tf.keras.Model):
    def __init__(self, dim_in, dim_h, dim_out, rank=2, **kwargs):
        super(VanillaGNN, self).__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.rank = rank

        self.gnn1 = VanillaGNNLayer(self.dim_in, self.dim_h, self.rank)
        self.gnn2 = VanillaGNNLayer(self.dim_h, self.dim_h, self.rank)
        self.gnn3 = VanillaGNNLayer(self.dim_h, self.dim_out, None)

    def call(self, x, adjacency):
        h = self.gnn1(x, adjacency)
        h = tf.nn.tanh(h)
        h = self.gnn2(h, adjacency)
        h = self.gnn3(h, adjacency)
        return tf.nn.softmax(h, axis=1)

    def f1_macro(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    def compute_f1_score(self, logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        true_labels = tf.cast(labels, tf.int32)
        return self.f1_macro(true_labels.numpy(), predictions.numpy())

    def evaluate(self, x, adjacency, y):
        y = tf.cast(y, tf.int32)
        out = self(x, adjacency)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out)
        loss = tf.reduce_mean(loss)
        f1 = round(self.compute_f1_score(out, y), 4)
        return loss.numpy(), f1

    def test(self, data):
        out = self(data.x, data.adjacency)
        test_f1 = self.compute_f1_score(out, data.y)
        return round(test_f1, 4)

    def predict(self, data):
        out = self(data.x, data.adjacency)
        return tf.argmax(out, axis=1, output_type=tf.int32).numpy()

    def get_config(self):
        config = {
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_out": self.dim_out,
            "rank": self.rank,
        }
        base_config = super(VanillaGNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            dim_in=config["dim_in"],
            dim_h=config["dim_h"],
            dim_out=config["dim_out"],
            rank=config["rank"],
        )

    @tf.function
    def train_step(self, batch_x, batch_adjacency, batch_y, optimizer):
        with tf.GradientTape() as tape:
            out = self(batch_x, batch_adjacency)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=out)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def fit(self, data, epochs, batch_size, test_size=0.2, optimizer="adam"):
        optimizers = {
            "sgd": tf.keras.optimizers.SGD(),
            "adam": tf.keras.optimizers.Adam(),
            "adamw": tf.keras.optimizers.AdamW(),
            "adadelta": tf.keras.optimizers.Adadelta(),
            "rmsprop": tf.keras.optimizers.RMSprop(),
        }
        optimizer = optimizers[optimizer]
        train_losses = []
        train_f1_scores = []
        val_losses = []
        val_f1_scores = []

        X_train, X_test, y_train, y_test = train_test_split(
            data.x, data.y, test_size=test_size, shuffle=False
        )
        adjacency_train = tf.sparse.slice(data.adjacency, [0, 0], [len(X_train), len(X_train)])
        adjacency_test = tf.sparse.slice(
            data.adjacency, [len(X_train), 0], [len(X_test), len(X_test)]
        )

        batch_starts = np.arange(0, len(X_train), batch_size)
        for epoch in range(epochs):
            np.random.shuffle(batch_starts)
            for start in batch_starts:
                end = start + batch_size
                batch_x = X_train[start:end, :]
                batch_adjacency = tf.sparse.slice(
                    adjacency_train, [start, start], [batch_size, batch_size]
                )
                batch_y = y_train[start:end]
                train_loss = self.train_step(batch_x, batch_adjacency, batch_y, optimizer)

            train_loss, train_f1 = self.evaluate(X_train, adjacency_train, y_train)
            train_losses.append(train_loss)
            train_f1_scores.append(train_f1)

            if epoch % 5 == 0:
                clear_output(wait=True)
                warnings.warn(
                    "It is normal for validation metrics to underperform during training. Use the test method to validate after training.",
                    UserWarning,
                )
                val_loss, val_f1 = self.evaluate(X_test, adjacency_test, y_test)
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1)
                print(
                    f"Epoch {epoch:>3} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
                )

        return train_losses, train_f1_scores, val_losses, val_f1_scores


if __name__ == "__main__":
    print("Examples will be running below")
