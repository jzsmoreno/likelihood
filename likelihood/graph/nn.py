import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress TensorFlow INFO logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from pandas.core.frame import DataFrame
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from likelihood.tools import generate_feature_yaml

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def compare_similarity(arr1: np.ndarray, arr2: np.ndarray) -> int:
    """Compares the similarity between two arrays of categories.

    Parameters
    ----------
    arr1 : `ndarray`
        The first array of categories.
    arr2 : `ndarray`
        The second array of categories.

    Returns
    -------
    count: `int`
        The number of categories that are the same in both arrays.
    """

    count = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            count += 1
    return count


def cal_adjacency_matrix(
    df: DataFrame, exclude_subset: List[str] = [], sparse: bool = True, **kwargs
) -> Tuple[dict, np.ndarray]:
    """Calculates the adjacency matrix for a given DataFrame.
    The adjacency matrix is a matrix that represents the similarity between each pair of categories.
    The similarity is calculated using the `compare_similarity` function.
    The resulting matrix is a square matrix with the same number of rows and columns as the input DataFrame.

    Parameters
    ----------
    df : `DataFrame`
        The input DataFrame containing the categories.
    exclude_subset : `List[str]`, optional
        A list of categories to exclude from the calculation of the adjacency matrix.
    sparse : `bool`, optional
        Whether to return a sparse matrix or a dense matrix.
    **kwargs : `dict`
        Additional keyword arguments to pass to the `compare_similarity` function.

    Keyword Arguments:
    ----------
    similarity: `int`
        The minimum number of categories that must be the same in both arrays to be considered similar.

    Returns
    -------
    adj_dict : `dict`
        A dictionary containing the categories.
    adjacency_matrix : `ndarray`
        The adjacency matrix.
    """

    yaml_ = generate_feature_yaml(df)
    categorical_columns = yaml_["categorical_features"]
    if len(exclude_subset) > 0:
        categorical_columns = [col for col in categorical_columns if col not in exclude_subset]

    if len(categorical_columns) > 1:
        df_categorical = df[categorical_columns].copy()
    else:
        categorical_columns = [
            col
            for col in df.columns
            if (
                col not in exclude_subset
                and pd.api.types.is_integer_dtype(df[col])
                and len(df[col].unique()) > 2
            )
        ]
        df_categorical = df[categorical_columns].copy()

    assert len(df_categorical) > 0

    similarity = kwargs["similarity"] if "similarity" in kwargs else len(df_categorical.columns) - 1
    assert similarity <= df_categorical.shape[1]

    adj_dict = {}
    for index, row in df_categorical.iterrows():
        adj_dict[index] = row.to_list()

    adjacency_matrix = np.zeros((len(df_categorical), len(df_categorical)))

    for i in range(len(df_categorical)):
        for j in range(len(df_categorical)):
            if compare_similarity(adj_dict[i], adj_dict[j]) >= similarity:
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
    ):
        _, adjacency = cal_adjacency_matrix(df, exclude_subset=exclude_subset, sparse=True)
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
    def __init__(self, dim_in, dim_out, kernel_initializer="glorot_uniform", **kwargs):
        super(VanillaGNNLayer, self).__init__(**kwargs)
        self.dim_out = dim_out
        self.kernel_initializer = kernel_initializer
        self.linear = None

    def build(self, input_shape):
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
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.linear.kernel_initializer
                ),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="VanillaGNN")
class VanillaGNN(tf.keras.Model):
    def __init__(self, dim_in, dim_h, dim_out, **kwargs):
        super(VanillaGNN, self).__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.gnn1 = VanillaGNNLayer(self.dim_in, self.dim_h)
        self.gnn2 = VanillaGNNLayer(self.dim_h, self.dim_h)
        self.gnn3 = VanillaGNNLayer(self.dim_h, self.dim_out)

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
        f1 = self.compute_f1_score(out, y)
        return loss.numpy(), f1

    def test(self, data):
        out = self(data.x, data.adjacency)
        test_f1 = self.compute_f1_score(out, data.y)
        return test_f1

    def predict(self, data):
        out = self(data.x, data.adjacency)
        return tf.argmax(out, axis=1, output_type=tf.int32).numpy()

    def get_config(self):
        config = {
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_out": self.dim_out,
        }
        base_config = super(VanillaGNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            dim_in=config["dim_in"],
            dim_h=config["dim_h"],
            dim_out=config["dim_out"],
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
        warnings.warn(
            "It is normal for validation metrics to underperform. Use the test method to validate after training.",
            UserWarning,
        )
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
                val_loss, val_f1 = self.evaluate(X_test, adjacency_test, y_test)
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1)
                clear_output(wait=True)
                print(
                    f"Epoch {epoch:>3} | Train Loss: {train_loss:.3f} | Train F1: {train_f1:.3f} | Val Loss: {val_loss:.3f} | Val F1: {val_f1:.3f}"
                )

        return train_losses, train_f1_scores, val_losses, val_f1_scores


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.datasets import load_iris

    # Load the dataset
    iris = load_iris()

    # Convert to a DataFrame for easy exploration
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target

    iris_df["sepal length (cm)"] = iris_df["sepal length (cm)"].astype("category")
    iris_df["sepal width (cm)"] = iris_df["sepal width (cm)"].astype("category")
    iris_df["petal length (cm)"] = iris_df["petal length (cm)"].astype("category")
    iris_df["petal width (cm)"] = iris_df["petal width (cm)"].astype("category")

    # Display the first few rows of the dataset
    print(iris_df.head())

    iris_df = iris_df.sample(frac=1, replace=False).reset_index(drop=True)

    data = Data(iris_df, "species")

    model = VanillaGNN(dim_in=data.x.shape[1], dim_h=8, dim_out=len(iris_df["species"].unique()))
    print("Before training F1:", model.test(data))
    model.fit(data, epochs=200, batch_size=32, test_size=0.5)
    model.save("./best_model", save_format="tf")
    print("After training F1:", model.test(data))
    best_model = tf.keras.models.load_model("./best_model")

    print("After loading F1:", best_model.test(data))
    df_results = pd.DataFrame()

    # Suppose we have a new dataset without the target variable
    iris_df = iris_df.drop(columns=["species"])
    data_new = Data(iris_df)
    print("Predictions:", best_model.predict(data_new))
    df_results["predicted"] = list(model.predict(data))
    df_results["actual"] = list(data.y)
    # df_results.to_csv("results.csv", index=False)
    breakpoint()
