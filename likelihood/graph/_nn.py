import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os
from typing import Any, Dict, List, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from IPython.display import clear_output
from sklearn.metrics import f1_score

from likelihood.tools import LoRALayer

from .nn import Data


@tf.keras.utils.register_keras_serializable(package="Custom", name="VanillaGNNLayer")
class VanillaGNNLayer(tf.keras.layers.Layer):
    """A vanilla graph neural network layer.

    Parameters
    ----------
    dim_in : `int`
        The number of input features.
    dim_out : `int`
        The number of output features.
    rank : `int` or `None`
        The rank for the LoRA adapter. ``None`` disables LoRA.
    kernel_initializer : `str`
        The kernel initializer for the dense layer.
    **kwargs : `dict`
        Additional keyword arguments passed to :class:`tf.keras.layers.Layer`.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        rank: int = None,
        kernel_initializer: str = "glorot_uniform",
        **kwargs,
    ) -> None:
        super(VanillaGNNLayer, self).__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        self.kernel_initializer = kernel_initializer
        self.linear = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer architecture based on the rank.

        Parameters
        ----------
        input_shape : `Tuple[int, ...]`
            The shape of the input tensors.
        """
        if self.rank:
            self.linear = LoRALayer(self.dim_out, rank=self.rank)
        else:
            self.linear = tf.keras.layers.Dense(
                self.dim_out, use_bias=False, kernel_initializer=self.kernel_initializer
            )
        super(VanillaGNNLayer, self).build(input_shape)

    def call(self, x: np.ndarray, adjacency: tf.SparseTensor) -> np.ndarray:
        """Forward pass of the layer over the graph.

        Parameters
        ----------
        x : `np.ndarray`
            The input node features.
        adjacency : `tf.SparseTensor`
            The graph adjacency matrix.

        Returns
        -------
        `np.ndarray`
            The layer output.
        """
        x = self.linear(x)
        x = tf.sparse.sparse_dense_matmul(adjacency, x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for serialization.

        Returns
        -------
        config : `Dict[str, Any]`
            The configuration dictionary containing layer parameters including
            ``dim_in``, ``dim_out``, ``rank``, and ``kernel_initializer``.
        """

        config = super(VanillaGNNLayer, self).get_config()
        config.update(
            {
                "dim_in": self.dim_in,
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VanillaGNNLayer":
        """Reconstruct the layer from its serialized config.

        Parameters
        ----------
        config : `Dict[str, Any]`
            The serialized configuration of the layer.

        Returns
        -------
        `VanillaGNNLayer`
            The reconstructed layer.
        """
        if config.get("kernel_initializer") is not None:
            config["kernel_initializer"] = tf.keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        return cls(**config)


class VanillaGNN:
    """A vanilla graph neural network model.

    Parameters
    ----------
    dim_in : `int`
        The number of input features.
    dim_h : `int`
        The number of hidden features.
    dim_out : `int`
        The number of output features.
    rank : `int`
        The rank for the LoRA adapter.
    **kwargs : `dict`
        Additional keyword arguments passed to the model.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int, rank: int = 2, **kwargs) -> None:
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.rank = rank

        self.gnn1 = VanillaGNNLayer(self.dim_in, self.dim_h, self.rank)
        self.gnn2 = VanillaGNNLayer(self.dim_h, self.dim_h, self.rank)
        self.gnn3 = VanillaGNNLayer(self.dim_h, self.dim_out, None)

        self.build()

    def build(self) -> None:
        """Build the functional model architecture."""
        x_in = tf.keras.Input(shape=(self.dim_in,), name="node_features")
        adjacency_in = tf.keras.Input(shape=(None,), sparse=True, name="adjacency")

        gnn1 = VanillaGNNLayer(self.dim_in, self.dim_h, self.rank)
        gnn2 = VanillaGNNLayer(self.dim_h, self.dim_h, self.rank)
        gnn3 = VanillaGNNLayer(self.dim_h, self.dim_out, rank=None)

        h = gnn1(x_in, adjacency_in)
        h = tf.keras.activations.tanh(h)
        h = gnn2(h, adjacency_in)
        h = gnn3(h, adjacency_in)
        out = tf.keras.activations.softmax(h, axis=-1)

        self.model = tf.keras.Model(
            inputs=[x_in, adjacency_in], outputs=out, name="VanillaGNN_Functional"
        )

    @tf.function
    def __call__(self, x: np.ndarray, adjacency: tf.SparseTensor) -> np.ndarray:
        """Forward pass over the model.

        Parameters
        ----------
        x : `np.ndarray`
            The node feature matrix.
        adjacency : `tf.SparseTensor`
            The graph adjacency matrix.

        Returns
        -------
        `np.ndarray`
            The model output.
        """
        return self.model([x, adjacency])

    def f1_macro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the macro-averaged F1 score.

        Parameters
        ----------
        y_true : `np.ndarray`
            The ground-truth labels.
        y_pred : `np.ndarray`
            The predicted labels.

        Returns
        -------
        `float`
            The macro-averaged F1 score.
        """
        return f1_score(y_true, y_pred, average="macro")

    def compute_f1_score(self, logits: tf.Tensor, labels: tf.Tensor) -> float:
        """Compute the F1 score from model logits.

        Parameters
        ----------
        logits : `tf.Tensor`
            The model logits.
        labels : `tf.Tensor`
            The ground-truth labels.

        Returns
        -------
        `float`
            The macro-averaged F1 score.
        """
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        true_labels = tf.cast(labels, tf.int32)
        return self.f1_macro(true_labels.numpy(), predictions.numpy())

    def evaluate(
        self, x: np.ndarray, adjacency: tf.SparseTensor, y: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate the model on the given inputs.

        Parameters
        ----------
        x : `np.ndarray`
            The node feature matrix.
        adjacency : `tf.SparseTensor`
            The graph adjacency matrix.
        y : `np.ndarray`
            The ground-truth labels.

        Returns
        -------
        `Tuple[float, float]`
            The mean loss and the F1 score.
        """
        y = tf.cast(y, tf.int32)
        out = self(x, adjacency)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out)
        loss = tf.reduce_mean(loss)
        f1 = round(self.compute_f1_score(out, y), 4)
        return loss.numpy(), f1

    def test(self, data: Data) -> float:
        """Evaluate the model on the test split of the data.

        Parameters
        ----------
        data : `Data`
            The dataset containing the test split.

        Returns
        -------
        `float`
            The macro-averaged F1 score on the test split.
        """
        data.x = tf.convert_to_tensor(data.x) if not tf.is_tensor(data.x) else data.x
        out = self(data.x, data.adjacency)
        test_f1 = self.compute_f1_score(out, data.y)
        return round(test_f1, 4)

    def predict(self, data: Data) -> np.ndarray:
        """Predict the labels for the test split of the data.

        Parameters
        ----------
        data : `Data`
            The dataset containing the test split.

        Returns
        -------
        `np.ndarray`
            The predicted labels for the test split.
        """
        data.x = tf.convert_to_tensor(data.x) if not tf.is_tensor(data.x) else data.x
        out = self(data.x, data.adjacency)
        return tf.argmax(out, axis=1, output_type=tf.int32).numpy()

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the complete model including all components.

        Parameters
        ----------
        filepath : str
            Path where to save the model.
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)

        self.model.save(os.path.join(filepath, "main_model.keras"))

        # Save configuration
        import json

        config = self.get_config()

        with open(os.path.join(filepath, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "VanillaGNN":
        """
        Load a complete model from saved components.

        Parameters
        ----------
        filepath : str
            Path where the model was saved.

        Returns
        -------
        VanillaGNN
            The loaded model instance.
        """
        import json
        import os

        # Load configuration
        with open(os.path.join(filepath, "config.json"), "r") as f:
            config = json.load(f)

        # Create new instance
        instance = cls(**config)

        instance.model = tf.keras.models.load_model(os.path.join(filepath, "main_model.keras"))

        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for serialization.

        Returns
        -------
        `Dict[str, Any]`
            The configuration dictionary containing model parameters including
            ``dim_in``, ``dim_h``, ``dim_out``, and ``rank``.
        """
        return {
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_out": self.dim_out,
            "rank": self.rank,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VanillaGNN":
        """Reconstruct the model from its config.

        Parameters
        ----------
        config : `Dict[str, Any]`
            The serialized configuration of the model.

        Returns
        -------
        `VanillaGNN`
            The reconstructed model.
        """
        return cls(
            dim_in=config["dim_in"],
            dim_h=config["dim_h"],
            dim_out=config["dim_out"],
            rank=config["rank"],
        )

    def get_build_config(self) -> Dict[str, Any]:
        """Get the configuration needed to build the model.

        Returns
        -------
        `Dict[str, Any]`
            The configuration of the model.
        """
        config = {
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_out": self.dim_out,
            "rank": self.rank,
        }
        return config

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "VanillaGNN":
        """Build the model from its configuration.

        Parameters
        ----------
        config : `Dict[str, Any]`
            The configuration of the model.

        Returns
        -------
        `VanillaGNN`
            The built model.
        """
        return cls(**config)

    @tf.function
    def train_step(
        self,
        batch_x: np.ndarray,
        batch_adjacency: tf.SparseTensor,
        batch_y: np.ndarray,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> tf.Tensor:
        """Perform a single training step on a batch of data.

        Parameters
        ----------
        batch_x : `np.ndarray`
            The input node features.
        batch_adjacency : `tf.SparseTensor`
            The graph adjacency matrix.
        batch_y : `np.ndarray`
            The ground-truth labels.
        optimizer : `tf.keras.optimizers.Optimizer`
            The optimizer to use.

        Returns
        -------
        `tf.Tensor`
            The loss value.
        """
        with tf.GradientTape() as tape:
            out = self(batch_x, batch_adjacency)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=out)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def fit(
        self,
        data: Data,
        epochs: int,
        batch_size: int,
        test_size: float = 0.2,
        optimizer: str = "adam",
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Fit the model on the dataset.

        Parameters
        ----------
        data : `Data`
            The dataset to train on.
        epochs : `int`
            The number of training epochs.
        batch_size : `int`
            The batch size for training.
        test_size : `float`
            The proportion of the dataset to use for testing. Default is 0.2.
        optimizer : `str`
            The optimizer to use. Default is "adam".

        Returns
        -------
        `Tuple[List[float], List[float], List[float], List[float]]`
            The lists of train losses, train F1 scores, val losses, and val F1 scores.
        """
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

        num_nodes = len(data.x)
        split_index = int((1 - test_size) * num_nodes)

        X_train, X_test = data.x[:split_index], data.x[split_index:]
        y_train, y_test = data.y[:split_index], data.y[split_index:]

        adjacency_train = tf.sparse.slice(data.adjacency, [0, 0], [split_index, split_index])
        adjacency_test = tf.sparse.slice(
            data.adjacency,
            [split_index, split_index],
            [num_nodes - split_index, num_nodes - split_index],
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
                val_loss, val_f1 = self.evaluate(X_test, adjacency_test, y_test)
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1)
                print(
                    f"Epoch {epoch:>3} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
                )

        return train_losses, train_f1_scores, val_losses, val_f1_scores


if __name__ == "__main__":
    print("Examples will be running below")
