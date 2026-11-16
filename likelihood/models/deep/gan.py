"""Generative adversarial network model definitions."""

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from IPython.display import clear_output
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

from likelihood.tools import get_metrics

from ._keras_io import load_keras_model, save_keras_model


@tf.keras.utils.register_keras_serializable(package="Custom", name="GANRegressor")
class GANRegressor(tf.keras.Model):
    """GANRegressor is a custom Keras model that combines a generator and a discriminator."""

    def __init__(
        self,
        input_shape_parm: int,
        output_shape_parm: int,
        num_neurons: int = 128,
        activation: str = "linear",
        depth: int = 5,
        dropout: float = 0.2,
        l2_reg: float = 0.0,
        **kwargs,
    ) -> None:
        super(GANRegressor, self).__init__()
        self.input_shape_parm = input_shape_parm
        self.output_shape_parm = output_shape_parm
        self.num_neurons = num_neurons
        self.activation = activation
        self.depth = depth
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.optimizer = kwargs.get("optimizer", "adam")

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        dummy_input = tf.convert_to_tensor(tf.random.normal([1, self.input_shape_parm]))
        self.build(dummy_input.shape)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the GAN model architecture.

        Parameters
        ----------
        input_shape : `Tuple[int, ...]`
            The shape of the input tensor.

        Returns
        -------
        None
        """
        self.gan = tf.keras.models.Sequential([self.generator, self.discriminator], name="gan")

        self._compile_models()
        super(GANRegressor, self).build(input_shape)

    def _compile_models(self) -> None:
        """Compile the generator, the discriminator and the adversarial model.

        ``build`` uses it for a fresh instance and :meth:`load_model` after
        restoring an archive, so a loaded model can keep training.
        """
        self.generator.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
        )

        self.discriminator.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        self.gan.compile(optimizer=self.optimizer, loss="binary_crossentropy")

    def _build_generator(self) -> tf.keras.Model:
        generator = tf.keras.Sequential(name="generator")
        generator.add(
            tf.keras.layers.Dense(
                self.num_neurons,
                activation="selu",
                input_shape=[self.input_shape_parm],
                kernel_regularizer=l2(self.l2_reg),
            )
        )
        generator.add(tf.keras.layers.Dropout(self.dropout))
        for _ in range(self.depth - 1):
            generator.add(
                tf.keras.layers.Dense(
                    self.num_neurons, activation="selu", kernel_regularizer=l2(self.l2_reg)
                ),
            )
            generator.add(tf.keras.layers.Dropout(self.dropout))
        generator.add(tf.keras.layers.Dense(2 * self.output_shape_parm, activation=self.activation))
        return generator

    def _build_discriminator(self) -> tf.keras.Model:
        discriminator = tf.keras.Sequential(name="discriminator")
        for _ in range(self.depth):
            discriminator.add(
                tf.keras.layers.Dense(
                    self.num_neurons, activation="selu", kernel_regularizer=l2(self.l2_reg)
                ),
            )
            discriminator.add(tf.keras.layers.Dropout(self.dropout))
        discriminator.add(tf.keras.layers.Dense(2, activation="softmax"))
        return discriminator

    def train_gan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        n_epochs: int,
        validation_split: float = 0.2,
        verbose: int = 1,
    ) -> pd.DataFrame:
        """Train the GAN model.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target data.
        batch_size : int
            Number of samples in each batch.
        n_epochs : int
            Number of training epochs.
        validation_split : float, optional
            Fraction of the data to be used for validation.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        history : pd.DataFrame
            Training history.
        """
        loss_history = []
        for epoch in tqdm(range(n_epochs)):
            batch_starts = np.arange(0, len(X), batch_size)
            for start in batch_starts:
                np.random.shuffle(batch_starts)
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end].reshape(-1, self.output_shape_parm)
                y_batch = np.concatenate((y_batch, y_batch**2), axis=1)
                X_batch = tf.cast(X_batch, "float32")
                noise = tf.random.normal(
                    shape=X_batch.shape, stddev=tf.math.reduce_std(X_batch, keepdims=False)
                )

                # Phase 1 - training the generator
                self.generator.train_on_batch(X_batch, y_batch)

                # Phase 2 - training the discriminator
                generated_y_fake = self.generator(noise)
                generated_y_real = self.generator(X_batch)
                fake_and_real = tf.concat([generated_y_fake, generated_y_real], axis=0)
                X_fake_and_real = tf.concat([noise, X_batch], axis=0)
                batch_size = int(fake_and_real.shape[0] / 2)
                indices_ = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)[:, 0]
                indices_ = tf.cast(indices_, "int32")
                y1 = tf.one_hot(indices_, 2)
                self.gan.train_on_batch(X_fake_and_real, y1)

            loss = self._cal_loss(generated_y_real, y_batch)
            loss_history.append([epoch, loss])

        if verbose:
            X_batch, y_batch, X_batch_val, y_batch_val = self._train_and_val(
                X_batch, y_batch, validation_split=validation_split
            )
            generated_y = self.generator(X_batch)
            generated_y_val = self.generator(X_batch_val)
            y_pred = self.discriminator.predict(fake_and_real, verbose=0)
            y_pred = list(np.argmax(y_pred, axis=1))

            metrics = get_metrics(self._get_frame(indices_.numpy().tolist(), y_pred), "y", "y_pred")
            loss = self._cal_loss(generated_y, y_batch)
            loss_val = self._cal_loss(generated_y_val, y_batch_val)
            clear_output(wait=True)
            metrics_list = [
                ("Epoch", f"{epoch}"),
                ("Loss", f"{loss:.2f} / {loss_val:.2f}"),
                ("Accuracy", f"{metrics['accuracy']:.2f} / {metrics['accuracy']:.2f}"),
                ("Precision", f"{metrics['precision']:.2f} / {metrics['precision']:.2f}"),
                ("Recall", f"{metrics['recall']:.2f} / {metrics['recall']:.2f}"),
                ("F1 Score", f"{metrics['f1_score']:.2f} / {metrics['f1_score']:.2f}"),
                ("Kappa", f"{metrics['kappa']:.2f} / {metrics['kappa']:.2f}"),
            ]

            metric_width = 15
            value_width = 30

            header = f"| {'Metric':<{metric_width}} | {'Value':<{value_width}} |"
            separator = "+" + "-" * (len(header) - 2) + "+"

            print(separator)
            print(header)
            print(separator)

            for metric_name, metric_values in metrics_list:
                data_row = f"| {metric_name:<{metric_width}} | {metric_values:<{value_width}} |"
                print(data_row)

            print(separator)

        return pd.DataFrame(loss_history, columns=["epoch", "loss"])

    def _get_frame(self, y: np.ndarray, y_pred: list) -> pd.DataFrame:
        df = pd.DataFrame()
        df["y"] = y
        df["y_pred"] = y_pred
        return df

    def _train_and_val(
        self, X: tf.Tensor, y: tf.Tensor, validation_split: float
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        split = int((1 - validation_split) * len(X))

        if len(X) > split and split > 0:
            X_train = X[:split]
            y_train = y[:split]
            X_val = X[split:]
            y_val = y[split:]
        else:
            X_train = X
            y_train = y
            X_val = X
            y_val = y

        X_train = tf.cast(X_train, "float32")
        X_val = tf.cast(X_val, "float32")

        return X_train, y_train, X_val, y_val

    def _cal_loss(self, generated: tf.Tensor, y: tf.Tensor) -> float:
        return tf.math.reduce_mean(100 * abs((y - generated) / y), keepdims=False).numpy().item()

    def train_gen(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        n_epochs: int,
        validation_split: float = 0.2,
        patience: int = 3,
    ) -> pd.DataFrame:
        """Train the generator model.

        Parameters
        ----------
        X_train : array-like
            Training data.
        y_train : array-like
            Training target data.
        batch_size : int
            Batch size for training.
        n_epochs : int
            Number of epochs for training.
        validation_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
        patience : int, optional
            Number of epochs to wait before early stopping. Default is 3.

        Returns
        -------
        history : pd.DataFrame
            Training history.
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
        # Prepare the target by extending it with its square
        self.discriminator.trainable = False
        y_train_extended = np.concatenate(
            (
                y_train.reshape(-1, self.output_shape_parm),
                y_train.reshape(-1, self.output_shape_parm) ** 2,
            ),
            axis=1,
        )

        history = self.generator.fit(
            X_train,
            y_train_extended,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=validation_split,
            callbacks=[callback],
        )

        return pd.DataFrame(history.history)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute the output tensor by passing inputs through the generator.

        Parameters
        ----------
        inputs : `tf.Tensor`
            Input tensor to the model.

        Returns
        -------
        `tf.Tensor`
            Output tensor of shape (batch_size, 1).
        """
        return self.generator(inputs)[:, 0]

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.

        Only JSON friendly constructor arguments are returned: the generator,
        discriminator and adversarial models are rebuilt by ``__init__`` from
        them, which keeps both the ``.keras`` archive and the SavedModel format
        of TensorFlow <= 2.15 serializable.

        Returns
        -------
        config : `Dict[str, Any]`
            Configuration dictionary.
        """
        config = {
            "input_shape_parm": self.input_shape_parm,
            "output_shape_parm": self.output_shape_parm,
            "num_neurons": self.num_neurons,
            "activation": self.activation,
            "depth": self.depth,
            "dropout": self.dropout,
            "l2_reg": self.l2_reg,
            "optimizer": self.optimizer,
        }
        base_config = super(GANRegressor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GANRegressor":
        """Reconstruct an instance from its configuration dictionary.

        Parameters
        ----------
        config : `Dict[str, Any]`
            Configuration dictionary containing the instance's parameters.

        Returns
        -------
        `GANRegressor`
            Reconstructed instance.
        """
        return cls(
            input_shape_parm=config["input_shape_parm"],
            output_shape_parm=config["output_shape_parm"],
            num_neurons=config.get("num_neurons", 128),
            activation=config.get("activation", "linear"),
            depth=config.get("depth", 5),
            dropout=config.get("dropout", 0.2),
            l2_reg=config.get("l2_reg", 0.0),
            optimizer=config.get("optimizer", "adam"),
        )

    def save_model(self, path: str = "./gan_regressor") -> None:
        """Serialize the GAN using the format supported by the installed TF.

        TensorFlow > 2.15 writes a ``.keras`` archive (the extension is appended
        when ``path`` does not have it); TensorFlow <= 2.15 writes a SavedModel
        directory named after ``path``.

        Parameters
        ----------
        path : str, default="./gan_regressor"
            Destination file or directory, depending on the TensorFlow version.

        Notes
        -----
        The artifact can also be produced with the matching Keras call
        (``model.save("name.keras")`` on TensorFlow > 2.15,
        ``model.save("name", save_format="tf")`` on older releases);
        :meth:`load_model` accepts both formats.
        """
        save_keras_model(self, path)

    @classmethod
    def load_model(cls, path: str = "./gan_regressor") -> "GANRegressor":
        """Rebuild a GAN regressor saved with :meth:`save_model`.

        Parameters
        ----------
        path : str, default="./gan_regressor"
            Keras archive, the same name without the extension, or a SavedModel
            directory written by TensorFlow <= 2.15.

        Returns
        -------
        GANRegressor
            Model with the stored generator, discriminator and optimizer
            settings, recompiled so it can keep training.

        Raises
        ------
        TypeError
            If the artifact does not contain a :class:`GANRegressor`.
        """
        restored = load_keras_model(path, {cls.__name__: cls}, compile=False)
        if not isinstance(restored, cls):
            raise TypeError(f"{path!r} does not contain a {cls.__name__} archive")
        restored._compile_models()
        return restored
